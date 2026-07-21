"""Model-agnostic adapter for the refusal / poetry direction pipelines.

The pipelines need only a narrow, backend-specific surface: chat formatting,
the end-of-instruction token window, refusal-token resolution, raw-activation
mean capture, last-position logits (under an open steering session), and plain
generation. ``DirectionModel`` is that surface; ``GemmaDirectionModel`` and
``QwenDirectionModel`` implement it over the two inference wrappers. Steering
itself stays on ``HookManager`` — it already takes ``decoder_layers``.

``build_direction_model(model_name)`` dispatches on the model id. Wrapper
imports are lazy so loading the Qwen path never imports the Gemma fork (and
vice versa).
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from tqdm import tqdm

from interpret.experiments.directions_common.sites import QWEN_SITE_MAP, CaptureSite

# Canonical refusal-cue words. The reference scores P(first response token) ==
# "I" / "As" (refusal openers). Token IDs differ per tokenizer, so each backend
# resolves these words to ids rather than hardcoding them.
REFUSAL_WORDS: tuple[str, ...] = ("I", "As")


@runtime_checkable
class DirectionModel(Protocol):
    """Backend surface the direction pipelines depend on."""

    n_layers: int
    d_model: int

    @property
    def decoder_layers(self) -> nn.ModuleList: ...

    def format_chat(self, instruction: str) -> str: ...

    def eoi_token_ids(self) -> list[int]: ...

    def refusal_token_ids(self, configured: tuple[int, ...]) -> tuple[int, ...]: ...

    def capture_means(
        self,
        instructions: list[str],
        sites: tuple[CaptureSite, ...],
        n_eoi: int,
    ) -> dict[str, torch.Tensor]: ...

    def last_position_logits(self, prompt: str) -> torch.Tensor: ...

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...


class _BaseDirectionModel:
    """Shared plumbing: decoder-layer access, layer count, and generation."""

    def __init__(self, wrapper) -> None:
        self.wrapper = wrapper

    @property
    def decoder_layers(self) -> nn.ModuleList:
        return self.wrapper.decoder_layers

    @property
    def n_layers(self) -> int:
        return len(self.wrapper.decoder_layers)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Greedy completion of `prompt` (chat-formatted). Honours any open session."""
        return self.wrapper.generate_from_template(
            self.format_chat(prompt),
            output_len=max_new_tokens,
            temperature=None,
        )

    # Subclasses must provide these.
    def format_chat(self, instruction: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


class GemmaDirectionModel(_BaseDirectionModel):
    """``DirectionModel`` over ``GemmaPytorchInference`` (Gemma-3, fork PyTorch).

    Behaviour is byte-for-byte equivalent to the pre-refactor Gemma pipeline:
    same chat template, same fork-cache capture loop, same final-norm logit
    path.
    """

    EOI_TEMPLATE_SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"

    def __init__(self, model_name: str, wrapper=None) -> None:
        if wrapper is None:
            from interpret.inference.gemma_pytorch import GemmaPytorchInference

            wrapper = GemmaPytorchInference(model_name)
        super().__init__(wrapper)

    @property
    def d_model(self) -> int:
        return int(self.wrapper.model.text_token_embedder.weight.shape[1])

    def format_chat(self, instruction: str) -> str:
        # ``format_prompt`` includes the trailing newline after
        # ``<start_of_turn>model`` so EOI_TEMPLATE_SUFFIX is exactly the
        # suffix the model sees (always fed via generate_from_template).
        return self.wrapper.format_prompt(instruction)

    def eoi_token_ids(self) -> list[int]:
        return self.wrapper.tokenize(self.EOI_TEMPLATE_SUFFIX, bos=False)

    def refusal_token_ids(self, configured: tuple[int, ...]) -> tuple[int, ...]:
        """Verify the configured ids map to ``"I"``; warn on drift, return as-is.

        Preserves the original ``verify_refusal_tokens`` behaviour exactly.
        """
        expected = self.wrapper.tokenize("I", bos=False)
        if tuple(expected) != tuple(configured):
            warnings.warn(
                f"Refusal token mismatch: configured {configured}, tokeniser "
                f"produced {expected} for 'I'. Update refusal_token_ids if wrong.",
                stacklevel=2,
            )
        return configured

    def capture_means(
        self,
        instructions: list[str],
        sites: tuple[CaptureSite, ...],
        n_eoi: int,
    ) -> dict[str, torch.Tensor]:
        n_layers, d_model = self.n_layers, self.d_model
        n_samples = len(instructions)
        if n_samples == 0:
            raise ValueError("capture_means: empty instruction list")
        if not sites:
            raise ValueError("capture_means: no sites requested")

        names = {site.value for site in sites}  # fork-cache intermediate names
        accum = {
            site.value: torch.zeros(
                (n_eoi, n_layers, d_model), dtype=torch.float64, device="cpu"
            )
            for site in sites
        }
        with self.wrapper.cache_activations(
            layers=set(range(n_layers)), intermediates=names, prefill=True
        ) as get_cache:
            for instruction in tqdm(instructions, desc=f"gemma capture {sorted(names)}"):
                self.wrapper.reset_prefill_cache()
                self.wrapper.generate_from_template(
                    self.format_chat(instruction), output_len=1
                )
                prefill = get_cache()["prefill"]
                for layer_idx in range(n_layers):
                    layer_cache = prefill[layer_idx]
                    for site in sites:
                        acts = layer_cache[site.value]  # (1, seq_len, d_model)
                        tail = acts[0, -n_eoi:, :].to(torch.float64).cpu()
                        accum[site.value][:, layer_idx, :] += tail / n_samples
        return accum

    def last_position_logits(self, prompt: str) -> torch.Tensor:
        """Last-position logits (CPU fp32) via the final-norm + tied-embedding path.

        Captures the post-final-norm hidden state and applies the model's tied
        embedding (with optional softcap) — the same logit pipeline ``Sampler``
        uses inside ``model.generate``. Any open ``HookManager`` session is in
        effect during this forward.
        """
        with self.wrapper.cache_activations(
            layers={0}, intermediates={"final_norm"}, prefill=True
        ) as get_cache:
            self.wrapper.reset_prefill_cache()
            self.wrapper.generate_from_template(
                self.format_chat(prompt), output_len=1
            )
            cache = get_cache()

        final_norm = cache["prefill"]["final_norm"]  # (1, seq_len, d_model), CPU
        embed = self.wrapper.model.text_token_embedder.weight  # (vocab, d_model)
        last = final_norm[0, -1, :].to(embed.device, embed.dtype)
        logits = last @ embed.T
        cap = getattr(self.wrapper.config, "final_logit_softcapping", None)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits.detach().to("cpu", torch.float32)


class QwenDirectionModel(_BaseDirectionModel):
    """``DirectionModel`` over ``Qwen3Inference`` (Qwen3 / Qwen3.5, HF transformers).

    Thinking is disabled in the chat template so the first response position is
    the actual answer opener (not a ``<think>`` token). The EOI window is read
    off the live template; refusal ids are recomputed for the Qwen tokenizer.
    """

    def __init__(self, model_name: str, wrapper=None) -> None:
        if wrapper is None:
            from interpret.inference.qwen3_transformers import Qwen3Inference

            wrapper = Qwen3Inference(model_name)
        super().__init__(wrapper)

    @property
    def d_model(self) -> int:
        return int(self.wrapper.model.get_input_embeddings().weight.shape[1])

    def format_chat(self, instruction: str) -> str:
        return self.wrapper.format_prompt(instruction, enable_thinking=False)

    def eoi_token_ids(self) -> list[int]:
        """The constant post-instruction token window of the chat template.

        Computed as the longest common token *suffix* of two formatted prompts
        with different instructions: whatever the template appends after the
        user content (``<|im_end|>\\n<|im_start|>assistant\\n`` and any
        thinking-block scaffold) is identical across prompts and shows up as
        that shared suffix. Robust to template internals and ``enable_thinking``
        scaffolding without hardcoding ChatML strings.
        """
        ids_a = self.wrapper.tokenize(self.format_chat("apple"), add_special_tokens=True)
        ids_b = self.wrapper.tokenize(
            self.format_chat("zebra orange table lamp"), add_special_tokens=True
        )
        suffix: list[int] = []
        for a, b in zip(reversed(ids_a), reversed(ids_b), strict=False):
            if a != b:
                break
            suffix.append(a)
        suffix.reverse()
        if not suffix:
            raise RuntimeError(
                "Could not derive an EOI window for this Qwen template "
                "(empty common suffix). Inspect format_chat output."
            )
        return suffix

    def refusal_token_ids(self, configured: tuple[int, ...]) -> tuple[int, ...]:
        """Recompute refusal-cue ids for the Qwen tokenizer (ignores Gemma ids).

        Takes the first token of each ``REFUSAL_WORDS`` entry (deduped, order
        preserved). The Gemma-default ``configured`` value does not transfer
        across tokenizers, so it is intentionally not used.
        """
        ids: list[int] = []
        for word in REFUSAL_WORDS:
            encoded = self.wrapper.tokenize(word, add_special_tokens=False)
            if not encoded:
                continue
            tid = encoded[0]
            if tid not in ids:
                ids.append(tid)
        if not ids:
            raise RuntimeError("Could not resolve any refusal token ids for Qwen.")
        if tuple(ids) != tuple(configured):
            decoded = [self.wrapper.tokenizer.decode([tid]) for tid in ids]
            warnings.warn(
                f"Qwen refusal tokens recomputed to {tuple(ids)} "
                f"(decode: {decoded}) for {REFUSAL_WORDS}; "
                f"configured {configured} not used.",
                stacklevel=2,
            )
        return tuple(ids)

    def capture_means(
        self,
        instructions: list[str],
        sites: tuple[CaptureSite, ...],
        n_eoi: int,
    ) -> dict[str, torch.Tensor]:
        from interpret.sae import HookType

        n_layers, d_model = self.n_layers, self.d_model
        n_samples = len(instructions)
        if n_samples == 0:
            raise ValueError("capture_means: empty instruction list")
        if not sites:
            raise ValueError("capture_means: no sites requested")

        hook_types: set[HookType] = {QWEN_SITE_MAP[site][0] for site in sites}
        accum = {
            site.value: torch.zeros(
                (n_eoi, n_layers, d_model), dtype=torch.float64, device="cpu"
            )
            for site in sites
        }
        layers_all = set(range(n_layers))
        site_desc = sorted(s.value for s in sites)
        for instruction in tqdm(instructions, desc=f"qwen capture {site_desc}"):
            with self.wrapper.cache_activations(
                layers=layers_all, hook_types=hook_types, prefill_only=True
            ) as get_cache:
                self.wrapper.generate_from_template(
                    self.format_chat(instruction), output_len=1
                )
                cache = get_cache()  # {layer: {HookType: (1, seq, d)}}
            for site in sites:
                hook_type, offset = QWEN_SITE_MAP[site]
                for layer_idx in range(n_layers):
                    src = layer_idx + offset
                    if src < 0 or src >= n_layers:
                        continue  # RESID_PRE[0]: no source -> leave zeros
                    layer_cache = cache.get(src)
                    if layer_cache is None or hook_type not in layer_cache:
                        raise RuntimeError(
                            f"missing capture for layer {src} {hook_type} "
                            f"(site {site.value})"
                        )
                    acts = layer_cache[hook_type]  # (1, seq_len, d_model)
                    # MPS doesn't support float64 — move to CPU before casting.
                    tail = acts[0, -n_eoi:, :].cpu().to(torch.float64)
                    accum[site.value][:, layer_idx, :] += tail / n_samples
        return accum

    def last_position_logits(self, prompt: str) -> torch.Tensor:
        """Last-position logits (CPU fp32) via a plain forward through the HF head.

        Any open ``HookManager`` session steers this forward (hooks fire on the
        decoder layers the model passes through). No final-logit softcap on Qwen.
        """
        input_ids = self.wrapper.tokenizer(
            self.format_chat(prompt), return_tensors="pt", add_special_tokens=True
        ).input_ids.to(self.wrapper.device)
        with torch.no_grad():
            out = self.wrapper.model(input_ids=input_ids)
        logits = out.logits[0, -1, :]
        return logits.detach().to("cpu", torch.float32)


def build_direction_model(model_name: str, wrapper=None) -> DirectionModel:
    """Construct the right ``DirectionModel`` for ``model_name``.

    Dispatches on the model id: ``gemma`` -> Gemma, ``qwen`` -> Qwen. Pass a
    pre-loaded ``wrapper`` to reuse weights across experiments (e.g. the three
    poetry runs).
    """
    lowered = model_name.lower()
    if "gemma" in lowered:
        return GemmaDirectionModel(model_name, wrapper=wrapper)
    if "qwen" in lowered:
        return QwenDirectionModel(model_name, wrapper=wrapper)
    raise ValueError(
        f"Cannot infer backend for model_name {model_name!r}. "
        "Expected an id containing 'gemma' or 'qwen'."
    )
