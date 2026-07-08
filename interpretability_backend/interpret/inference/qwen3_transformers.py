"""Qwen3 / Qwen3.5 inference wrapper over HuggingFace transformers with activation hooks.

Wraps `AutoModelForCausalLM` for Qwen3 (0.6B–14B) and dense Qwen3.5
(0.8B / 2B / 4B / 9B / 27B), runs in bfloat16 on MPS / CUDA / CPU, and
exposes a `cache_activations()` context manager that captures residual /
post-attn / attention-out / MLP-out hidden states per layer via PyTorch
forward hooks. No fork of transformers; the stock model is used
unchanged. Capture machinery shares `HookType` and `ActivationStore` with
`interpret.sae.hook_manager.HookManager` but bypasses SAE loading — raw
hidden states only.

Hook points (see `_RawActivationCapture` docstring for full semantics):
- `RESID_POST` — output of the full decoder layer (post second residual add).
- `POST_ATTN`  — residual stream right after the attn-residual-add, before
                 `post_attention_layernorm`. Captured via forward-pre-hook
                 on `post_attention_layernorm`. Uniform across Qwen3.5's
                 full- and linear-attention layer types.
- `ATTN_OUT`   — raw output of the attention sub-block. For Qwen3.5
                 linear-attention layers, hooks `linear_attn` instead of
                 `self_attn` — captured tensor semantics differ.
- `MLP_OUT`    — raw output of the MLP sub-block (pre-residual-add).

Verified against transformers 5.4.0; the layer attribute path
`model.model.layers` is asserted in __init__ to fail fast on future drift.

Notes:
- bfloat16 is the default precision (fp16 overflows on Qwen3's QK-norm —
  RMSNorm-scaled attention logits exceed fp16's max).
- Per-layer sliding-window attention (config.layer_types) is encapsulated
  inside Qwen3Attention and is invisible to forward hooks. Captured
  attn_out semantics differ across layer types — relevant for SAE
  training, not for hook mechanics.
- Qwen3's chat template is non-trivial (think blocks, tool-use sections);
  use `tokenizer.apply_chat_template` exclusively, never hand-rolled.

Usage:
    from interpret.inference.qwen3_transformers import Qwen3Inference
    from interpret.sae.sae_config import HookType

    wrapper = Qwen3Inference("Qwen/Qwen3-0.6B")
    print(wrapper.generate("What colour is the sky?"))

    with wrapper.cache_activations(
        layers={5}, hook_types={HookType.RESID_POST}
    ) as get_cache:
        wrapper.generate("What colour is the sky?")
        cache = get_cache()
    # cache == {5: {HookType.RESID_POST: Tensor[1, prompt_len, hidden_size]}}

CLI:
    uv run python -m interpret.inference.qwen3_transformers \\
        --model Qwen/Qwen3-0.6B \\
        --prompt "What colour is the sky?" \\
        --output-len 100
"""

import argparse
import contextlib
import queue
import threading
from collections.abc import Callable, Generator
from typing import Literal

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.generation.streamers import BaseStreamer

from interpret.inference.streaming import TokenStreamEvent
from interpret.sae.activation_store import ActivationStore
from interpret.sae.sae_config import HookType


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class _RawActivationCapture:
    """Forward-hook based raw hidden-state capture for transformer decoder layers.

    Hook-target resolution:
        RESID_POST -> forward hook on `layer` (post-MLP residual stream).
        MLP_OUT    -> forward hook on `layer.mlp` (raw MLP out, pre-residual-add).
        ATTN_OUT   -> forward hook on `layer.self_attn` (full-attention layers)
                      or `layer.linear_attn` (Qwen3.5 linear-attention layers).
                      Captured tensor semantics differ across the two.
        POST_ATTN  -> forward-pre hook on `layer.post_attention_layernorm`,
                      capturing the residual stream right after the attn-residual-add.
                      Uniform across full- and linear-attention layers.

    Raw hidden states (no SAE encode, no reconstruction) are stored in an
    `ActivationStore` keyed by (layer_index, hook_type).

    With prefill_only=True (default), each hook becomes a no-op after the
    first capture — the prefill captures the full prompt with seq_len > 1;
    subsequent decode steps are dropped (the handle stays attached but
    early-returns on a `_captured` set lookup).
    """

    _PRE_HOOK_TYPES: frozenset[HookType] = frozenset({HookType.POST_ATTN})

    def __init__(
        self,
        layers: set[int],
        hook_types: set[HookType],
        prefill_only: bool = True,
    ) -> None:
        self.layers = layers
        self.hook_types = hook_types
        self.prefill_only = prefill_only
        self.store = ActivationStore()
        self._handles: list[RemovableHandle] = []
        # The store key widened to ``(layer, hook_type, sae_id)`` when SAEs
        # became per-site-co-attachable. The raw capture path never has an
        # SAE — leave the slot empty so ``store.prefill(layer, hook_type)``
        # (which defaults ``sae_id=""``) reads the matching record back.
        self._captured: set[tuple[int, HookType, str]] = set()

    @staticmethod
    def _resolve_target(layer: nn.Module, hook_type: HookType) -> nn.Module:
        if hook_type is HookType.RESID_POST:
            return layer
        if hook_type is HookType.MLP_OUT:
            return layer.mlp
        if hook_type is HookType.ATTN_OUT:
            # full_attention layers: layer.self_attn; linear_attention: layer.linear_attn
            attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
            if attn is None:
                raise AttributeError(
                    "decoder layer has neither `self_attn` nor `linear_attn` — "
                    "unsupported architecture for ATTN_OUT capture"
                )
            return attn
        if hook_type is HookType.POST_ATTN:
            return layer.post_attention_layernorm
        raise ValueError(f"unsupported hook_type: {hook_type}")

    def _record(self, key: tuple[int, HookType, str], tensor: torch.Tensor) -> None:
        with torch.no_grad():
            self.store.record(key, tensor.detach().clone())
        self._captured.add(key)

    def _make_forward_hook(self, key: tuple[int, HookType, str]):
        def hook_fn(module: nn.Module, inputs: tuple, output):
            if self.prefill_only and key in self._captured:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            self._record(key, hidden)
            return None

        return hook_fn

    def _make_pre_forward_hook(self, key: tuple[int, HookType, str]):
        def hook_fn(module: nn.Module, args: tuple):
            if self.prefill_only and key in self._captured:
                return None
            hidden = args[0]
            self._record(key, hidden)
            return None

        return hook_fn

    def attach(self, decoder_layers: nn.ModuleList) -> None:
        for layer_idx in self.layers:
            if layer_idx >= len(decoder_layers):
                raise ValueError(
                    f"layer_index {layer_idx} out of range (model has {len(decoder_layers)} layers)"
                )
            for hook_type in self.hook_types:
                target = self._resolve_target(decoder_layers[layer_idx], hook_type)
                key = (layer_idx, hook_type, "")
                if hook_type in self._PRE_HOOK_TYPES:
                    handle = target.register_forward_pre_hook(self._make_pre_forward_hook(key))
                else:
                    handle = target.register_forward_hook(self._make_forward_hook(key))
                self._handles.append(handle)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def collect(self) -> dict[int, dict[HookType, torch.Tensor]]:
        result: dict[int, dict[HookType, torch.Tensor]] = {}
        for layer_idx in self.layers:
            for hook_type in self.hook_types:
                record = self.store.prefill(layer_idx, hook_type)
                if record is not None:
                    result.setdefault(layer_idx, {})[hook_type] = record.feature_acts
        return result


class _TokenIdStreamer(BaseStreamer):
    """Collects generated token IDs into a queue for incremental consumption.

    Mirrors ``TextStreamer``'s prompt handling: the first ``put`` (the prompt
    token IDs, a 2-D tensor) is dropped; each subsequent ``put`` carries the
    newly generated token id(s). ``end()`` enqueues a sentinel so the consumer
    knows generation finished. Batch size 1 only (the wrapper never batches).
    """

    SENTINEL = object()

    def __init__(self) -> None:
        self.queue: queue.Queue = queue.Queue()
        self._prompt_seen = False

    def put(self, value: torch.Tensor) -> None:
        if value.dim() > 1:
            value = value[0]  # drop batch dim; prompt arrives as (1, prompt_len)
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        for token_id in value.tolist():
            self.queue.put(int(token_id))

    def end(self) -> None:
        self.queue.put(self.SENTINEL)


class _CancelCriteria(StoppingCriteria):
    """Stops generation after the current token once ``cancel_event`` is set."""

    def __init__(self, cancel_event: threading.Event) -> None:
        self._cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._cancel_event.is_set()


def _drain_until_sentinel(q: "queue.Queue") -> Generator[int, None, None]:
    """Yield token ids from a queue until the streamer's sentinel appears."""
    while True:
        item = q.get()
        if item is _TokenIdStreamer.SENTINEL:
            return
        yield item


def _stream_token_events(
    token_source,
    decode: Callable[..., str],
    *,
    skip_special_tokens: bool = True,
) -> Generator[TokenStreamEvent, None, None]:
    """Turn a stream of token ids into ``TokenStreamEvent``s via decode + diff.

    Decodes the full growing token list each step and yields only the new
    characters — the same diff strategy ``GemmaPytorchInference`` uses, which
    keeps multi-byte / BPE pieces from being split across deltas. The last
    emitted event carries ``is_done=True``; an empty source yields nothing.
    Pure (no model / threads) so it can be unit-tested directly.
    """
    token_ids: list[int] = []
    prev_text = ""
    pending: TokenStreamEvent | None = None
    idx = 0
    for token_id in token_source:
        if pending is not None:
            yield pending
        token_ids.append(token_id)
        full_text = decode(token_ids, skip_special_tokens=skip_special_tokens)
        text_delta = full_text[len(prev_text) :]
        prev_text = full_text
        pending = TokenStreamEvent(idx, token_id, text_delta, False)
        idx += 1
    if pending is not None:
        yield pending._replace(is_done=True)


class Qwen3Inference:
    """Qwen3 wrapper over HuggingFace transformers with activation hooks.

    Attributes:
        model: The HF AutoModelForCausalLM instance (Qwen3ForCausalLM).
        tokenizer: The fast tokenizer.
        device: torch.device the model runs on.
        dtype: torch.dtype the model parameters are cast to.
    """

    def __init__(
        self,
        model_name: str,
        dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16",
        device: str | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.dtype = _DTYPE_MAP[dtype]

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype)
            .to(self.device)
            .eval()
        )

        assert hasattr(self.model, "model") and hasattr(self.model.model, "layers"), (
            "Qwen3 layer path drifted — expected `model.model.layers`. "
            "Pin the verified transformers version (5.4.0) or update the wrapper."
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def decoder_layers(self) -> nn.ModuleList:
        """The decoder-layer ModuleList — entry point for raw forward-hook attachment."""
        return self.model.model.layers

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Tokenize a string. Defaults to no special tokens (for offset use)."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    @property
    def prepends_bos(self) -> bool:
        """Whether ``generate_from_template`` sequences start with a BOS token.

        Probed from the tokenizer rather than assumed: Qwen3/Qwen3.5
        tokenizers define no BOS (``bos_token_id`` is None), so position 0
        of a prefill is real content there — callers that mask the first
        position (e.g. BOS-excluding aggregation) must check this first.
        """
        bos_id = self._tokenizer.bos_token_id
        if bos_id is None:
            return False
        ids = self._tokenizer.encode("probe", add_special_tokens=True)
        return len(ids) > 0 and ids[0] == bos_id

    def format_prompt(self, prompt: str | list[dict], enable_thinking: bool = True) -> str:
        """Apply Qwen3's chat template. Accepts a raw user string or a messages list.

        ``enable_thinking`` maps to the Qwen3 template variable of the same
        name; the streaming chat methods default it to False to suppress
        ``<think>`` blocks in the visible output.
        """
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def generate(
        self,
        prompt: str,
        output_len: int = 256,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_thinking: bool = True,
    ) -> str:
        """Generate a text response from a single-turn user prompt."""
        formatted = self.format_prompt(prompt, enable_thinking=enable_thinking)
        return self._generate(formatted, output_len, temperature, top_p, top_k)

    def generate_chat(
        self,
        turns: list[tuple[str, str]],
        output_len: int = 256,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> str:
        """Generate from a multi-turn conversation. Roles: 'user', 'assistant', 'system'."""
        messages = [{"role": role, "content": content} for role, content in turns]
        formatted = self.format_prompt(messages)
        return self._generate(formatted, output_len, temperature, top_p, top_k)

    def generate_from_template(
        self,
        prompt: str,
        output_len: int = 1,
        add_bos: bool = True,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> str:
        """Run the raw prompt through the model without chat-template wrapping.

        Mirrors ``GemmaPytorchInference.generate_from_template`` so the
        autointerpreter collector can call the same method on either
        wrapper. Used when ``use_chat_template: false`` is set on the
        base-model config (e.g. SAE feature-activation collection where
        we want the raw ``"{word}: {definition}."`` template to reach the
        model unwrapped).

        ``add_bos`` controls the tokenizer's ``add_special_tokens`` flag.
        Qwen3 tokenizers may or may not prepend a BOS depending on the
        variant; explicit control keeps the collector's behaviour
        identical to Gemma's collect path.
        """
        input_ids = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=add_bos,
        ).input_ids.to(self.device)
        do_sample = temperature is not None
        gen_kwargs: dict = {
            "max_new_tokens": output_len,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        with torch.no_grad():
            output_ids = self.model.generate(input_ids=input_ids, **gen_kwargs)

        prompt_len = input_ids.shape[1]
        generated = output_ids[0, prompt_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _generate(
        self,
        formatted_prompt: str,
        output_len: int,
        temperature: float | None,
        top_p: float,
        top_k: int,
    ) -> str:
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        do_sample = temperature is not None
        gen_kwargs = {
            "max_new_tokens": output_len,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, prompt_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_stream(
        self,
        formatted_prompt: str,
        output_len: int,
        temperature: float | None,
        top_p: float,
        top_k: int,
        cancel_event: threading.Event | None,
    ) -> Generator[TokenStreamEvent, None, None]:
        """Stream tokens from a pre-formatted prompt.

        Runs ``model.generate`` on a background thread feeding a token-id queue,
        and yields clean text deltas via full-decode + diff. The last real token
        carries ``is_done=True``. A cancel criterion is always attached: an
        explicit ``cancel_event`` (set by the caller on disconnect), or an
        internal one set on generator close so abandoning the stream early halts
        the background generate instead of running it to completion.
        """
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        do_sample = temperature is not None
        gen_kwargs: dict = {
            "max_new_tokens": output_len,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        cancel = cancel_event if cancel_event is not None else threading.Event()
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_CancelCriteria(cancel)])

        streamer = _TokenIdStreamer()
        errors: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.no_grad():
                    self.model.generate(**inputs, streamer=streamer, **gen_kwargs)
            except BaseException as exc:  # surface to the consumer below
                errors.append(exc)
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            yield from _stream_token_events(
                _drain_until_sentinel(streamer.queue), self._tokenizer.decode
            )
        finally:
            cancel.set()  # halt the background generate if abandoned early
            thread.join(timeout=30.0)
        if errors:
            raise errors[0]

    def generate_stream(
        self,
        prompt: str,
        output_len: int = 256,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
        cancel_event: threading.Event | None = None,
        enable_thinking: bool = False,
    ) -> Generator[TokenStreamEvent, None, None]:
        """Stream tokens from a single-turn user prompt (thinking off by default)."""
        formatted = self.format_prompt(prompt, enable_thinking=enable_thinking)
        yield from self._generate_stream(
            formatted, output_len, temperature, top_p, top_k, cancel_event
        )

    def generate_chat_stream(
        self,
        turns: list[tuple[str, str]],
        output_len: int = 256,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
        cancel_event: threading.Event | None = None,
        enable_thinking: bool = False,
    ) -> Generator[TokenStreamEvent, None, None]:
        """Stream tokens from a multi-turn conversation. Roles: 'user'/'assistant'/'system'.

        ``enable_thinking`` defaults to False so ``<think>`` blocks stay out of
        the visible output; callers (e.g. the backend chat toggle) may opt in.
        """
        messages = [{"role": role, "content": content} for role, content in turns]
        formatted = self.format_prompt(messages, enable_thinking=enable_thinking)
        yield from self._generate_stream(
            formatted, output_len, temperature, top_p, top_k, cancel_event
        )

    @contextlib.contextmanager
    def cache_activations(
        self,
        layers: set[int],
        hook_types: set[HookType] | None = None,
        prefill_only: bool = True,
    ) -> Generator[Callable[[], dict[int, dict[HookType, torch.Tensor]]], None, None]:
        """Context manager: capture raw hidden states at the requested layers.

        Yields a callable that, after generation, returns
        `{layer_idx: {hook_type: Tensor[batch, seq_len, hidden_size]}}`.
        Tensors are detached + cloned, on the model's device and dtype.

        With prefill_only=True (default), each hook becomes a no-op after
        its first capture — the prompt prefill (seq_len > 1) is recorded;
        decode steps are dropped (the handle stays attached for the
        lifetime of the context).

        Args:
            layers: Layer indices to capture (e.g. {5, 12}).
            hook_types: Subset of {RESID_POST, ATTN_OUT, MLP_OUT}.
                None = {RESID_POST}.
            prefill_only: If True, hooks auto-remove after the first pass.
        """
        types = hook_types if hook_types is not None else {HookType.RESID_POST}
        capture = _RawActivationCapture(layers, types, prefill_only=prefill_only)
        capture.attach(self.decoder_layers)
        try:
            yield capture.collect
        finally:
            capture.detach()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen3 inference via HuggingFace transformers with activation hooks"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model ID")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--capture-layer",
        type=int,
        default=5,
        help="Layer index to capture activations from for the smoke test",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    wrapper = Qwen3Inference(args.model, dtype=args.dtype)
    print(f"Model loaded on {wrapper.device}, dtype {wrapper.dtype}.")

    with wrapper.cache_activations(
        layers={args.capture_layer},
        hook_types={
            HookType.RESID_POST,
            HookType.POST_ATTN,
            HookType.ATTN_OUT,
            HookType.MLP_OUT,
        },
    ) as get_cache:
        result = wrapper.generate(
            args.prompt,
            output_len=args.output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        cache = get_cache()

    print(f"\n{'=' * 40}")
    print(result)
    print(f"{'=' * 40}\n")

    layer_cache = cache[args.capture_layer]
    for hook_type, tensor in layer_cache.items():
        print(
            f"layer {args.capture_layer} {hook_type.value}: "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"finite={torch.isfinite(tensor).all().item()} "
            f"nonzero={torch.any(tensor != 0).item()}"
        )


if __name__ == "__main__":
    main()
