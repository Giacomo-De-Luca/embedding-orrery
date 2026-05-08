"""Raw PyTorch Gemma3 4b inference wrapper with interpretability hook support.

Wraps the reference gemma_pytorch implementation (interpret/forked/gemma_pytorch/)
to run Gemma3 4b with bfloat16 precision on MPS. Exposes the raw model for
attaching forward/backward hooks to any layer, and provides built-in
activation caching for residual stream extraction.

Usage:
    from interpret.inference.gemma_pytorch import GemmaPytorchInference

    model = GemmaPytorchInference("/path/to/checkpoint")
    print(model.generate("What colour is the sky?"))

    # Extract prefill activations (default) from specific layers:
    with model.cache_activations(layers={0, 17, 33}, intermediates={"post_attn", "post_mlp"}) as get_cache:
        model.generate("What colour is the sky?")
        cache = get_cache()
    # cache == {"prefill": {0: {"post_attn": Tensor, "post_mlp": Tensor}, ...}}

    # Extract both prefill and last decode step:
    with model.cache_activations(layers={30}, prefill=True, last=True) as get_cache:
        model.generate("prompt")
        cache = get_cache()
    # cache["prefill"][30]["post_mlp"].shape == [1, seq_len, 2560]
    # cache["last"][30]["post_mlp"].shape    == [1, 1, 2560]

    # Or attach a hook to a decoder layer directly:
    handle = model.model.model.layers[0].register_forward_hook(my_hook_fn)

CLI:
    uv run python -m interpret.inference.gemma_pytorch \\
        --checkpoint /path/to/weights \\
        --prompt "What colour is the sky?" \\
        --output-len 100
"""

import argparse
import contextlib
import gc
import json
import re
import sys
from pathlib import Path

from PIL import Image
from typing import Literal

# Add gemma_pytorch fork to import path
_GEMMA_PYTORCH_ROOT = Path(__file__).resolve().parents[1] / "forked" / "gemma_pytorch"


@contextlib.contextmanager
def _set_default_tensor_type(dtype):
    """Temporarily sets the default torch dtype."""
    import torch
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(torch.float)


class GemmaPytorchInference:
    """Raw PyTorch Gemma3 4b wrapper for inference with interpretability hook support.

    Attributes:
        model: The Gemma3ForMultimodalLM instance — full access to all layers,
               attention heads, MLP blocks, vision encoder, embeddings, etc.
        config: The GemmaConfig used to build the model.
        device: The torch.device the model runs on (mps).
    """

    def __init__(self, checkpoint_path: str, model_size: str = "4b", precision: Literal["float16", "bfloat16", "float32"] = "bfloat16") -> None:
        import torch
        gemma_root = str(_GEMMA_PYTORCH_ROOT)
        if gemma_root not in sys.path:
            sys.path.insert(0, gemma_root)
        from gemma import config as gemma_config
        from gemma import gemma3_model


        ## this crashes without metal or cuda, - but I honestly don't want it to work on cpu
        
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
        self.model_size = model_size

        # Resolve HF model ID to local cache path if needed
        checkpoint_path = self._resolve_checkpoint(checkpoint_path)

        # Build config for 4b variant
        # Using bfloat16: fp16 overflows on MPS due to narrow dynamic range
        # (attention logits after RMSNorm scaling exceed fp16 max of 65504).
        # bfloat16 has the same exponent range as float32, avoiding this issue.
        self.config = gemma_config.get_model_config(model_size, dtype=precision)

        # Resolve tokenizer path: checkpoint dir first, then reference repo fallback
        tokenizer_rel = Path(self.config.tokenizer)
        tokenizer_in_ckpt = Path(checkpoint_path) / tokenizer_rel
        tokenizer_in_ref = _GEMMA_PYTORCH_ROOT / tokenizer_rel

        if tokenizer_in_ckpt.is_file():
            self.config.tokenizer = str(tokenizer_in_ckpt)
        elif tokenizer_in_ref.is_file():
            self.config.tokenizer = str(tokenizer_in_ref)
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_in_ckpt} or {tokenizer_in_ref}. "
                f"Expected: {tokenizer_rel}"
            )

        # Construct model with bfloat16 default tensor type
        with _set_default_tensor_type(torch.bfloat16):
            self.model = gemma3_model.Gemma3ForMultimodalLM(self.config)
            self._load_weights(checkpoint_path)

        self.model = self.model.to(self.device).eval()

    @property
    def tokenizer(self):
        """Expose the underlying sentencepiece tokenizer for offset computation."""
        return self.model.tokenizer

    def tokenize(self, text: str, bos: bool = False) -> list[int]:
        """Tokenize a string without BOS/EOS tokens (for offset computation).

        Args:
            text: Raw string to tokenize.
            bos: Whether to prepend the BOS token.

        Returns:
            List of token IDs.
        """
        return self.tokenizer.encode(text, bos=bos)

    
    @staticmethod
    def format_prompt(text: str) -> str:
        return (
            f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
        )

    @staticmethod
    def _resolve_checkpoint(checkpoint_path: str) -> str:
        """Resolve a checkpoint path, supporting HuggingFace model IDs.

        If the path looks like a HF model ID (e.g. "google/gemma-3-4b-it"),
        resolves it to the local HF cache snapshot directory. Otherwise
        returns the path unchanged.
        """
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            return checkpoint_path

        # Looks like an HF model ID (contains "/" but doesn't exist as a path)
        if "/" in checkpoint_path:
            try:
                from huggingface_hub import snapshot_download
                return snapshot_download(checkpoint_path)
            except Exception as e:
                raise FileNotFoundError(
                    f"Checkpoint not found at '{checkpoint_path}' and could not "
                    f"resolve as HuggingFace model ID: {e}"
                ) from e

        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    @property
    def _gemma_model(self):
        """The inner GemmaModel (contains the decoder layers and cache state)."""
        return self.model.model

    def configure_cache(
        self,
        layers: set[int] | None = None,
        intermediates: set[str] | None = None,
        prefill: bool = True,
        last: bool = False,
    ) -> None:
        """Configure activation caching for specific layers and intermediates.

        Args:
            layers: Layer indices to capture (e.g. {0, 17, 33}). None = all layers.
            intermediates: Which points to capture. None = all intermediates.
                Valid: "pre_attn", "post_attn", "mlp_out", "post_mlp", "final_norm".
            prefill: Capture activations from the first forward call (the prompt).
                Shape will be [B, seq_len, hidden_size] with full sequence.
            last: Capture activations from the final decode step.
                Shape will be [B, 1, hidden_size].
        """
        self._gemma_model.configure_cache(
            layers=layers, intermediates=intermediates, prefill=prefill, last=last,
        )

    def clear_cache(self) -> None:
        """Disable caching and release stored activations."""
        self._gemma_model.clear_cache()

    def reset_prefill_cache(self) -> None:
        """Reset the prefill cache so the next forward pass captures fresh activations.

        Call this between samples when using configure_cache() in a loop.
        The cache configuration (layers, intermediates, prefill/last flags) is preserved.
        """
        gm = self._gemma_model
        gm._prefill_cache = {}
        gm._prefill_final_norm = None
        gm._prefill_captured = False

    @staticmethod
    def _build_cache_dict(
        layer_cache: dict, final_norm: "torch.Tensor | None",
    ) -> dict:
        """Build a cache dict from layer cache and optional final_norm."""
        result = dict(layer_cache)
        if final_norm is not None:
            result["final_norm"] = final_norm
        return result

    def get_cached_activations(self) -> dict[str, dict]:
        """Return cached activations keyed by generation step.

        Returns:
            Dict with "prefill" and/or "last" keys (depending on configuration),
            each mapping to {layer_idx: {"intermediate_name": Tensor}, ...}.
            If "final_norm" was requested, it appears as a top-level key within
            each step dict.

            Example with prefill=True, last=True:
            {
                "prefill": {30: {"post_attn": Tensor[1, seq_len, 2560]}, ...},
                "last":    {30: {"post_attn": Tensor[1, 1, 2560]}, ...},
            }
        """
        gm = self._gemma_model
        result = {}
        if gm.cache_prefill and gm._prefill_cache:
            result["prefill"] = self._build_cache_dict(gm._prefill_cache, gm._prefill_final_norm)
        if gm.cache_last and gm._last_cache:
            result["last"] = self._build_cache_dict(gm._last_cache, gm._last_final_norm)
        return result

    @contextlib.contextmanager
    def cache_activations(
        self,
        layers: set[int] | None = None,
        intermediates: set[str] | None = None,
        prefill: bool = True,
        last: bool = False,
    ):
        """Context manager for one-shot activation capture.

        Yields a callable that returns the populated cache dict after generation.

        Usage:
            with model.cache_activations(layers={17}, intermediates={"post_mlp"}) as get_cache:
                model.generate("prompt")
                cache = get_cache()
            # cache == {"prefill": {17: {"post_mlp": Tensor}}}
        """
        self.configure_cache(
            layers=layers, intermediates=intermediates, prefill=prefill, last=last,
        )
        try:
            yield self.get_cached_activations
        finally:
            self.clear_cache()

    def _load_weights(self, checkpoint_path: str) -> None:
        """Load weights from checkpoint — supports .pt, .bin shards, and .safetensors."""
        import torch

        ckpt = Path(checkpoint_path)

        # Single .pt file (original gemma_pytorch format)
        if ckpt.is_file() and ckpt.suffix == ".pt":
            self.model.load_weights(checkpoint_path)
            return

        # Directory — check for safetensors first, then fall back to bin shards
        safetensors_index = ckpt / "model.safetensors.index.json"
        bin_index = ckpt / "pytorch_model.bin.index.json"

        if safetensors_index.is_file():
            self._load_safetensors_hf(ckpt, safetensors_index)
        elif bin_index.is_file():
            self.model.load_weights(checkpoint_path)
        else:
            raise FileNotFoundError(
                f"No valid checkpoint found in {ckpt}. Expected one of: "
                "single .pt file, model.safetensors.index.json, or "
                "pytorch_model.bin.index.json"
            )

    def _load_safetensors_hf(self, ckpt_dir: Path, index_path: Path) -> None:
        """Load HuggingFace safetensors weights, mapping keys to reference format."""
        import torch
        from safetensors.torch import load_file

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        shard_files = sorted(set(index["weight_map"].values()))

        for shard_file in shard_files:
            shard_path = ckpt_dir / shard_file
            hf_state = load_file(str(shard_path))
            mapped_state = self._map_hf_to_reference(hf_state)
            self.model.load_state_dict(mapped_state, strict=False)
            del hf_state, mapped_state
            gc.collect()

    def _map_hf_to_reference(self, hf_state: dict) -> dict:
        """Map HuggingFace weight keys to reference gemma_pytorch format.

        Handles prefix remapping, q/k/v → fused qkv_proj concatenation,
        norm renaming, and vision encoder key differences.
        """
        import torch

        ref_state = {}
        # Collect q/k/v per layer for fusion
        qkv_parts: dict[int, dict[str, torch.Tensor]] = {}
        vocab_size = self.config.vocab_size

        for hf_key, tensor in hf_state.items():
            ref_key = self._remap_key(hf_key)
            if ref_key is None:
                continue

            # Detect separate q/k/v projections that need fusing
            layer_qkv = re.match(
                r"model\.layers\.(\d+)\.self_attn\.([qkv])_proj\.weight", ref_key
            )
            if layer_qkv:
                layer_idx = int(layer_qkv.group(1))
                proj = layer_qkv.group(2)
                qkv_parts.setdefault(layer_idx, {})[proj] = tensor
                continue

            # Trim embedding if HF checkpoint has padded vocab
            if ref_key == "text_token_embedder.weight" and tensor.shape[0] > vocab_size:
                tensor = tensor[:vocab_size]

            # Transpose mm_input_projection (HF stores as [in, out], ref expects [out, in])
            if ref_key == "mm_input_projection.weight" and tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.t()

            ref_state[ref_key] = tensor

        # Fuse q/k/v into qkv_proj for each layer
        for layer_idx, parts in qkv_parts.items():
            if "q" in parts and "k" in parts and "v" in parts:
                fused = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
                ref_state[f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"] = fused

        return ref_state

    @staticmethod
    def _remap_key(hf_key: str) -> str | None:
        """Remap a single HuggingFace key to reference gemma_pytorch key.

        Returns None for keys that should be skipped (e.g. freqs_cis buffers).
        """
        # Language model layers: strip "language_model." prefix
        if hf_key.startswith("language_model.model.embed_tokens."):
            return hf_key.replace(
                "language_model.model.embed_tokens", "text_token_embedder"
            )
        if hf_key.startswith("language_model.model."):
            key = hf_key.replace("language_model.model.", "model.", 1)
            # Rename QK norms
            key = key.replace(".self_attn.q_norm.", ".self_attn.query_norm.")
            key = key.replace(".self_attn.k_norm.", ".self_attn.key_norm.")
            return key

        # Final norm
        if hf_key.startswith("language_model."):
            return hf_key.replace("language_model.", "", 1)

        # Multimodal projector (marked for transpose in _map_hf_to_reference)
        if hf_key == "multi_modal_projector.mm_input_projection_weight":
            return "mm_input_projection.weight"
        if hf_key.startswith("multi_modal_projector.mm_soft_emb_norm."):
            return hf_key.replace(
                "multi_modal_projector.mm_soft_emb_norm",
                "mm_soft_embedding_norm",
            )

        # Vision tower
        if hf_key.startswith("vision_tower.vision_model."):
            key = hf_key.replace(
                "vision_tower.vision_model.", "siglip_vision_model.", 1
            )
            # Embeddings prefix
            key = key.replace(
                "siglip_vision_model.embeddings.", "siglip_vision_model."
            )
            # Encoder layers → encoder_blocks
            key = key.replace(
                "siglip_vision_model.encoder.layers.",
                "siglip_vision_model.encoder_blocks.",
            )
            # Vision attention out_proj → o_proj
            key = key.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
            # Post layernorm → final_norm
            key = key.replace(
                "siglip_vision_model.post_layernorm.",
                "siglip_vision_model.final_norm.",
            )
            return key

        return None

    def _generate(
        self,
        sequence: list,
        output_len: int = 256,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> str:
        """Run generation on a pre-formatted input sequence.

        Args:
            sequence: List of interleaved str/PIL.Image elements for a single prompt.
            output_len: Maximum number of tokens to generate.
            temperature: Sampling temperature. None for greedy decoding.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold.

        Returns:
            The generated model response text (prompt stripped).
        """
        results = self.model.generate(
            [sequence],
            self.device,
            output_len=output_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return self._strip_prompt(results[0])

    def generate(
        self,
        prompt: str,
        output_len: int = 256,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> str:
        """Generate a text response from a text-only prompt."""
        formatted = (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model"
        )
        return self._generate([formatted], output_len, temperature, top_p, top_k)

    def generate_from_template(
        self,
        formatted_prompt: str,
        output_len: int = 256,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> str:
        """Generate from a pre-formatted prompt string (no chat wrapping).

        Use this when the caller has already applied the chat template or
        wants a custom template. The prompt is passed directly to the model.

        Args:
            formatted_prompt: Complete prompt string including any chat markers.
            output_len: Maximum tokens to generate.
        """
        return self._generate([formatted_prompt], output_len, temperature, top_p, top_k)

    def generate_chat(
        self,
        turns: list[tuple[str, str]],
        output_len: int = 256,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> str:
        """Generate a response from a multi-turn conversation.

        Args:
            turns: List of (role, content) tuples. Role is "user" or "model".
                The last turn should be ("user", ...) — the model response
                will be generated. If the last turn is ("model", ...) it is
                treated as a prefill and generation continues from there.

        Example:
            result = model.generate_chat([
                ("user", "What colour is the sky?"),
                ("model", "Blue."),
                ("user", "And grass?"),
            ])
        """
        parts = []
        for role, content in turns:
            parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
        # If the last turn was from the user, open the model turn
        if turns[-1][0] != "model":
            parts.append("<start_of_turn>model")
        else:
            # Last turn is a model prefill — reopen it without end_of_turn
            parts[-1] = f"<start_of_turn>model\n{turns[-1][1]}"
        formatted = "".join(parts)
        return self._generate([formatted], output_len, temperature, top_p, top_k)

    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        output_len: int = 256,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> str:
        """Generate a text response from a prompt with an image."""
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        interleaved = [
            "<start_of_turn>user\n",
            image,
            f"{prompt}<end_of_turn>\n<start_of_turn>model",
        ]
        return self._generate(interleaved, output_len, temperature, top_p, top_k)

    @staticmethod
    def _strip_prompt(text: str) -> str:
        """Strip the prompt template from generated output."""
        marker = "<start_of_turn>model"
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
        return text.strip()


def _resolve_default_checkpoint() -> str | None:
    """Find the gemma-3-4b-it checkpoint in the HuggingFace cache."""
    hf_model_dir = Path.home() / ".cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots"
    if not hf_model_dir.is_dir():
        return None
    snapshots = sorted(hf_model_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(snapshots[0]) if snapshots else None


def main() -> None:
    default_ckpt = _resolve_default_checkpoint()

    parser = argparse.ArgumentParser(
        description="Run Gemma3 4b inference with PyTorch on MPS (bfloat16)"
    )
    parser.add_argument(
        "--checkpoint", default=default_ckpt,
        help="Path to checkpoint directory or single .pt file "
             f"(default: {default_ckpt or 'none found — required'})",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Text prompt to send to the model",
    )
    parser.add_argument(
        "--image", default=None,
        help="Optional path to an image file for multimodal queries",
    )
    parser.add_argument(
        "--output-len", type=int, default=256,
        help="Maximum number of tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature. Omit for greedy decoding.",
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)

    args = parser.parse_args()

    if args.checkpoint is None:
        parser.error(
            "No checkpoint found. Provide --checkpoint or download the model:\n"
            "  huggingface-cli download google/gemma-3-4b-it"
        )

    print("Loading model...")
    model = GemmaPytorchInference(args.checkpoint)
    print("Model loaded.")

    if args.image:
        result = model.generate_with_image(
            args.prompt, args.image,
            output_len=args.output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    else:
        result = model.generate(
            args.prompt,
            output_len=args.output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    print(f"\n{'=' * 40}")
    print(result)
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
