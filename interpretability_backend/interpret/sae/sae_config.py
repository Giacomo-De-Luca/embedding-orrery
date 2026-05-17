"""Configuration for SAE hook attachment.

Two SAE families are supported, each with its own config dataclass:

- ``GemmaScopeSAEConfig`` — Google's Gemma-scope JumpReLU SAE suite for
  Gemma3. Hosted on HuggingFace under ``google/gemma-scope-2-{size}-{variant}``
  with per-(hook, layer, width, l0) folders. Indexed by Neuronpedia.
- ``QwenScopeSAEConfig`` — Qwen's TopK SAE suite for Qwen3. Hosted on
  HuggingFace under ``Qwen/SAE-Res-Qwen3-{size}-{variant}-W{width}-L0_{k}``
  with one ``layer{N}.sae.pt`` per layer. Not indexed by Neuronpedia.

Both expose the fields ``HookManager`` and ``loading.load_sae`` consume:
``layer_index, hook_type, d_in, d_sae, dtype, device, prefill_only,
read_only, collect_last_only``. The legacy ``SAEConfig`` symbol is
retained as an alias for ``GemmaScopeSAEConfig`` to avoid breaking
existing call sites that pre-date the multi-family split.
"""

from dataclasses import dataclass
from enum import Enum


class HookType(Enum):
    """Where in a decoder layer to attach the SAE hook."""

    RESID_POST = "resid_post"  # after full layer output (both residual adds)
    MLP_OUT = "mlp_out"  # after MLP block output (raw, pre-residual-add)
    ATTN_OUT = "attn_out"  # after attention block output (raw, pre-residual-add)
    POST_ATTN = "post_attn"  # residual stream after attn-residual-add, before MLP norm


# Reverse lookup: string -> HookType. Used by service layers that accept
# hook_type as a string parameter.
HOOK_TYPE_FROM_STR: dict[str, HookType] = {ht.value: ht for ht in HookType}


# d_sae value for each width suffix shared across SAE families.
WIDTH_TO_D_SAE: dict[str, int] = {
    "16k": 16384,
    "32k": 32768,
    "65k": 65536,
    "262k": 262144,
}

# Per-model-size hidden dimension (d_in) for Gemma-3.
MODEL_SIZE_TO_D_IN: dict[str, int] = {
    "1b": 1152,
    "4b": 2560,
    "12b": 3840,
    "27b": 5376,
}

# Per-model-size number of decoder layers for Gemma-3.
MODEL_SIZE_TO_LAYERS: dict[str, int] = {
    "1b": 26,
    "4b": 34,
    "12b": 48,
    "27b": 62,
}


@dataclass
class GemmaScopeSAEConfig:
    """Configuration for a single pretrained Gemma-scope SAE.

    The model_size, variant, hook_type, layer_index, width, and l0_size
    determine the HuggingFace repo and path:
        google/gemma-scope-2-{model_size}-{variant}
        {hook_type}/layer_{N}_width_{W}k_l0_{size}/params.safetensors

    The Neuronpedia model ID is derived as:
        gemma-3-{model_size}       (variant="pt")
        gemma-3-{model_size}-it    (variant="it")
    """

    layer_index: int
    hook_type: HookType = HookType.RESID_POST
    model_size: str = "4b"
    variant: str = "it"  # "pt" (pretrained/base) or "it" (instruction-tuned)
    width: str = "16k"  # "16k", "65k", or "262k"
    l0_size: str = "medium"  # "small", "medium", or "big"
    d_in: int | None = None  # auto-derived from model_size if None
    dtype: str = "bfloat16"
    device: str = "mps"
    collect_last_only: bool = False
    prefill_only: bool = False
    read_only: bool = True

    def __post_init__(self) -> None:
        if self.d_in is None:
            self.d_in = MODEL_SIZE_TO_D_IN.get(self.model_size, 2560)

    @property
    def repo_id(self) -> str:
        """HuggingFace repository ID for SAE weights."""
        return f"google/gemma-scope-2-{self.model_size}-{self.variant}"

    @property
    def neuronpedia_model_id(self) -> str:
        """Neuronpedia model ID for label lookups."""
        base = f"gemma-3-{self.model_size}"
        return base if self.variant == "pt" else f"{base}-{self.variant}"

    @property
    def d_sae(self) -> int:
        return WIDTH_TO_D_SAE[self.width]


# Backwards-compatible alias. Existing imports `from interpret.sae import
# SAEConfig` continue to work and resolve to the Gemma-scope config —
# correct for every current caller in this repo (probing engine,
# autointerpreter, diagnostics) which is Gemma-bound today.
SAEConfig = GemmaScopeSAEConfig


@dataclass
class QwenScopeSAEConfig:
    """Configuration for a single pretrained Qwen-scope TopK SAE.

    The repo layout is flat:
        Qwen/SAE-Res-Qwen3-{model_size}-{variant}-W{width}-L0_{k}
        layer{N}.sae.pt

    Only ``RESID_POST`` is meaningful for Qwen-scope (the SAEs are
    trained on the residual stream after each decoder layer).
    """

    layer_index: int
    k: int = 50  # 50 or 100 — selects the L0_50 or L0_100 trained variant
    width: str = "32k"  # only "32k" shipped today
    model_size: str = "1.7B"  # "1.7B" today; future: "0.6B", "4B", ...
    variant: str = "Base"  # "Base" today
    hook_type: HookType = HookType.RESID_POST
    d_in: int = 2048  # Qwen3-1.7B hidden size
    dtype: str = "bfloat16"
    device: str = "mps"
    collect_last_only: bool = False
    prefill_only: bool = False
    read_only: bool = True

    @property
    def repo_id(self) -> str:
        """HuggingFace repository ID for SAE weights."""
        return (
            f"Qwen/SAE-Res-Qwen3-{self.model_size}-{self.variant}-W{self.width.upper()}-L0_{self.k}"
        )

    def weights_filename(self, layer: int | None = None) -> str:
        """Filename of the per-layer weights inside the repo."""
        layer = self.layer_index if layer is None else layer
        return f"layer{layer}.sae.pt"

    @property
    def d_sae(self) -> int:
        return WIDTH_TO_D_SAE[self.width]
