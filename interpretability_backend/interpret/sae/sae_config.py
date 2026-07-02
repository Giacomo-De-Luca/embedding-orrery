"""Configuration for SAE hook attachment.

Two SAE families are supported, each with its own config dataclass:

- ``GemmaScopeSAEConfig`` — Google's Gemma-scope JumpReLU SAE suite for
  Gemma3. Hosted on HuggingFace under ``google/gemma-scope-2-{size}-{variant}``
  with per-(hook, layer, width, l0) folders. Indexed by Neuronpedia.
- ``QwenScopeSAEConfig`` — Qwen's TopK SAE suite for Qwen3 / Qwen3.5. Hosted
  on HuggingFace under ``Qwen/SAE-Res-{family}-{size}[-{variant}]-W{width}-L0_{k}``
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


# d_sae value for each width suffix, shared across SAE families. Gemma-scope
# uses 16k/32k/65k/262k; Qwen-scope adds 64k/80k/128k. Width *labels* differ per
# family (e.g. Gemma "65k" vs Qwen "64k") but each resolves to a latent count.
WIDTH_TO_D_SAE: dict[str, int] = {
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "65k": 65536,
    "80k": 81920,
    "128k": 131072,
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

    def identity(self) -> str:
        """Stable per-(family, width, l0) slug.

        Used by HookManager / ActivationStore as the third element of
        ``(layer, hook_type, sae_id)`` keys so two SAEs at the same
        ``(layer, hook_type)`` site can be disambiguated.
        """
        return f"w{self.width}_l0_{self.l0_size}"


# Backwards-compatible alias. Existing imports `from interpret.sae import
# SAEConfig` continue to work and resolve to the Gemma-scope config —
# correct for every current caller in this repo (probing engine,
# autointerpreter, diagnostics) which is Gemma-bound today.
SAEConfig = GemmaScopeSAEConfig


@dataclass(frozen=True)
class QwenScopeModelInfo:
    """Per-model Qwen-Scope facts that can't be inferred without a download.

    ``family`` is the segment in the SAE repo *name* ("Qwen3" or "Qwen3.5");
    the HuggingFace org is always "Qwen". ``variant`` is the "-Base" segment
    when present (``None`` for Qwen3.5-27B, whose repo has no "-Base" suffix).
    ``d_in`` is advisory — the loader validates it against the downloaded
    tensor and treats the tensor shape as the source of truth.
    """

    family: str
    variant: str | None
    d_in: int
    widths: tuple[str, ...]
    ks: tuple[int, ...]
    n_layers: int


# Registry of Qwen-Scope SAE checkpoints. Adding a model is a one-line entry;
# dims are confirmed against the downloaded weights at load time. The larger
# ``d_in``/``n_layers`` values should be checked against the model card on
# first download (the loader's shape check is the safety net for ``d_in``).
QWEN_SCOPE_MODELS: dict[str, QwenScopeModelInfo] = {
    "1.7B": QwenScopeModelInfo("Qwen3", "Base", 2048, ("32k",), (50, 100), 28),
    "2B": QwenScopeModelInfo("Qwen3.5", "Base", 2048, ("32k",), (50, 100), 24),
    "8B": QwenScopeModelInfo("Qwen3", "Base", 4096, ("64k",), (50, 100), 36),
    "27B": QwenScopeModelInfo("Qwen3.5", None, 5120, ("80k",), (50, 100), 64),
}


@dataclass
class QwenScopeSAEConfig:
    """Configuration for a single pretrained Qwen-scope TopK SAE.

    The repo layout is flat — one ``layer{N}.sae.pt`` per layer under:
        Qwen/SAE-Res-{family}-{model_size}[-{variant}]-W{width}-L0_{k}

    Naming is irregular across the suite (family prefix, the optional "-Base"
    segment, and width all vary by model), so those facts live in
    ``QWEN_SCOPE_MODELS`` keyed by ``model_size`` rather than being derived by
    string rule. ``d_in``/``d_sae`` here are advisory: the loader reads the
    real dims from the downloaded tensor and validates against the config.

    Only ``RESID_POST`` is meaningful for Qwen-scope (the SAEs are trained on
    the residual stream after each decoder layer).
    """

    layer_index: int
    model_size: str = "1.7B"  # key into QWEN_SCOPE_MODELS
    k: int = 50  # 50 or 100 — selects the L0_50 or L0_100 trained variant
    width: str | None = None  # None -> the model's first/only width
    hook_type: HookType = HookType.RESID_POST
    d_in: int | None = None  # None -> filled from the registry; loader validates
    dtype: str = "bfloat16"
    device: str = "mps"
    collect_last_only: bool = False
    prefill_only: bool = False
    read_only: bool = True

    def __post_init__(self) -> None:
        info = QWEN_SCOPE_MODELS.get(self.model_size)
        if info is None:
            raise ValueError(
                f"Unknown Qwen model_size {self.model_size!r}. Valid: {sorted(QWEN_SCOPE_MODELS)}"
            )
        if self.hook_type is not HookType.RESID_POST:
            raise ValueError(
                "Qwen-scope SAEs are residual-stream only; hook_type must be "
                f"RESID_POST, got {self.hook_type}."
            )
        if self.width is None:
            self.width = info.widths[0]
        if self.width not in info.widths:
            raise ValueError(
                f"width {self.width!r} not available for Qwen {self.model_size}. "
                f"Valid: {list(info.widths)}"
            )
        if self.k not in info.ks:
            raise ValueError(
                f"k={self.k} not available for Qwen {self.model_size}. Valid: {list(info.ks)}"
            )
        if self.d_in is None:
            self.d_in = info.d_in

    @property
    def variant(self) -> str | None:
        """The "-Base" segment for this model (``None`` when the repo has none)."""
        return QWEN_SCOPE_MODELS[self.model_size].variant

    @property
    def neuronpedia_model_id(self) -> str:
        """Stable model id for ``FeatureLabelStore`` keying.

        Qwen-scope has no Neuronpedia listing; the name keeps the attribute
        contract that ``FeatureLabelStore.params_from_config`` reads. E.g.
        ``qwen3.5-2B-base``.
        """
        info = QWEN_SCOPE_MODELS[self.model_size]
        base = f"{info.family.lower()}-{self.model_size}"
        return f"{base}-{info.variant.lower()}" if info.variant else base

    @property
    def repo_id(self) -> str:
        """HuggingFace repository ID for SAE weights."""
        info = QWEN_SCOPE_MODELS[self.model_size]
        variant_seg = f"-{info.variant}" if info.variant else ""
        return (
            f"Qwen/SAE-Res-{info.family}-{self.model_size}"
            f"{variant_seg}-W{self.width.upper()}-L0_{self.k}"
        )

    def weights_filename(self, layer: int | None = None) -> str:
        """Filename of the per-layer weights inside the repo."""
        layer = self.layer_index if layer is None else layer
        return f"layer{layer}.sae.pt"

    @property
    def d_sae(self) -> int:
        return WIDTH_TO_D_SAE[self.width]

    def identity(self) -> str:
        """Stable per-(width, k) slug (Qwen-scope uses TopK, not L0)."""
        return f"w{self.width}_l0_{self.k}"
