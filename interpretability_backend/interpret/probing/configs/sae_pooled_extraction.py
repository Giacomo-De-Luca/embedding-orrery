"""Pooled-SAE extraction config — the two-stage pipeline's stage 2.

Consumes a ``token_residuals`` extraction: encodes every token position of
one site through the family's SAE (JumpReLU for gemma-scope, TopK for
qwen-scope) and pools per sample, producing a probeable
``[N, d_kept]`` dataset per layer under ``(layer, "sae_max"|"sae_last")``.

Pooling semantics match the production document-activation path
(`interpret.sae.activation_store.max_pool_feature_acts`): ``max`` takes the
per-feature max over the sample's tokens with the BOS position masked when
the source model prepends BOS (the BOS activation sink otherwise tops every
sample identically); ``last`` encodes only the last-token residual.

The SAE family is read from the source dataset's metadata; the gemma-only
(``l0_size``/``variant``) and qwen-only (``k``) knobs are ignored by the
other family. ``width`` must be set explicitly per family in the YAML
("16k"-style for gemma, "32k" for qwen-1.7B) — the sae_analysis label
lookup reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from interpret.probing.configs.token_extraction import CANONICAL_SITES


@dataclass
class SAEPooledExtractionConfig:
    """SAE-encode token-level residuals and pool per sample."""

    name: str  # required, drives cache filename + probe folder
    source_extraction: str  # a token_residuals extraction
    site: str = "resid_post"  # which source site's keys to encode
    type: Literal["sae_pooled"] = "sae_pooled"
    pooling: Literal["max", "last"] = "max"
    layers: list[int] | None = None  # None -> all source layers for `site`

    # SAE identity. Family comes from the source dataset's metadata.
    width: str = "16k"  # gemma: 16k/65k/262k; qwen-1.7B: 32k (set in YAML)
    l0_size: str = "medium"  # gemma only
    variant: str = "it"  # gemma only: "pt" | "it"
    model_size: str = "4b"  # gemma: "4b"...; qwen: "1.7B"/"2B"/"8B"/"27B"
    k: int = 50  # qwen only (TopK)

    # Store pooled activations as scipy CSR instead of dense tensors.
    # Pooled SAE rows are typically 1-2% nonzero (TopK k=50 over ~15
    # tokens; JumpReLU short prompts), so CSR cuts disk/RAM ~25-50x —
    # required for large widths (64k/262k) and safety-scale sample counts.
    # Probes consume CSR natively (liblinear/lbfgs/libsvm); the trainer
    # then standardises scale-only (no mean centering — see
    # `sklearn_probes._fit_one`). Note: this field is part of the cache
    # identity — pre-existing sae_pooled sidecars (which lack the key)
    # fail loud with CacheMismatchError; delete the listed .pt/.yaml to
    # recompute stage 2 (stage-1 token caches are unaffected).
    sparse: bool = False

    exclude_bos: bool = True  # effective only when the source prepends BOS
    drop_dead_features: bool = True
    # A feature survives iff its pooled activation is > 0 in at least this
    # many samples. 1 == the classic any-sample-alive filter.
    min_active_samples: int = 1
    device: str = "cpu"  # "cuda" on A100, "mps" for local smoke
    batch_size_tokens: int = 8192  # encode chunk size for the max path

    def __post_init__(self) -> None:
        if self.type != "sae_pooled":
            raise ValueError(
                f"SAEPooledExtractionConfig.type must be 'sae_pooled', got {self.type!r}",
            )
        if not self.name:
            raise ValueError("SAEPooledExtractionConfig.name is required.")
        if not self.source_extraction:
            raise ValueError(
                "SAEPooledExtractionConfig.source_extraction is required — "
                "must reference a token_residuals extraction's name.",
            )
        if self.site not in CANONICAL_SITES:
            raise ValueError(
                f"Unknown site {self.site!r}. Valid: {list(CANONICAL_SITES)}",
            )
        if self.min_active_samples < 1:
            raise ValueError("min_active_samples must be >= 1.")

    @property
    def intermediate_key(self) -> str:
        """The intermediate string used in the output dataset's keys."""
        return f"sae_{self.pooling}"

    def cache_filename(self) -> str:
        return self.name
