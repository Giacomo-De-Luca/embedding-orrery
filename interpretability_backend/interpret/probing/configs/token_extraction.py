"""Token-level residual extraction config — the two-stage pipeline's stage 1.

Unlike the pooled `gemma` extraction, this type keeps EVERY token position:
per `(layer, site)` key it stores one ragged ``Tensor[total_tokens, hidden]``
plus a shared ``metadata["token_offsets"]`` (length N+1) mapping samples to
row ranges. Downstream consumers (``sae_pooled``, ``residual_pooled``)
choose their own pooling — the extension the pooled path's docstrings have
always pointed at.

The dataset is NOT probeable (its dim 0 is tokens, not samples), so
``skip_probes`` defaults to True and the orchestrator excludes it from the
probe stage.

Sites are canonical, family-agnostic names mapped per family at extraction
time: ``resid_post`` (residual stream after the full layer), ``mlp_out``
and ``attn_out`` (raw sub-block outputs, pre-residual-add — the sites
gemma-scope mlp/attn SAEs are trained on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CANONICAL_SITES = ("resid_post", "mlp_out", "attn_out")

DEFAULT_QWEN_CHECKPOINT = "Qwen/Qwen3-1.7B"


@dataclass
class TokenLevelExtractionConfig:
    """Capture token-level residuals from a Gemma or Qwen checkpoint."""

    name: str  # required, drives cache filename
    family: Literal["gemma", "qwen"] = "gemma"
    type: Literal["token_residuals"] = "token_residuals"
    # gemma: None -> auto-resolve gemma-3-4b-it from the HF cache.
    # qwen: None -> DEFAULT_QWEN_CHECKPOINT.
    checkpoint: str | None = None
    layers: list[int] = field(default_factory=list)  # required non-empty
    sites: list[str] = field(default_factory=lambda: ["resid_post"])
    storage_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    device: str | None = None  # wrapper auto-resolves when None
    # Token-level datasets are excluded from the probe stage — dim 0 is
    # tokens, not samples. Exposed as a field so the orchestrator's check
    # stays a plain getattr.
    skip_probes: bool = True

    def __post_init__(self) -> None:
        if self.type != "token_residuals":
            raise ValueError(
                f"TokenLevelExtractionConfig.type must be 'token_residuals', got {self.type!r}",
            )
        if not self.name:
            raise ValueError("TokenLevelExtractionConfig.name is required.")
        if self.family not in ("gemma", "qwen"):
            raise ValueError(
                f"TokenLevelExtractionConfig.family must be 'gemma' or 'qwen', got {self.family!r}",
            )
        if not self.layers:
            raise ValueError(
                "TokenLevelExtractionConfig.layers is required — list the "
                "layer indices to capture.",
            )
        bad = [s for s in self.sites if s not in CANONICAL_SITES]
        if bad:
            raise ValueError(
                f"Unknown sites {bad}. Valid: {list(CANONICAL_SITES)}",
            )
        if not self.sites:
            raise ValueError("TokenLevelExtractionConfig.sites is required.")
        if self.family == "qwen" and self.sites != ["resid_post"]:
            # Qwen-scope SAEs are residual-stream only and the non-residual
            # qwen capture sites are untested in this pipeline. Relax this
            # check if raw-site qwen probing is ever needed.
            raise ValueError(
                f"family='qwen' supports sites=['resid_post'] only (got {self.sites}).",
            )
        if self.family == "qwen" and self.checkpoint is None:
            self.checkpoint = DEFAULT_QWEN_CHECKPOINT

    def cache_filename(self) -> str:
        return self.name
