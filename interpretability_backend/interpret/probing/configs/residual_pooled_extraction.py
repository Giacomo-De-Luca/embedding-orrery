"""Residual-pooled extraction config — raw-residual baselines from the
token-level cache.

Pools a ``token_residuals`` extraction's ragged token tensors per sample
WITHOUT SAE encoding, producing the ``[N, hidden]`` per-layer baseline the
SAE probes are compared against — for both families, from the same forward
pass that fed the SAEs (no second model run, unlike the legacy pooled
``gemma`` extraction, which also has no qwen counterpart).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from interpret.probing.configs.token_extraction import CANONICAL_SITES


@dataclass
class ResidualPooledExtractionConfig:
    """Pool token-level residuals per sample (no SAE)."""

    name: str  # required
    source_extraction: str  # a token_residuals extraction
    site: str = "resid_post"
    type: Literal["residual_pooled"] = "residual_pooled"
    pooling: Literal["last", "max", "mean"] = "last"
    layers: list[int] | None = None  # None -> all source layers for `site`
    exclude_bos: bool = True  # effective for max/mean when source prepends BOS

    def __post_init__(self) -> None:
        if self.type != "residual_pooled":
            raise ValueError(
                f"ResidualPooledExtractionConfig.type must be 'residual_pooled', got {self.type!r}",
            )
        if not self.name:
            raise ValueError("ResidualPooledExtractionConfig.name is required.")
        if not self.source_extraction:
            raise ValueError(
                "ResidualPooledExtractionConfig.source_extraction is required.",
            )
        if self.site not in CANONICAL_SITES:
            raise ValueError(
                f"Unknown site {self.site!r}. Valid: {list(CANONICAL_SITES)}",
            )

    @property
    def intermediate_key(self) -> str:
        """The intermediate string used in the output dataset's keys."""
        return f"res_{self.pooling}"

    def cache_filename(self) -> str:
        return self.name
