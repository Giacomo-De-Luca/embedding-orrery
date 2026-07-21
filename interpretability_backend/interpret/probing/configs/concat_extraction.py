"""Concat extraction config — stack pooled SAE features across layers/sites.

Concatenates the ``(layer, intermediate)`` keys of one or more pooled
extractions (typically ``sae_pooled``) into a single wide
``[N, sum(d_kept)]`` key ``(0, "concat")`` so a probe can select
task-relevant features jointly across the whole depth of the model.
Per-column identity is preserved in ``metadata["feature_names"]``
(``"L{layer}_{site}_f{true_index}"``), which the probe stage picks up for
`feature_importance.csv`.

Note: with a single (layer, intermediate) key, the sklearn CSV writer drops
the constant layer/intermediate columns and the cross-experiment
``consolidate`` pivots skip this extraction — same pre-existing constraint
as ``csv_features``. Per-probe CSVs, summary.json and feature importance
are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ConcatExtractionConfig:
    """Concatenate pooled extractions' keys into one wide feature matrix."""

    name: str  # required
    # One or more pooled extraction names, concatenated in list order.
    source_extractions: list[str] = field(default_factory=list)
    type: Literal["concat"] = "concat"
    layers: list[int] | None = None  # optional filter applied to every source

    def __post_init__(self) -> None:
        if self.type != "concat":
            raise ValueError(
                f"ConcatExtractionConfig.type must be 'concat', got {self.type!r}",
            )
        if not self.name:
            raise ValueError("ConcatExtractionConfig.name is required.")
        if not self.source_extractions:
            raise ValueError(
                "ConcatExtractionConfig.source_extractions is required — "
                "list at least one pooled extraction name.",
            )

    def cache_filename(self) -> str:
        return self.name
