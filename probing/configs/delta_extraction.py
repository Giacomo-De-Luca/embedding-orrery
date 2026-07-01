"""Delta-extraction config — pairwise activation subtraction.

Consumes another extraction's `[N, d]` activations, pairs each non-baseline
row with a baseline row by a manifest column (e.g. each tinted image
paired with its grayscale counterpart sharing `source_image`), and emits
the per-row difference. Output keeps the source's layer/intermediate
keys; only the row dimension shrinks (baseline rows are removed).

Use cases:
  * Subtract grayscale-baseline activations from tinted-object
    activations to isolate per-image colour signal.
  * Subtract a "neutral prompt" activation from a "stimulus" activation
    in text-mode experiments to cancel framing/format effects.

The source extraction must produce activations for both baseline and
non-baseline rows; the engine selects baselines via `baseline_filter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DeltaExtractionConfig:
    """Pairwise subtraction of another extraction's activations."""

    name: str  # required, drives cache filename + folder
    source_extraction: str  # name of the extraction whose activations are subtracted

    # Manifest column that pairs baseline ↔ non-baseline rows. Each
    # non-baseline row's value of this column must match exactly one
    # baseline row's value.
    pairing_column: str

    # Manifest column-equality filter selecting baseline rows.
    # Example: `{is_grayscale: true}` picks the grayscale-baseline rows.
    # Multiple entries are AND-combined.
    baseline_filter: dict[str, Any] = field(default_factory=dict)

    type: Literal["delta"] = "delta"

    def __post_init__(self) -> None:
        if self.type != "delta":
            raise ValueError(
                f"DeltaExtractionConfig.type must be 'delta', got {self.type!r}",
            )
        if not self.name:
            raise ValueError("DeltaExtractionConfig.name is required.")
        if not self.source_extraction:
            raise ValueError(
                "DeltaExtractionConfig.source_extraction is required — "
                "must reference another extraction's name.",
            )
        if not self.pairing_column:
            raise ValueError(
                "DeltaExtractionConfig.pairing_column is required.",
            )
        if not self.baseline_filter:
            raise ValueError(
                "DeltaExtractionConfig.baseline_filter is required and "
                "must select at least one row per pairing_column value.",
            )

    def cache_filename(self) -> str:
        return self.name
