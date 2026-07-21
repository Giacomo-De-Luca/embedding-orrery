"""Extraction config for already-extracted feature vectors stored in a CSV.

This extraction does no model inference. It pulls a list of numeric
columns from the manifest's DataFrame, stacks them into an `[N, F]`
tensor, and emits an `ActivationDataset` keyed by a single
`(layer=0, intermediate=<intermediate_label>)` pair so the downstream
probe pipeline treats it like any other extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class CSVFeaturesExtractionConfig:
    """Pull pre-computed feature vectors from the manifest's DataFrame.

    Args:
        name: Cache filename + probe folder name (must be unique within
            the experiment).
        feature_columns: Explicit list of feature column names. When
            None, falls back to ``manifest.feature_columns``.
        intermediate_label: Name used in the single
            ``(layer, intermediate)`` activation key. Defaults to
            ``"features"`` to keep output paths human-readable.
    """

    name: str
    type: Literal["csv_features"] = "csv_features"
    feature_columns: list[str] | None = None
    intermediate_label: str = "features"

    def __post_init__(self) -> None:
        if self.type != "csv_features":
            raise ValueError(
                f"CSVFeaturesExtractionConfig.type must be 'csv_features', got {self.type!r}",
            )
        if not self.name:
            raise ValueError("CSVFeaturesExtractionConfig.name is required.")
        if self.feature_columns is not None and not self.feature_columns:
            raise ValueError(
                "CSVFeaturesExtractionConfig.feature_columns, if set, must be a non-empty list.",
            )

    def cache_filename(self) -> str:
        return self.name
