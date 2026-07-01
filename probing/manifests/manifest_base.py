"""Abstract base for manifest builders.

A `ManifestBuilder` produces a (sample, target) table for the extraction
pipeline. Each subclass owns one data domain (psycholinguistic words,
colour names, etc.) and is the sole experiment-specific component in the
probing engine.

Subclasses must:
  * Declare `prompt_column`, `target_columns`, `samples`.
  * Implement `build_dataframe()` producing the wide manifest.
  * Implement `get_rated_samples(source, column)` returning aligned
    (samples, values) for a specific (data source, target column) pair.
    Single-source manifests should accept any source string and ignore it,
    or use a fixed source identifier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd


class ManifestBuilder(ABC):
    """Produces a manifest DataFrame with a prompt column + target columns."""

    @property
    @abstractmethod
    def prompt_column(self) -> str:
        """Name of the column containing the text input to the model."""

    @property
    @abstractmethod
    def target_columns(self) -> list[str]:
        """Column names treated as default regression targets."""

    @property
    @abstractmethod
    def samples(self) -> list[str]:
        """Canonical ordering of unique samples — drives activation tensor index."""

    @abstractmethod
    def build_dataframe(self) -> pd.DataFrame:
        """Construct the full manifest DataFrame.

        Must include `prompt_column` and every entry of `target_columns`.
        May include additional metadata columns.
        """

    @abstractmethod
    def get_rated_samples(
        self, source: str, column: str,
    ) -> tuple[list[str], np.ndarray]:
        """Return samples + target values for a (source, column) pair.

        Only returns samples with non-null values. Result arrays are aligned
        (`values[i]` is the target for `samples[i]`).

        Args:
            source: Identifier of the underlying data source (e.g. "glasgow",
                "concreteness"). For single-source manifests, the value is
                accepted but may be ignored.
            column: Target column name within that source.
        """

    @cached_property
    def sample_to_idx(self) -> dict[str, int]:
        """Reverse lookup from sample to its index in `self.samples`.

        Duplicates map to the first occurrence. Subclasses must return a
        deterministic `samples` list.
        """
        result: dict[str, int] = {}
        for i, s in enumerate(self.samples):
            if s not in result:
                result[s] = i
        return result

    def get_sample_indices(self, samples: list[str]) -> np.ndarray:
        """Map samples to their indices in `self.samples`.

        Raises KeyError if any sample is missing.
        """
        return np.array([self.sample_to_idx[s] for s in samples])

    def save(self, path: Path) -> Path:
        """Write the manifest DataFrame to a CSV at `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.build_dataframe().to_csv(path, index=False)
        return path
