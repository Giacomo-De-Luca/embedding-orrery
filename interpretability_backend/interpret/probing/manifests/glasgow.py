"""Glasgow + Brysbaert concreteness psycholinguistic manifest builder.

Union of two psycholinguistic norms datasets — different methodologies and
scales, never merged but the sample list is the union so a single activation
tensor can serve probes against either source.

Sources:
  * `RatingSource.CONCRETENESS` — Brysbaert et al. concreteness norms
    (~40K words). Key column: `Conc.M`.
  * `RatingSource.GLASGOW` — Glasgow psycholinguistic norms (~4.7K words).
    Key columns: `concreteness`, `imageability`, `valence`, `arousal`,
    `familiarity`, `aoa`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from interpret.probing.manifests.manifest_base import ManifestBuilder


class RatingSource(Enum):
    """Psycholinguistic rating dataset identifier."""

    CONCRETENESS = "concreteness"
    GLASGOW = "glasgow"


@dataclass
class GlasgowPaths:
    """Paths to the two source CSVs."""

    concreteness: Path = Path("resources/psycolinguistics/concreteness.tsv")
    glasgow: Path = Path("resources/psycolinguistics/glasgow_norm.csv")


class GlasgowManifestBuilder(ManifestBuilder):
    """Union of Brysbaert concreteness + Glasgow norms.

    `target_columns` exposes a default selectable column set (Glasgow
    `concreteness` by default). Any column from either source is accessible
    via `get_rated_samples(source, column)`.
    """

    def __init__(
        self,
        paths: GlasgowPaths | None = None,
        default_targets: list[str] | None = None,
        glasgow_only: bool = False,
    ) -> None:
        """Build the manifest.

        Args:
            paths: Source CSV paths. Defaults to `GlasgowPaths()`.
            default_targets: Columns to expose as `target_columns`.
            glasgow_only: If True, restrict `samples` (and the manifest
                DataFrame) to Glasgow words only — excludes Brysbaert
                concreteness words. Useful for re-using a Gemma extraction
                cached over Glasgow words alone.
        """
        cfg = paths if paths is not None else GlasgowPaths()

        self._concreteness_df = self._load_source(
            cfg.concreteness,
            source_name="concreteness",
            word_column="Word",
            sep="\t",
        )
        self._glasgow_df = self._load_source(
            cfg.glasgow,
            source_name="glasgow",
            word_column="word",
            sep=",",
        )

        conc_words = set(self._concreteness_df["word_lower"])
        glas_words = set(self._glasgow_df["word_lower"])
        if glasgow_only:
            self._samples: list[str] = sorted(glas_words)
        else:
            self._samples = sorted(conc_words | glas_words)

        self._default_targets = list(default_targets) if default_targets else ["concreteness"]
        self._glasgow_only = glasgow_only

        overlap = conc_words & glas_words
        print(
            f"GlasgowManifestBuilder: {len(conc_words)} concreteness words, "
            f"{len(glas_words)} Glasgow words, "
            f"{len(overlap)} overlap, "
            f"{len(self._samples)} samples "
            f"({'glasgow_only' if glasgow_only else 'union'})"
        )

    @property
    def prompt_column(self) -> str:
        return "word"

    @property
    def target_columns(self) -> list[str]:
        return list(self._default_targets)

    @property
    def samples(self) -> list[str]:
        return self._samples

    def build_dataframe(self) -> pd.DataFrame:
        """Build a wide-form manifest with one row per word rated on every default target.

        Default targets are resolved via `get_rated_samples(source, column)` —
        callers must therefore set `default_targets` to columns that
        unambiguously belong to one source. Words missing any default target
        are dropped.
        """
        rated_sets: dict[str, dict[str, float]] = {}
        for col in self._default_targets:
            source = self._infer_source(col)
            words, ratings = self.get_rated_samples(source, col)
            rated_sets[col] = dict(zip(words, ratings, strict=True))

        common_words = sorted(set.intersection(*(set(d.keys()) for d in rated_sets.values())))

        rows = [
            {
                self.prompt_column: w,
                **{col: rated_sets[col][w] for col in self._default_targets},
            }
            for w in common_words
        ]
        return pd.DataFrame(rows)

    def get_rated_samples(
        self,
        source: str,
        column: str,
    ) -> tuple[list[str], np.ndarray]:
        """Return words + ratings for `(source, column)`.

        Args:
            source: One of `"concreteness"` or `"glasgow"`. The string form
                of `RatingSource`.
            column: Target column within that source.
        """
        try:
            src_enum = RatingSource(source)
        except ValueError as e:
            valid = [s.value for s in RatingSource]
            raise ValueError(
                f"Unknown source {source!r}. Valid: {valid}",
            ) from e

        df = self._concreteness_df if src_enum is RatingSource.CONCRETENESS else self._glasgow_df
        if column not in df.columns:
            raise ValueError(
                f"Column {column!r} not in {src_enum.value} dataset. "
                f"Available: {[c for c in df.columns if c != 'word_lower']}",
            )
        return self._filter_rated(df, column)

    @staticmethod
    def _filter_rated(
        df: pd.DataFrame,
        column: str,
    ) -> tuple[list[str], np.ndarray]:
        """Drop nulls, return aligned words + values.

        Raises if duplicates remain on `word_lower` after filtering — the
        manifest constructor already de-duplicates the source data; a
        late duplicate signals a data problem the caller should know about.
        """
        valid = df[df[column].notna()]
        if valid.empty:
            raise ValueError(f"No non-null values for column {column!r}.")
        dups = valid["word_lower"].value_counts()
        dups = dups[dups > 1]
        if not dups.empty:
            raise ValueError(
                f"Duplicate words after filtering on column {column!r}: "
                f"{dups.head(10).to_dict()}{' ...' if len(dups) > 10 else ''}",
            )
        return (
            valid["word_lower"].tolist(),
            valid[column].to_numpy(dtype=np.float32),
        )

    @staticmethod
    def _load_source(
        path: Path,
        *,
        source_name: str,
        word_column: str,
        sep: str,
    ) -> pd.DataFrame:
        """Load a source CSV, normalise word column, raise on duplicates.

        Replaces the previous silent dedup behaviour. Genuine duplicates
        in the source data should be investigated — usually casing
        variants (e.g. "Apple" + "apple" both lowercase to "apple") that
        the dataset author should have resolved upstream.
        """
        df = pd.read_csv(path, sep=sep)
        df = df.dropna(subset=[word_column]).copy()
        df["word_lower"] = df[word_column].astype(str).str.lower().str.strip()
        df = df[df["word_lower"].str.len() > 0]

        dups = df["word_lower"].value_counts()
        dups = dups[dups > 1]
        if not dups.empty:
            raise ValueError(
                f"{source_name}: {len(dups)} duplicate words after "
                f"lowercase + strip. First 10: {dups.head(10).to_dict()}. "
                f"Resolve upstream or pass a cleaned file.",
            )
        return df

    def _infer_source(self, column: str) -> str:
        """Disambiguate which source a column belongs to.

        Used only by `build_dataframe()` for the default-targets path.
        Raises if the column appears in both (e.g. `concreteness` exists
        in both sources) — callers must specify explicitly via
        `get_rated_samples(source, column)`.
        """
        in_conc = column in self._concreteness_df.columns and column != "word_lower"
        in_glas = column in self._glasgow_df.columns and column != "word_lower"
        if in_conc and in_glas:
            raise ValueError(
                f"Column {column!r} exists in both concreteness and glasgow "
                f"sources. Use get_rated_samples(source, column) to "
                f"disambiguate.",
            )
        if in_conc:
            return RatingSource.CONCRETENESS.value
        if in_glas:
            return RatingSource.GLASGOW.value
        raise ValueError(f"Column {column!r} not in either source.")

    @property
    def concreteness_columns(self) -> list[str]:
        return [c for c in self._concreteness_df.columns if c != "word_lower"]

    @property
    def glasgow_columns(self) -> list[str]:
        return [c for c in self._glasgow_df.columns if c != "word_lower"]
