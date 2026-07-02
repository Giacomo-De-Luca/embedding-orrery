"""Manifest for already-extracted feature vectors stored in a CSV.

Use case: a CSV ships per-row feature vectors plus categorical labels
(e.g. ``resources/features.csv`` with 72 numeric columns + ``condition``
and ``safety_label``). The probing engine treats each row as one sample,
keyed by a synthetic ``row_id`` (``row_0000``, ``row_0001``, ...) so
sample IDs are unique even when the source CSV has repeating identifiers.

Categorical targets are encoded to integer class indices in
``get_rated_samples``. Encoding is alphabetical so the mapping is
deterministic without per-experiment configuration: condition
``poetry=0, prose=1`` and safety_label ``safe=0, unsafe=1``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from interpret.probing.manifests.manifest_base import ManifestBuilder

ROW_ID_COLUMN = "row_id"
SOURCE_NAME = "features_csv"
DEFAULT_TARGET_COLUMNS = ("condition", "safety_label")
METADATA_COLUMNS = ("index", "sample_id", "condition", "safety_label", "verified")


class FeatureCSVManifestBuilder(ManifestBuilder):
    """Manifest backed by a single CSV with feature columns + label columns.

    Args:
        features_path: Path to the CSV. Must contain every name in
            ``DEFAULT_TARGET_COLUMNS`` plus at least one numeric feature
            column. Any column outside ``METADATA_COLUMNS`` is treated as
            a feature.
        target_columns: Subset of ``DEFAULT_TARGET_COLUMNS`` to expose.
        filters: Optional column-equality filter applied after loading.
            ``{"condition": "poetry"}`` keeps rows where ``condition``
            equals ``"poetry"``. Multiple keys are AND-combined.
            ``row_id`` values are assigned BEFORE filtering so they keep
            referencing the original CSV row positions.
        drop_sample_ids_csv: Optional path to a CSV with ``sample_id``
            (and optionally ``reason``) columns. Rows whose
            ``sample_id`` matches one in this CSV are dropped from the
            manifest. Applied after ``filters``. Sample IDs aren't
            unique in the source data (each maps to up to one poetry
            row + one prose row), and dropping covers every matching
            row regardless of condition.
        drop_sample_ids_reasons: When ``drop_sample_ids_csv`` is given
            and the CSV has a ``reason`` column, restrict the drop to
            rows whose reason is in this list (e.g. ``["refusal"]`` or
            ``["refusal", "Italian"]``). When None, every ID in the
            CSV is dropped.
        balance_classes_on: Optional column name to balance against by
            random undersampling. After ``filters`` and
            ``drop_sample_ids_csv`` have run, every non-minority class
            on this column is downsampled to the minority count using
            ``balance_seed``. Useful when one class (e.g. ``safe`` in
            prose) outnumbers the other and we want a probe to see
            equal counts without relying on sklearn's
            ``class_weight=balanced`` reweighting.
        balance_seed: RNG seed for the undersampling. Set explicitly
            so re-runs are deterministic.
    """

    def __init__(
        self,
        features_path: str | Path = "resources/features.csv",
        target_columns: list[str] | None = None,
        filters: dict[str, str | int | float | bool] | None = None,
        drop_sample_ids_csv: str | Path | None = None,
        drop_sample_ids_reasons: list[str] | None = None,
        balance_classes_on: str | None = None,
        balance_seed: int = 42,
    ) -> None:
        self._path = Path(features_path)
        if not self._path.exists():
            raise FileNotFoundError(f"features CSV not found: {self._path}")

        df = pd.read_csv(self._path)

        targets = list(target_columns) if target_columns else list(DEFAULT_TARGET_COLUMNS)
        missing = [c for c in targets if c not in df.columns]
        if missing:
            raise ValueError(
                f"Target columns {missing} missing from {self._path}. "
                f"Available: {df.columns.tolist()}",
            )
        self._target_columns = targets

        feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
        if not feature_cols:
            raise ValueError(
                f"No feature columns found in {self._path}. "
                f"Expected columns outside {METADATA_COLUMNS}.",
            )
        self._feature_columns = feature_cols

        df = df.reset_index(drop=True).copy()
        df[ROW_ID_COLUMN] = [f"row_{i:04d}" for i in range(len(df))]

        self._filters = dict(filters) if filters else {}
        if self._filters:
            for col, value in self._filters.items():
                if col not in df.columns:
                    raise ValueError(
                        f"Filter column {col!r} not in CSV. "
                        f"Available: {df.columns.tolist()}",
                    )
                df = df[df[col] == value]
            df = df.reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    f"Filters {self._filters} produced an empty manifest "
                    f"from {self._path}.",
                )

        self._drop_csv_path = (
            Path(drop_sample_ids_csv)
            if drop_sample_ids_csv is not None else None
        )
        self._drop_reasons = (
            list(drop_sample_ids_reasons)
            if drop_sample_ids_reasons else None
        )
        self._n_dropped = 0
        if self._drop_csv_path is not None:
            if not self._drop_csv_path.exists():
                raise FileNotFoundError(
                    f"drop_sample_ids_csv not found: {self._drop_csv_path}",
                )
            drop_df = pd.read_csv(self._drop_csv_path)
            if "sample_id" not in drop_df.columns:
                raise ValueError(
                    f"{self._drop_csv_path} must have a 'sample_id' column. "
                    f"Got: {drop_df.columns.tolist()}",
                )
            if self._drop_reasons is not None:
                if "reason" not in drop_df.columns:
                    raise ValueError(
                        f"drop_sample_ids_reasons set but "
                        f"{self._drop_csv_path} has no 'reason' column.",
                    )
                drop_df = drop_df[drop_df["reason"].isin(self._drop_reasons)]
            drop_ids = set(drop_df["sample_id"].astype(str))
            before = len(df)
            df = df[~df["sample_id"].astype(str).isin(drop_ids)]
            df = df.reset_index(drop=True)
            self._n_dropped = before - len(df)
            if df.empty:
                raise ValueError(
                    f"After dropping sample_ids from "
                    f"{self._drop_csv_path} (reasons={self._drop_reasons}), "
                    f"manifest is empty.",
                )

        self._balance_classes_on = balance_classes_on
        self._balance_seed = balance_seed
        self._n_balanced_dropped = 0
        if balance_classes_on is not None:
            if balance_classes_on not in df.columns:
                raise ValueError(
                    f"balance_classes_on={balance_classes_on!r} not in "
                    f"CSV columns {df.columns.tolist()}.",
                )
            counts_before = df[balance_classes_on].value_counts()
            if counts_before.empty:
                raise ValueError(
                    f"Column {balance_classes_on!r} has no non-null "
                    f"values; cannot balance.",
                )
            min_count = int(counts_before.min())
            rng = np.random.default_rng(balance_seed)
            keep_indices: list[int] = []
            for class_value, _ in counts_before.items():
                class_idx = df.index[df[balance_classes_on] == class_value].to_numpy()
                if len(class_idx) > min_count:
                    chosen = rng.choice(class_idx, size=min_count, replace=False)
                else:
                    chosen = class_idx
                keep_indices.extend(chosen.tolist())
            before = len(df)
            df = df.loc[sorted(keep_indices)].reset_index(drop=True)
            self._n_balanced_dropped = before - len(df)
            counts_after = df[balance_classes_on].value_counts().to_dict()
            print(
                f"FeatureCSVManifestBuilder balanced on "
                f"{balance_classes_on!r} (seed={balance_seed}): "
                f"{counts_before.to_dict()} -> {counts_after}",
            )

        self._df = df

        # Build deterministic class-index encoders (alphabetical) for each
        # categorical target. Stored so get_rated_samples can encode and
        # downstream consumers can recover the label names from
        # `target_label_maps`.
        self._label_maps: dict[str, dict[str, int]] = {
            col: {label: i for i, label in enumerate(sorted(df[col].dropna().unique()))}
            for col in self._target_columns
        }

        bits = []
        if self._filters:
            bits.append(f"filters={self._filters}")
        if self._n_dropped:
            reason_str = (
                f"reasons={self._drop_reasons}"
                if self._drop_reasons else "all reasons"
            )
            bits.append(f"dropped {self._n_dropped} rows ({reason_str})")
        if self._n_balanced_dropped:
            bits.append(
                f"balanced on {self._balance_classes_on!r} "
                f"(removed {self._n_balanced_dropped} majority rows)",
            )
        suffix = f" ({'; '.join(bits)})" if bits else ""
        print(
            f"FeatureCSVManifestBuilder: {len(df)} rows{suffix}, "
            f"{len(self._feature_columns)} features, "
            f"targets={self._target_columns}, "
            f"label_maps={self._label_maps}",
        )

    @property
    def prompt_column(self) -> str:
        return ROW_ID_COLUMN

    @property
    def target_columns(self) -> list[str]:
        return list(self._target_columns)

    @property
    def samples(self) -> list[str]:
        return self._df[ROW_ID_COLUMN].tolist()

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def target_label_maps(self) -> dict[str, dict[str, int]]:
        return {col: dict(m) for col, m in self._label_maps.items()}

    def build_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def get_rated_samples(
        self, source: str, column: str,
    ) -> tuple[list[str], np.ndarray]:
        if source != SOURCE_NAME:
            raise ValueError(
                f"Unknown source {source!r}. Expected {SOURCE_NAME!r}.",
            )
        if column not in self._target_columns:
            raise ValueError(
                f"Column {column!r} not in target_columns "
                f"{self._target_columns}.",
            )
        valid = self._df[self._df[column].notna()]
        if valid.empty:
            raise ValueError(f"No non-null values for column {column!r}.")
        mapping = self._label_maps[column]
        encoded = valid[column].map(mapping).to_numpy(dtype=np.int64)
        return valid[ROW_ID_COLUMN].tolist(), encoded
