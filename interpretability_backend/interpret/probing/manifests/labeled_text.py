"""Generic manifest for labeled text datasets (TSV / CSV / parquet).

Use case: a flat file ships one text column plus one or more label columns
(e.g. ``resources/datasets/SAE/trec.tsv`` with ``text``, ``coarse_label``,
``fine_label``, ``split``). Each row is one sample; sample IDs are the raw
text strings (the probing engine keys activation rows by prompt), so
duplicate texts are dropped by default.

Target encoding in ``get_rated_samples``:
  * integer / float columns pass through as ``int64`` class indices —
    TREC-style datasets that ship numeric labels keep their original ids;
  * object (string) columns are encoded alphabetically to ``int64``
    (deterministic without per-experiment configuration), with the mapping
    exposed on ``target_label_maps``.

``min_class_count`` exists because ``StratifiedKFold(n_splits=k)`` raises
when any class has fewer than ``k`` members (TREC ``fine_label``'s smallest
class has 4). It filters per target — other targets keep the full manifest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from interpret.probing.manifests.manifest_base import ManifestBuilder

SOURCE_NAME_DEFAULT = "labeled_text"


class LabeledTextManifestBuilder(ManifestBuilder):
    """Manifest backed by a single delimited/parquet file with text + labels.

    Args:
        path: Data file. Loader chosen by suffix: ``.tsv`` (tab), ``.csv``
            (comma), ``.parquet``.
        text_column: Column holding the prompt text.
        target_columns: Label columns to expose as probe targets. Required.
        source_name: Identifier expected as ``source`` in
            ``get_rated_samples`` / the experiment YAML ``targets`` block.
        split_column: Optional column naming the canonical split.
        splits: When set (requires ``split_column``), keep only rows whose
            split value is in this list. ``None`` keeps all rows.
        dedupe: Drop duplicate texts, keeping the first occurrence. Sample
            IDs are the raw texts, so duplicates would collide downstream
            (``ActivationDataset.subset`` requires unique IDs).
        limit: Deterministic head-N applied after all filtering — for smoke
            runs on a small slice.
        min_class_count: Per-target minimum class size, e.g.
            ``{"fine_label": 5}``. Applied inside ``get_rated_samples`` for
            that target only: rows whose class has fewer members (counted
            after dedupe/split/limit) are excluded from that target's
            returned samples.
    """

    def __init__(
        self,
        path: str | Path,
        text_column: str = "text",
        target_columns: list[str] | None = None,
        source_name: str = SOURCE_NAME_DEFAULT,
        split_column: str | None = None,
        splits: list[str] | None = None,
        dedupe: bool = True,
        limit: int | None = None,
        min_class_count: dict[str, int] | None = None,
    ) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"labeled-text data file not found: {self._path}")
        if not target_columns:
            raise ValueError("LabeledTextManifestBuilder: target_columns is required.")

        df = self._load(self._path)

        missing = [c for c in [text_column, *target_columns] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} missing from {self._path}. Available: {df.columns.tolist()}",
            )
        self._text_column = text_column
        self._target_columns = list(target_columns)
        self._source_name = source_name

        n_start = len(df)
        if splits is not None:
            if split_column is None:
                raise ValueError("splits filter requires split_column.")
            if split_column not in df.columns:
                raise ValueError(
                    f"split_column {split_column!r} not in {self._path}. "
                    f"Available: {df.columns.tolist()}",
                )
            df = df[df[split_column].isin(splits)]
            if df.empty:
                raise ValueError(
                    f"splits={splits} on column {split_column!r} produced an "
                    f"empty manifest from {self._path}.",
                )

        n_dupes = 0
        if dedupe:
            before = len(df)
            df = df.drop_duplicates(subset=[text_column], keep="first")
            n_dupes = before - len(df)

        df = df.reset_index(drop=True)
        if limit is not None:
            df = df.head(limit).reset_index(drop=True)

        self._df = df
        self._min_class_count = dict(min_class_count) if min_class_count else {}
        unknown = [c for c in self._min_class_count if c not in self._target_columns]
        if unknown:
            raise ValueError(
                f"min_class_count keys {unknown} are not target columns {self._target_columns}.",
            )

        # Alphabetical class-index encoders for string-typed targets only;
        # numeric targets pass through and get no entry here.
        self._label_maps: dict[str, dict[str, int]] = {}
        for col in self._target_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                values = sorted(df[col].dropna().astype(str).unique())
                self._label_maps[col] = {label: i for i, label in enumerate(values)}

        bits = [f"{n_start} rows read"]
        if splits is not None:
            bits.append(f"splits={splits}")
        if n_dupes:
            bits.append(f"dropped {n_dupes} duplicate texts")
        if limit is not None:
            bits.append(f"limit={limit}")
        print(
            f"LabeledTextManifestBuilder: {len(df)} samples ({'; '.join(bits)}), "
            f"targets={self._target_columns}, "
            f"encoded={sorted(self._label_maps)}",
        )

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(
            f"Unsupported file type {suffix!r} for {path} (expected .tsv/.csv/.parquet).",
        )

    @property
    def prompt_column(self) -> str:
        return self._text_column

    @property
    def target_columns(self) -> list[str]:
        return list(self._target_columns)

    @property
    def samples(self) -> list[str]:
        return self._df[self._text_column].astype(str).tolist()

    @property
    def target_label_maps(self) -> dict[str, dict[str, int]]:
        return {col: dict(m) for col, m in self._label_maps.items()}

    def build_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def get_rated_samples(
        self,
        source: str,
        column: str,
    ) -> tuple[list[str], np.ndarray]:
        if source != self._source_name:
            raise ValueError(
                f"Unknown source {source!r}. Expected {self._source_name!r}.",
            )
        if column not in self._target_columns:
            raise ValueError(
                f"Column {column!r} not in target_columns {self._target_columns}.",
            )
        valid = self._df[self._df[column].notna()]
        if valid.empty:
            raise ValueError(f"No non-null values for column {column!r}.")

        min_count = self._min_class_count.get(column)
        if min_count:
            counts = valid[column].value_counts()
            keep_classes = counts[counts >= min_count].index
            dropped = len(valid) - int(valid[column].isin(keep_classes).sum())
            if dropped:
                print(
                    f"LabeledTextManifestBuilder[{column}]: dropped {dropped} "
                    f"rows in classes with < {min_count} members "
                    f"({len(counts) - len(keep_classes)} classes).",
                )
            valid = valid[valid[column].isin(keep_classes)]
            if valid.empty:
                raise ValueError(
                    f"min_class_count={min_count} removed every row for column {column!r}.",
                )

        if column in self._label_maps:
            encoded = (
                valid[column].astype(str).map(self._label_maps[column]).to_numpy(dtype=np.int64)
            )
        else:
            encoded = valid[column].to_numpy()
            if np.issubdtype(encoded.dtype, np.floating):
                if not np.all(encoded == np.round(encoded)):
                    raise ValueError(
                        f"Column {column!r} has non-integral float values — "
                        "classification targets must be class indices.",
                    )
                encoded = encoded.astype(np.int64)
            else:
                encoded = encoded.astype(np.int64)
        return valid[self._text_column].astype(str).tolist(), encoded
