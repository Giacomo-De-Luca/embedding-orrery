"""Local data file client for loading parquet, JSON, CSV and TSV files."""

import json as json_module
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..utils.media_kinds import MediaKind, MediaKindDetector, MediaReference


@dataclass
class LocalFileInfo:
    """Information about a local data file."""

    file_path: str
    file_type: str  # parquet, json, jsonl, csv, tsv
    columns: list[str]
    num_rows: int
    file_size_bytes: int
    error: str | None = None


@dataclass
class LocalFilePreview:
    """Preview rows from a local file."""

    file_path: str
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    error: str | None = None
    # Media columns of the preview: column -> "image" / "audio" / "video".
    column_kinds: dict[str, str] = field(default_factory=dict)


def _detect_file_type(file_path: str) -> str:
    """Detect file type from extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    elif suffix == ".json":
        return "json"
    elif suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    elif suffix == ".csv":
        return "csv"
    elif suffix == ".tsv":
        return "tsv"
    else:
        return "unknown"


def _count_lines(file_path: str) -> int:
    """Count lines in a file efficiently."""
    with open(file_path, "rb") as f:
        return sum(1 for _ in f) - 1  # Subtract header


def load_local_dataframe(file_path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load data from a local file into a pandas DataFrame.

    Supports parquet, JSON, JSONL/NDJSON, CSV and TSV formats.
    Optimized to read only required rows where possible.
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == ".parquet":
        # OPTIMIZATION: Use pyarrow to read only specific rows for preview
        if nrows is not None:
            # Limit reading to nrows
            table = pq.read_table(file_path)
            if len(table) > nrows:
                table = table.slice(0, nrows)
            df = table.to_pandas()
        else:
            df = pd.read_parquet(file_path)

    elif suffix == ".json":
        if nrows is not None and nrows <= 100:
            # Optimization: for small previews, avoid full pandas parse of large JSON
            with open(file_path) as f:
                data = json_module.load(f)
            if isinstance(data, list):
                data = data[:nrows]
            df = pd.DataFrame(data)
        else:
            df = pd.read_json(file_path)
            if nrows:
                df = df.head(nrows)

    elif suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(file_path, lines=True, nrows=nrows)

    elif suffix == ".csv":
        df = pd.read_csv(file_path, nrows=nrows)

    elif suffix == ".tsv":
        df = pd.read_csv(file_path, sep="\t", nrows=nrows)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return df


def get_local_file_info(file_path: str) -> LocalFileInfo:
    """
    Get information about a local data file efficiently WITHOUT reading the whole file.
    """
    path = Path(file_path)

    if not path.exists():
        return LocalFileInfo(
            file_path=file_path,
            file_type="unknown",
            columns=[],
            num_rows=0,
            file_size_bytes=0,
            error=f"File not found: {file_path}",
        )

    file_type = _detect_file_type(file_path)
    file_size = path.stat().st_size

    try:
        columns = []
        num_rows = 0

        if file_type == "parquet":
            # OPTIMIZATION: Read only metadata
            metadata = pq.read_metadata(file_path)
            num_rows = metadata.num_rows
            # Use read_schema for logical column names (read_metadata gives
            # physical leaf names which flatten nested types like list<float>)
            schema = pq.read_schema(file_path)
            columns = schema.names

        elif file_type == "csv":
            # OPTIMIZATION: Read only header for columns
            df_iter = pd.read_csv(file_path, nrows=0)
            columns = list(df_iter.columns)
            # Count lines for rows (approximation if quoting is complex, but usually good)
            # For huge files, maybe we just estimate? But exact count is expected.
            num_rows = _count_lines(file_path)

        elif file_type == "tsv":
            # OPTIMIZATION: Read only header
            df_iter = pd.read_csv(file_path, sep="\t", nrows=0)
            columns = list(df_iter.columns)
            num_rows = _count_lines(file_path)

        elif file_type == "jsonl":
            # OPTIMIZATION: Read first line for columns
            df_head = pd.read_json(file_path, lines=True, nrows=1)
            columns = list(df_head.columns)
            num_rows = _count_lines(file_path) + 1  # No header in JSONL usually

        elif file_type == "json":
            # Use stdlib json to avoid heavy pandas DataFrame construction
            with open(file_path) as f:
                data = json_module.load(f)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                columns = list(data[0].keys())
                num_rows = len(data)
            elif isinstance(data, dict):
                columns = list(data.keys())
                num_rows = 1
            else:
                columns = []
                num_rows = 0

        else:
            return LocalFileInfo(
                file_path=file_path,
                file_type=file_type,
                columns=[],
                num_rows=0,
                file_size_bytes=file_size,
                error=f"Unsupported file type: {path.suffix}",
            )

        return LocalFileInfo(
            file_path=file_path,
            file_type=file_type,
            columns=columns,
            num_rows=num_rows,
            file_size_bytes=file_size,
        )

    except Exception as e:
        return LocalFileInfo(
            file_path=file_path,
            file_type=file_type,
            columns=[],
            num_rows=0,
            file_size_bytes=file_size,
            error=str(e),
        )


def _is_missing(value: Any) -> bool:
    """None or a float NaN (pandas' missing value); bytes and dicts never are."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def get_local_file_preview(file_path: str, n_rows: int = 5) -> LocalFilePreview:
    """Get preview rows from a local data file.

    ``column_kinds`` names the columns holding media (``"image"`` / ``"audio"``
    / ``"video"``, detected from the preview rows); their cells show a path or
    a placeholder such as ``[image 34 KB]`` instead of raw bytes.
    """
    try:
        # Optimization: Only load necessary rows for preview
        # This now uses the optimized load_local_dataframe inside
        df = load_local_dataframe(file_path, nrows=n_rows)

        # Get preview rows
        preview_df = df.head(n_rows)
        detector = MediaKindDetector()
        column_kinds = {}
        for col in df.columns:
            kind = detector.kind_of_column(v for v in preview_df[col] if not _is_missing(v))
            if kind is not None:
                column_kinds[col] = kind.value

        # Convert to list of dicts, handling non-serializable types
        rows = []
        for _, row in preview_df.iterrows():
            row_dict = {}
            for col in df.columns:
                value = row[col]
                if col in column_kinds and not _is_missing(value):
                    kind = MediaKind(column_kinds[col])
                    row_dict[col] = MediaReference.describe(value, kind)
                # Check array-like types BEFORE pd.isna() to avoid
                # "truth value of array is ambiguous" error
                elif isinstance(value, np.ndarray):
                    row_dict[col] = value[:10].tolist() if len(value) > 10 else value.tolist()
                elif isinstance(value, list):
                    row_dict[col] = value[:10] if len(value) > 10 else value
                elif pd.isna(value):
                    row_dict[col] = None
                elif isinstance(value, (str, int, float, bool)):
                    row_dict[col] = value
                else:
                    row_dict[col] = str(value)[:200]
            rows.append(row_dict)

        return LocalFilePreview(
            file_path=file_path,
            columns=list(df.columns),
            rows=rows,
            total_rows=len(df),  # Logic might be weird if we only loaded partial.
            # But Preview doesn't strictly need total_rows if Info provides it.
            # We can leave it as len(df) which is partial.
            column_kinds=column_kinds,
        )
    except Exception as e:
        return LocalFilePreview(
            file_path=file_path, columns=[], rows=[], total_rows=0, error=str(e)
        )


def load_local_file_portion(
    file_path: str,
    n_rows: int | None = None,
    start_row: int | None = None,
    end_row: int | None = None,
    sample_n: int | None = None,
    sample_seed: int = 42,
    raw_columns: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], int]:
    """
    Load a portion of a local data file.

    Cells of ``raw_columns`` (the media columns of a multimodal job), and in
    every mode any media cell (``MediaReference.for_metadata``: bytes, media
    dicts with or without bytes), keep their value as read: ``extract_metadata``
    stores them as a path or a placeholder, where ``str(...)`` stored a dump of
    the whole file. Every other non-primitive cell is converted to a list or string.
    """
    raw = set(raw_columns)
    # For embedding, we likely need to read more.
    # If standard load, it reads all then samples.
    # Future optimization: push down limit to read_parquet/csv

    df = load_local_dataframe(file_path)
    total = len(df)

    if sample_n is not None:
        # Random sample (seeded through random_state)
        n = min(sample_n, total)
        df = df.sample(n=n, random_state=sample_seed)
    elif start_row is not None or end_row is not None:
        # Row range
        start = max(0, start_row or 0)
        end = min(total, end_row or total)
        df = df.iloc[start:end]
    elif n_rows is not None:
        # First N rows
        df = df.head(n_rows)

    # Convert to list of dicts
    rows = []
    for _, row in df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            if col in raw:
                row_dict[col] = None if _is_missing(value) else value
            # Check array-like types BEFORE pd.isna() to avoid
            # "truth value of array is ambiguous" error
            elif isinstance(value, np.ndarray):
                row_dict[col] = value.tolist()
            elif isinstance(value, list):
                row_dict[col] = value
            elif pd.isna(value):
                row_dict[col] = None
            elif isinstance(value, (str, int, float, bool)):
                row_dict[col] = value
            elif MediaReference.for_metadata(value) is not None:
                # Bytes or a media dict: extract_metadata stores its path or a placeholder
                row_dict[col] = value
            else:
                try:
                    row_dict[col] = value.tolist() if hasattr(value, "tolist") else str(value)
                except Exception:
                    row_dict[col] = str(value)
        rows.append(row_dict)

    return rows, total
