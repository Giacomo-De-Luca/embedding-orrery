"""Shared I/O utilities for results storage.

Provides CSV writing and statistics computation used by both
BenchmarkResults and ExtractionResults.

Usage:
    from interpret.utils.results_io import write_csv, compute_stats
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to a CSV file with headers from the first row."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict) -> None:
    """Append a single row to a CSV file, writing the header if the file is new.

    All rows appended to the same file must have identical keys in the same order.
    """
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def compute_stats(values: list[float]) -> dict:
    """Compute mean, median, std for a list of values."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
    }
