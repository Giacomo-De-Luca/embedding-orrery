"""Shared probe metric helpers.

Single source of truth for which metrics are higher-is-better and which
metric to surface as the primary one in summaries / plots. Also exposes
the per-experiment probe-results walker reused by
`consolidate.py` and `visualisations.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HIGHER_IS_BETTER: dict[str, bool] = {
    "val_r2": True,
    "val_accuracy": True,
    "val_f1_weighted": True,
    "val_pearson": True,
    "val_spearman": True,
    "val_loss": False,
    "val_mse": False,
    "val_mae": False,
    "val_lab_distance": False,
}

METRIC_PREFERENCE: tuple[str, ...] = (
    "val_lab_distance",
    "val_r2",
    "val_spearman",
    "val_accuracy",
    "val_loss",
)


def pick_metric(df: pd.DataFrame) -> str | None:
    """First metric in `METRIC_PREFERENCE` that has any non-NaN value in df."""
    for metric in METRIC_PREFERENCE:
        if metric in df.columns and df[metric].notna().any():
            return metric
    return None


def best_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    """Row from df with the best value of `metric`; None if no valid rows."""
    if metric not in df.columns:
        return None
    valid = df[df[metric].notna()]
    if valid.empty:
        return None
    higher = HIGHER_IS_BETTER.get(metric, True)
    idx = valid[metric].idxmax() if higher else valid[metric].idxmin()
    return valid.loc[idx]


def best_indices(
    df: pd.DataFrame,
    metric: str,
    group_cols: list[str] | None = None,
) -> pd.Index:
    """Indices of best-`metric` rows, optionally one per `group_cols` group.

    Returns an empty index when no rows have non-NaN `metric` values.
    `metric` direction comes from `HIGHER_IS_BETTER` (defaults to True).
    """
    if metric not in df.columns:
        return df.index[:0]
    valid = df[df[metric].notna()]
    if valid.empty:
        return valid.index
    higher = HIGHER_IS_BETTER.get(metric, True)
    if not group_cols:
        idx = valid[metric].idxmax() if higher else valid[metric].idxmin()
        return pd.Index([idx])
    grouped = valid.groupby(group_cols, dropna=False)[metric]
    return grouped.idxmax() if higher else grouped.idxmin()


def load_probe_results(
    experiment_dir: Path | str,
) -> dict[str, pd.DataFrame]:
    """Walk `<exp>/probes/<extraction>/<target>/<probe_kind>/probe_results.csv`.

    Returns `{"<extraction>/<target>/<probe_kind>": dataframe}`. Each frame
    is augmented with `extraction`, `target`, `probe_kind` columns so it
    can be concatenated downstream.
    """
    experiment_dir = Path(experiment_dir)
    probes_root = experiment_dir / "probes"
    if not probes_root.exists():
        return {}
    frames: dict[str, pd.DataFrame] = {}
    for ext_dir in sorted(probes_root.iterdir()):
        if not ext_dir.is_dir():
            continue
        for target_dir in sorted(ext_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            for probe_dir in sorted(target_dir.iterdir()):
                if not probe_dir.is_dir():
                    continue
                csv_path = probe_dir / "probe_results.csv"
                if not csv_path.exists():
                    continue
                try:
                    df = pd.read_csv(csv_path)
                except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                    print(f"  [skip] {csv_path}: {e}")
                    continue
                df["extraction"] = ext_dir.name
                df["target"] = target_dir.name
                df["probe_kind"] = probe_dir.name
                frames[f"{ext_dir.name}/{target_dir.name}/{probe_dir.name}"] = df
    return frames
