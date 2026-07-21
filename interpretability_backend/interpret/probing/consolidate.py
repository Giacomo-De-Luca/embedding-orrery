"""Cross-experiment results aggregation.

Walks an experiments root, reads each `<experiment>/experiment.yaml` +
all `<experiment>/probes/<probe_name>/probe_results.csv`, and writes
four tables + figures under `<output_dir>`:

  * `consolidated_long.csv` — every probe row, tagged with experiment
    name + extraction type + manifest source.
  * `wide_<metric>.csv`     — pivoted: experiment × probe → value of
    the chosen metric.
  * `best_per_condition.csv` — best (probe, layer, intermediate) per
    experiment + per probe folder.
  * `summary.md` — human-readable cross-experiment table + embedded
    cross-experiment figures.
  * `figures/*.png` — seaborn plots emitted by `ConsolidatedVisualiser`.

Produces the cross-experiment summary table that lives at
`<experiments_root>/_consolidated/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from interpret.probing.utils.metrics import (
    HIGHER_IS_BETTER,
    load_probe_results,
)


@dataclass
class ConsolidateConfig:
    experiments_root: Path = Path("resources/experiments")
    output_dir: Path = Path("resources/experiments/_consolidated")
    primary_metric: str = "val_r2"
    secondary_metric: str = "val_spearman"
    exclude_dirs: tuple[str, ...] = field(
        default_factory=lambda: ("_consolidated",),
    )

    def __post_init__(self) -> None:
        self.experiments_root = Path(self.experiments_root)
        self.output_dir = Path(self.output_dir)


# ── Public API ───────────────────────────────────────────────────────────────


def consolidate(config: ConsolidateConfig | None = None) -> dict[str, Path]:
    """Run the consolidation pipeline; return paths to written outputs."""
    config = config or ConsolidateConfig()
    if not config.experiments_root.exists():
        raise FileNotFoundError(
            f"Experiments root not found: {config.experiments_root}",
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    long_df = _collect(config)
    if long_df.empty:
        print("No probe results found.")
        return {}

    outputs: dict[str, Path] = {}
    outputs["long"] = _write_long(long_df, config)
    outputs["wide_primary"] = _write_wide(
        long_df,
        config.primary_metric,
        config,
    )
    if config.secondary_metric != config.primary_metric:
        outputs["wide_secondary"] = _write_wide(
            long_df,
            config.secondary_metric,
            config,
        )
    outputs["best"] = _write_best(long_df, config)

    # Render figures *before* the markdown so the summary can embed them.
    try:
        from interpret.probing.visualisations import (
            ConsolidatedVisualiser,
        )

        ConsolidatedVisualiser(
            config.output_dir,
            primary_metric=config.primary_metric,
        ).render()
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb

        print(f"  [warn] consolidated visualisations failed: {exc}")
        _tb.print_exc()

    outputs["markdown"] = _write_markdown(long_df, config)

    print("\nOutputs:")
    for k, p in outputs.items():
        print(f"  {k:14s} -> {p}")
    return outputs


def refresh_summary(consolidated_dir: Path | str) -> Path:
    """Rebuild `summary.md` from an existing `consolidated_long.csv`.

    Used by the visualisations CLI so the user can regenerate the summary
    + figures without re-walking the experiments tree.
    """
    consolidated_dir = Path(consolidated_dir)
    long_path = consolidated_dir / "consolidated_long.csv"
    if not long_path.exists():
        raise FileNotFoundError(
            f"{long_path} not found — run consolidate() first.",
        )
    long_df = pd.read_csv(long_path)
    config = ConsolidateConfig(
        experiments_root=consolidated_dir.parent,
        output_dir=consolidated_dir,
    )
    return _write_markdown(long_df, config)


# ── Collection ───────────────────────────────────────────────────────────────


def _collect(config: ConsolidateConfig) -> pd.DataFrame:
    """Walk experiments root, pull every probe_results.csv into one frame.

    Reuses `utils.metrics.load_probe_results` per experiment and decorates
    each row with experiment-level metadata. Recurses one level into
    non-underscore-prefixed group folders (e.g. ``poetry/``) so grouped
    experiments are picked up alongside flat ones.
    """
    frames: list[pd.DataFrame] = []
    for child in sorted(config.experiments_root.iterdir()):
        if not _is_walkable(child, config):
            continue
        if (child / "experiment.yaml").exists():
            _ingest_experiment(child, frames, prefix=child.name)
            continue
        for nested in sorted(child.iterdir()):
            if not _is_walkable(nested, config):
                continue
            if (nested / "experiment.yaml").exists():
                _ingest_experiment(
                    nested,
                    frames,
                    prefix=f"{child.name}/{nested.name}",
                )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _is_walkable(path: Path, config: ConsolidateConfig) -> bool:
    """Skip files, leading-underscore folders, and explicit excludes."""
    if not path.is_dir():
        return False
    if path.name.startswith("_"):
        return False
    if path.name in config.exclude_dirs:
        return False
    return True


def _ingest_experiment(
    exp_dir: Path,
    frames: list[pd.DataFrame],
    *,
    prefix: str,
) -> None:
    """Read one experiment's probe results into ``frames``."""
    meta = _load_experiment_meta(exp_dir)
    per_exp = load_probe_results(exp_dir)
    for key, df in per_exp.items():
        df = df.copy()
        df["experiment"] = meta["name"]
        df["manifest_source"] = meta["manifest_source"]
        frames.append(df)
        print(f"  {prefix}/probes/{key}: {len(df)} rows")


def _load_experiment_meta(exp_dir: Path) -> dict:
    """Pull experiment-level metadata out of experiment.yaml."""
    cfg_path = exp_dir / "experiment.yaml"
    meta = {"name": exp_dir.name, "manifest_source": ""}
    if not cfg_path.exists():
        return meta
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    meta["name"] = cfg.get("name", exp_dir.name)
    manifest = cfg.get("manifest", {}) or {}
    meta["manifest_source"] = manifest.get("path", "").split(":")[-1]
    return meta


# ── Writers ──────────────────────────────────────────────────────────────────


def _write_long(df: pd.DataFrame, config: ConsolidateConfig) -> Path:
    out = config.output_dir / "consolidated_long.csv"
    df.to_csv(out, index=False)
    print(f"\nLong-format: {len(df)} rows -> {out}")
    return out


def _write_wide(
    df: pd.DataFrame,
    metric: str,
    config: ConsolidateConfig,
) -> Path:
    if metric not in df.columns:
        raise ValueError(f"Metric {metric!r} not in consolidated columns.")
    pivot = df.pivot_table(
        index=["experiment", "extraction", "manifest_source", "target", "layer", "intermediate"],
        columns="probe_kind",
        values=metric,
        aggfunc="max",
        dropna=False,
    ).reset_index()
    probe_cols = [
        c
        for c in pivot.columns
        if c
        not in {"experiment", "extraction", "manifest_source", "target", "layer", "intermediate"}
    ]
    if probe_cols:
        pivot = pivot.dropna(subset=probe_cols, how="all")
    pivot = pivot.sort_values(
        ["experiment", "extraction", "target", "layer", "intermediate"],
    )
    out = config.output_dir / f"wide_{metric}.csv"
    pivot.to_csv(out, index=False)
    print(f"Wide ({metric}): {pivot.shape[0]} rows × {pivot.shape[1]} cols -> {out}")
    return out


def _write_best(df: pd.DataFrame, config: ConsolidateConfig) -> Path:
    """Best (layer, intermediate) per (experiment, target, probe_kind)."""
    primary = config.primary_metric
    out = config.output_dir / "best_per_condition.csv"
    if primary not in df.columns:
        pd.DataFrame().to_csv(out, index=False)
        return out
    valid = df[df[primary].notna()]
    if valid.empty:
        pd.DataFrame().to_csv(out, index=False)
        return out
    higher = HIGHER_IS_BETTER.get(primary, True)
    group_cols = [
        "experiment",
        "extraction",
        "manifest_source",
        "target",
        "probe_kind",
    ]
    if higher:
        idx = valid.groupby(group_cols, dropna=False)[primary].idxmax()
    else:
        idx = valid.groupby(group_cols, dropna=False)[primary].idxmin()
    best = valid.loc[idx]
    keep = group_cols + ["layer", "intermediate", primary]
    keep = [c for c in keep if c in best.columns]
    secondary = config.secondary_metric
    if secondary in best.columns:
        keep.append(secondary)
    best = best[keep].sort_values(group_cols)
    best.to_csv(out, index=False)
    print(f"Best per condition: {len(best)} rows -> {out}")
    return out


def _write_markdown(df: pd.DataFrame, config: ConsolidateConfig) -> Path:
    primary = config.primary_metric
    lines: list[str] = ["# Cross-experiment summary", ""]
    if primary not in df.columns:
        lines.append(f"_Primary metric `{primary}` not present._")
        out = config.output_dir / "summary.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    valid = df[df[primary].notna()]
    higher = HIGHER_IS_BETTER.get(primary, True)
    group_cols = [
        "experiment",
        "extraction",
        "manifest_source",
        "target",
        "probe_kind",
    ]

    lines.append(
        f"## Best per (experiment, target, probe_kind) — by `{primary}`",
    )
    lines.append("")
    lines.append(
        "| experiment | extraction | manifest | target | probe | layer | "
        f"intermediate | {primary} |",
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    if not valid.empty:
        if higher:
            idx = valid.groupby(group_cols, dropna=False)[primary].idxmax()
        else:
            idx = valid.groupby(group_cols, dropna=False)[primary].idxmin()
        best = valid.loc[idx].sort_values(group_cols)
        for _, r in best.iterrows():
            lines.append(
                f"| {r['experiment']} | {r['extraction']} | "
                f"{r['manifest_source']} | {r['target']} | "
                f"{r['probe_kind']} | {r['layer']} | {r['intermediate']} | "
                f"{r[primary]:.4f} |",
            )
    lines.append("")

    figures_dir = config.output_dir / "figures"
    figure_pngs = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []
    if figure_pngs:
        lines.append("## Figures")
        lines.append("")
        for png in figure_pngs:
            label = png.stem.replace("_", " ")
            lines.append(f"![{label}](figures/{png.name})")
            lines.append("")

    out = config.output_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown: -> {out}")
    return out


def main() -> None:
    consolidate()


if __name__ == "__main__":
    main()
