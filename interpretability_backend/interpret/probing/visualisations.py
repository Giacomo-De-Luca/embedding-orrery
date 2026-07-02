"""Seaborn-based figures for probing experiments.

Two visualisers:

  * `ExperimentVisualiser(experiment_dir).render()` — emits per-extraction
    layer curves, probe×target heatmaps, best-metric bar charts, and
    optional cross-extraction comparisons + RGB/LAB channel panels into
    `<experiment_dir>/figures/`.
  * `ConsolidatedVisualiser(consolidated_dir).render()` — emits two
    cross-experiment plots into `<consolidated_dir>/figures/` from the
    `consolidated_long.csv` produced by `consolidate.py`.

The orchestrator calls `ExperimentVisualiser` after `report.write_summary`.
Both classes are also runnable standalone:

    uv run python -m interpret.probing.visualisations \\
        resources/experiments/glasgow_psycholinguistic_norms

    uv run python -m interpret.probing.visualisations \\
        resources/experiments/_consolidated --consolidated
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from interpret.probing.utils.metrics import (
    HIGHER_IS_BETTER,
    best_indices,
    load_probe_results,
    pick_metric,
)

_RGB_TARGETS = ("R", "G", "B")
_LAB_TARGETS = ("L", "a", "b")


def _setup_style() -> None:
    """Apply a consistent seaborn theme across every figure."""
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="deep",
        font_scale=0.9,
    )


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    """Tight-layout, savefig at 150 dpi, close — used by every plot."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Per-experiment
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Layout:
    """Parameters that scale figure size with the data."""

    base_width: float = 4.0
    base_height: float = 3.2
    max_facets_per_row: int = 4


class ExperimentVisualiser:
    """Render seaborn figures for a single experiment directory.

    Reads the experiment's probe outputs and `experiment.yaml`; writes
    PNGs to `<experiment_dir>/figures/`.
    """

    def __init__(
        self,
        experiment_dir: Path | str,
        layout: _Layout | None = None,
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.figures_dir = self.experiment_dir / "figures"
        self.layout = layout or _Layout()

    def render(self) -> list[Path]:
        """Render every figure for this experiment. Returns written paths."""
        _setup_style()
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        long_df = self._build_long_df()
        if long_df.empty:
            print(
                f"[visualisations] No probe results under "
                f"{self.experiment_dir / 'probes'} — nothing to plot.",
            )
            return []

        written: list[Path] = []
        extractions = sorted(long_df["extraction"].unique())
        targets = sorted(long_df["target"].unique())

        for extraction in extractions:
            ext_df = long_df[long_df["extraction"] == extraction]
            written += self._render_layer_curves(ext_df, extraction)
            written += self._render_probe_target_heatmap(ext_df, extraction)
            written += self._render_best_metric_bars(ext_df, extraction)
            written += self._render_colour_channel_panels(ext_df, extraction, targets)

        if len(extractions) > 1:
            written += self._render_extraction_comparison(long_df, extractions, targets)

        print(
            f"[visualisations] Wrote {len(written)} figure(s) to "
            f"{self.figures_dir}",
        )
        return written

    def figure_index(self) -> dict[str, list[Path]]:
        """Return `{group_label: [figure_paths]}` for embedding into summary.md.

        Groups by extraction when figure stems are produced via
        `_extraction_figure_path()` (kind__extraction.png); cross-extraction
        figures fall under the `"cross-extraction"` key.
        """
        if not self.figures_dir.exists():
            return {}
        index: dict[str, list[Path]] = {}
        for png in sorted(self.figures_dir.glob("*.png")):
            stem = png.stem
            if "__" in stem:
                _kind, extraction = stem.split("__", 1)
                index.setdefault(extraction, []).append(png)
            else:
                index.setdefault("cross-extraction", []).append(png)
        return index

    def _extraction_figure_path(self, kind: str, extraction: str) -> Path:
        """Canonical filename for a per-extraction figure."""
        return self.figures_dir / f"{kind}__{extraction}.png"

    # ── data prep ────────────────────────────────────────────────────────────

    def _build_long_df(self) -> pd.DataFrame:
        """Concat every probe_results.csv into one tidy frame."""
        frames = load_probe_results(self.experiment_dir)
        if not frames:
            return pd.DataFrame()
        long_df = pd.concat(frames.values(), ignore_index=True)
        for col in ("layer", "intermediate", "probe_kind", "extraction", "target"):
            if col not in long_df.columns:
                long_df[col] = pd.NA
        return long_df

    @staticmethod
    def _best_per_layer(
        df: pd.DataFrame,
        metric: str,
        group_cols: list[str],
    ) -> pd.DataFrame:
        """Within each `group_cols`, keep the best row per `layer` by `metric`."""
        idx = best_indices(df, metric, [*group_cols, "layer"])
        if idx.empty:
            return df.iloc[0:0]
        return df.loc[idx].reset_index(drop=True)

    @staticmethod
    def _aggregate_best(
        df: pd.DataFrame,
        metric: str,
        group_cols: list[str],
    ) -> pd.DataFrame:
        """One row per `group_cols`: best value of `metric` across all rows."""
        sub = df[df[metric].notna()]
        if sub.empty:
            return sub
        higher = HIGHER_IS_BETTER.get(metric, True)
        agg = "max" if higher else "min"
        return (
            sub.groupby(group_cols, dropna=False)[metric]
            .agg(agg)
            .reset_index()
        )

    # ── plot 1: layer curves (regressors + classifiers stacked) ─────────────

    def _render_layer_curves(
        self,
        ext_df: pd.DataFrame,
        extraction: str,
    ) -> list[Path]:
        panels: list[tuple[str, pd.DataFrame]] = []
        for metric in ("val_r2", "val_accuracy", "val_lab_distance"):
            if metric not in ext_df.columns:
                continue
            sub = ext_df[ext_df[metric].notna()]
            if sub.empty:
                continue
            panels.append((metric, sub))
        if not panels:
            return []

        targets = sorted(ext_df["target"].unique())
        n_targets = len(targets)
        ncols = min(self.layout.max_facets_per_row, max(1, n_targets))
        nrows_per_metric = (n_targets + ncols - 1) // ncols
        n_metrics = len(panels)

        fig, axes = plt.subplots(
            n_metrics * nrows_per_metric, ncols,
            figsize=(
                self.layout.base_width * ncols,
                self.layout.base_height * n_metrics * nrows_per_metric,
            ),
            squeeze=False,
            sharex=True,
        )

        for m_idx, (metric, sub) in enumerate(panels):
            best_layers = self._best_per_layer(
                sub, metric, group_cols=["target", "probe_kind"],
            )
            for t_idx, target in enumerate(targets):
                row = m_idx * nrows_per_metric + (t_idx // ncols)
                col = t_idx % ncols
                ax = axes[row, col]
                target_df = best_layers[best_layers["target"] == target]
                if target_df.empty:
                    ax.set_visible(False)
                    continue
                sns.lineplot(
                    data=target_df,
                    x="layer", y=metric,
                    hue="probe_kind",
                    marker="o",
                    ax=ax,
                )
                ax.set_title(f"{target} · {metric}", fontsize=10)
                ax.set_xlabel("layer")
                ax.set_ylabel(metric)
                if t_idx != 0:
                    legend = ax.get_legend()
                    if legend is not None:
                        legend.remove()

            # Hide leftover empty axes in the metric block.
            used_axes = n_targets
            block_axes = nrows_per_metric * ncols
            for slot in range(used_axes, block_axes):
                row = m_idx * nrows_per_metric + (slot // ncols)
                col = slot % ncols
                axes[row, col].set_visible(False)

        fig.suptitle(
            f"{self.experiment_dir.name} · extraction={extraction} · layer curves",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()
        out = self.figures_dir / f"layer_curves__{extraction}.png"
        _save_fig(fig, out)
        return [out]

    # ── plot 2: probe × target heatmap of best metric ────────────────────────

    def _render_probe_target_heatmap(
        self,
        ext_df: pd.DataFrame,
        extraction: str,
    ) -> list[Path]:
        metric = pick_metric(ext_df)
        if metric is None:
            return []
        agg = self._aggregate_best(
            ext_df, metric, group_cols=["probe_kind", "target"],
        )
        if agg.empty:
            return []
        pivot = agg.pivot(index="probe_kind", columns="target", values=metric)
        if pivot.empty:
            return []
        cmap = "viridis" if HIGHER_IS_BETTER.get(metric, True) else "viridis_r"
        height = max(2.5, 0.5 * len(pivot.index) + 1.5)
        width = max(4.0, 0.8 * len(pivot.columns) + 2)
        fig, ax = plt.subplots(figsize=(width, height))
        sns.heatmap(
            pivot, annot=True, fmt=".3f",
            cmap=cmap, ax=ax, cbar_kws={"label": metric},
        )
        ax.set_title(
            f"{self.experiment_dir.name} · extraction={extraction} · "
            f"best {metric} per (probe, target)",
        )
        ax.set_xlabel("target")
        ax.set_ylabel("probe")
        fig.tight_layout()
        out = self.figures_dir / f"probe_target_heatmap__{extraction}.png"
        _save_fig(fig, out)
        return [out]

    # ── plot 3: best-metric bar chart ────────────────────────────────────────

    def _render_best_metric_bars(
        self,
        ext_df: pd.DataFrame,
        extraction: str,
    ) -> list[Path]:
        metric = pick_metric(ext_df)
        if metric is None:
            return []
        agg = self._aggregate_best(
            ext_df, metric, group_cols=["target", "probe_kind"],
        )
        if agg.empty:
            return []
        target_order = sorted(agg["target"].unique())
        n_targets = len(target_order)
        width = max(6.0, 1.1 * n_targets + 2)
        fig, ax = plt.subplots(figsize=(width, 4.0))
        sns.barplot(
            data=agg,
            x="target", y=metric, hue="probe_kind",
            order=target_order, ax=ax,
        )
        ax.set_title(
            f"{self.experiment_dir.name} · extraction={extraction} · "
            f"best {metric} per probe",
        )
        ax.set_xlabel("target")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="probe_kind", bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        out = self.figures_dir / f"best_metric_bars__{extraction}.png"
        _save_fig(fig, out)
        return [out]

    # ── plot 4: extraction comparison (only for multi-extraction experiments) ─

    def _render_extraction_comparison(
        self,
        long_df: pd.DataFrame,
        extractions: list[str],
        targets: list[str],
    ) -> list[Path]:
        metric = pick_metric(long_df)
        if metric is None:
            return []
        best_layers = self._best_per_layer(
            long_df, metric, group_cols=["extraction", "target"],
        )
        if best_layers.empty:
            return []
        n_targets = len(targets)
        ncols = min(self.layout.max_facets_per_row, max(1, n_targets))
        nrows = (n_targets + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(
                self.layout.base_width * ncols,
                self.layout.base_height * nrows,
            ),
            squeeze=False,
            sharex=True,
        )
        for t_idx, target in enumerate(targets):
            row = t_idx // ncols
            col = t_idx % ncols
            ax = axes[row, col]
            target_df = best_layers[best_layers["target"] == target]
            if target_df.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(
                data=target_df,
                x="layer", y=metric, hue="extraction",
                marker="o", ax=ax,
            )
            ax.set_title(target, fontsize=10)
            ax.set_xlabel("layer")
            ax.set_ylabel(metric)
            if t_idx != 0:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
        for slot in range(n_targets, nrows * ncols):
            axes[slot // ncols, slot % ncols].set_visible(False)
        fig.suptitle(
            f"{self.experiment_dir.name} · extraction comparison · {metric}",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()
        out = self.figures_dir / "extraction_comparison.png"
        _save_fig(fig, out)
        return [out]

    # ── plot 5: RGB / LAB channel panels (xkcd-style experiments) ────────────

    def _render_colour_channel_panels(
        self,
        ext_df: pd.DataFrame,
        extraction: str,
        targets: list[str],
    ) -> list[Path]:
        target_set = set(targets)
        written: list[Path] = []
        for label, channels in (("rgb", _RGB_TARGETS), ("lab", _LAB_TARGETS)):
            if not all(c in target_set for c in channels):
                continue
            metric = pick_metric(ext_df)
            if metric is None:
                return written
            sub = ext_df[ext_df["target"].isin(channels)]
            best_layers = self._best_per_layer(
                sub, metric, group_cols=["target"],
            )
            if best_layers.empty:
                continue
            best_layers = best_layers.assign(
                target=pd.Categorical(
                    best_layers["target"], categories=list(channels), ordered=True,
                ),
            )
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            sns.lineplot(
                data=best_layers,
                x="layer", y=metric, hue="target",
                marker="o", ax=ax,
            )
            ax.set_title(
                f"{self.experiment_dir.name} · extraction={extraction} · "
                f"{label.upper()} channels · best {metric}",
            )
            ax.set_xlabel("layer")
            ax.set_ylabel(metric)
            ax.legend(title="channel")
            fig.tight_layout()
            out = (
                self.figures_dir
                / f"colour_channels_{label}__{extraction}.png"
            )
            _save_fig(fig, out)
            written.append(out)
        return written


# ─────────────────────────────────────────────────────────────────────────────
# Cross-experiment
# ─────────────────────────────────────────────────────────────────────────────


class ConsolidatedVisualiser:
    """Render cross-experiment figures from `consolidated_long.csv`."""

    def __init__(
        self,
        consolidated_dir: Path | str,
        primary_metric: str = "val_r2",
    ) -> None:
        self.consolidated_dir = Path(consolidated_dir)
        self.figures_dir = self.consolidated_dir / "figures"
        self.primary_metric = primary_metric

    def render(self) -> list[Path]:
        """Render the two cross-experiment figures. Returns written paths."""
        _setup_style()
        long_path = self.consolidated_dir / "consolidated_long.csv"
        if not long_path.exists():
            print(
                f"[visualisations] {long_path} not found — run "
                f"`interpret.probing.consolidate` first.",
            )
            return []
        df = pd.read_csv(long_path)
        if df.empty:
            return []
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        written += self._render_best_metric_heatmap(df)
        written += self._render_layer_curves_facet(df)
        print(
            f"[visualisations] Wrote {len(written)} cross-experiment figure(s) "
            f"to {self.figures_dir}",
        )
        return written

    def figure_paths(self) -> list[Path]:
        if not self.figures_dir.exists():
            return []
        return sorted(self.figures_dir.glob("*.png"))

    def _render_best_metric_heatmap(self, df: pd.DataFrame) -> list[Path]:
        metric = self.primary_metric
        if metric not in df.columns:
            return []
        valid = df[df[metric].notna()]
        if valid.empty:
            return []
        agg_func = "max" if HIGHER_IS_BETTER.get(metric, True) else "min"
        agg = (
            valid.groupby(["experiment", "target", "probe_kind"], dropna=False)[metric]
            .agg(agg_func)
            .reset_index()
        )
        agg["row"] = agg["experiment"] + " · " + agg["target"]
        pivot = agg.pivot(index="row", columns="probe_kind", values=metric)
        if pivot.empty:
            return []
        height = max(3.0, 0.4 * len(pivot.index) + 1.5)
        width = max(5.0, 0.9 * len(pivot.columns) + 3)
        fig, ax = plt.subplots(figsize=(width, height))
        cmap = "viridis" if HIGHER_IS_BETTER.get(metric, True) else "viridis_r"
        sns.heatmap(
            pivot, annot=True, fmt=".3f",
            cmap=cmap, ax=ax, cbar_kws={"label": f"best {metric}"},
        )
        ax.set_title(
            f"Cross-experiment: best {metric} per (experiment·target, probe)",
        )
        ax.set_xlabel("probe")
        ax.set_ylabel("experiment · target")
        fig.tight_layout()
        out = self.figures_dir / f"cross_experiment_best_{metric}.png"
        _save_fig(fig, out)
        return [out]

    def _render_layer_curves_facet(self, df: pd.DataFrame) -> list[Path]:
        metric = self.primary_metric
        if metric not in df.columns or "layer" not in df.columns:
            return []
        idx = best_indices(
            df, metric, ["experiment", "target", "extraction", "layer"],
        )
        if idx.empty:
            return []
        best = df.loc[idx].copy()
        # Combine experiment + target into a single facet so we only allocate
        # cells for (experiment, target) pairs that actually have data —
        # otherwise the grid is mostly empty when experiments use different
        # target sets.
        best["facet"] = best["experiment"] + " · " + best["target"]
        facet_order = sorted(best["facet"].unique())
        col_wrap = min(4, max(1, len(facet_order)))
        g = sns.relplot(
            data=best,
            x="layer", y=metric,
            hue="extraction", style="extraction",
            col="facet", col_wrap=col_wrap, col_order=facet_order,
            kind="line", marker="o",
            facet_kws={"sharey": False, "sharex": False},
            height=2.8, aspect=1.2,
        )
        g.set_titles("{col_name}")
        g.figure.suptitle(
            f"Cross-experiment best-probe {metric} over layers",
            fontsize=13, y=1.02,
        )
        out = self.figures_dir / f"cross_experiment_layer_curves_{metric}.png"
        _save_fig(g.figure, out)
        return [out]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _is_consolidated_dir(path: Path) -> bool:
    return (path / "consolidated_long.csv").exists()


def _is_experiment_dir(path: Path) -> bool:
    return (path / "experiment.yaml").exists() or (path / "probes").exists()


def main() -> None:
    """CLI: regenerate plots for an experiment directory or _consolidated dir."""
    if len(sys.argv) < 2:
        print(
            "Usage: python -m interpret.probing.visualisations "
            "<experiment_or_consolidated_dir> [--consolidated]",
            file=sys.stderr,
        )
        sys.exit(2)
    target = Path(sys.argv[1])
    forced_consolidated = "--consolidated" in sys.argv[2:]

    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(2)

    if forced_consolidated or _is_consolidated_dir(target):
        ConsolidatedVisualiser(target).render()
        # Refresh consolidated summary.md so it picks up the new figures.
        try:
            from interpret.probing import consolidate as _consolidate
            _consolidate.refresh_summary(target)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not refresh consolidated summary: {exc}")
            traceback.print_exc()
        return

    if _is_experiment_dir(target):
        ExperimentVisualiser(target).render()
        return

    print(
        f"{target} doesn't look like an experiment dir or a consolidated dir. "
        "Pass --consolidated to force.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
