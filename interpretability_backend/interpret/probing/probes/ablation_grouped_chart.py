"""Render the 3-panel grouped-feature ablation chart for one probe folder.

Reads ``group_{category,context,section}_importance_summary.csv`` from
an ablation folder and writes a paper-style horizontal bar chart with
auto-computed x-limits and ticks (no per-axis hardcoding required).

CLI: ``uv run python -m interpret.probing.visualisations.ablation_grouped_chart \\
        resources/experiments/<exp>/ablation/csv_features/<target>/<probe>/``

Outputs land in a sibling ``figures/`` folder under the probe folder
in five flavours (all red negative): clean × {plain, labelled},
plus three error-bar overlays — ``_errorbars`` (±1σ across folds) on
both clean and labelled, and ``_sem`` (±1 SEM, labelled only) for the
paper-style figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from interpret.probing.probes._chart_style import (
    NEG_COLOR_RED,
    POS_COLOR,
    apply_theme,
    humanize,
)

# Axis configuration: (panel name, source CSV, panel title).
AXES: tuple[tuple[str, str, str], ...] = (
    ("category", "group_category_importance_summary.csv", "Category"),
    ("context", "group_context_importance_summary.csv", "Context"),
    ("section", "group_section_importance_summary.csv", "Section"),
)


def _fmt_tick(x: float, _pos: int) -> str:
    if x == 0:
        return "0"
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "-0.") else s


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.sort_values("accuracy_drop", ascending=True).reset_index(drop=True)


def _auto_xlim(
    means: np.ndarray,
    *,
    label_pad_frac: float,
    min_pad_frac: float = 0.04,
    extra_right: float = 0.0,
    extra_left: float = 0.0,
) -> tuple[float, float]:
    """Compute (xlim_lo, xlim_hi) from the data with a buffer for labels.

    Both ends are extended by ``min_pad_frac`` of the data range so
    bars don't touch the spine. The right end gets an extra
    ``label_pad_frac`` so the end-of-bar text labels (when shown) fit
    inside the axes. ``extra_right`` / ``extra_left`` accept absolute
    additional padding (e.g. the largest error-bar std) so the bars
    plus their ribbons stay inside the axes. ``0`` is forced into the
    visible range so the axvline reference is always drawn.
    """
    data_lo = float(np.nanmin(means))
    data_hi = float(np.nanmax(means))
    lo = min(data_lo, 0.0)
    hi = max(data_hi, 0.0)
    span = hi - lo if hi > lo else max(abs(hi), abs(lo), 1.0) * 0.1
    pad = span * min_pad_frac
    return (
        lo - pad - extra_left,
        hi + pad + span * label_pad_frac + extra_right,
    )


N_FOLDS = 5  # all probes in this group run StratifiedKFold(n_splits=5).


def _render(
    *,
    probe_dir: Path,
    out_dir: Path,
    neg_color: str,
    show_labels: bool,
    errorbar_kind: str,  # "none" | "stddev" | "sem"
    out_stem: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True)

    label_pad_frac = 0.20 if show_labels else 0.05

    for ax, (_axis_name, csv_name, title) in zip(axes, AXES, strict=True):
        csv_path = probe_dir / csv_name
        if not csv_path.exists():
            ax.set_axis_off()
            ax.set_title(f"{title}\n(missing: {csv_name})", fontsize=8)
            continue
        df = _load(csv_path)
        groups = [humanize(g) for g in df["group"].tolist()]
        means = df["accuracy_drop"].to_numpy()
        # Per-fold std of the leave-one-group-out drop, divided by
        # sqrt(N_FOLDS) when the SEM variant is requested.
        if errorbar_kind != "none" and "std_accuracy_drop" in df.columns:
            stds = df["std_accuracy_drop"].to_numpy()
            if errorbar_kind == "sem":
                stds = stds / np.sqrt(N_FOLDS)
            max_std = float(np.nanmax(stds)) if stds.size else 0.0
            xerr_arg = stds
        else:
            stds = np.zeros_like(means)
            max_std = 0.0
            xerr_arg = None
        colors = [POS_COLOR if m > 0 else neg_color for m in means]
        y = np.arange(len(groups))

        xlim = _auto_xlim(
            means,
            label_pad_frac=label_pad_frac,
            extra_right=max_std,
            extra_left=max_std,
        )
        ax.barh(
            y,
            means,
            xerr=xerr_arg,
            color=colors,
            edgecolor="none",
            error_kw={"ecolor": "0.35", "elinewidth": 0.7, "capsize": 2.0},
        )

        if show_labels:
            text_pad = 0.012 * (xlim[1] - xlim[0])
            for yi, m, s in zip(y, means, stds, strict=True):
                # Place the label past the error-bar tip on the
                # positive side so std bars don't overlap the text.
                rightmost = max(m, 0.0) + (s if np.isfinite(s) else 0.0)
                ax.text(
                    rightmost + text_pad,
                    yi,
                    f"{m:.3f}",
                    ha="left",
                    va="center",
                    fontsize=6,
                    color="0.25",
                    clip_on=False,
                )

        ax.axvline(0, color="0.3", lw=0.8, zorder=1)
        ax.set_xlim(xlim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tick))
        ax.set_yticks(y)
        ax.set_yticklabels(groups)
        ax.set_title(title, fontsize=9, pad=4)
        ax.xaxis.grid(True, alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=2.5, width=0.6)
        ax.tick_params(axis="y", length=0)

    base_caption = r"$\Delta$ val. accuracy (baseline $-$ ablated)"
    if errorbar_kind == "stddev":
        caption = base_caption + rf"; error bars: $\pm 1\sigma$ across {N_FOLDS} folds"
    elif errorbar_kind == "sem":
        caption = base_caption + rf"; error bars: $\pm 1\,$SEM, $n={N_FOLDS}$"
    else:
        caption = base_caption
    fig.supxlabel(caption, fontsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig((out_dir / out_stem).with_suffix(".png"), dpi=300)
    plt.close(fig)


def render_probe_dir(probe_dir: Path, *, out_dir: Path | None = None) -> Path:
    """Render every chart variant for a single ablation probe folder.

    Five PNGs total, all using the red negative-bar palette:
    ``ablation_red_{clean,labeled}.png`` baselines, ``_errorbars``
    overlays (±1σ across folds) on both, and ``ablation_red_labeled_sem.png``
    (±1 SEM) for the paper-style figure.

    Args:
        probe_dir: Folder containing
            ``group_{category,context,section}_importance_summary.csv``.
        out_dir: Where to write the PNG outputs. Defaults to a
            ``figures`` sibling next to the CSV files.
    Returns:
        The output directory.
    """
    apply_theme()
    if out_dir is None:
        out_dir = probe_dir / "figures"

    # (show_labels, errorbar_kind, out_stem)
    variants: tuple[tuple[bool, str, str], ...] = (
        (False, "none", "ablation_red_clean"),
        (True, "none", "ablation_red_labeled"),
        (False, "stddev", "ablation_red_clean_errorbars"),
        (True, "stddev", "ablation_red_labeled_errorbars"),
        (True, "sem", "ablation_red_labeled_sem"),
    )
    for labels, kind, stem in variants:
        _render(
            probe_dir=probe_dir,
            out_dir=out_dir,
            neg_color=NEG_COLOR_RED,
            show_labels=labels,
            errorbar_kind=kind,
            out_stem=stem,
        )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "probe_dir",
        type=Path,
        help=(
            "Path to one ablation probe folder, e.g. "
            "resources/experiments/<exp>/ablation/csv_features/<target>/<probe>/"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: <probe_dir>/figures)",
    )
    args = parser.parse_args()
    out_dir = render_probe_dir(args.probe_dir, out_dir=args.out_dir)
    print(f"wrote 5 chart variants -> {out_dir}")


if __name__ == "__main__":
    main()
