"""Exploratory layouts for the per-feature β bar chart.

The canonical chart family lives in
:mod:`interpret.probing.probes.ablation_feature_bars`. This
module is a sandbox for alternative layouts evaluated alongside it:

* ``gap``         — insert a horizontal gap (``C_GAP``) between each
  c-group within a category, so c0/c1/c2/c3 read as visually distinct
  strips. β is still encoded by the diverging RdBu cmap.
* ``gap_small``   — same idea as ``gap`` but with the tighter
  ``C_GAP_SMALL`` spacing (the c-blocks read as separate without
  giving up horizontal space to whitespace).
* ``p_pastel``    — replace the diverging cmap with three fixed pastel
  colours, one per p-section (p0/p1/p2). The bar's *height* still encodes
  signed β; bar *colour* now encodes p.
* ``p_pastel_gap`` — pastel-by-p AND ``C_GAP_SMALL`` between c-groups,
  AND drops the per-bar p0/p1/p2 row from the x-axis (the colour /
  legend already encode p so the labels are redundant).
* ``p_gray_bg``   — keep the diverging β cmap on the foreground bars,
  but draw a faint gray rectangle behind each bar whose shade encodes p.
  Reader can see β (foreground) and p (background tint) at a glance.

All variants reuse the ``enhanced_text`` size/padding block from the
canonical ``_gridlines_minor`` chart — they're meant to be looked at
side-by-side with that variant.

Driven from
``scripts.interpretability.experiments.poetry._compare.aggregate_results_variants``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory

from interpret.probing.probes._chart_style import (
    apply_theme,
    humanize,
    parse_feature_name,
)
from interpret.probing.probes.ablation_feature_bars import (
    _attach_axes,
    _ordered_features,
)

# Pastel palette used by the ``p_pastel`` variant. Three perceptually
# similar tints so no single p dominates by saturation; ordered p0→p2.
P_PASTEL: dict[str, str] = {
    "p0": "#F4B6C2",  # pastel pink
    "p1": "#A8DADC",  # pastel teal
    "p2": "#C9C9E8",  # pastel periwinkle
}

# Gray shades used by the ``p_gray_bg`` variant. Faint enough to read as
# a background hint, with monotonic darkening p0→p2 so the order maps to
# perceived depth.
P_GRAY_BG: dict[str, str] = {
    "p0": "#F2F2F2",
    "p1": "#E2E2E2",
    "p2": "#CFCFCF",
}

# Extra space inserted between c-groups (in bar-slot units). The
# ``gap`` variant uses ``C_GAP`` (more breathing room); the
# ``gap_small`` variant and the combined ``p_pastel_gap`` variant use
# ``C_GAP_SMALL`` — tight enough that the c-blocks read as separate
# strips without wasting horizontal space, while keeping p0/p1/p2 inside
# a c-block close enough that their labels don't conflate with the next
# c-group's labels.
C_GAP = 0.5
C_GAP_SMALL = 0.3


def _positions_with_c_gaps(
    parsed: list[tuple[str, str, str]],
    gap: float,
) -> np.ndarray:
    """Return one x-position per parsed feature, with extra spacing across c-groups.

    Within a (cat, ctx) block consecutive bars are 1.0 apart. Crossing a
    ctx boundary (same category, new context) or a category boundary
    adds an additional ``gap`` to the running x.
    """
    positions: list[float] = []
    prev_ctx: str | None = None
    prev_cat: str | None = None
    x = 0.0
    for ctx, _sec, cat in parsed:
        if prev_ctx is not None and (ctx != prev_ctx or cat != prev_cat):
            x += gap
        positions.append(x)
        x += 1.0
        prev_ctx, prev_cat = ctx, cat
    return np.asarray(positions)


def _decorate_x_axis(
    ax: Axes,
    feature_order: list[str],
    positions: np.ndarray,
    *,
    cat_band_y: float = 1.04,
    ctx_underline_y: float = -0.065,
    section_label_y: float = -0.075,
    section_label_size: float = 9.5,
    ctx_label_size: float = 11,
    cat_band_size: float = 16,
    cat_band_color: str = "#000000",
    hide_section_labels: bool = False,
) -> None:
    """3-tier x-axis (category band, ctx tick + underline, p label) using explicit positions.

    Mirrors :func:`ablation_feature_bars._decorate_x_axis` but consumes a
    pre-computed ``positions`` array so the ``gap`` variant (uneven
    spacing) lays out correctly.

    When ``hide_section_labels`` is True the per-bar p0/p1/p2 row is
    dropped — used by the ``p_pastel_gap`` variant where bar colour
    already encodes p so the labels are redundant.
    """
    parsed = [parse_feature_name(f) for f in feature_order]
    cats_in_order = [p[2] for p in parsed]

    boundaries: list[tuple[int, str]] = []
    for i, cat in enumerate(cats_in_order):
        if i == 0 or cat != cats_in_order[i - 1]:
            boundaries.append((i, cat))
    boundaries.append((len(feature_order), ""))

    for i, _ in boundaries[1:-1]:
        x = (positions[i - 1] + positions[i]) / 2
        ax.axvline(x, color="0.55", lw=0.8, alpha=0.7)

    for k in range(len(boundaries) - 1):
        i, cat = boundaries[k]
        j = boundaries[k + 1][0]
        mid = (positions[i] + positions[j - 1]) / 2
        ax.text(
            mid,
            cat_band_y,
            humanize(cat),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=cat_band_size,
            color=cat_band_color,
        )

    ctx_block_positions: dict[tuple[str, str], list[int]] = {}
    for i, (ctx, _sec, cat) in enumerate(parsed):
        ctx_block_positions.setdefault((cat, ctx), []).append(i)

    ctx_tick_positions: list[float] = []
    ctx_tick_labels: list[str] = []
    for (_cat, ctx), idxs in ctx_block_positions.items():
        xs = [float(positions[i]) for i in idxs]
        ctx_tick_positions.append(float(np.mean(xs)))
        ctx_tick_labels.append(ctx)
        ax.plot(
            [min(xs) - 0.35, max(xs) + 0.35],
            [ctx_underline_y, ctx_underline_y],
            transform=ax.get_xaxis_transform(),
            color="0.7",
            lw=0.4,
            clip_on=False,
        )

    ax.set_xticks(ctx_tick_positions)
    ax.set_xticklabels(ctx_tick_labels, fontsize=ctx_label_size)
    ax.tick_params(axis="x", length=0)

    if not hide_section_labels:
        for i, (_ctx, sec, _cat) in enumerate(parsed):
            ax.text(
                float(positions[i]),
                section_label_y,
                sec,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=section_label_size,
                color="0.4",
            )


def render_variant(
    long_df: pd.DataFrame,
    out_path: Path,
    *,
    variant: str,
    experiments: list[str],
    value_col: str,
    error_col: str | None = None,
    show_errorbars: bool = False,
    y_label: str,
    diverging_cmap: str = "RdBu",
    legend_handles: Sequence[Patch] | None = None,
    legend_kwargs: dict | None = None,
    y_major_step: float | None = None,
) -> Path:
    """Render one chart in the requested ``variant`` and write it to ``out_path``.

    Single-experiment only (uses ``experiments[0]``); the canonical
    multi-experiment renderer in ``ablation_feature_bars`` covers the
    grouped-bar use case. Output dpi / figure size mirror the canonical
    chart so the variants drop in alongside it visually.

    Args:
        long_df: Long-format frame with ``feature_name``, ``experiment``,
            ``value_col`` (and ``error_col`` when ``show_errorbars=True``).
        variant: ``"gap"`` | ``"p_pastel"`` | ``"p_gray_bg"``.
        experiments: One-element list, the experiment to plot.
        value_col: Column with bar heights.
        error_col: Column with per-fold std for ±1σ ribbons.
        show_errorbars: Overlay ribbons; ignored when ``error_col`` is
            None or missing.
        y_label: y-axis label (TeX-friendly).
        diverging_cmap: matplotlib cmap name used for ``gap`` and
            ``p_gray_bg`` (which keep β-by-colour). Ignored by
            ``p_pastel`` since p drives colour there.
        legend_handles: Pre-built handles. Used as-is by ``gap`` /
            ``p_gray_bg``; replaced by a p-swatch legend for
            ``p_pastel``.
        legend_kwargs: Forwarded to ``ax.legend(...)``.
        y_major_step: If set, snap y-axis major ticks to multiples of
            this step (and minor ticks to half-steps).

    Returns:
        ``out_path``.
    """
    valid_variants = {"gap", "gap_small", "p_pastel", "p_pastel_gap", "p_gray_bg"}
    if variant not in valid_variants:
        raise ValueError(
            f"render_variant: unknown variant {variant!r} "
            f"(expected one of {sorted(valid_variants)})",
        )

    use_pastel = variant in {"p_pastel", "p_pastel_gap"}
    use_gray_bg = variant == "p_gray_bg"
    hide_section_labels = variant == "p_pastel_gap"
    if variant == "gap":
        gap_size: float | None = C_GAP
    elif variant in {"gap_small", "p_pastel_gap"}:
        gap_size = C_GAP_SMALL
    else:
        gap_size = None

    apply_theme()
    df = _attach_axes(long_df)
    primary = experiments[0]
    feature_order = _ordered_features(df[df["experiment"] == primary])
    parsed = [parse_feature_name(f) for f in feature_order]

    if gap_size is not None:
        positions = _positions_with_c_gaps(parsed, gap_size)
    else:
        positions = np.arange(len(feature_order), dtype=float)

    sub = df[df["experiment"] == primary].set_index("feature_name").reindex(feature_order)
    values = sub[value_col].to_numpy()
    yerr = (
        sub[error_col].to_numpy()
        if show_errorbars and error_col is not None and error_col in sub.columns
        else None
    )

    # Bar colour depends on variant: pastel-by-p for the pastel variants,
    # diverging by β for the others.
    if use_pastel:
        bar_colour = [P_PASTEL[sec] for _, sec, _ in parsed]
    else:
        finite = values[np.isfinite(values)]
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
        vmax = vmax or 1e-9
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap(diverging_cmap)
        bar_colour = [cmap(norm(v)) if np.isfinite(v) else "0.8" for v in values]

    fig, ax = plt.subplots(figsize=(14.0, 4.6), constrained_layout=True)
    ax.margins(x=0.005)

    # Gray background bars are drawn first so foreground bars + axhline
    # stack on top.
    if use_gray_bg:
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        bg_width = 1.0
        for i, (_ctx, sec, _cat) in enumerate(parsed):
            ax.add_patch(
                Rectangle(
                    (float(positions[i]) - bg_width / 2, 0),
                    bg_width,
                    1.0,
                    transform=trans,
                    facecolor=P_GRAY_BG[sec],
                    edgecolor="none",
                    zorder=0,
                ),
            )

    bar_width = 0.8
    ax.bar(
        positions,
        values,
        width=bar_width,
        yerr=yerr,
        color=bar_colour,
        edgecolor="0.4",
        linewidth=0.4,
        error_kw={"ecolor": "0.35", "elinewidth": 0.6, "capsize": 1.5},
        zorder=2,
    )

    if (df[value_col].dropna() < 0).any():
        ax.axhline(0, color="0.3", lw=0.6, zorder=2)

    # Enhanced-text sizing block — kept in sync with the canonical
    # `_gridlines_minor` variant in ablation_feature_bars.py so the
    # comparison is apples-to-apples.
    text_color = "#1a1a1a"
    bold_text_color = "#000000"
    label_size = 15
    tick_label_size = 10
    ctx_tick_pad = 4.5

    _decorate_x_axis(
        ax,
        feature_order,
        positions,
        cat_band_color=bold_text_color,
        hide_section_labels=hide_section_labels,
    )

    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=label_size, color=bold_text_color)

    ax.set_axisbelow(True)
    ax.yaxis.grid(
        True,
        color="#D8D8D8",
        linewidth=0.7,
        linestyle="-",
        which="major",
        zorder=0,
    )
    ax.xaxis.grid(False)
    if y_major_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))
        ax.yaxis.set_minor_locator(MultipleLocator(y_major_step / 2))
        ax.yaxis.grid(
            True,
            color="#ECECEC",
            linewidth=0.5,
            linestyle="-",
            which="minor",
            zorder=0,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            length=2.5,
            width=0.5,
            color=text_color,
        )

    ax.tick_params(
        axis="x",
        labelsize=11,
        labelcolor=text_color,
        pad=ctx_tick_pad,
    )
    ax.tick_params(
        axis="y",
        labelsize=tick_label_size,
        labelcolor=text_color,
        pad=ctx_tick_pad,
    )

    # Top headroom so the upper-right legend doesn't sit on the
    # tallest bars (matches the canonical chart family).
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(top=ymax + (ymax - ymin) * 0.18)

    # Legend handling. Pastel variants need their own (p0/p1/p2 swatches);
    # gap / gap_small / p_gray_bg keep the caller's β-direction handles
    # since bar colour still encodes β there.
    if use_pastel:
        handles = [
            Patch(
                facecolor=P_PASTEL[sec],
                edgecolor="0.4",
                linewidth=0.4,
                label=sec,
            )
            for sec in ("p0", "p1", "p2")
        ]
        kwargs = dict(legend_kwargs or {"loc": "upper right"})
        kwargs["fontsize"] = 10
        kwargs.setdefault("borderpad", 0.7)
        kwargs.setdefault("labelspacing", 0.7)
        ax.legend(handles=handles, **kwargs)
    elif legend_handles is not None:
        kwargs = dict(legend_kwargs or {"loc": "upper right"})
        kwargs["fontsize"] = 10
        kwargs.setdefault("borderpad", 0.7)
        kwargs.setdefault("labelspacing", 0.7)
        ax.legend(handles=list(legend_handles), **kwargs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
