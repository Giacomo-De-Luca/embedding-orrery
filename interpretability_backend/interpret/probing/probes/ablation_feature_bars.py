"""Paper-style per-feature ablation / |β| bar charts.

Two render entry-points share the same x-axis layout (categories →
contexts → sections, derived from feature names of the form
``c{ctx}_p{section}_{CATEGORY}``):

* :func:`render_single_experiment` — one bar per feature for a single
  ``(experiment, probe)`` cell. Replaces the per-probe
  ``feature_importance.png`` rendered by ``mlp_ablation`` (which was a
  72-bar alphabetical strip).
* :func:`render_multi_experiment` — grouped bars (one bar per
  experiment per feature) for cross-experiment comparisons. Used by
  ``aggregate_results`` to replace ``feature_importance_grouped_bar_*``.

Both produce a clean variant and an ``_errorbars`` companion when given
a per-fold std column.

A heatmap variant (:func:`render_heatmap`) renders the same data as a
``categories × (context · section)`` grid stacked one row per
experiment — used by the cross-experiment view.

Adapted from the reference implementation provided in the conversation;
the theme / palette comes from :mod:`._chart_style` so it stays in sync
with the 3-panel ``ablation_grouped_chart``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm

from interpret.probing.probes._chart_style import (
    CATEGORIES,
    CONTEXTS,
    NEG_COLOR_GRAY,
    POS_COLOR,
    SECTIONS,
    apply_theme,
    humanize,
    parse_feature_name,
)

_CONTEXT_SECTION = tuple(f"{c}_{p}" for c in CONTEXTS for p in SECTIONS)


def _attach_axes(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``ctx``/``sec``/``cat`` columns derived from ``feature_name``."""
    parts = df["feature_name"].apply(parse_feature_name).tolist()
    return df.assign(
        ctx=[p[0] for p in parts],
        sec=[p[1] for p in parts],
        cat=[p[2] for p in parts],
    )


def _ordered_features(df: pd.DataFrame) -> list[str]:
    """Sort feature names by paper-order (category) then ctx, then section."""
    cat_rank = {c: i for i, c in enumerate(CATEGORIES)}
    return (
        df.assign(_cat_rank=df["cat"].map(cat_rank).fillna(len(CATEGORIES)))
        .sort_values(["_cat_rank", "ctx", "sec"])["feature_name"]
        .drop_duplicates()
        .tolist()
    )


def _decorate_x_axis(
    ax: Axes,
    feature_order: list[str],
    *,
    cat_band_y: float = 1.04,
    ctx_underline_y: float = -0.045,
    section_label_y: float = -0.055,
    section_label_size: float = 4.5,
    ctx_label_size: float = 6,
    cat_band_size: float = 10,
    cat_band_color: str = "0.15",
    vertical_gridlines: bool = False,
) -> None:
    """Apply the 3-tier x-axis: section ticks, ctx underlines, category bands.

    When ``vertical_gridlines`` is True, faint vertical lines are drawn
    at every within-category context boundary (i.e. between each c0/c1/
    c2/c3 sub-group). Cross-category dividers are unchanged.
    """
    cats_in_order = [parse_feature_name(f)[2] for f in feature_order]

    # Category bands + dividers between adjacent categories.
    boundaries: list[tuple[int, str]] = []
    for i, cat in enumerate(cats_in_order):
        if i == 0 or cat != cats_in_order[i - 1]:
            boundaries.append((i, cat))
    boundaries.append((len(feature_order), ""))

    for i, _ in boundaries[1:-1]:
        ax.axvline(i - 0.5, color="0.55", lw=0.8, alpha=0.7)

    for k in range(len(boundaries) - 1):
        i, cat = boundaries[k]
        j = boundaries[k + 1][0]
        ax.text(
            (i + j - 1) / 2,
            cat_band_y,
            humanize(cat),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=cat_band_size,
            color=cat_band_color,
        )

    # Within each (category, context) block, underline + a section label
    # under each individual bar position. We compute positions from the
    # actual feature_order (instead of the constant grid) so the layout
    # adapts when a feature is missing.
    parsed = [parse_feature_name(f) for f in feature_order]
    ctx_block_positions: dict[tuple[str, str], list[int]] = {}
    for i, (ctx, _sec, cat) in enumerate(parsed):
        ctx_block_positions.setdefault((cat, ctx), []).append(i)

    ctx_tick_positions: list[float] = []
    ctx_tick_labels: list[str] = []
    for (_cat, ctx), idxs in ctx_block_positions.items():
        ctx_tick_positions.append(float(np.mean(idxs)))
        ctx_tick_labels.append(ctx)
        # Underline the (cat, ctx) span.
        ax.plot(
            [min(idxs) - 0.35, max(idxs) + 0.35],
            [ctx_underline_y, ctx_underline_y],
            transform=ax.get_xaxis_transform(),
            color="0.7",
            lw=0.4,
            clip_on=False,
        )

    # Faint vertical gridlines between within-category context groups.
    if vertical_gridlines:
        for i in range(1, len(parsed)):
            prev_ctx, _, prev_cat = parsed[i - 1]
            curr_ctx, _, curr_cat = parsed[i]
            if prev_ctx != curr_ctx and prev_cat == curr_cat:
                ax.axvline(
                    i - 0.5,
                    color="#E5E5E5",
                    lw=0.5,
                    zorder=0,
                )

    ax.set_xticks(ctx_tick_positions)
    ax.set_xticklabels(ctx_tick_labels, fontsize=ctx_label_size)
    ax.tick_params(axis="x", length=0)

    for i, (_ctx, sec, _cat) in enumerate(parsed):
        ax.text(
            i,
            section_label_y,
            sec,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=section_label_size,
            color="0.4",
        )


# ── Single-experiment per-feature bars ──────────────────────────────────────


def render_single_experiment(
    summary_csv: Path,
    out_path: Path,
    *,
    show_errorbars: bool = False,
    title: str | None = None,
    pos_color: str = POS_COLOR,
    neg_color: str = NEG_COLOR_GRAY,
) -> Path:
    """Render one paper-style chart of per-feature accuracy drops.

    Reads ``feature_importance_summary.csv`` from a probe ablation
    folder (columns: ``feature_name``, ``accuracy_drop``,
    ``std_accuracy_drop``), draws one signed bar per feature in
    paper-order, and writes a single PNG to ``out_path``. Bars are
    coloured teal for positive drops (feature was important) and grey
    for negatives (ablating it helped). When ``show_errorbars`` is set,
    ±1σ ribbons from ``std_accuracy_drop`` are overlaid.

    Args:
        summary_csv: Path to ``feature_importance_summary.csv``.
        out_path: Output PNG path. Parent dirs are created if missing.
        show_errorbars: Overlay ±1σ ribbons from ``std_accuracy_drop``.
        title: Optional figure title.
        pos_color, neg_color: Bar fill colours.

    Returns:
        The written PNG path.
    """
    apply_theme()
    df = _attach_axes(pd.read_csv(summary_csv))
    feature_order = _ordered_features(df)
    sub = df.set_index("feature_name").reindex(feature_order).reset_index()
    means = sub["accuracy_drop"].to_numpy()
    if show_errorbars and "std_accuracy_drop" in sub.columns:
        yerr = sub["std_accuracy_drop"].to_numpy()
    else:
        yerr = None

    colors = [pos_color if m > 0 else neg_color for m in means]
    x = np.arange(len(feature_order))

    fig, ax = plt.subplots(figsize=(14.0, 4.6), constrained_layout=True)
    ax.bar(
        x,
        means,
        yerr=yerr,
        color=colors,
        edgecolor="0.4",
        linewidth=0.4,
        error_kw={"ecolor": "0.35", "elinewidth": 0.6, "capsize": 1.5},
    )
    ax.axhline(0, color="0.3", lw=0.6, zorder=2)

    _decorate_x_axis(ax, feature_order)
    ax.set_xlabel("")
    ax.set_ylabel(
        r"$\Delta$ val. accuracy (baseline $-$ ablated)",
        fontsize=9,
    )
    if title:
        # Above the cat-band labels (which sit at y≈1.04 in axes
        # coords) so they don't collide with the title.
        ax.set_title(title, y=1.12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def render_single_experiment_pair(
    summary_csv: Path,
    *,
    out_dir: Path,
    out_stem: str = "feature_importance",
    title: str | None = None,
) -> tuple[Path, Path]:
    """Convenience wrapper: emit clean + ``_errorbars`` variants together.

    Returns ``(clean_path, errorbars_path)``.
    """
    clean = render_single_experiment(
        summary_csv,
        out_dir / f"{out_stem}.png",
        show_errorbars=False,
        title=title,
    )
    errorbars = render_single_experiment(
        summary_csv,
        out_dir / f"{out_stem}_errorbars.png",
        show_errorbars=True,
        title=title,
    )
    return clean, errorbars


# ── Multi-experiment per-feature bars ───────────────────────────────────────


def render_multi_experiment(
    long_df: pd.DataFrame,
    out_path: Path,
    *,
    experiments: list[str],
    experiment_labels: dict[str, str] | None = None,
    value_col: str,
    error_col: str | None = None,
    show_errorbars: bool = False,
    y_label: str,
    title: str | None = None,
    palette_name: str = "pastel",
    diverging: bool = False,
    diverging_cmap: str = "RdBu",
    color_col: str | None = None,
    show_legend: bool = True,
    legend_handles: list | None = None,
    legend_kwargs: dict | None = None,
    gridlines: bool = False,
    minor_gridlines: bool = False,
    vertical_gridlines: bool = False,
    y_major_step: float | None = None,
    enhanced_text: bool = False,
) -> Path:
    """Render grouped bars (one bar per experiment per feature) to a PNG.

    Args:
        long_df: Long-format DataFrame with at least
            ``feature_name``, ``experiment``, ``value_col`` (and
            ``error_col`` when ``show_errorbars=True``).
        out_path: Output PNG path.
        experiments: Experiments to plot in order; controls hue order.
        experiment_labels: Optional mapping for the legend (defaults to
            ``experiments`` themselves).
        value_col: Column with bar heights.
        error_col: Column with per-fold std for ±1σ ribbons.
        show_errorbars: Overlay ribbons. Silently ignored when
            ``error_col`` is None or missing from the frame.
        y_label: y-axis label (TeX-friendly).
        title: Optional figure title.
        palette_name: seaborn palette name; pastel by default to match
            the reference style.
        diverging: When True, colour each bar by a value using a
            diverging cmap centred at 0 (instead of one colour per
            experiment). Useful for single-experiment signed-β charts.
        diverging_cmap: matplotlib cmap name used when ``diverging=True``.
        color_col: Column to drive bar colour when ``diverging=True``.
            Defaults to ``value_col``. Pass a different column when bar
            heights and bar colours should encode different quantities
            (e.g. ``value_col`` is a sign-flipped β for the chart's
            directional framing, while ``color_col`` keeps the original
            β so the same feature stays the same colour across both
            directional variants).
        show_legend: Set to False to suppress the legend.
        legend_handles: Optional pre-built handles (e.g.
            ``matplotlib.patches.Patch`` instances) used in place of the
            default per-experiment colour legend.
        legend_kwargs: Optional kwargs forwarded to ``ax.legend(...)``;
            useful with ``legend_handles``.
        gridlines: When True, draw faint horizontal major gridlines
            behind the bars (``axisbelow=True``, light gray, no minor
            ticks unless ``minor_gridlines`` is also set).
        minor_gridlines: When True (only effective with
            ``gridlines=True``), additionally draw fainter minor
            gridlines at half the major step plus minor tick marks
            on the y-axis.
        vertical_gridlines: When True, draw faint vertical gridlines
            at the within-category context boundaries (between c0/c1/
            c2/c3 sub-groups). Independent of ``gridlines`` (which
            controls horizontal gridlines).
        y_major_step: When set together with ``gridlines=True``, snap
            the y-axis major ticks to multiples of this step.
        enhanced_text: When True, render axis labels / tick labels /
            legend / category band at a larger size and in a darker
            colour, AND suppress the figure title (the legend tells
            the reader which class is which). Used by presentation-
            grade variants where the chart needs to read at a lower
            zoom.

    Returns:
        The written PNG path.
    """
    apply_theme()
    df = _attach_axes(long_df)
    # Order driven by the first experiment in the list — every other
    # experiment is reindexed against this order so missing features
    # produce empty slots instead of misaligning the grid.
    feature_order = _ordered_features(df[df["experiment"] == experiments[0]])
    palette = sns.color_palette(palette_name, n_colors=len(experiments))
    n_exp = len(experiments)
    bar_width = 0.8 / n_exp
    centers = np.arange(len(feature_order))

    color_source = color_col or value_col
    if diverging:
        vmax = float(df[color_source].abs().max() or 1e-9)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap(diverging_cmap)
    else:
        norm = None
        cmap = None

    fig, ax = plt.subplots(figsize=(14.0, 4.6), constrained_layout=True)
    for i, exp in enumerate(experiments):
        sub = df[df["experiment"] == exp].set_index("feature_name").reindex(feature_order)
        offsets = centers + (i - (n_exp - 1) / 2) * bar_width
        values = sub[value_col].to_numpy()
        if show_errorbars and error_col is not None and error_col in sub.columns:
            yerr = sub[error_col].to_numpy()
        else:
            yerr = None
        if diverging:
            assert cmap is not None and norm is not None
            color_values = sub[color_source].to_numpy()
            bar_color = [cmap(norm(v)) for v in color_values]
        else:
            bar_color = palette[i]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            yerr=yerr,
            color=bar_color,
            label=(experiment_labels or {}).get(exp, exp),
            edgecolor="0.4",
            linewidth=0.4,
            error_kw={"ecolor": "0.35", "elinewidth": 0.6, "capsize": 1.5},
        )

    if (df[value_col].dropna() < 0).any():
        ax.axhline(0, color="0.3", lw=0.6, zorder=2)

    # Tight horizontal margins so the first / last bars sit close to
    # the plot edges (default xmargin=0.05 wastes ~5 bar-widths of
    # whitespace on each side, leaving no room for bigger p labels).
    if enhanced_text:
        ax.margins(x=0.005)

    # Enhanced sizes are modest bumps over the seaborn-paper defaults
    # (label 9, cat band 10, ctx 6, section 4.5, legend 8): +2 on the
    # bigger labels (y-axis label, category band), +1 on legend / c /
    # p. Padding gets a small bump too so the bigger glyphs don't
    # crowd the chart.
    if enhanced_text:
        text_color = "#1a1a1a"  # c / p / ticks / legend
        bold_text_color = "#000000"  # y-axis label + feature-name band
        label_size = 15  # y-axis label (default 9, +6)
        tick_label_size = 10  # y tick labels
        section_label_size = 9.5  # p0/p1/p2 (default 4.5, +5)
        section_label_y = -0.075  # axes-y for top of p text
        ctx_label_size = 11  # c0..c3 (default 6, +5)
        ctx_tick_pad = 4.5  # ~1px more than default 3.5
        # Bigger c labels span further below the x-axis, so push the
        # underline ("horizontal bar" between c and p) further down to
        # restore breathing room above it AND tighten the gap below it
        # to p.
        ctx_underline_y = -0.065  # default -0.045
        cat_band_size = 16  # feature-name bands (default 10, +6)
        legend_size = 12  # legend (default 10, +2)
        legend_pad_bump = {"borderpad": 0.7, "labelspacing": 0.7}
    else:
        text_color = None  # leave seaborn default
        bold_text_color = None
        label_size = 9
        tick_label_size = None
        section_label_size = 4.5
        section_label_y = -0.055
        ctx_label_size = 6
        ctx_tick_pad = None
        ctx_underline_y = -0.045
        cat_band_size = 10
        legend_size = None
        legend_pad_bump = {}

    _decorate_x_axis(
        ax,
        feature_order,
        section_label_size=section_label_size,
        section_label_y=section_label_y,
        ctx_label_size=ctx_label_size,
        ctx_underline_y=ctx_underline_y,
        cat_band_size=cat_band_size,
        cat_band_color=bold_text_color or "0.15",
        vertical_gridlines=vertical_gridlines,
    )
    ax.set_xlabel("")
    ylabel_kwargs: dict = {"fontsize": label_size}
    if bold_text_color is not None:
        ylabel_kwargs["color"] = bold_text_color
    ax.set_ylabel(y_label, **ylabel_kwargs)
    if gridlines:
        from matplotlib.ticker import MultipleLocator

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
        if minor_gridlines:
            minor_step = (y_major_step or 2.0) / 2.0
            ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
            ax.yaxis.grid(
                True,
                color="#ECECEC",
                linewidth=0.5,
                linestyle="-",
                which="minor",
                zorder=0,
            )
            # Make minor tick marks visible to match the finer grid.
            ax.tick_params(
                axis="y",
                which="minor",
                length=2.5,
                width=0.5,
                color=text_color or "0.4",
            )
    # ``enhanced_text`` mode lets the legend swatches name the classes,
    # so we drop the title to free vertical space for the larger labels.
    if title and not enhanced_text:
        # Above the cat-band labels (which sit at y≈1.04 in axes
        # coords) so they don't collide with the title.
        ax.set_title(title, y=1.12, fontsize=10)
    if enhanced_text:
        # x-axis ticks carry the c0..c3 labels (set by _decorate_x_axis).
        # Reapply size/colour here in case a tick_params elsewhere clobbered
        # them, plus extra `pad` for the tick → label gap.
        ax.tick_params(
            axis="x",
            labelsize=ctx_label_size,
            labelcolor=text_color,
            pad=ctx_tick_pad,
        )
        ax.tick_params(
            axis="y",
            labelsize=tick_label_size,
            labelcolor=text_color,
            pad=ctx_tick_pad,
        )
    if show_legend:
        # Top headroom so the upper-right legend doesn't sit on the
        # tallest bars (the bigger legend after the +2pt bump made the
        # collision more visible).
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(top=ymax + (ymax - ymin) * 0.18)
        # When ``enhanced_text`` is on, override the caller's legend
        # font size + add extra inner padding so the bigger glyphs
        # don't crowd the frame.
        if legend_handles is not None:
            kwargs: dict[str, Any] = dict(legend_kwargs or {})
            kwargs.setdefault("loc", "upper right")
            # Caller-supplied fontsize takes priority (lets the loop
            # in ``aggregate_results`` emit alternate legend sizes
            # without subclassing). Otherwise pick the enhanced
            # default, falling back to a readable 10pt minimum.
            if "fontsize" not in kwargs:
                kwargs["fontsize"] = legend_size or 10
            for k, v in legend_pad_bump.items():
                kwargs.setdefault(k, v)
            ax.legend(handles=legend_handles, **kwargs)
        else:
            ax.legend(
                title="Experiment",
                loc="upper right",
                fontsize=legend_size or 10,
                title_fontsize=legend_size or 10,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ── Multi-experiment heatmap ────────────────────────────────────────────────


def render_heatmap(
    long_df: pd.DataFrame,
    out_path: Path,
    *,
    experiments: list[str],
    experiment_labels: dict[str, str] | None = None,
    value_col: str,
    cbar_label: str,
    title: str | None = None,
    cmap: str | None = None,
    diverging: bool = False,
) -> Path:
    """Render the cross-experiment heatmap (cat × ctx_sec, stacked per experiment).

    When ``diverging`` is True, the colour scale is symmetric around 0
    and a diverging cmap (default ``vlag``) is used — appropriate for
    signed values like ``beta_mean``. Otherwise a sequential cmap
    (default ``crest``) is used over the data's [min, max] range.
    """
    apply_theme()
    df = _attach_axes(long_df)
    df = df[df["experiment"].isin(experiments)]
    if df.empty:
        raise ValueError(
            "render_heatmap: no rows match the requested experiments.",
        )

    data_min = float(df[value_col].min())
    data_max = float(df[value_col].max())
    if diverging:
        # Use TwoSlopeNorm so the colorbar bounds match the data while 0
        # stays at the colormap centre. Forcing symmetric ±max(|x|) (the
        # naive approach) wastes one half of the bar when the data is
        # asymmetric (e.g. positives reach +7 but negatives only -2).
        if data_min < 0 < data_max:
            norm = TwoSlopeNorm(vcenter=0.0, vmin=data_min, vmax=data_max)
        elif data_min >= 0:
            norm = Normalize(vmin=0.0, vmax=data_max)
        else:
            norm = Normalize(vmin=data_min, vmax=0.0)
        # Viridis even for signed data: TwoSlopeNorm pins 0 to the
        # colormap centre, so negatives render in purple/blue and
        # positives in green/yellow — perceptually uniform and
        # colour-blind safe, no white/light bleach near zero.
        cmap = cmap or "viridis"
    else:
        norm = Normalize(vmin=data_min, vmax=data_max)
        cmap = cmap or "viridis"

    n_axes = len(experiments)
    fig, axes = plt.subplots(
        n_axes,
        1,
        figsize=(8.0, 1.9 * n_axes),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"hspace": 0.10},
    )
    if n_axes == 1:
        axes = np.array([axes])

    last_mesh = None
    for i, (ax, exp) in enumerate(zip(axes, experiments)):
        sub = df[df["experiment"] == exp].copy()
        sub["col"] = sub["ctx"] + "_" + sub["sec"]
        pivot = sub.pivot(index="cat", columns="col", values=value_col).reindex(
            index=list(CATEGORIES), columns=list(_CONTEXT_SECTION)
        )
        last_mesh = ax.pcolormesh(
            pivot.values,
            cmap=cmap,
            norm=norm,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.set_aspect("equal")
        ax.set_yticks(np.arange(len(CATEGORIES)) + 0.5)
        ax.set_yticklabels([humanize(c) for c in CATEGORIES])
        n_per_ctx = len(SECTIONS)
        ax.set_xticks([j * n_per_ctx + (n_per_ctx - 1) / 2 + 0.5 for j in range(len(CONTEXTS))])
        if i == n_axes - 1:
            ax.set_xticklabels(list(CONTEXTS), fontsize=8)
            for ctx_idx in range(len(CONTEXTS)):
                base = ctx_idx * n_per_ctx
                ax.plot(
                    [base + 0.15, base + n_per_ctx - 0.15],
                    [-0.14, -0.14],
                    transform=ax.get_xaxis_transform(),
                    color="0.7",
                    lw=0.4,
                    clip_on=False,
                )
            for j in range(len(_CONTEXT_SECTION)):
                ax.text(
                    j + 0.5,
                    -0.18,
                    SECTIONS[j % n_per_ctx],
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=6,
                    color="0.35",
                )
        else:
            ax.set_xticklabels([])

        label = (experiment_labels or {}).get(exp, exp)
        ax.set_ylabel(
            label.replace(": ", ":\n"),
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
            labelpad=12,
        )
        ax.tick_params(length=0)
        ax.invert_yaxis()
        for x in range(n_per_ctx, len(_CONTEXT_SECTION), n_per_ctx):
            ax.axvline(x, color="white", lw=2.0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    assert last_mesh is not None
    cbar = fig.colorbar(
        last_mesh,
        ax=list(axes),
        orientation="vertical",
        shrink=0.7,
        pad=0.02,
        fraction=0.025,
    )
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
