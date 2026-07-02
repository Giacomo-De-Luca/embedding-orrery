"""Shared paper-style theme + helpers for probe ablation charts.

Used by `ablation_grouped_chart` (3-panel leave-one-group-out summary)
and `ablation_feature_bars` (per-feature grouped bars + heatmap). Kept
in one place so the figures stay visually consistent and the rcParams
update happens through a single function call.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Bar palette shared across the per-probe single-experiment chart and
# the 3-panel leave-one-group-out chart.
POS_COLOR = "#3a6e8f"        # muted teal
NEG_COLOR_GRAY = "#bdbdbd"
NEG_COLOR_RED = "#d4928a"    # muted pastel red

# Feature-name parsing convention. Names look like
# `c{ctx}_p{section}_{CATEGORY}` — the third field uses ``maxsplit=2``
# so multi-token category labels (e.g. ``HARMFUL_PAYLOAD``) stay intact.
CATEGORIES: tuple[str, ...] = (
    "FIGURATIVE", "FUNCTION_WORD", "HARMFUL_PAYLOAD",
    "PUNCTUATION", "SETUP", "TECHNICAL",
)
CONTEXTS: tuple[str, ...] = ("c0", "c1", "c2", "c3")
SECTIONS: tuple[str, ...] = ("p0", "p1", "p2")


def apply_theme() -> None:
    """Paper-style seaborn + matplotlib rc setup. Idempotent."""
    sns.set_theme(context="paper", style="ticks", font="serif")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    })


def humanize(label: str) -> str:
    """Render a feature/group label for display: ``HARMFUL_PAYLOAD`` → ``Harmful Payload``."""
    if "_" in label or label.isupper():
        return label.replace("_", " ").title()
    return label


def parse_feature_name(feature_name: str) -> tuple[str, str, str]:
    """Split ``c{ctx}_p{section}_{CATEGORY}`` into ``(ctx, section, category)``.

    Falls back to ``("", "", feature_name)`` when the convention isn't
    matched, so callers can render unknown features without crashing.
    """
    parts = feature_name.split("_", 2)
    if len(parts) < 3:
        return ("", "", feature_name)
    return parts[0], parts[1], parts[2]
