"""Per-feature Spearman correlation between SAE feature activations and targets.

For each layer that has SAE feature activations, compute the Spearman
correlation of every feature's activation pattern (across samples) with
each target column. Enriches with Neuronpedia labels and writes per-layer
CSVs + a combined summary JSON + heatmap/distribution plots.

This is *not* a probe — there is no train/val split, no fit. It's a
descriptive analysis intended to surface SAE features whose activation
patterns track a target variable across the corpus.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.sae_analysis import (
    CorrelationMapConfig,
)

# SAE intermediate name produced by extract_sae_activations.
from interpret.probing.sae_analysis.constants import SAE_INTERMEDIATES
from interpret.probing.sae_analysis.labels import (
    load_feature_labels,
)


def run_correlation_map(
    sae_dataset: ActivationDataset,
    targets: dict[str, np.ndarray],
    config: CorrelationMapConfig,
    output_dir: Path,
    *,
    width: str,
) -> dict[int, pd.DataFrame]:
    """Compute per-feature correlations across all SAE-encoded layers.

    Args:
        sae_dataset: SAE-encoded activations with `(layer, intermediate)` (any intermediate in `SAE_INTERMEDIATES`)
            keys. `metadata["kept_by_layer"]` is required for label
            enrichment (maps filtered column index -> original SAE feature
            index).
        targets: `{column_name: aligned_array_of_length_N}` mapping.
        config: Top-K, density filter, label parquet directory.
        output_dir: Where per-layer CSVs + plots + summary go.
        width: SAE width string (e.g. "16k") used for label lookup.

    Returns:
        Dict layer -> per-feature correlation DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_by_layer = sae_dataset.metadata.get("kept_by_layer", {})
    target_columns = list(targets)

    all_corrs: dict[int, pd.DataFrame] = {}
    for layer, intermediate in sorted(sae_dataset.layer_intermediate_keys()):
        if intermediate not in SAE_INTERMEDIATES:
            continue
        feat_tensor, _ = sae_dataset.get(layer, intermediate)
        feat = feat_tensor.numpy()

        kept = _to_kept_array(kept_by_layer.get(layer), feat.shape[1])
        labels = load_feature_labels(config.sae_vectors_dir, layer, width)

        df = _correlate_features(
            feat,
            targets,
            max_density=config.max_density,
            kept=kept,
        )
        df["layer"] = layer
        df["label"] = df["feature_idx"].map(labels).fillna("")
        all_corrs[layer] = df

        df.to_csv(output_dir / f"correlations_layer{layer}.csv", index=False)
        _print_top_per_target(df, target_columns, layer, config.top_k)

    if all_corrs:
        combined = pd.concat(all_corrs.values(), ignore_index=True)
        combined.to_csv(output_dir / "all_correlations.csv", index=False)
        _write_summary(all_corrs, target_columns, config.top_k, output_dir)
        _plot_topk_heatmap(
            all_corrs,
            target_columns,
            config.top_k,
            output_dir / "topk_heatmap.png",
        )
        _plot_distribution(
            all_corrs,
            target_columns,
            output_dir / "correlation_distribution.png",
        )

    return all_corrs


# ── Internals ────────────────────────────────────────────────────────────────


def _to_kept_array(
    kept: list[int] | None,
    n_columns: int,
) -> np.ndarray:
    """Resolve a kept-feature mapping; default to identity if missing."""
    if kept is None:
        return np.arange(n_columns, dtype=np.int64)
    arr = np.asarray(kept, dtype=np.int64)
    if len(arr) != n_columns:
        raise ValueError(
            f"kept_by_layer length {len(arr)} != feature columns {n_columns}",
        )
    return arr


def _correlate_features(
    feat: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    max_density: float | None,
    kept: np.ndarray,
) -> pd.DataFrame:
    """Spearman correlation between every feature column and every target."""
    n_samples, n_features = feat.shape
    rows = []
    for col_idx in range(n_features):
        col = feat[:, col_idx]
        nonzero = np.count_nonzero(col)
        density = nonzero / n_samples
        if max_density is not None and density > max_density:
            continue
        row = {
            "feature_idx": int(kept[col_idx]),
            "nonzero_frac": float(density),
            "mean_activation": float(col.mean()),
        }
        for name, vals in targets.items():
            row[f"rho_{name}"] = _safe_spearman(col, vals)
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(a, b)
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return float("nan")
    return float(rho)  # type: ignore[arg-type]


def _print_top_per_target(
    df: pd.DataFrame,
    target_columns: list[str],
    layer: int,
    top_k: int,
) -> None:
    for target in target_columns:
        rho_col = f"rho_{target}"
        if rho_col not in df.columns:
            continue
        top = df.reindex(
            df[rho_col].abs().sort_values(ascending=False).index,
        ).head(top_k)
        print(f"\nLayer {layer} — top {top_k} features for {target}:")
        for _, r in top.iterrows():
            lbl = f"  [{r.label[:55]}]" if r.label else ""
            print(
                f"  F{int(r.feature_idx):>5d}  "
                f"rho={r[rho_col]:+.3f}  "
                f"active={r.nonzero_frac:.1%}  "
                f"mean_act={r.mean_activation:.3f}"
                f"{lbl}",
            )


def _write_summary(
    all_corrs: dict[int, pd.DataFrame],
    target_columns: list[str],
    top_k: int,
    output_dir: Path,
) -> None:
    summary: dict = {}
    for layer, df in sorted(all_corrs.items()):
        layer_summary: dict = {}
        for target in target_columns:
            rho_col = f"rho_{target}"
            if rho_col not in df.columns:
                continue
            top = df.nlargest(top_k, rho_col)
            bottom = df.nsmallest(top_k, rho_col)
            layer_summary[target] = {
                "top_positive": [
                    {
                        "feature_idx": int(r.feature_idx),
                        "rho": round(float(r[rho_col]), 4),
                        "label": str(r.get("label", "")),
                    }
                    for _, r in top.iterrows()
                ],
                "top_negative": [
                    {
                        "feature_idx": int(r.feature_idx),
                        "rho": round(float(r[rho_col]), 4),
                        "label": str(r.get("label", "")),
                    }
                    for _, r in bottom.iterrows()
                ],
            }
        summary[f"layer{layer}"] = layer_summary

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _plot_topk_heatmap(
    all_corrs: dict[int, pd.DataFrame],
    target_columns: list[str],
    top_k: int,
    output_path: Path,
) -> None:
    layers = sorted(all_corrs)
    n_targets = len(target_columns)
    fig, axes = plt.subplots(
        n_targets,
        len(layers),
        figsize=(5 * len(layers), 0.35 * top_k * n_targets),
        squeeze=False,
    )
    fig.suptitle(
        "Top SAE features by |Spearman ρ| with manifest targets",
        fontsize=14,
        y=1.01,
    )
    for col_idx, layer in enumerate(layers):
        df = all_corrs[layer]
        for row_idx, target in enumerate(target_columns):
            ax = axes[row_idx, col_idx]
            rho_col = f"rho_{target}"
            if rho_col not in df.columns:
                ax.set_visible(False)
                continue
            top = df.reindex(
                df[rho_col].abs().sort_values(ascending=False).index,
            ).head(top_k)
            labels = [
                f"F{int(r.feature_idx)}"
                + (f": {str(r.get('label', ''))[:40]}" if r.get("label") else "")
                for _, r in top.iterrows()
            ]
            values = top[rho_col].values
            colors = plt.cm.RdBu_r((values + 1) / 2)
            ax.barh(range(len(values)), values, color=colors)
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlim(-1, 1)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.invert_yaxis()
            if row_idx == 0:
                ax.set_title(f"Layer {layer}", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(target, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_distribution(
    all_corrs: dict[int, pd.DataFrame],
    target_columns: list[str],
    output_path: Path,
) -> None:
    layers = sorted(all_corrs)
    n_targets = len(target_columns)
    fig, axes = plt.subplots(
        n_targets,
        len(layers),
        figsize=(4 * len(layers), 3.5 * n_targets),
        squeeze=False,
    )
    fig.suptitle(
        "Distribution of feature–target Spearman ρ",
        fontsize=13,
        y=1.01,
    )
    for col_idx, layer in enumerate(layers):
        df = all_corrs[layer]
        for row_idx, target in enumerate(target_columns):
            ax = axes[row_idx, col_idx]
            rho_col = f"rho_{target}"
            if rho_col not in df.columns:
                ax.set_visible(False)
                continue
            vals = df[rho_col].dropna().values
            ax.hist(vals, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("ρ")
            if col_idx == 0:
                ax.set_ylabel(target)
            if row_idx == 0:
                ax.set_title(f"Layer {layer} (n={len(vals)})")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
