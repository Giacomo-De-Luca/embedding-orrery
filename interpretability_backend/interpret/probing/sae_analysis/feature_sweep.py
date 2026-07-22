"""Top-K refit sweep: how well does the prediction scale with feature count?

For each layer, rank SAE features by either |Spearman ρ| or |lasso coef|,
then refit plain OLS on the top-K columns for K in `k_values` and report
cross-validated R²/Spearman per K. Surfaces the prediction-quality curve
as a function of feature count — answers "how many sparse features do I
need to predict the target?".

Differs from `top_features` (which only ranks features by an existing
probe's |coef|) and `correlation_map` (which only reports per-feature
ρ): this analysis actually refits a small linear model on the chosen
subset and tells you the predictive value of using just those features.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.sae_analysis import (
    FeatureSweepConfig,
)
from interpret.probing.sae_analysis.constants import SAE_INTERMEDIATES
from interpret.probing.sae_analysis.directions import load_direction_coef
from interpret.probing.sae_analysis.labels import (
    load_feature_labels,
)
from interpret.probing.utils.matrix_ops import to_dense_numpy


def run_feature_sweep(
    sae_dataset: ActivationDataset,
    target_values: np.ndarray,
    config: FeatureSweepConfig,
    output_dir: Path,
    *,
    width: str,
    target_name: str,
    directions_dir: Path | None = None,
) -> dict[str | int, dict]:
    """Refit OLS on top-K features and report CV R² for each K.

    Default mode: per-layer sweep. When `config.pool_layers` is True,
    features from every SAE layer are concatenated into one wide matrix
    and a single global ranking + sweep runs over the pool.

    Args:
        sae_dataset: SAE-encoded dataset; expects `(layer, intermediate)` (any intermediate in `SAE_INTERMEDIATES`) keys
            and `metadata["kept_by_layer"]` mapping filtered column index ->
            original SAE feature index.
        target_values: Aligned target array of length N.
        config: Sweep configuration (ranking strategy, k_values, n_splits,
            pool_layers).
        output_dir: Where `feature_sweep.json` and the R²-vs-K plot go.
        width: SAE width string (e.g. "16k") for label lookup.
        target_name: Display name (used in plot titles).
        directions_dir: Required when `ranking == "lasso"`. Path to the
            `directions/` folder of the source probe (typically
            `<output>/probes/<extraction>/<target>/<source_probe>/directions`).

    Returns:
        - Per-layer mode: `{layer: {ranking, n_features_total, results}}`.
        - Pooled mode: `{"pooled": {ranking, layers_in_pool,
          n_features_total, results}}`. Each result also includes
          `feature_layers` parallel to `feature_indices`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.ranking == "lasso" and directions_dir is None:
        raise ValueError(
            "feature_sweep with ranking='lasso' requires directions_dir.",
        )

    if config.pool_layers:
        result = _run_pooled(
            sae_dataset=sae_dataset,
            target_values=target_values,
            config=config,
            width=width,
            directions_dir=directions_dir,
        )
    else:
        result = _run_per_layer(
            sae_dataset=sae_dataset,
            target_values=target_values,
            config=config,
            width=width,
            directions_dir=directions_dir,
        )

    out_path = output_dir / "feature_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if result:
        _plot_r2_vs_k(result, target_name, output_dir / "r2_vs_k.png")
        _print_summary(result, target_name)

    return result


def _run_per_layer(
    *,
    sae_dataset: ActivationDataset,
    target_values: np.ndarray,
    config: FeatureSweepConfig,
    width: str,
    directions_dir: Path | None,
) -> dict[str | int, dict]:
    kept_by_layer = sae_dataset.metadata.get("kept_by_layer", {})
    result: dict[str | int, dict] = {}
    for layer, intermediate in sorted(sae_dataset.layer_intermediate_keys()):
        if intermediate not in SAE_INTERMEDIATES:
            continue
        feat_tensor, _ = sae_dataset.get(layer, intermediate)
        # Small-data analysis: densify sparse input (never configured on
        # concat-scale matrices).
        feat = to_dense_numpy(feat_tensor)
        kept = _resolve_kept(kept_by_layer.get(layer), feat.shape[1])

        ranking_scores = _compute_ranking(
            feat=feat,
            target_values=target_values,
            ranking=config.ranking,
            directions_dir=directions_dir,
            layer=layer,
            intermediate=intermediate,
            source_probe=config.source_probe,
        )
        if ranking_scores is None:
            print(
                f"  feature_sweep layer {layer}: no ranking available "
                f"(missing lasso directions?), skipping",
            )
            continue

        order = np.argsort(np.abs(ranking_scores))[::-1]
        labels = load_feature_labels(config.sae_vectors_dir, layer, width)

        layer_results = _kfold_sweep(
            feat=feat,
            target_values=target_values,
            order=order,
            ranking_scores=ranking_scores,
            kept=kept,
            labels_per_kept=lambda kept_idx, lbls=labels: lbls.get(int(kept_idx), ""),
            k_values=config.k_values,
            n_splits=config.n_splits,
        )
        result[layer] = {
            "ranking": config.ranking,
            "n_features_total": int(feat.shape[1]),
            "results": layer_results,
        }
    return result


def _run_pooled(
    *,
    sae_dataset: ActivationDataset,
    target_values: np.ndarray,
    config: FeatureSweepConfig,
    width: str,
    directions_dir: Path | None,
) -> dict[str | int, dict]:
    """Concatenate features across SAE layers, rank pooled, sweep top-K."""
    kept_by_layer = sae_dataset.metadata.get("kept_by_layer", {})

    feat_blocks: list[np.ndarray] = []
    layer_of_pool: list[int] = []
    kept_of_pool: list[int] = []
    coef_of_pool: list[np.ndarray] = []  # only used when ranking == "lasso"
    layers_in_pool: list[int] = []
    labels_per_layer: dict[int, dict[int, str]] = {}

    for layer, intermediate in sorted(sae_dataset.layer_intermediate_keys()):
        if intermediate not in SAE_INTERMEDIATES:
            continue
        feat_tensor, _ = sae_dataset.get(layer, intermediate)
        feat = to_dense_numpy(feat_tensor)
        kept = _resolve_kept(kept_by_layer.get(layer), feat.shape[1])
        feat_blocks.append(feat)
        layer_of_pool.extend([layer] * feat.shape[1])
        kept_of_pool.extend(kept.tolist())
        layers_in_pool.append(layer)
        labels_per_layer[layer] = load_feature_labels(
            config.sae_vectors_dir,
            layer,
            width,
        )
        if config.ranking == "lasso":
            assert directions_dir is not None
            raw_coef = load_direction_coef(
                directions_dir,
                layer,
                intermediate,
                config.source_probe,
            )
            if raw_coef is None:
                print(
                    f"  feature_sweep[pooled] layer {layer}: no lasso "
                    f"directions, skipping pooled sweep",
                )
                return {}
            coef = raw_coef.reshape(-1)
            if coef.shape[0] != feat.shape[1]:
                print(
                    f"  feature_sweep[pooled] layer {layer}: lasso coef "
                    f"shape {coef.shape[0]} != feature columns "
                    f"{feat.shape[1]}, skipping pooled sweep",
                )
                return {}
            coef_of_pool.append(coef)

    if not feat_blocks:
        return {}

    feat_pool = np.concatenate(feat_blocks, axis=1)
    layer_arr = np.asarray(layer_of_pool, dtype=np.int64)
    kept_arr = np.asarray(kept_of_pool, dtype=np.int64)

    if config.ranking == "lasso":
        ranking_scores = np.concatenate(coef_of_pool)
    else:
        # Correlation-based ranking computed over the pooled feature matrix.
        ranking_fn = _pearson_per_feature if config.ranking == "pearson" else _spearman_per_feature
        ranking_scores = ranking_fn(feat_pool, target_values)

    order = np.argsort(np.abs(ranking_scores))[::-1]

    def label_lookup(pool_idx: int) -> str:
        layer = int(layer_arr[pool_idx])
        kept_idx = int(kept_arr[pool_idx])
        return labels_per_layer.get(layer, {}).get(kept_idx, "")

    pool_results = _kfold_sweep(
        feat=feat_pool,
        target_values=target_values,
        order=order,
        ranking_scores=ranking_scores,
        kept=kept_arr,
        labels_per_kept=lambda _kept_idx: "",  # filled below per pool position
        k_values=config.k_values,
        n_splits=config.n_splits,
        layer_of_pool=layer_arr,
        label_pool_lookup=label_lookup,
    )

    return {
        "pooled": {
            "ranking": config.ranking,
            "layers_in_pool": layers_in_pool,
            "n_features_total": int(feat_pool.shape[1]),
            "results": pool_results,
        },
    }


def _kfold_sweep(
    *,
    feat: np.ndarray,
    target_values: np.ndarray,
    order: np.ndarray,
    ranking_scores: np.ndarray,
    kept: np.ndarray,
    labels_per_kept,
    k_values: list[int],
    n_splits: int,
    layer_of_pool: np.ndarray | None = None,
    label_pool_lookup=None,
) -> list[dict]:
    """Run the K-sweep loop given a feature matrix and a feature ordering.

    `layer_of_pool` and `label_pool_lookup` are only used in pooled mode;
    when set, each result row carries `feature_layers` alongside
    `feature_indices` and labels are resolved via the pool-aware lookup.
    """
    results: list[dict] = []
    for k in k_values:
        if k > feat.shape[1]:
            continue
        top_idx = order[:k]
        X_k = feat[:, top_idx]
        r2_mean, r2_std, spear_mean = _kfold_ols(
            X_k,
            target_values,
            n_splits=n_splits,
        )
        feat_indices = [int(kept[i]) for i in top_idx]
        if layer_of_pool is not None:
            assert label_pool_lookup is not None
            labels = [label_pool_lookup(int(i)) for i in top_idx]
            row: dict = {
                "k": int(k),
                "feature_indices": feat_indices,
                "feature_layers": [int(layer_of_pool[i]) for i in top_idx],
                "labels": labels,
            }
        else:
            row = {
                "k": int(k),
                "feature_indices": feat_indices,
                "labels": [labels_per_kept(fi) for fi in feat_indices],
            }
        row["ranking_scores"] = [round(float(ranking_scores[i]), 6) for i in top_idx]
        row["val_r2_mean"] = round(float(r2_mean), 6)
        row["val_r2_std"] = round(float(r2_std), 6)
        row["val_spearman_mean"] = round(float(spear_mean), 6)
        results.append(row)
    return results


# ── Internals ────────────────────────────────────────────────────────────────


def _resolve_kept(
    kept: list[int] | None,
    n_columns: int,
) -> np.ndarray:
    if kept is None:
        return np.arange(n_columns, dtype=np.int64)
    arr = np.asarray(kept, dtype=np.int64)
    if len(arr) != n_columns:
        raise ValueError(
            f"kept_by_layer length {len(arr)} != feature columns {n_columns}",
        )
    return arr


def _compute_ranking(
    *,
    feat: np.ndarray,
    target_values: np.ndarray,
    ranking: str,
    directions_dir: Path | None,
    layer: int,
    intermediate: str,
    source_probe: str,
) -> np.ndarray | None:
    """Per-feature score whose absolute value drives the top-K selection."""
    if ranking == "rho":
        return _spearman_per_feature(feat, target_values)
    if ranking == "pearson":
        return _pearson_per_feature(feat, target_values)
    if ranking == "lasso":
        assert directions_dir is not None
        raw_coef = load_direction_coef(
            directions_dir,
            layer,
            intermediate,
            source_probe,
        )
        if raw_coef is None:
            return None
        coef = raw_coef.reshape(-1)
        if coef.shape[0] != feat.shape[1]:
            print(
                f"  feature_sweep layer {layer}: lasso coef shape "
                f"{coef.shape[0]} != feature columns {feat.shape[1]}, "
                f"skipping",
            )
            return None
        return coef
    raise ValueError(f"Unknown ranking: {ranking!r}")


def _spearman_per_feature(
    feat: np.ndarray,
    target_values: np.ndarray,
) -> np.ndarray:
    n_features = feat.shape[1]
    out = np.zeros(n_features, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_features):
            col = feat[:, i]
            if np.std(col) < 1e-12:
                continue
            rho, _ = spearmanr(col, target_values)
            if rho is not None and not np.isnan(rho):
                out[i] = float(rho)
    return out


def _pearson_per_feature(
    feat: np.ndarray,
    target_values: np.ndarray,
) -> np.ndarray:
    """Vectorised Pearson r between each feature column and the target."""
    y = np.asarray(target_values, dtype=np.float64)
    y_centered = y - y.mean()
    y_std = float(y.std())
    if y_std < 1e-12:
        return np.zeros(feat.shape[1], dtype=np.float64)

    feat_f = feat.astype(np.float64, copy=False)
    feat_centered = feat_f - feat_f.mean(axis=0, keepdims=True)
    feat_std = feat_f.std(axis=0)
    # 0-std columns -> r=0; avoid divide-by-zero with a safe denominator.
    safe = feat_std > 1e-12
    denom = np.where(safe, feat_std, 1.0) * y_std
    cov = (feat_centered * y_centered[:, None]).mean(axis=0)
    out = cov / denom
    out[~safe] = 0.0
    return out


def _kfold_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
) -> tuple[float, float, float]:
    """K-fold CV with plain OLS; returns (mean R², std R², mean Spearman)."""
    n = X.shape[0]
    n_splits = min(n_splits, n)
    if n_splits < 2:
        raise ValueError(
            f"feature_sweep needs >= 2 samples per fold, got n={n}",
        )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    r2_scores: list[float] = []
    spear_scores: list[float] = []
    for train_idx, val_idx in kf.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        r2 = _r2(y[val_idx], pred)
        r2_scores.append(r2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = spearmanr(pred, y[val_idx])
        spear_scores.append(
            float(rho) if rho is not None and not np.isnan(rho) else 0.0,
        )
    return (
        float(np.mean(r2_scores)),
        float(np.std(r2_scores)),
        float(np.mean(spear_scores)),
    )


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _plot_r2_vs_k(
    result: dict[str | int, dict],
    target_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for key in sorted(result, key=str):
        rows = result[key]["results"]
        ks = [r["k"] for r in rows]
        means = [r["val_r2_mean"] for r in rows]
        stds = [r["val_r2_std"] for r in rows]
        ax.errorbar(
            ks,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=_legend_label(key),
        )
    ranking = next(iter(result.values()))["ranking"]
    ax.set_xscale("log")
    ax.set_xlabel(f"K (top features by |{ranking}|)")
    ax.set_ylabel("CV val R²")
    ax.set_title(
        f"Top-K SAE features → OLS prediction of {target_name}",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Group")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_summary(result: dict[str | int, dict], target_name: str) -> None:
    print(f"\nFeature sweep — {target_name}:")
    for key, payload in sorted(result.items(), key=lambda kv: str(kv[0])):
        rows = payload["results"]
        if not rows:
            continue
        line = "  ".join(
            f"K={r['k']}: R²={r['val_r2_mean']:+.3f}±{r['val_r2_std']:.3f}" for r in rows
        )
        print(f"  {_legend_label(key)}: {line}")


def _legend_label(key: str | int) -> str:
    return f"L{key}" if isinstance(key, int) else str(key)
