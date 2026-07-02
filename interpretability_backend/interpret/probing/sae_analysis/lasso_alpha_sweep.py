"""Lasso alpha sweep on SAE features — find the natural sparsity point.

For each layer, fit `sklearn.linear_model.Lasso` at a series of alpha
values, run K-fold CV to record val R² + Spearman + n_nonzero per
alpha, then refit on full data and save the resulting `coef_` as a
per-alpha `.npz` direction file using the same schema as the existing
sklearn probes (`coef`, `intercept`, `scaler_mean`, `scaler_scale`).

Differs from `feature_sweep`:
  * `feature_sweep` ranks features marginally (ρ / Pearson r / |lasso
    coef|), then refits OLS on top-K. K is the explicit knob.
  * `lasso_alpha_sweep` lets Lasso *itself* pick which features to keep
    via L1 regularisation. Alpha is the explicit knob; n_nonzero is
    the consequence.

The two analyses give different feature subsets at the same nominal K
because Lasso accounts for feature redundancy whereas marginal ranking
does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.sae_analysis import (
    LassoAlphaSweepConfig,
)

_SAE_INTERMEDIATE = "sae_feat"


def run_lasso_alpha_sweep(
    sae_dataset: ActivationDataset,
    target_values: np.ndarray,
    config: LassoAlphaSweepConfig,
    output_dir: Path,
    *,
    target_name: str,
) -> dict[int, dict]:
    """Sweep Lasso alpha per SAE layer, save curves + per-alpha directions.

    Args:
        sae_dataset: SAE-encoded dataset with `(layer, "sae_feat")` keys.
        target_values: Aligned target array of length N.
        config: alphas, n_splits.
        output_dir: Where `lasso_alpha_sweep.json`, plots, and the
            per-alpha `directions/` folder are written.
        target_name: Display name for plot titles.

    Returns:
        Dict layer -> {results: [{alpha, val_r2_mean, val_r2_std,
        val_spearman_mean, n_nonzero_mean, n_nonzero_max}, ...]}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directions_dir = output_dir / "directions"
    directions_dir.mkdir(parents=True, exist_ok=True)

    result: dict[int, dict] = {}
    for layer, intermediate in sorted(sae_dataset.layer_intermediate_keys()):
        if intermediate != _SAE_INTERMEDIATE:
            continue
        feat_tensor, _ = sae_dataset.get(layer, intermediate)
        feat = feat_tensor.numpy().astype(np.float64, copy=False)

        layer_results: list[dict] = []
        for alpha in config.alphas:
            stats = _kfold_lasso(
                feat, target_values, alpha=alpha, n_splits=config.n_splits,
            )
            _save_full_fit_direction(
                directions_dir=directions_dir,
                layer=layer,
                intermediate=intermediate,
                feat=feat,
                target_values=target_values,
                alpha=alpha,
            )
            layer_results.append({
                "alpha": float(alpha),
                "val_r2_mean": round(stats["r2_mean"], 6),
                "val_r2_std": round(stats["r2_std"], 6),
                "val_spearman_mean": round(stats["spear_mean"], 6),
                "n_nonzero_mean": round(stats["nz_mean"], 4),
                "n_nonzero_max": int(stats["nz_max"]),
            })

        result[layer] = {
            "n_features_total": int(feat.shape[1]),
            "results": layer_results,
        }

    out_path = output_dir / "lasso_alpha_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if result:
        _plot_r2_vs_alpha(
            result, target_name, output_dir / "r2_vs_alpha.png",
        )
        _plot_nnz_vs_alpha(
            result, target_name, output_dir / "n_nonzero_vs_alpha.png",
        )
        _print_summary(result, target_name)

    return result


# ── Internals ────────────────────────────────────────────────────────────────


def _kfold_lasso(
    X: np.ndarray, y: np.ndarray, *, alpha: float, n_splits: int,
) -> dict[str, float]:
    """K-fold CV with StandardScaler + Lasso(alpha)."""
    n = X.shape[0]
    n_splits = min(n_splits, n)
    if n_splits < 2:
        raise ValueError(
            f"lasso_alpha_sweep needs >= 2 samples per fold, got n={n}",
        )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    r2_scores: list[float] = []
    spear_scores: list[float] = []
    nz_counts: list[int] = []
    for train_idx, val_idx in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        model = Lasso(alpha=alpha, max_iter=20000)
        model.fit(X_tr, y[train_idx])
        pred = model.predict(X_val)
        r2_scores.append(_r2(y[val_idx], pred))
        rho, _ = spearmanr(pred, y[val_idx])
        spear_scores.append(
            float(rho) if rho is not None and not np.isnan(rho) else 0.0,
        )
        nz_counts.append(int(np.count_nonzero(model.coef_)))

    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "spear_mean": float(np.mean(spear_scores)),
        "nz_mean": float(np.mean(nz_counts)),
        "nz_max": int(np.max(nz_counts)),
    }


def _save_full_fit_direction(
    *,
    directions_dir: Path,
    layer: int,
    intermediate: str,
    feat: np.ndarray,
    target_values: np.ndarray,
    alpha: float,
) -> None:
    """Refit on full data; save coef + scaler stats with same schema as sklearn probes."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(feat)
    model = Lasso(alpha=alpha, max_iter=20000)
    model.fit(X_s, target_values)

    kind = f"lasso_alpha={alpha:g}"
    path = directions_dir / f"L{layer}_{intermediate}_{kind}.npz"
    arrays: dict[str, np.ndarray] = {
        "coef": np.asarray(model.coef_, dtype=np.float64),
        "intercept": np.atleast_1d(np.asarray(model.intercept_, dtype=np.float64)),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
    }
    np.savez(str(path), **arrays)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _plot_r2_vs_alpha(
    result: dict[int, dict], target_name: str, output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for layer in sorted(result):
        rows = result[layer]["results"]
        alphas = [r["alpha"] for r in rows]
        means = [r["val_r2_mean"] for r in rows]
        stds = [r["val_r2_std"] for r in rows]
        ax.errorbar(
            alphas, means, yerr=stds, marker="o", capsize=3,
            label=f"L{layer}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Lasso alpha (log)")
    ax.set_ylabel("CV val R²")
    ax.set_title(f"Lasso alpha → R² for {target_name}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Layer")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_nnz_vs_alpha(
    result: dict[int, dict], target_name: str, output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for layer in sorted(result):
        rows = result[layer]["results"]
        alphas = [r["alpha"] for r in rows]
        nnz = [r["n_nonzero_mean"] for r in rows]
        ax.plot(alphas, nnz, marker="o", label=f"L{layer}")
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel("Lasso alpha (log)")
    ax.set_ylabel("# nonzero coefficients (mean over folds)")
    ax.set_title(f"Lasso sparsity for {target_name}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(title="Layer")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_summary(result: dict[int, dict], target_name: str) -> None:
    print(f"\nLasso alpha sweep — {target_name}:")
    for layer, payload in sorted(result.items()):
        rows = payload["results"]
        if not rows:
            continue
        line = "  ".join(
            f"α={r['alpha']:g}: R²={r['val_r2_mean']:+.3f} "
            f"(nnz={r['n_nonzero_mean']:.0f})"
            for r in rows
        )
        print(f"  L{layer}: {line}")
