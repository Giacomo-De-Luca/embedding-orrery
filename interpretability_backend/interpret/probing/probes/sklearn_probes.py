"""Sklearn-family probes: ridge / lasso / svr / logreg / massmean.

One probe per (layer, intermediate, spec). Per-spec output folder under
`<output_dir>/probes/<spec.name>/`.

Differences vs the old `run_sklearn_probes._train_probes_on_layers`:

1. Spec-driven: caller passes a single `SklearnProbeSpec`; each call
   trains one probe-kind across all layers (instead of all kinds at once).
2. Lean CSV schema: only metric columns that actually got populated for
   the task at hand are emitted (classification probes drop the regression
   metrics, etc.). When k-fold is in use, two summary rows per
   (layer, intermediate) are appended at the bottom — `fold=mean` and
   `fold=std` — so the per-fold values and their cross-fold spread sit
   in the same CSV.
3. Per-probe try/except: on fit failure, record an `error` row with NaN
   metrics and continue. Enables per-probe resumability.
4. Optional distance metric (e.g. `val_lab_distance`, mean CIEDE2000) when
   `spec.distance` names an `ExperimentalDistance` whose `target_dim` matches.
5. Optional k-fold cross-validation: when `spec.n_folds` is set, the
   probe trains once per fold (StratifiedKFold for classification,
   KFold otherwise), writes one CSV row per (layer, intermediate, fold),
   and aggregates mean/std into `summary.json`. Per-fold directions are
   saved as `directions/L{layer}_{intermediate}_{kind}_fold{i}.npz`.
6. Standardised |β| feature-importance: when k-fold is run on a logreg
   probe with `save_directions=True`, a `feature_importance.csv` is
   written next to `directions/` with mean/std of standardised
   coefficients across folds (the saved coefs are already in
   `StandardScaler`-units, so they are directly comparable).
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.probe import SklearnProbeSpec
from interpret.probing.utils.cross_validation import resolve_folds
from interpret.utils.distances import resolve_distance

# Probes whose `coef_` is meaningful for downstream feature ranking.
_LINEAR_PROBE_KINDS = {"ridge", "lasso", "logreg"}

# Probe kinds that take integer class labels rather than continuous targets.
_CLASSIFICATION_KINDS = {"logreg", "svc"}


def train_sklearn_probe(
    dataset: ActivationDataset,
    spec: SklearnProbeSpec,
    targets: np.ndarray,
    output_dir: Path,
    *,
    groups: np.ndarray | None = None,
    indices_override: tuple[np.ndarray, np.ndarray] | None = None,
    feature_names: list[str] | None = None,
) -> Path:
    """Train one sklearn probe of `spec.kind` across all (layer, intermediate).

    Args:
        dataset: Activations container. `dataset.targets` is ignored.
        spec: Probe kind + hyperparams.
        targets: Aligned targets (1-D for most kinds; may be 2-D for ridge).
        output_dir: Per-probe folder. CSV + summary.json + (if requested)
            directions/ subfolder go here.
        groups: Optional per-sample group labels. When provided, the
            train/val split uses `GroupShuffleSplit` so members of the
            same group never end up on opposite sides of the split.
            Required when probing paired data (e.g. tinted variants of
            the same source image) to prevent shape leakage.
        indices_override: Optional explicit (train_idx, val_idx) pair.
            When provided, bypasses internal split logic — used by
            callers that drive their own k-fold or other custom split
            schedule. Mutually exclusive with `spec.n_folds`.
        feature_names: Optional list aligned with the feature axis of
            each layer's tensor. When provided AND k-fold runs on a
            logreg probe with `save_directions=True`, the function
            additionally writes `feature_importance.csv` with one row
            per (layer, intermediate, feature_name).

    Returns:
        The output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "probe_results.csv"
    directions_dir = (
        output_dir / "directions" if spec.save_directions else None
    )
    if directions_dir is not None:
        directions_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models" if spec.save_models else None
    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute classification labels BEFORE splitting so that
    # StratifiedKFold gets the actual class labels (not raw continuous
    # values that would be misinterpreted as one class per unique
    # float). When `targets` are already integer class indices (e.g.
    # from a categorical manifest column), use them directly — re-
    # binning binary 0/1 with n_bins=2 collapses everything onto one
    # class because the median equals one of the two values.
    is_classification = spec.kind in _CLASSIFICATION_KINDS
    if is_classification and targets.ndim == 1:
        if np.issubdtype(targets.dtype, np.integer):
            y_binned = targets.astype(int, copy=False)
        else:
            y_binned = _bin_targets(targets, spec.classification_bins)
    else:
        y_binned = None

    folds = resolve_folds(
        n=len(targets),
        n_folds=spec.n_folds,
        seed=spec.seed,
        is_classification=is_classification,
        stratify_y=y_binned,
        train_split=spec.train_split,
        groups=groups,
        indices_override=indices_override,
    )

    keys = sorted(dataset.layer_intermediate_keys())
    rows: list[dict] = []
    # Per-(layer, intermediate) bucket of standardised coefficient rows
    # captured during the fold loop. Drained into feature_importance.csv
    # at the end if conditions are met.
    coef_buckets: dict[tuple[int, str], list[np.ndarray]] = {}

    for layer, intermediate in keys:
        tensor, _ = dataset.get(layer, intermediate)
        X = tensor.numpy()

        for fold_label, train_idx, val_idx in folds:
            row: dict = {
                "layer": layer,
                "intermediate": intermediate,
                "probe_kind": spec.kind,
                "fold": fold_label,
            }
            try:
                metrics, fold_coef = _fit_one(
                    X, targets, y_binned,
                    train_idx, val_idx,
                    spec=spec,
                    layer=layer,
                    intermediate=intermediate,
                    directions_dir=directions_dir,
                    models_dir=models_dir,
                    fold_label=fold_label if len(folds) > 1 else None,
                )
                row.update(metrics)
                if fold_coef is not None and spec.kind == "logreg":
                    coef_buckets.setdefault(
                        (layer, intermediate), [],
                    ).append(fold_coef)
            except Exception as exc:  # noqa: BLE001 — per-probe isolation
                msg = f"{type(exc).__name__}: {exc}"
                print(
                    f"  [error] L{layer}/{intermediate}/{spec.kind}/"
                    f"{fold_label}: {msg}",
                )
                row["error"] = msg
            rows.append(row)
            print(
                f"  [done]  L{layer}/{intermediate}/{spec.kind}/"
                f"{fold_label}",
            )

    multi_fold = len(folds) > 1
    _write_probe_results(csv_path, rows, probe_kind=spec.kind, multi_fold=multi_fold)
    _maybe_write_feature_importance(
        output_dir,
        spec=spec,
        coef_buckets=coef_buckets,
        feature_names=feature_names,
        multi_fold=multi_fold,
    )
    _write_summary(
        output_dir, spec=spec, rows=rows, multi_fold=multi_fold,
    )
    return output_dir


# ── Internals ────────────────────────────────────────────────────────────────


# Canonical column order for the CSV. Only metric columns that actually
# got populated for the run (i.e. have at least one non-NaN value across
# the per-fold rows) are emitted — classification probes drop the
# regression metrics and vice versa.
_FIXED_COLUMNS: tuple[str, ...] = ("layer", "intermediate", "probe_kind", "fold")
_CANONICAL_METRIC_ORDER: tuple[str, ...] = (
    "val_accuracy", "val_f1_weighted", "val_auc",
    "val_r2", "val_mse", "val_mae", "train_r2",
    "val_pearson", "train_pearson",
    "val_spearman", "train_spearman",
    "val_lab_distance",
)


def _write_probe_results(
    csv_path: Path,
    rows: list[dict],
    *,
    probe_kind: str,
    multi_fold: bool,
) -> None:
    """Render per-fold rows + optional mean/std summary rows to CSV.

    Drops metric columns that are all-NaN. Also drops ``layer`` and
    ``intermediate`` columns when each is constant across all rows
    (e.g. csv_features extractions where ``layer=0`` /
    ``intermediate=features`` carry no information). When
    ``multi_fold`` is True, appends two summary rows per
    (layer, intermediate) pair labelled ``fold=mean`` and ``fold=std``
    so the CSV captures cross-fold spread alongside the raw per-fold
    values.
    """
    populated = {
        k for r in rows for k, v in r.items()
        if k not in _FIXED_COLUMNS and k != "error" and pd.notna(v)
    }
    metric_columns = [m for m in _CANONICAL_METRIC_ORDER if m in populated]
    saw_error = any(pd.notna(r.get("error")) for r in rows)

    # Suppress fixed columns whose values are constant across all rows.
    # `probe_kind` and `fold` always carry information (kind is the row's
    # probe and fold distinguishes the per-fold and summary rows), so
    # only `layer` and `intermediate` are eligible for dropping.
    drop_constant = {
        col for col in ("layer", "intermediate")
        if len({r.get(col) for r in rows}) <= 1
    }
    fixed_kept = tuple(c for c in _FIXED_COLUMNS if c not in drop_constant)

    columns = [
        *fixed_kept, *metric_columns,
        *(["error"] if saw_error else []),
    ]

    out_rows = list(rows)
    if multi_fold:
        groups: dict[tuple[int, str], list[dict]] = {}
        for r in rows:
            groups.setdefault((r["layer"], r["intermediate"]), []).append(r)
        for (layer, intermediate), bucket in groups.items():
            for stat_name in ("mean", "std"):
                summary_row: dict = {
                    "layer": layer,
                    "intermediate": intermediate,
                    "probe_kind": probe_kind,
                    "fold": stat_name,
                }
                for m in metric_columns:
                    vals = [r[m] for r in bucket if pd.notna(r.get(m))]
                    if not vals:
                        summary_row[m] = np.nan
                    elif stat_name == "mean":
                        summary_row[m] = float(np.mean(vals))
                    else:
                        # ddof=0 (population std), matches summary.json.
                        summary_row[m] = (
                            float(np.std(vals, ddof=0))
                            if len(vals) > 1 else np.nan
                        )
                out_rows.append(summary_row)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)


def _build_estimator(spec: SklearnProbeSpec) -> tuple[BaseEstimator, bool]:
    """Return (estimator_template, is_classification) for spec.kind."""
    kind = spec.kind
    if kind == "ridge":
        return Ridge(alpha=spec.alpha), False
    if kind == "lasso":
        return Lasso(alpha=spec.alpha, max_iter=spec.max_iter), False
    if kind == "svr":
        return SVR(C=spec.C, kernel=spec.kernel), False
    if kind == "svc":
        return SVC(
            C=spec.C, kernel=spec.kernel, class_weight=spec.class_weight,
        ), True
    if kind == "logreg":
        return LogisticRegression(
            C=spec.C, max_iter=spec.logreg_max_iter, class_weight=spec.class_weight,
        ), True
    if kind == "massmean":
        # Closed-form: handled separately in `_fit_one`.
        return DummyRegressor(strategy="mean"), False
    raise ValueError(f"Unknown sklearn probe kind: {kind!r}")


def _fit_one(
    X: np.ndarray,
    y: np.ndarray,
    y_binned: np.ndarray | None,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    spec: SklearnProbeSpec,
    layer: int,
    intermediate: str,
    directions_dir: Path | None,
    models_dir: Path | None = None,
    fold_label: str | None,
) -> tuple[dict, np.ndarray | None]:
    """Fit a single (layer, intermediate, spec, fold) probe.

    Returns the metric dict and (when applicable) the standardised
    coefficient vector — needed by the caller to aggregate per-fold |β|
    into a feature-importance table.
    """
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    if spec.standardise:
        scaler = StandardScaler(with_std=not spec.center_only)
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        scaler_mean = scaler.mean_  # type: ignore[attr-defined]
        scaler_scale = getattr(scaler, "scale_", None)
    else:
        # No scaling: fit + report β in the feature's native units.
        # Persist zero-mean / unit-scale params alongside saved
        # directions so downstream loaders see a consistent .npz schema.
        X_train_s = X_train
        X_val_s = X_val
        scaler_mean = np.zeros(X_train.shape[1], dtype=X_train.dtype)
        scaler_scale = None

    metrics: dict[str, float] = {}
    coef_for_aggregation: np.ndarray | None = None

    if spec.kind == "massmean":
        # Closed-form direction; no fit() call.
        if y.ndim != 1:
            raise ValueError("massmean probe requires 1-D targets.")
        direction = _mass_mean_direction(X_train_s, y_train)
        if np.linalg.norm(direction) < 1e-10:
            metrics["val_pearson"] = float("nan")
            metrics["train_pearson"] = float("nan")
            metrics["val_spearman"] = float("nan")
            metrics["train_spearman"] = float("nan")
        else:
            proj_val = X_val_s @ direction
            proj_train = X_train_s @ direction
            metrics["val_pearson"] = _safe_pearson(y_val, proj_val)
            metrics["train_pearson"] = _safe_pearson(y_train, proj_train)
            metrics["val_spearman"] = _safe_spearman(y_val, proj_val)
            metrics["train_spearman"] = _safe_spearman(y_train, proj_train)
        if directions_dir is not None:
            _save_probe_direction(
                directions_dir, layer, intermediate, "massmean",
                coef=direction, intercept=0.0,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                fold_label=fold_label,
            )
        return metrics, coef_for_aggregation

    estimator, is_classification = _build_estimator(spec)

    if is_classification:
        if y_binned is None:
            raise ValueError(
                f"{spec.kind!r} requires binned 1-D classification targets.",
            )
        y_bin_train = y_binned[train_idx]
        y_bin_val = y_binned[val_idx]
        estimator.fit(X_train_s, y_bin_train)
        y_pred = estimator.predict(X_val_s)
        metrics["val_accuracy"] = float(accuracy_score(y_bin_val, y_pred))
        metrics["val_f1_weighted"] = float(
            f1_score(y_bin_val, y_pred, average="weighted"),
        )
        auc = _binary_auc(estimator, X_val_s, y_bin_val)
        if auc is not None:
            metrics["val_auc"] = auc
    else:
        estimator.fit(X_train_s, y_train)
        y_pred_val = estimator.predict(X_val_s)
        y_pred_train = estimator.predict(X_train_s)

        metrics["val_r2"] = float(r2_score(y_val, y_pred_val))
        metrics["val_mse"] = float(mean_squared_error(y_val, y_pred_val))
        metrics["val_mae"] = float(mean_absolute_error(y_val, y_pred_val))
        metrics["train_r2"] = float(r2_score(y_train, y_pred_train))
        if y.ndim == 1:
            metrics["val_pearson"] = _safe_pearson(y_val, y_pred_val)
            metrics["train_pearson"] = _safe_pearson(y_train, y_pred_train)
            metrics["val_spearman"] = _safe_spearman(y_val, y_pred_val)
            metrics["train_spearman"] = _safe_spearman(y_train, y_pred_train)

        dist = resolve_distance(spec.distance) if spec.distance else None
        if dist is not None and y.ndim == 2 and dist.applies_to(y.shape[1]):
            per_sample = dist.batch(
                y_pred_val.astype(np.float64),
                y_val.astype(np.float64),
            )
            metrics[dist.metric_key] = float(per_sample.mean())

    if spec.kind in _LINEAR_PROBE_KINDS:
        coef = getattr(estimator, "coef_", None)
        if coef is not None:
            # `coef` is in StandardScaler-transformed units, so it is the
            # standardised β. Aggregation only makes sense for binary
            # logreg (1-D coefficient vector); skip multi-class shapes.
            coef_arr = np.asarray(coef)
            if spec.kind == "logreg" and coef_arr.shape[0] == 1:
                coef_for_aggregation = coef_arr.ravel()
            if directions_dir is not None:
                intercept = getattr(estimator, "intercept_", 0.0)
                _save_probe_direction(
                    directions_dir, layer, intermediate, spec.kind,
                    coef=coef_arr, intercept=intercept,
                    scaler_mean=scaler_mean,
                    scaler_scale=scaler_scale,
                    fold_label=fold_label,
                )

    if models_dir is not None:
        # Persist the fitted estimator itself (with the scaler params needed
        # to reproduce its input space) — the non-linear analogue of
        # save_directions, mirroring the MLP checkpoint artifact.
        suffix = f"_{fold_label}" if fold_label else ""
        joblib.dump(
            {
                "estimator": estimator,
                "scaler_mean": np.asarray(scaler_mean),
                "scaler_scale": (
                    np.asarray(scaler_scale) if scaler_scale is not None else None
                ),
            },
            models_dir / f"L{layer}_{intermediate}_{spec.kind}{suffix}.joblib",
        )

    return metrics, coef_for_aggregation


def _binary_auc(
    estimator: BaseEstimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float | None:
    """Compute binary ROC-AUC from `predict_proba` or `decision_function`.

    Returns None when the val set has fewer than 2 unique classes
    (AUC is undefined) or when neither score function is available.
    """
    if len(np.unique(y_val)) < 2:
        return None
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X_val)
        if scores.ndim == 2 and scores.shape[1] >= 2:
            scores = scores[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_val)
        # Multi-class decision_function returns 2-D; binary returns 1-D.
        if np.asarray(scores).ndim != 1:
            return None
    else:
        return None
    try:
        return float(roc_auc_score(y_val, scores))
    except ValueError:
        return None


def _bin_targets(y: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    return np.digitize(y, edges[1:-1])


def _mass_mean_direction(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    median = np.median(y_train)
    high_mask = y_train >= median
    mu_high = X_train[high_mask].mean(axis=0)
    mu_low = X_train[~high_mask].mean(axis=0)
    direction = mu_high - mu_low
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return direction
    return direction / norm


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r that returns NaN for constant arrays."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman that returns NaN silently for constant arrays."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore", message=".*constant.*", category=Warning,
        )
        rho, _ = spearmanr(a, b)
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return float("nan")
    return float(rho)  # type: ignore[arg-type]


def _save_probe_direction(
    directions_dir: Path,
    layer: int,
    intermediate: str,
    kind: str,
    *,
    coef: np.ndarray,
    intercept: float | np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray | None,
    fold_label: str | None = None,
) -> None:
    """Save coef + intercept + scaler params as .npz for downstream analysis."""
    suffix = f"_{fold_label}" if fold_label else ""
    path = directions_dir / f"L{layer}_{intermediate}_{kind}{suffix}.npz"
    arrays = {
        "coef": coef,
        "intercept": np.atleast_1d(intercept),
        "scaler_mean": scaler_mean,
    }
    if scaler_scale is not None:
        arrays["scaler_scale"] = scaler_scale
    np.savez(str(path), **arrays)


def _maybe_write_feature_importance(
    output_dir: Path,
    *,
    spec: SklearnProbeSpec,
    coef_buckets: dict[tuple[int, str], list[np.ndarray]],
    feature_names: list[str] | None,
    multi_fold: bool,
) -> None:
    """Aggregate per-fold standardised |β| into `feature_importance.csv`.

    Only fires when (a) the probe is logreg, (b) `save_directions` was
    requested (so directions exist anyway), and (c) more than one fold
    ran. With a single fold there's nothing to aggregate; the .npz in
    `directions/` already exposes the same coefficients.
    """
    if (
        spec.kind != "logreg"
        or not spec.save_directions
        or not multi_fold
        or not coef_buckets
    ):
        return

    rows: list[dict] = []
    for (layer, intermediate), coefs in sorted(coef_buckets.items()):
        if not coefs:
            continue
        stack = np.stack(coefs, axis=0)  # [n_folds, n_features]
        beta_mean = stack.mean(axis=0)
        beta_std = stack.std(axis=0, ddof=0)
        abs_stack = np.abs(stack)
        abs_mean = abs_stack.mean(axis=0)
        abs_std = abs_stack.std(axis=0, ddof=0)

        n_features = stack.shape[1]
        if feature_names is None or len(feature_names) != n_features:
            names = [f"feature_{i}" for i in range(n_features)]
        else:
            names = list(feature_names)

        for i, name in enumerate(names):
            rows.append(
                {
                    "layer": layer,
                    "intermediate": intermediate,
                    "feature_index": i,
                    "feature_name": name,
                    "beta_mean": float(beta_mean[i]),
                    "beta_std": float(beta_std[i]),
                    "abs_beta_mean": float(abs_mean[i]),
                    "abs_beta_std": float(abs_std[i]),
                    "n_folds": stack.shape[0],
                },
            )

    if not rows:
        return
    pd.DataFrame(rows).to_csv(
        output_dir / "feature_importance.csv", index=False,
    )


_AGGREGATABLE_METRICS = (
    "val_r2", "val_mse", "val_mae",
    "train_r2",
    "val_pearson", "train_pearson",
    "val_spearman", "train_spearman",
    "val_accuracy", "val_f1_weighted", "val_auc",
    "val_lab_distance",
)


def _write_summary(
    output_dir: Path,
    *,
    spec: SklearnProbeSpec,
    rows: list[dict],
    multi_fold: bool,
) -> None:
    summary: dict = {
        "spec": {
            "kind": spec.kind,
            "name": spec.name,
            "alpha": spec.alpha,
            "C": spec.C,
            "kernel": spec.kernel,
            "max_iter": spec.max_iter,
            "logreg_max_iter": spec.logreg_max_iter,
            "classification_bins": spec.classification_bins,
            "train_split": spec.train_split,
            "seed": spec.seed,
            "save_directions": spec.save_directions,
            "save_models": spec.save_models,
            "center_only": spec.center_only,
            "standardise": spec.standardise,
            "n_folds": spec.n_folds,
            "distance": spec.distance,
        },
        "num_probes": len(rows),
        "num_errors": sum(
            1 for r in rows if isinstance(r.get("error"), str)
        ),
    }
    df = pd.DataFrame(rows)

    # A distance plug-in contributes its own metric column (e.g. val_lab_distance);
    # surface it in aggregation + best-metric selection using its own direction.
    dist = resolve_distance(spec.distance) if spec.distance else None
    agg_metrics = _AGGREGATABLE_METRICS
    if dist is not None and dist.metric_key not in agg_metrics:
        agg_metrics = (*agg_metrics, dist.metric_key)

    if multi_fold and not df.empty:
        # Aggregate per (layer, intermediate) across folds. Reported as
        # `aggregated[layer][intermediate][metric] = {mean, std, n}`.
        agg_tree: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
        for (layer, inter), sub in df.groupby(["layer", "intermediate"]):
            inter_tree = agg_tree.setdefault(int(layer), {})
            metric_tree = inter_tree.setdefault(str(inter), {})
            for metric in agg_metrics:
                if metric not in sub.columns:
                    continue
                values = sub[metric].dropna()
                if values.empty:
                    continue
                metric_tree[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n": int(values.shape[0]),
                }
        summary["aggregated"] = agg_tree

    # Direction per metric: r2/spearman/accuracy/auc maximise; a distance metric
    # uses its own `higher_is_better` (LAB CIEDE2000 → minimise).
    best_metric_dirs: dict[str, bool] = {
        "val_r2": True, "val_spearman": True, "val_accuracy": True, "val_auc": True,
    }
    if dist is not None:
        best_metric_dirs[dist.metric_key] = dist.higher_is_better
    for metric, higher_is_better in best_metric_dirs.items():
        if metric not in df.columns:
            continue
        valid = df[df[metric].notna()]
        if valid.empty:
            continue
        idx = (
            valid[metric].idxmax() if higher_is_better
            else valid[metric].idxmin()
        )
        best = df.loc[idx]
        summary[f"best_{metric}"] = {
            "value": float(best[metric]),
            "layer": int(best["layer"]),
            "intermediate": str(best["intermediate"]),
            "fold": str(best.get("fold", "fold_0")),
        }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
