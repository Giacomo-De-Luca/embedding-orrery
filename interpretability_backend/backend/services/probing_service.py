"""Embedding-space probing service.

Bridges the `interpret/probing` toolkit to stored collections: X is the
collection's ChromaDB vectors, y is a metadata field from DuckDB — numeric
for the regression kinds, categorical labels for the classifier kinds
(logreg / svc / mlp_clf; binary or multiclass, see ``probe_targets.py``).
Training reuses the toolkit trainers (`train_sklearn_probe` for the sklearn
kinds, `train_mlp_probes` for both MLPs) on an in-memory ActivationDataset;
this module only assembles data, scores all items with the trained probe, and
persists metrics + direction + per-item scores to DuckDB.

Classifier probes score every item with class probabilities: binary targets
persist P(class 1) as the score; multiclass targets persist the confidence
(max probability) as the score plus the predicted class index and the full
probability vector (fetched per class, lazily, by the UI).

Toolkit artifacts (probe_results.csv, summary.json, directions/, checkpoints/)
are written under ``PROBING_RESULTS_DIR/collections/<collection>/<field>/<kind>/``
so the offline tooling (consolidate.py, visualisations) keeps working on
platform-trained probes.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, softmax

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.probe import MLPProbeSpec, SklearnProbeSpec
from interpret.probing.probes.mlp_probe import ProbeModel, train_mlp_probes
from interpret.probing.probes.sklearn_probes import train_sklearn_probe
from interpret.probing.utils.enums import TaskType

from ..utils.classification_metrics import ClassificationMetrics
from ..utils.duckdb_sync import _get_db as _get_duckdb
from ..utils.embedding_loader import load_embeddings_for_ids
from ..utils.probe_split import ProbeSplitter
from ..utils.resource_paths import PROBING_RESULTS_DIR
from .job_ids import JobIds
from .probe_targets import CategoricalLabel, CategoricalTarget, ProbeTargetError
from .probing_types import (
    _CALIBRATED_KINDS,
    _PREDICTIVE_KINDS,
    CLASSIFICATION_KINDS,
    MIN_CLASS_COUNT_FLOOR,
    PROBE_KINDS,
    ProbeConfig,
    ProbeTask,
    sanitize_field_key,
    score_field_names,
)
from .progress_emitter import emit_progress

logger = logging.getLogger("orrery." + __name__)

MIN_VALID_SAMPLES = 50

_MLP_SCORING_BATCH = 8192
# RBF SVR training is O(n^2); cap its training pool well below the linear cap.
_SVR_MAX_TRAIN = 10_000
# RBF SVC is O(n^2) too, and Platt scaling refits it ~5 more times (internal
# CV) — keep it well inside the UI's 10-minute mutation timeout.
_SVC_MAX_TRAIN = 5_000

# Score bundle for kinds with no persisted linear direction (mlp, svr, svc,
# multinomial logreg — whose [C, D] coefficients stay in the .npz artifact).
_EMPTY_BUNDLE = {
    "direction": None,
    "scaler_mean": None,
    "scaler_scale": None,
    "intercept": None,
}


class ProbeTrainingError(Exception):
    """Raised by the core for invalid probe inputs (caught by the orchestrator)."""


@dataclass
class ProbeCoreOutput:
    """Result of training + scoring, before persistence."""

    metrics: dict
    scores: list[float]
    residuals: list[float | None] | None
    direction: list[float] | None
    scaler_mean: list[float] | None
    scaler_scale: list[float] | None
    intercept: float | None
    n_train: int
    n_val: int
    # Mass-mean family only: the univariate readout {"slope", "intercept"}
    # mapping projection scores to target units; None for other kinds.
    calibration: dict | None = None
    task: str = ProbeTask.REGRESSION
    # Classifier kinds: the categorical target the probe was trained on.
    target: CategoricalTarget | None = None
    # Multiclass only: predicted class index [N] and probabilities [N, C].
    predicted: np.ndarray | None = None
    probs: np.ndarray | None = None


@dataclass
class ProbeRunResult:
    """Outcome of a full probe run (mirrors the GraphQL result shape)."""

    collection_name: str
    target_field: str
    kind: str
    metrics: dict | None = None
    n_train: int = 0
    n_val: int = 0
    n_scored: int = 0
    score_field: str | None = None
    residual_field: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    # For binary categorical targets: the applied value->0/1 mapping
    # (e.g. {"safe": 0.0, "unsafe": 1.0}); None for numeric targets.
    target_mapping: dict | None = None


def _build_spec(config: ProbeConfig) -> SklearnProbeSpec | MLPProbeSpec:
    """Map a ProbeConfig to the toolkit probe spec for its kind."""
    if config.kind in ("ridge", "lasso", "massmean", "massmean_cov"):
        # alpha applies to ridge/lasso and is ignored by the closed-form
        # mass-mean kinds; all four persist a linear direction.
        return SklearnProbeSpec(
            kind=config.kind,
            alpha=config.alpha,
            standardise=True,
            save_directions=True,
            seed=config.seed,
            train_split=config.train_split,
        )
    if config.kind == "svr":
        # RBF SVR is nonlinear: no coef_/direction, so persist the fitted
        # estimator (with scaler params) to score the whole collection.
        return SklearnProbeSpec(
            kind="svr",
            C=config.c,
            kernel=config.kernel,
            standardise=True,
            save_models=True,
            seed=config.seed,
            train_split=config.train_split,
        )
    if config.kind == "logreg":
        # Binary or multinomial classifier; the saved direction file holds
        # the hyperplane normal(s), and scores are recomputed from it.
        return SklearnProbeSpec(
            kind="logreg",
            C=config.c,
            class_weight=config.class_weight,
            standardise=True,
            save_directions=True,
            seed=config.seed,
            train_split=config.train_split,
        )
    if config.kind == "svc":
        # RBF classifier: no coef_, so persist the estimator; seeded Platt
        # scaling gives the class probabilities every item is scored with.
        return SklearnProbeSpec(
            kind="svc",
            C=config.c,
            kernel=config.kernel,
            class_weight=config.class_weight,
            standardise=True,
            save_models=True,
            probability=True,
            seed=config.seed,
            train_split=config.train_split,
        )
    if config.kind in ("mlp", "mlp_clf"):
        return MLPProbeSpec(
            hidden_dims=list(config.hidden_dims) if config.hidden_dims else [256],
            epochs=config.epochs,
            patience=config.patience,
            dev_split=config.dev_split,
            activation=config.activation,
            class_weight=config.class_weight if config.kind == "mlp_clf" else None,
            seed=config.seed,
            train_split=config.train_split,
        )
    raise ValueError(f"Unknown probe kind {config.kind!r}. Expected one of {PROBE_KINDS}.")


def _artifact_path(spec: SklearnProbeSpec | MLPProbeSpec, output_dir: Path) -> Path:
    """The trained-probe file the toolkit writes for this spec (single fold)."""
    if isinstance(spec, MLPProbeSpec):
        return output_dir / "checkpoints" / "layer_0_embedding.pt"
    if spec.save_models:
        return output_dir / "models" / f"L0_embedding_{spec.kind}.joblib"
    return output_dir / "directions" / f"L0_embedding_{spec.kind}.npz"


def _clean_number(value) -> float | None:
    """NaN/inf -> None so metrics survive json.dumps and GraphQL."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _parse_metrics(output_dir: Path) -> dict:
    """Read the single-row probe_results.csv into a {metric: value|None} dict.

    Only ``val_*``/``train_*`` columns are kept: the sklearn writer drops
    constant layer/intermediate columns for single-run probes while the MLP
    writer keeps them, so column positions cannot be relied on.
    """
    csv_path = output_dir / "probe_results.csv"
    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    return {
        col: _clean_number(row[col]) for col in df.columns if col.startswith(("val_", "train_"))
    }


def _standardize_chunks(X: np.ndarray, mean: np.ndarray, scale: np.ndarray | None):
    """Yield standardized float64 chunks of X without a full second copy."""
    denom = scale if scale is not None else 1.0
    for start in range(0, X.shape[0], _MLP_SCORING_BATCH):
        chunk = X[start : start + _MLP_SCORING_BATCH].astype(np.float64)
        yield (chunk - mean) / denom


class LinearProbeArtifact:
    """A toolkit direction file (.npz): coef [K, D], intercept [K], scaler params.

    K is 1 for the regression kinds, mass-mean and binary logreg, and C for
    multinomial logreg.
    """

    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.coef = np.atleast_2d(np.asarray(data["coef"], dtype=np.float64))
        self.intercept = np.asarray(data["intercept"], dtype=np.float64).ravel()
        self.scaler_mean = np.asarray(data["scaler_mean"], dtype=np.float64)
        self.scaler_scale = (
            np.asarray(data["scaler_scale"], dtype=np.float64) if "scaler_scale" in data else None
        )

    def project(self, X: np.ndarray) -> np.ndarray:
        """Standardised projections [N, K] (no intercept), scored in chunks."""
        out = np.empty((X.shape[0], self.coef.shape[0]), dtype=np.float64)
        pos = 0
        for chunk in _standardize_chunks(X, self.scaler_mean, self.scaler_scale):
            out[pos : pos + len(chunk)] = chunk @ self.coef.T
            pos += len(chunk)
        return out

    def bundle(self) -> dict:
        """The single persisted direction + scaler params (K == 1 only)."""
        return {
            "direction": [float(v) for v in self.coef[0]],
            "scaler_mean": [float(v) for v in self.scaler_mean],
            "scaler_scale": (
                [float(v) for v in self.scaler_scale] if self.scaler_scale is not None else None
            ),
            "intercept": float(self.intercept[0]),
        }


def _score_linear(X: np.ndarray, npz_path: Path, kind: str) -> tuple[np.ndarray, dict]:
    """Score all rows with a saved linear direction; return (scores, bundle)."""
    artifact = LinearProbeArtifact(npz_path)
    scores = artifact.project(X)[:, 0]
    if kind in _PREDICTIVE_KINDS:
        scores = scores + artifact.intercept[0]
    return scores, artifact.bundle()


def _logreg_probs(X: np.ndarray, npz_path: Path) -> tuple[np.ndarray, dict]:
    """Class probabilities [N, C] from a saved logreg hyperplane file.

    Binary files hold one hyperplane (sigmoid -> [1 - p, p]) whose normal is
    persisted as the probe direction; multinomial files hold C rows (softmax)
    that don't fit the 1-D direction column, so no direction is persisted.
    """
    artifact = LinearProbeArtifact(npz_path)
    logits = artifact.project(X) + artifact.intercept
    if logits.shape[1] == 1:
        p1 = expit(logits[:, 0])
        return np.column_stack([1.0 - p1, p1]), artifact.bundle()
    return softmax(logits, axis=1), dict(_EMPTY_BUNDLE)


def _calibrated_readout(
    scores: np.ndarray,
    y: np.ndarray,
    *,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
) -> tuple[dict, dict | None]:
    """Univariate readout for an uncalibrated projection.

    Fits ``y ≈ slope·score + intercept`` by least squares on the train rows,
    then reports R² of that readout on train and validation rows. Returns
    ``(metrics, {"slope", "intercept"})``, or ``({}, None)`` when the train
    projections are (near-)constant.
    """
    s_train, y_train = scores[train_rows], y[train_rows]
    if np.std(s_train) < 1e-12:
        return {}, None
    slope, intercept = np.polyfit(s_train, y_train, 1)

    def _r2(rows: np.ndarray) -> float | None:
        y_true = y[rows]
        y_pred = slope * scores[rows] + intercept
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        if ss_tot < 1e-12:
            return None
        return _clean_number(1.0 - float(((y_true - y_pred) ** 2).sum()) / ss_tot)

    metrics = {"val_r2": _r2(val_rows), "train_r2": _r2(train_rows)}
    return metrics, {"slope": float(slope), "intercept": float(intercept)}


def _score_mlp(
    X: np.ndarray, checkpoint_path: Path, spec: MLPProbeSpec, output_dim: int = 1
) -> np.ndarray:
    """Raw MLP outputs [N, output_dim] from the best checkpoint (batched, no grad)."""
    state = torch.load(checkpoint_path, weights_only=True)
    model = ProbeModel(
        input_dim=X.shape[1],
        output_dim=output_dim,
        hidden_dims=spec.hidden_dims,
        dropout=spec.dropout,
        # Must match training: activation layers are parameterless, so a
        # mismatched rebuild loads the checkpoint silently but scores wrong.
        activation=spec.activation,
    )
    model.load_state_dict(state)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, X.shape[0], _MLP_SCORING_BATCH):
            batch = torch.from_numpy(X[start : start + _MLP_SCORING_BATCH]).float()
            chunks.append(model(batch).numpy())
    return np.concatenate(chunks).astype(np.float64)


def _load_sklearn_model(model_path: Path) -> tuple[object, np.ndarray, np.ndarray | None]:
    """Load a persisted sklearn estimator + its scaler params."""
    bundle = joblib.load(model_path)
    scale = bundle["scaler_scale"]
    return (
        bundle["estimator"],
        np.asarray(bundle["scaler_mean"], dtype=np.float64),
        np.asarray(scale, dtype=np.float64) if scale is not None else None,
    )


def _score_sklearn_model(
    X: np.ndarray, model_path: Path, n_classes: int | None = None
) -> np.ndarray:
    """Score all rows with a persisted estimator, batched.

    Regression (``n_classes`` None): predictions [N]. Classification:
    ``predict_proba`` [N, n_classes], columns placed via ``classes_``.
    """
    estimator, mean, scale = _load_sklearn_model(model_path)
    if n_classes is None:
        chunks = [estimator.predict(c) for c in _standardize_chunks(X, mean, scale)]
        return np.concatenate(chunks).astype(np.float64)
    columns = np.asarray(estimator.classes_, dtype=np.int64)
    probs = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    pos = 0
    for chunk in _standardize_chunks(X, mean, scale):
        probs[pos : pos + len(chunk), columns] = estimator.predict_proba(chunk)
        pos += len(chunk)
    return probs


def _classification_target(
    y: np.ndarray, config: ProbeConfig, target: CategoricalTarget | None
) -> CategoricalTarget:
    """The categorical target for a classifier kind (derived from numeric y if absent)."""
    if target is not None:
        return target
    try:
        return CategoricalTarget.from_numeric(y, config.min_class_fraction)
    except ProbeTargetError as exc:
        raise ProbeTrainingError(str(exc)) from exc


def run_probe_core(
    X: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    config: ProbeConfig,
    output_dir: Path,
    target: CategoricalTarget | None = None,
) -> ProbeCoreOutput:
    """Train one probe and score every row.

    Args:
        X: [N, D] embedding matrix (any float dtype; cast as needed).
        y: [N] float64 targets with NaN where the item has no usable value.
            Invalid rows are excluded from training but still scored.
        ids: Item ids aligned to X/y rows.
        config: Probe configuration; ``kind`` selects the trainer.
        output_dir: Toolkit artifact directory for this probe.
        target: Classifier kinds only — the categorical target (built by the
            orchestrator from the field's labels). When None it is derived
            from numeric ``y``; either way ``y`` becomes its class codes.

    Raises:
        ProbeTrainingError: fewer than MIN_VALID_SAMPLES valid targets, or a
            target the kind cannot train on.
    """
    spec = _build_spec(config)
    output_dir = Path(output_dir)
    is_classification = config.kind in CLASSIFICATION_KINDS
    if is_classification:
        target = _classification_target(y, config, target)
        y = target.codes
        task = target.task
    else:
        target = None
        task = ProbeTask.REGRESSION
    multiclass = task == ProbeTask.MULTICLASS

    valid_idx = np.flatnonzero(np.isfinite(y))
    if len(valid_idx) < MIN_VALID_SAMPLES:
        raise ProbeTrainingError(
            f"Field {config.target_field!r} has {len(valid_idx)} usable "
            f"values; at least {MIN_VALID_SAMPLES} are required."
        )

    # RBF SVR/SVC are O(n^2); cap their training pool tighter than the linear cap.
    train_cap = config.max_train_samples
    if config.kind == "svr":
        train_cap = min(train_cap, _SVR_MAX_TRAIN)
    elif config.kind == "svc":
        train_cap = min(train_cap, _SVC_MAX_TRAIN)

    # Multiclass splits are stratified (every class on both sides, rare
    # classes survive the cap); everything else keeps the original random
    # split, so kinds on the same field share it and metrics compare.
    splitter = ProbeSplitter(config.seed, config.train_split, min_per_class=MIN_CLASS_COUNT_FLOOR)
    pool_idx, train_rel, val_rel = splitter.split(
        valid_idx, train_cap, labels=y[valid_idx] if multiclass else None
    )
    indices_override = (train_rel, val_rel)
    n_train, n_val = len(train_rel), len(val_rel)

    X_pool = np.ascontiguousarray(X[pool_idx], dtype=np.float32)
    # Classification kinds need integer class labels; the toolkit passes an
    # integer target through verbatim (a float 0/1 would be percentile-binned).
    y_pool = y[pool_idx].astype(np.int64 if is_classification else np.float64)

    if is_classification and not multiclass:
        # The shared binary split is not stratified; near the sample floor a
        # random 80/20 cut can strand one class entirely, which sklearn
        # reports as an opaque fit failure. Fail with a targeted message.
        for part, name in ((train_rel, "train"), (val_rel, "validation")):
            if len(np.unique(y_pool[part])) < 2:
                raise ProbeTrainingError(
                    f"The random split left only one class in the {name} set — "
                    "the target is too imbalanced or too small for a classifier. "
                    "Try another seed, more data, or ridge/SVR."
                )
    dataset = ActivationDataset(
        activations={(0, "embedding"): torch.from_numpy(X_pool)},
        sample_ids=[ids[i] for i in pool_idx],
    )

    # The toolkit trainers swallow per-fit failures into an error CSV row, so
    # a failed retrain would otherwise silently reuse the previous run's
    # artifact. Remove it up front and require the trainer to recreate it.
    artifact_path = _artifact_path(spec, output_dir)
    artifact_path.unlink(missing_ok=True)

    if isinstance(spec, MLPProbeSpec) and is_classification:
        train_mlp_probes(
            dataset,
            spec,
            torch.from_numpy(y_pool).long(),
            output_dir,
            task_type=TaskType.CLASSIFICATION,
            num_classes=len(target.classes),
            target_columns=[config.target_field],
            indices_override=indices_override,
        )
    elif isinstance(spec, MLPProbeSpec):
        train_mlp_probes(
            dataset,
            spec,
            torch.tensor(y_pool, dtype=torch.float32).reshape(-1, 1),
            output_dir,
            task_type=TaskType.REGRESSION,
            target_columns=[config.target_field],
            indices_override=indices_override,
        )
    else:
        train_sklearn_probe(
            dataset,
            spec,
            y_pool,
            output_dir,
            indices_override=indices_override,
        )
    if not artifact_path.exists():
        raise ProbeTrainingError(
            f"The {config.kind} fit failed and produced no probe artifact — "
            "see the server logs for the underlying error."
        )

    train_rows, val_rows = pool_idx[train_rel], pool_idx[val_rel]
    if is_classification:
        return _classifier_output(
            X, y, spec, artifact_path, target, train_rows, val_rows, n_train, n_val
        )

    if isinstance(spec, MLPProbeSpec):
        scores = _score_mlp(X.astype(np.float32, copy=False), artifact_path, spec).ravel()
        bundle = dict(_EMPTY_BUNDLE)
    elif config.kind == "svr":
        scores = _score_sklearn_model(X, artifact_path)
        bundle = dict(_EMPTY_BUNDLE)
    else:
        scores, bundle = _score_linear(X, artifact_path, config.kind)

    if not np.all(np.isfinite(scores)):
        raise ProbeTrainingError(
            "Probe produced non-finite scores — the target may be constant or otherwise degenerate."
        )

    metrics = _parse_metrics(output_dir)
    if not metrics:
        # Every successful fit emits val_* columns; an empty dict means the
        # fit errored (error-only CSV row) or was degenerate (all-NaN metrics
        # dropped by the CSV writer, e.g. a ~zero mass-mean direction).
        raise ProbeTrainingError(
            f"The {config.kind} fit produced no usable metrics — the target "
            "may be degenerate or the fit failed (see server logs)."
        )
    calibration: dict | None = None
    if config.kind in _CALIBRATED_KINDS:
        # Mass-mean scores are an uncalibrated projection, so the toolkit only
        # reports correlations. A univariate calibration (slope/intercept fit
        # on the train split) gives it a comparable validation R² and
        # predictions in target units for residuals.
        cal_metrics, calibration = _calibrated_readout(
            np.asarray(scores), y, train_rows=train_rows, val_rows=val_rows
        )
        metrics.update(cal_metrics)

    _, residual_field = score_field_names(config.target_field, config.kind)
    residuals: list[float | None] | None = None
    if residual_field is not None:
        if config.kind in _CALIBRATED_KINDS:
            preds = (
                calibration["slope"] * np.asarray(scores) + calibration["intercept"]
                if calibration is not None
                else None
            )
        else:
            preds = np.asarray(scores)
        if preds is not None:
            residuals = [
                _clean_number(preds[i] - y[i]) if np.isfinite(y[i]) else None
                for i in range(len(ids))
            ]

    return ProbeCoreOutput(
        metrics=metrics,
        scores=[float(s) for s in scores],
        residuals=residuals,
        direction=bundle["direction"],
        scaler_mean=bundle["scaler_mean"],
        scaler_scale=bundle["scaler_scale"],
        intercept=bundle["intercept"],
        n_train=n_train,
        n_val=n_val,
        calibration=calibration,
    )


def _classifier_output(
    X: np.ndarray,
    y: np.ndarray,
    spec: SklearnProbeSpec | MLPProbeSpec,
    artifact_path: Path,
    target: CategoricalTarget,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    n_train: int,
    n_val: int,
) -> ProbeCoreOutput:
    """Score every item with class probabilities and compute classifier metrics.

    The predicted class is always argmax(probabilities) — for every kind —
    and the metrics are computed from the same probabilities, so what the UI
    shows and what the badge reports agree.
    """
    n_classes = len(target.classes)
    if isinstance(spec, MLPProbeSpec):
        logits = _score_mlp(X.astype(np.float32, copy=False), artifact_path, spec, n_classes)
        probs, bundle = softmax(logits, axis=1), dict(_EMPTY_BUNDLE)
    elif spec.kind == "svc":
        probs, bundle = _score_sklearn_model(X, artifact_path, n_classes), dict(_EMPTY_BUNDLE)
    else:
        probs, bundle = _logreg_probs(X, artifact_path)

    if probs.shape[1] != n_classes:
        raise ProbeTrainingError(
            f"Probe produced {probs.shape[1]} class probabilities for {n_classes} classes — "
            "a class was missing from training."
        )
    if not np.all(np.isfinite(probs)):
        raise ProbeTrainingError(
            "Probe produced non-finite probabilities — the target may be degenerate."
        )
    # Unlabelled rows (NaN codes) are never in train/val; -1 keeps the cast clean.
    labels = np.where(np.isfinite(y), y, -1.0)
    metrics = ClassificationMetrics(probs, labels, train_rows, val_rows).compute()

    multiclass = target.task == ProbeTask.MULTICLASS
    scores = probs.max(axis=1) if multiclass else probs[:, 1]
    return ProbeCoreOutput(
        metrics=metrics,
        scores=[float(s) for s in scores],
        residuals=None,
        direction=bundle["direction"],
        scaler_mean=bundle["scaler_mean"],
        scaler_scale=bundle["scaler_scale"],
        intercept=bundle["intercept"],
        n_train=n_train,
        n_val=n_val,
        task=target.task,
        target=target,
        predicted=probs.argmax(axis=1) if multiclass else None,
        probs=probs if multiclass else None,
    )


def _load_categorical_target(
    db, dataset_name: str, config: ProbeConfig
) -> tuple[list[str], CategoricalTarget]:
    """Item ids + the categorical target built from the field's canonical labels.

    Raises ProbeTargetError when the field can't serve as a categorical target.
    """
    rows = db.get_categorical_metadata_field(dataset_name, config.target_field)
    if not rows:
        raise ProbeTrainingError(f"Dataset {dataset_name!r} has no items to probe.")
    labels = [CategoricalLabel.canonical(text, json_type) for _, text, json_type in rows]
    target = CategoricalTarget.from_labels(labels, config.min_class_fraction)
    return [r[0] for r in rows], target


def _load_target(
    db, dataset_name: str, config: ProbeConfig
) -> tuple[list[str], np.ndarray, CategoricalTarget | None]:
    """Item ids, float targets (NaN = unusable) and the categorical target if any.

    Classifier kinds always read the field as categorical labels. Regression
    kinds read numbers, falling back to a binary categorical column (e.g.
    "safe"/"unsafe" -> 0/1, alphabetical) when the field isn't numeric.
    """
    if config.kind in CLASSIFICATION_KINDS:
        ids, target = _load_categorical_target(db, dataset_name, config)
        return ids, target.codes, target

    rows = db.get_numeric_metadata_field(dataset_name, config.target_field)
    if not rows:
        raise ProbeTrainingError(f"Dataset {dataset_name!r} has no items to probe.")
    ids = [r[0] for r in rows]
    y = np.array([np.nan if r[1] is None else float(r[1]) for r in rows], dtype=np.float64)
    n_valid = int(np.isfinite(y).sum())
    if n_valid >= MIN_VALID_SAMPLES:
        return ids, y, None

    # A numeric field that is merely too small must report that, not be
    # reinterpreted as categorical (whose classifier would hit the same floor).
    target = None
    if n_valid == 0:
        try:
            ids, target = _load_categorical_target(db, dataset_name, config)
        except ProbeTargetError:
            target = None
    if target is not None and (target.task != ProbeTask.BINARY or target.excluded):
        n_classes = len(target.classes) + len(target.excluded)
        raise ProbeTrainingError(
            f"Field {config.target_field!r} is categorical with {n_classes} classes; "
            "regression probes need a numeric or two-class target. Use a classifier: "
            "Logistic, SVC, or MLP (classifier)."
        )
    if target is None:
        raise ProbeTrainingError(
            f"Field {config.target_field!r} has {n_valid} usable values; at least "
            f"{MIN_VALID_SAMPLES} are required. Targets must be numeric or a binary "
            "categorical column (exactly two distinct values)."
        )
    return ids, target.codes, target


def train_probe_for_collection(config: ProbeConfig) -> ProbeRunResult:
    """Full probe run: load y (DuckDB) + X (ChromaDB), train, score, persist.

    Never raises: every failure emits a terminal "failed" progress event and
    comes back as a ProbeRunResult with ``error`` set.
    """
    start_time = time.time()
    job_id = JobIds.probe(config.collection_name)
    score_field, residual_field = score_field_names(config.target_field, config.kind)
    result = ProbeRunResult(
        collection_name=config.collection_name,
        target_field=config.target_field,
        kind=config.kind,
        score_field=score_field,
        residual_field=residual_field,
    )

    def _progress(stage: int, message: str, status: str = "running"):
        emit_progress(
            job_id=job_id,
            status=status,
            items_processed=stage,
            total_items=4,
            current_batch=stage,
            total_batches=4,
            message=message,
        )

    try:
        _build_spec(config)  # validate kind before any I/O

        db = _get_duckdb()
        if db is None:
            raise ProbeTrainingError("DuckDB is not available.")
        dataset_name = db.get_dataset_name_for_collection(config.collection_name)
        if not dataset_name:
            raise ProbeTrainingError(f"No dataset found for collection {config.collection_name!r}.")

        _progress(0, f"Loading targets for '{config.target_field}'...")
        ids, y, target = _load_target(db, dataset_name, config)
        n_valid = int(np.isfinite(y).sum())

        _progress(1, f"Loading {len(ids)} embedding vectors...")
        X = load_embeddings_for_ids(config.collection_name, ids)
        if X is None:
            raise ProbeTrainingError(
                f"Could not load embeddings for collection {config.collection_name!r}."
            )
        X = X.astype(np.float32)

        message = f"Training {config.kind} probe on {n_valid} samples..."
        if config.kind == "svc":
            message += " (SVC calibrates class probabilities — this can take a few minutes)"
        _progress(2, message)
        output_dir = (
            Path(PROBING_RESULTS_DIR)
            / "collections"
            / sanitize_field_key(config.collection_name)
            / sanitize_field_key(config.target_field)
            / config.kind
        )
        core = run_probe_core(X, y, ids, config, output_dir, target=target)
        target_mapping = target.target_mapping if target is not None else None

        _progress(3, "Saving probe and scores...")
        db.upsert_probe(
            config.collection_name,
            config.target_field,
            config.kind,
            config={
                **asdict(config),
                "task": core.task,
                "target_mapping": target_mapping,
                "calibration": core.calibration,
                "classes": core.target.classes if core.target else None,
                "excluded_classes": (core.target.excluded or None) if core.target else None,
                "class_threshold": core.target.threshold if core.target else None,
            },
            metrics=core.metrics,
            direction=core.direction,
            scaler_mean=core.scaler_mean,
            scaler_scale=core.scaler_scale,
            intercept=core.intercept,
            artifact_path=str(output_dir),
            n_train=core.n_train,
            n_val=core.n_val,
        )
        scores_df = pd.DataFrame(
            {
                "item_id": ids,
                "score": core.scores,
                "residual": core.residuals if core.residuals is not None else [None] * len(ids),
            }
        )
        if core.predicted is not None:
            scores_df["predicted"] = core.predicted
        db.insert_probe_scores_bulk(
            config.collection_name, config.target_field, config.kind, scores_df, probs=core.probs
        )

        result.metrics = core.metrics
        result.n_train = core.n_train
        result.n_val = core.n_val
        result.n_scored = len(ids)
        result.target_mapping = target_mapping
        result.duration_seconds = time.time() - start_time
        _progress(4, "Probe complete.", status="completed")
        logger.info(
            "Trained %s probe on %s/%s: %s",
            config.kind,
            config.collection_name,
            config.target_field,
            core.metrics,
        )
        return result

    except Exception as e:
        logger.exception(
            "Probe training failed for %s/%s (%s)",
            config.collection_name,
            config.target_field,
            config.kind,
        )
        emit_progress(
            job_id=job_id,
            status="failed",
            items_processed=0,
            total_items=4,
            current_batch=0,
            total_batches=4,
            error=str(e),
        )
        result.error = str(e)
        result.duration_seconds = time.time() - start_time
        return result
