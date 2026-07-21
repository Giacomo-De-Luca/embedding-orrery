"""Embedding-space probing service.

Bridges the `interpret/probing` toolkit to stored collections: X is the
collection's ChromaDB vectors, y is a numeric metadata field from DuckDB.
Training reuses the toolkit trainers (`train_sklearn_probe` for ridge and
massmean, `train_mlp_probes` for the MLP) on an in-memory ActivationDataset;
this module only assembles data, scores all items with the trained probe, and
persists metrics + direction + per-item scores to DuckDB.

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

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.probe import MLPProbeSpec, SklearnProbeSpec
from interpret.probing.probes.mlp_probe import ProbeModel, train_mlp_probes
from interpret.probing.probes.sklearn_probes import train_sklearn_probe
from interpret.probing.utils.enums import TaskType

from ..utils.duckdb_sync import _get_db as _get_duckdb
from ..utils.embedding_loader import load_embeddings_for_ids
from ..utils.resource_paths import PROBING_RESULTS_DIR
from .probing_types import (
    _BINARY_KINDS,
    _CALIBRATED_KINDS,
    _PREDICTIVE_KINDS,
    PROBE_KINDS,
    ProbeConfig,
    binary_target_mapping,
    sanitize_field_key,
    score_field_names,
)
from .progress_emitter import emit_progress

logger = logging.getLogger("orrery." + __name__)

MIN_VALID_SAMPLES = 50

_MLP_SCORING_BATCH = 8192
# RBF SVR training is O(n^2); cap its training pool well below the linear cap.
_SVR_MAX_TRAIN = 10_000

# Score bundle for kinds with no persisted linear direction (mlp, svr).
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
        # Binary classifier; the saved direction is the separating hyperplane
        # normal, and scores are P(class 1) recomputed from it.
        return SklearnProbeSpec(
            kind="logreg",
            C=config.c,
            class_weight=config.class_weight,
            standardise=True,
            save_directions=True,
            seed=config.seed,
            train_split=config.train_split,
        )
    if config.kind == "mlp":
        return MLPProbeSpec(
            hidden_dims=list(config.hidden_dims) if config.hidden_dims else [256],
            epochs=config.epochs,
            patience=config.patience,
            seed=config.seed,
            train_split=config.train_split,
        )
    raise ValueError(f"Unknown probe kind {config.kind!r}. Expected one of {PROBE_KINDS}.")


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


def _score_linear(X: np.ndarray, npz_path: Path, kind: str) -> tuple[np.ndarray, dict]:
    """Score all rows with a saved linear direction; return (scores, bundle).

    Scoring is chunked (like ``_score_mlp``) so the float64 standardization
    never materializes a second full-size copy of X.
    """
    data = np.load(npz_path)
    coef = np.asarray(data["coef"], dtype=np.float64).ravel()
    intercept = float(np.asarray(data["intercept"]).ravel()[0])
    scaler_mean = np.asarray(data["scaler_mean"], dtype=np.float64)
    scaler_scale = (
        np.asarray(data["scaler_scale"], dtype=np.float64)
        if "scaler_scale" in data
        else np.ones_like(scaler_mean)
    )
    scores = np.empty(X.shape[0], dtype=np.float64)
    for start in range(0, X.shape[0], _MLP_SCORING_BATCH):
        chunk = X[start : start + _MLP_SCORING_BATCH].astype(np.float64)
        scores[start : start + _MLP_SCORING_BATCH] = ((chunk - scaler_mean) / scaler_scale) @ coef
    if kind in _PREDICTIVE_KINDS:
        scores += intercept
    bundle = {
        "direction": [float(v) for v in coef],
        "scaler_mean": [float(v) for v in scaler_mean],
        "scaler_scale": [float(v) for v in scaler_scale],
        "intercept": intercept,
    }
    return scores, bundle


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


def _score_mlp(X: np.ndarray, checkpoint_path: Path, spec: MLPProbeSpec) -> np.ndarray:
    """Score all rows with the best MLP checkpoint (batched, no grad)."""
    state = torch.load(checkpoint_path, weights_only=True)
    model = ProbeModel(
        input_dim=X.shape[1],
        output_dim=1,
        hidden_dims=spec.hidden_dims,
        dropout=spec.dropout,
    )
    model.load_state_dict(state)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, X.shape[0], _MLP_SCORING_BATCH):
            batch = torch.from_numpy(X[start : start + _MLP_SCORING_BATCH]).float()
            chunks.append(model(batch).numpy().ravel())
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


def _standardize_chunks(X: np.ndarray, mean: np.ndarray, scale: np.ndarray | None):
    """Yield standardized float64 chunks of X without a full second copy."""
    denom = scale if scale is not None else 1.0
    for start in range(0, X.shape[0], _MLP_SCORING_BATCH):
        chunk = X[start : start + _MLP_SCORING_BATCH].astype(np.float64)
        yield (chunk - mean) / denom


def _score_svr(X: np.ndarray, model_path: Path) -> np.ndarray:
    """Score all rows with a persisted (rbf) SVR estimator, batched."""
    estimator, mean, scale = _load_sklearn_model(model_path)
    chunks = [estimator.predict(c) for c in _standardize_chunks(X, mean, scale)]
    return np.concatenate(chunks).astype(np.float64)


def _score_logreg(X: np.ndarray, npz_path: Path) -> tuple[np.ndarray, dict]:
    """Score all rows as P(class 1) from a saved logreg hyperplane, batched.

    The toolkit saves the standardised coef/intercept; P(class 1) is the
    sigmoid of the linear logit. Returns (scores, direction bundle).
    """
    data = np.load(npz_path)
    coef = np.asarray(data["coef"], dtype=np.float64).ravel()
    intercept = float(np.asarray(data["intercept"]).ravel()[0])
    mean = np.asarray(data["scaler_mean"], dtype=np.float64)
    scale = np.asarray(data["scaler_scale"], dtype=np.float64) if "scaler_scale" in data else None
    scores = np.empty(X.shape[0], dtype=np.float64)
    pos = 0
    # exp overflow at extreme logits saturates cleanly to 0/1; silence the warning.
    with np.errstate(over="ignore"):
        for chunk in _standardize_chunks(X, mean, scale):
            logits = chunk @ coef + intercept
            scores[pos : pos + len(logits)] = 1.0 / (1.0 + np.exp(-logits))
            pos += len(logits)
    bundle = {
        "direction": [float(v) for v in coef],
        "scaler_mean": [float(v) for v in mean],
        "scaler_scale": [float(v) for v in scale] if scale is not None else None,
        "intercept": intercept,
    }
    return scores, bundle


def _binarize_for_logreg(y: np.ndarray, valid_idx: np.ndarray) -> np.ndarray:
    """Validate a binary target and remap the two classes to {0, 1}.

    Returns a float64 copy of y with valid rows in {0.0, 1.0} (larger source
    value -> 1) and NaN elsewhere. Raises if not exactly two classes.
    """
    distinct = np.unique(y[valid_idx])
    if len(distinct) != 2:
        raise ProbeTrainingError(
            "Logistic regression requires a binary target (exactly two distinct "
            f"values); {len(distinct)} found. Use ridge, SVR, or MLP for "
            "continuous or multi-class fields."
        )
    out = np.full_like(y, np.nan)
    out[y == distinct[1]] = 1.0
    out[y == distinct[0]] = 0.0
    return out


def run_probe_core(
    X: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    config: ProbeConfig,
    output_dir: Path,
) -> ProbeCoreOutput:
    """Train one probe and score every row.

    Args:
        X: [N, D] embedding matrix (any float dtype; cast as needed).
        y: [N] float64 targets with NaN where the item has no usable value.
            Invalid rows are excluded from training but still scored.
        ids: Item ids aligned to X/y rows.
        config: Probe configuration; ``kind`` selects the trainer.
        output_dir: Toolkit artifact directory for this probe.

    Raises:
        ProbeTrainingError: fewer than MIN_VALID_SAMPLES valid targets.
    """
    spec = _build_spec(config)
    output_dir = Path(output_dir)
    is_classification = config.kind in _BINARY_KINDS

    valid_idx = np.flatnonzero(np.isfinite(y))
    if len(valid_idx) < MIN_VALID_SAMPLES:
        raise ProbeTrainingError(
            f"Field {config.target_field!r} has {len(valid_idx)} usable numeric "
            f"values; at least {MIN_VALID_SAMPLES} are required."
        )

    if config.kind in _BINARY_KINDS:
        # Validate + remap the two classes to {0, 1} before splitting.
        y = _binarize_for_logreg(y, valid_idx)

    # RBF SVR is O(n^2); cap its training pool tighter than the linear cap.
    train_cap = config.max_train_samples
    if config.kind == "svr":
        train_cap = min(train_cap, _SVR_MAX_TRAIN)

    rng = np.random.default_rng(config.seed)
    if len(valid_idx) > train_cap:
        pool_idx = rng.choice(valid_idx, size=train_cap, replace=False)
        pool_idx.sort()
    else:
        pool_idx = valid_idx

    perm = rng.permutation(len(pool_idx))
    cut = int(len(pool_idx) * config.train_split)
    indices_override = (perm[:cut], perm[cut:])
    n_train, n_val = len(perm[:cut]), len(perm[cut:])

    X_pool = np.ascontiguousarray(X[pool_idx], dtype=np.float32)
    # Classification kinds need integer class labels; the toolkit passes an
    # integer target through verbatim (a float 0/1 would be percentile-binned).
    y_pool = y[pool_idx].astype(np.int64 if is_classification else np.float64)

    if is_classification:
        # The shared split is not stratified; near the sample floor a random
        # 80/20 cut can strand one class entirely, which sklearn reports as an
        # opaque fit failure. Fail with a targeted message instead.
        for part, name in ((indices_override[0], "train"), (indices_override[1], "validation")):
            if len(np.unique(y_pool[part])) < 2:
                raise ProbeTrainingError(
                    f"The random split left only one class in the {name} set — "
                    "the target is too imbalanced or too small for logistic "
                    "regression. Try another seed, more data, or ridge/SVR."
                )
    dataset = ActivationDataset(
        activations={(0, "embedding"): torch.from_numpy(X_pool)},
        sample_ids=[ids[i] for i in pool_idx],
    )

    # The toolkit trainers swallow per-fit failures into an error CSV row, so
    # a failed retrain would otherwise silently reuse the previous run's
    # artifact. Remove it up front and require the trainer to recreate it.
    if isinstance(spec, MLPProbeSpec):
        artifact_path = output_dir / "checkpoints" / "layer_0_embedding.pt"
    elif config.kind == "svr":
        artifact_path = output_dir / "models" / f"L0_embedding_{spec.kind}.joblib"
    else:
        artifact_path = output_dir / "directions" / f"L0_embedding_{spec.kind}.npz"
    artifact_path.unlink(missing_ok=True)

    if isinstance(spec, MLPProbeSpec):
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

    if isinstance(spec, MLPProbeSpec):
        scores = _score_mlp(X.astype(np.float32, copy=False), artifact_path, spec)
        bundle = dict(_EMPTY_BUNDLE)
    elif config.kind == "svr":
        scores = _score_svr(X, artifact_path)
        bundle = dict(_EMPTY_BUNDLE)
    elif config.kind == "logreg":
        scores, bundle = _score_logreg(X, artifact_path)
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
            np.asarray(scores),
            y,
            train_rows=pool_idx[indices_override[0]],
            val_rows=pool_idx[indices_override[1]],
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


def train_probe_for_collection(config: ProbeConfig) -> ProbeRunResult:
    """Full probe run: load y (DuckDB) + X (ChromaDB), train, score, persist.

    Never raises: every failure emits a terminal "failed" progress event and
    comes back as a ProbeRunResult with ``error`` set.
    """
    start_time = time.time()
    job_id = f"{config.collection_name}_probe"
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
        rows = db.get_numeric_metadata_field(dataset_name, config.target_field)
        if not rows:
            raise ProbeTrainingError(f"Dataset {dataset_name!r} has no items to probe.")
        ids = [r[0] for r in rows]
        y = np.array([np.nan if r[1] is None else float(r[1]) for r in rows], dtype=np.float64)
        n_valid = int(np.isfinite(y).sum())
        target_mapping: dict[str, float] | None = None
        if n_valid < MIN_VALID_SAMPLES:
            # Not usable as numbers — try a binary categorical column
            # (e.g. "safe"/"unsafe" -> 0/1). Mapping is deterministic
            # (alphabetical: first value -> 0) and surfaced in the UI.
            text_rows = db.get_text_metadata_field(dataset_name, config.target_field)
            values = [r[1] for r in text_rows]
            target_mapping = binary_target_mapping(values)
            if target_mapping is not None:
                ids = [r[0] for r in text_rows]
                y = np.array(
                    [np.nan if v is None else target_mapping[v] for v in values],
                    dtype=np.float64,
                )
                n_valid = int(np.isfinite(y).sum())
        if n_valid < MIN_VALID_SAMPLES:
            raise ProbeTrainingError(
                f"Field {config.target_field!r} has {n_valid} usable values; at least "
                f"{MIN_VALID_SAMPLES} are required. Targets must be numeric or a binary "
                "categorical column (exactly two distinct values)."
            )

        _progress(1, f"Loading {len(ids)} embedding vectors...")
        X = load_embeddings_for_ids(config.collection_name, ids)
        if X is None:
            raise ProbeTrainingError(
                f"Could not load embeddings for collection {config.collection_name!r}."
            )
        X = X.astype(np.float32)

        _progress(2, f"Training {config.kind} probe on {n_valid} samples...")
        output_dir = (
            Path(PROBING_RESULTS_DIR)
            / "collections"
            / sanitize_field_key(config.collection_name)
            / sanitize_field_key(config.target_field)
            / config.kind
        )
        core = run_probe_core(X, y, ids, config, output_dir)

        _progress(3, "Saving probe and scores...")
        db.upsert_probe(
            config.collection_name,
            config.target_field,
            config.kind,
            config={
                **asdict(config),
                "target_mapping": target_mapping,
                "calibration": core.calibration,
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
        db.insert_probe_scores_bulk(
            config.collection_name, config.target_field, config.kind, scores_df
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
