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
    _PREDICTIVE_KINDS,
    PROBE_KINDS,
    ProbeConfig,
    sanitize_field_key,
    score_field_names,
)
from .progress_emitter import emit_progress

logger = logging.getLogger("orrery." + __name__)

MIN_VALID_SAMPLES = 50

_MLP_SCORING_BATCH = 8192


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


def _build_spec(config: ProbeConfig) -> SklearnProbeSpec | MLPProbeSpec:
    """Map a ProbeConfig to the toolkit probe spec for its kind."""
    if config.kind in ("ridge", "massmean"):
        return SklearnProbeSpec(
            kind=config.kind,
            alpha=config.alpha,
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


def _calibrated_r2(
    scores: np.ndarray,
    y: np.ndarray,
    *,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
) -> dict:
    """R² for an uncalibrated projection via a univariate readout.

    Fits ``y ≈ slope·score + intercept`` by least squares on the train rows,
    then reports R² of that readout on train and validation rows. Returns {}
    when the train projections are (near-)constant.
    """
    s_train, y_train = scores[train_rows], y[train_rows]
    if np.std(s_train) < 1e-12:
        return {}
    slope, intercept = np.polyfit(s_train, y_train, 1)

    def _r2(rows: np.ndarray) -> float | None:
        y_true = y[rows]
        y_pred = slope * scores[rows] + intercept
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        if ss_tot < 1e-12:
            return None
        return _clean_number(1.0 - float(((y_true - y_pred) ** 2).sum()) / ss_tot)

    return {"val_r2": _r2(val_rows), "train_r2": _r2(train_rows)}


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

    valid_idx = np.flatnonzero(np.isfinite(y))
    if len(valid_idx) < MIN_VALID_SAMPLES:
        raise ProbeTrainingError(
            f"Field {config.target_field!r} has {len(valid_idx)} usable numeric "
            f"values; at least {MIN_VALID_SAMPLES} are required."
        )

    rng = np.random.default_rng(config.seed)
    if len(valid_idx) > config.max_train_samples:
        pool_idx = rng.choice(valid_idx, size=config.max_train_samples, replace=False)
        pool_idx.sort()
    else:
        pool_idx = valid_idx

    perm = rng.permutation(len(pool_idx))
    cut = int(len(pool_idx) * config.train_split)
    indices_override = (perm[:cut], perm[cut:])
    n_train, n_val = len(perm[:cut]), len(perm[cut:])

    X_pool = np.ascontiguousarray(X[pool_idx], dtype=np.float32)
    y_pool = y[pool_idx].astype(np.float64)
    dataset = ActivationDataset(
        activations={(0, "embedding"): torch.from_numpy(X_pool)},
        sample_ids=[ids[i] for i in pool_idx],
    )

    # The toolkit trainers swallow per-fit failures into an error CSV row, so
    # a failed retrain would otherwise silently reuse the previous run's
    # artifact. Remove it up front and require the trainer to recreate it.
    if isinstance(spec, MLPProbeSpec):
        artifact_path = output_dir / "checkpoints" / "layer_0_embedding.pt"
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
        bundle = {
            "direction": None,
            "scaler_mean": None,
            "scaler_scale": None,
            "intercept": None,
        }
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
    if config.kind == "massmean":
        # Massmean scores are an uncalibrated projection, so the toolkit only
        # reports correlations. A univariate calibration (slope/intercept fit
        # on the train split) gives it a comparable validation R².
        metrics.update(
            _calibrated_r2(
                np.asarray(scores),
                y,
                train_rows=pool_idx[indices_override[0]],
                val_rows=pool_idx[indices_override[1]],
            )
        )

    _, residual_field = score_field_names(config.target_field, config.kind)
    residuals: list[float | None] | None = None
    if residual_field is not None:
        residuals = [
            _clean_number(scores[i] - y[i]) if np.isfinite(y[i]) else None for i in range(len(ids))
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
        if n_valid < MIN_VALID_SAMPLES:
            raise ProbeTrainingError(
                f"Field {config.target_field!r} has {n_valid} usable numeric "
                f"values; at least {MIN_VALID_SAMPLES} are required."
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
            config=asdict(config),
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
