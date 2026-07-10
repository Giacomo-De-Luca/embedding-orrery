"""Bridge between the interpret/ SAE pipeline and the backend GraphQL API.

Runs the standalone pipeline (download → merge → extract decoder vectors),
ingests features + activations into DuckDB (sae_features / sae_activations
tables), and returns output file paths. Does NOT store vectors in ChromaDB
(no projections/topics) — the user can import the parquet via the Local
Files flow for visualization.

Two SAE families are supported, dispatched on ``family``:

- ``"gemma"`` — Gemma-scope via Neuronpedia S3 (features + labels +
  optional activation examples), then decoder-vector extraction.
- ``"qwen"`` — Qwen-scope TopK SAEs. Not on Neuronpedia, so there is no
  download stage: decoder vectors are extracted straight from the HF
  weights with empty labels/densities (backfilled later by the autointerp
  pass), and activation examples are unavailable.
"""

import logging
import time
from collections.abc import Callable

from interpret.sae.paths import vectors_parquet_path
from interpret.sae.pipeline.prepare_sae_data import (
    SAEPipelineConfig,
    SAEPipelineRunner,
)
from interpret.sae.sae_config import (
    HOOK_TYPE_FROM_STR,
    MODEL_SIZE_TO_D_IN,
    MODEL_SIZE_TO_LAYERS,
    QWEN_SCOPE_MODELS,
    GemmaScopeSAEConfig,
    QwenScopeSAEConfig,
)
from interpret.sae.source_ids import neuronpedia_source_id, qwen_source_id

from ..services.progress_emitter import emit_progress

logger = logging.getLogger("orrery." + __name__)


def _build_stage_offsets(weights: dict[str, int]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for stage, weight in weights.items():
        offsets[stage] = offset
        offset += weight
    return offsets


# Total progress split across stages (100 units).
# Download sub-stages are reported as "download:download_features" etc.
_STAGE_WEIGHTS = {
    "download:download_features": 15,
    "download:download_explanations": 15,
    "download:download_activations": 15,
    "merge_activations": 5,
    "extract_vectors": 25,
    "ingest_features": 15,
    "ingest_activations": 10,
}
_STAGE_OFFSETS = _build_stage_offsets(_STAGE_WEIGHTS)

# Qwen-scope has no download/merge stages — extraction dominates.
_QWEN_STAGE_WEIGHTS = {
    "extract_vectors": 85,
    "ingest_features": 15,
}
_QWEN_STAGE_OFFSETS = _build_stage_offsets(_QWEN_STAGE_WEIGHTS)


def _failed(error: str, model_id: str = "", sae_id: str = "") -> dict:
    return {
        "model_id": model_id,
        "sae_id": sae_id,
        "features_parquet": None,
        "activations_jsonl": None,
        "features_inserted": 0,
        "activations_inserted": 0,
        "duration_seconds": 0.0,
        "status": "failed",
        "error": error,
    }


def _make_progress(
    job_id: str | None,
    weights: dict[str, int],
    offsets: dict[str, int],
) -> Callable[[str, int, int], None]:
    def _progress(stage: str, done: int, total: int) -> None:
        if not job_id:
            return
        stage_offset = offsets.get(stage, 0)
        stage_weight = weights.get(stage, 10)
        overall = stage_offset + (done * stage_weight // max(total, 1))
        emit_progress(
            job_id=job_id,
            status="running",
            items_processed=overall,
            total_items=100,
            current_batch=0,
            total_batches=0,
            message=f"SAE pipeline: {stage} ({done}/{total})",
        )

    return _progress


def _ingest_features(
    result: dict,
    parquet_path: str,
    model_id: str,
    sae_id: str,
    progress: Callable[[str, int, int], None],
) -> bool:
    """Ingest a features parquet into DuckDB ``sae_features``. Mutates
    ``result`` in place; returns False (with result marked failed) on error."""
    try:
        from ..embedding_functions.ingest_sae import ingest_sae_features

        progress("ingest_features", 0, 1)
        feat_result = ingest_sae_features(
            parquet_path=parquet_path,
            model_id=model_id,
            sae_id=sae_id,
            store_vectors=False,  # no ChromaDB — user imports parquet for viz
        )
        result["features_inserted"] = feat_result.get("records_inserted", 0)
        if feat_result.get("error"):
            result["error"] = feat_result["error"]
            result["status"] = "failed"
            return False
        progress("ingest_features", 1, 1)
        return True
    except Exception as e:
        result["error"] = f"Feature ingestion failed: {e}"
        result["status"] = "failed"
        logger.exception("Feature ingestion failed for %s/%s", model_id, sae_id)
        return False


def _finalize(result: dict, job_id: str | None, start: float) -> dict:
    result["duration_seconds"] = round(time.time() - start, 2)
    if job_id:
        emit_progress(
            job_id=job_id,
            status="completed",
            items_processed=100,
            total_items=100,
            current_batch=0,
            total_batches=0,
            message="SAE pipeline complete",
        )
    logger.info(
        "SAE pipeline complete for %s/%s in %.1fs — parquet: %s",
        result["model_id"],
        result["sae_id"],
        result["duration_seconds"],
        result["features_parquet"],
    )
    return result


def prepare_sae_data(
    layer: int,
    width: str = "16k",
    hook_type: str = "resid_post",
    model_size: str = "4b",
    variant: str = "it",
    family: str = "gemma",
    skip_download: bool = False,
    include_activations: bool = False,
    job_id: str | None = None,
) -> dict:
    """Run the SAE pipeline: download, extract, and ingest into DuckDB.

    This is a **synchronous** function intended to be called via
    ``asyncio.to_thread()`` from the GraphQL mutation layer.

    Ingests features + activations into DuckDB sae_features/sae_activations
    tables (without storing vectors in ChromaDB). Returns output file paths
    so the user can import the parquet for visualization separately.

    ``family="qwen"`` dispatches to the qwen-scope path (extract-only, no
    Neuronpedia download — see :func:`prepare_qwen_sae_data`).
    """
    if family == "qwen":
        if include_activations:
            return _failed(
                "Qwen-scope SAEs have no activation examples "
                "(not indexed by Neuronpedia) — disable includeActivations."
            )
        return prepare_qwen_sae_data(
            layer=layer,
            width=width,
            hook_type=hook_type,
            model_size=model_size,
            job_id=job_id,
        )
    if family != "gemma":
        return _failed(f"Unknown SAE family '{family}'. Valid: ['gemma', 'qwen']")

    start = time.time()

    if model_size not in MODEL_SIZE_TO_LAYERS:
        return _failed(
            f"Unknown model_size '{model_size}'. Valid: {list(MODEL_SIZE_TO_LAYERS.keys())}"
        )

    ht = HOOK_TYPE_FROM_STR.get(hook_type)
    if ht is None:
        return _failed(f"Unknown hook_type '{hook_type}'. Valid: {list(HOOK_TYPE_FROM_STR.keys())}")

    max_layers = MODEL_SIZE_TO_LAYERS[model_size]
    if layer >= max_layers:
        return _failed(
            f"Layer {layer} exceeds max ({max_layers - 1}) for model size '{model_size}'"
        )

    sae_config = GemmaScopeSAEConfig(
        layer_index=layer,
        width=width,
        hook_type=ht,
        model_size=model_size,
        variant=variant,
        d_in=MODEL_SIZE_TO_D_IN.get(model_size, 2560),
        device="cpu",
    )

    model_id = sae_config.neuronpedia_model_id
    sae_id = neuronpedia_source_id(sae_config)

    result: dict = {
        "model_id": model_id,
        "sae_id": sae_id,
        "features_parquet": None,
        "activations_jsonl": None,
        "features_inserted": 0,
        "activations_inserted": 0,
        "duration_seconds": 0.0,
        "status": "completed",
        "error": None,
    }

    # Build progress callback
    # Note: no early-return for existing files — downloads have resume support
    # and DuckDB inserts use INSERT OR REPLACE, so re-runs are safe.
    _progress = _make_progress(job_id, _STAGE_WEIGHTS, _STAGE_OFFSETS)

    # Run pipeline (download → merge → extract)
    try:
        pipeline_config = SAEPipelineConfig(
            sae=sae_config,
            skip_download=skip_download,
            skip_activations=not include_activations,
            skip_extract=False,
        )
        pipeline_result = SAEPipelineRunner(pipeline_config).run(
            progress_callback=_progress,
        )

        if pipeline_result.error:
            result["error"] = f"Pipeline failed: {pipeline_result.error}"
            result["status"] = "failed"
            return result

        # Populate output paths
        if pipeline_result.features_parquet:
            result["features_parquet"] = str(pipeline_result.features_parquet)
        if pipeline_result.activations_jsonl:
            result["activations_jsonl"] = str(pipeline_result.activations_jsonl)

    except Exception as e:
        result["error"] = f"Pipeline failed: {e}"
        result["status"] = "failed"
        logger.exception("SAE pipeline failed for %s/%s", model_id, sae_id)
        return result

    # ── Ingest features into DuckDB (without ChromaDB vectors) ────────
    if pipeline_result.features_parquet and pipeline_result.features_parquet.exists():
        if not _ingest_features(
            result, str(pipeline_result.features_parquet), model_id, sae_id, _progress
        ):
            return result

    # ── Ingest activations into DuckDB ────────────────────────────────
    if (
        include_activations
        and pipeline_result.activations_jsonl
        and pipeline_result.activations_jsonl.exists()
    ):
        try:
            from ..embedding_functions.ingest_sae import ingest_sae_activations

            _progress("ingest_activations", 0, 1)
            act_result = ingest_sae_activations(
                jsonl_path=str(pipeline_result.activations_jsonl),
                model_id=model_id,
                sae_id=sae_id,
            )
            result["activations_inserted"] = act_result.get("records_inserted", 0)
            if act_result.get("error"):
                result["error"] = act_result["error"]
                result["status"] = "failed"
                return result
            _progress("ingest_activations", 1, 1)
        except Exception as e:
            result["error"] = f"Activation ingestion failed: {e}"
            result["status"] = "failed"
            logger.exception("Activation ingestion failed for %s/%s", model_id, sae_id)
            return result

    return _finalize(result, job_id, start)


def prepare_qwen_sae_data(
    layer: int,
    width: str = "32k",
    hook_type: str = "resid_post",
    model_size: str = "1.7B",
    job_id: str | None = None,
) -> dict:
    """Extract + ingest a qwen-scope SAE (synchronous, like the gemma path).

    Qwen-scope isn't on Neuronpedia, so there is no download stage: decoder
    vectors come straight from the HF SAE weights (``extract_and_merge`` with
    ``skip_labels=True``) and labels/densities/logits stay empty until the
    Phase-2 autointerp pass backfills them. Same ids as the offline
    ``scripts/extract_qwen_decoder_vectors.py`` bootstrap
    (``qwen3-1.7B-base`` / ``{layer}-qwenscope-1-res-32k``), so re-running
    over an existing ingest is a safe no-op (INSERT OR REPLACE).
    """
    start = time.time()

    if hook_type != "resid_post":
        return _failed(
            f"Qwen-scope SAEs are residual-stream only; hook_type must be "
            f"'resid_post', got '{hook_type}'."
        )

    info = QWEN_SCOPE_MODELS.get(model_size)
    if info is None:
        return _failed(
            f"Unknown Qwen model_size '{model_size}'. Valid: {sorted(QWEN_SCOPE_MODELS)}"
        )
    if layer >= info.n_layers:
        return _failed(f"Layer {layer} exceeds max ({info.n_layers - 1}) for Qwen '{model_size}'")

    try:
        # k is pinned to 50, matching InterpretService._make_sae_config —
        # qwen_source_id doesn't encode k, so the two L0 variants would
        # collide in DuckDB under the same sae_id.
        sae_config = QwenScopeSAEConfig(
            layer_index=layer,
            width=width,
            model_size=model_size,
            device="cpu",
        )
    except ValueError as e:
        return _failed(str(e))

    model_id = sae_config.neuronpedia_model_id
    sae_id = qwen_source_id(sae_config)

    result: dict = {
        "model_id": model_id,
        "sae_id": sae_id,
        "features_parquet": None,
        "activations_jsonl": None,
        "features_inserted": 0,
        "activations_inserted": 0,
        "duration_seconds": 0.0,
        "status": "completed",
        "error": None,
    }

    _progress = _make_progress(job_id, _QWEN_STAGE_WEIGHTS, _QWEN_STAGE_OFFSETS)
    out_path = vectors_parquet_path(sae_config)

    try:
        if out_path.exists():
            logger.info("Qwen decoder-vector parquet already exists: %s", out_path)
        else:
            # Lazy: pulls torch via load_sae (same boundary as the gemma
            # pipeline's extract stage).
            from interpret.sae.extract_decoder_vectors import extract_and_merge
            from interpret.sae.loading import clear_sae_cache

            _progress("extract_vectors", 0, 1)
            extract_and_merge(sae_config, out_path, skip_labels=True)
            clear_sae_cache()  # free the cpu weight tensors
        result["features_parquet"] = str(out_path)
        _progress("extract_vectors", 1, 1)
    except Exception as e:
        result["error"] = f"Pipeline failed: {e}"
        result["status"] = "failed"
        logger.exception("Qwen SAE extraction failed for %s/%s", model_id, sae_id)
        return result

    if not _ingest_features(result, str(out_path), model_id, sae_id, _progress):
        return result

    return _finalize(result, job_id, start)
