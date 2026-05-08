"""Bridge between the interpret/ SAE pipeline and the backend GraphQL API.

Runs the standalone pipeline (download → merge → extract decoder vectors)
and returns output file paths. Does NOT auto-ingest into DuckDB — the user
imports the parquet via the Local Files flow to get projections and topics.
"""

import logging
import time

from interpret.sae.paths import vectors_parquet_path
from interpret.sae.pipeline.prepare_sae_data import (
    SAEPipelineConfig,
    SAEPipelineRunner,
)
from interpret.sae.sae_config import HOOK_TYPE_FROM_STR, GemmaScopeSAEConfig
from interpret.sae.source_ids import neuronpedia_source_id

from ..services.progress_emitter import emit_progress

logger = logging.getLogger("star_map." + __name__)

# Total progress split across stages (100 units).
# Download sub-stages are reported as "download:download_features" etc.
_STAGE_WEIGHTS = {
    "download:download_features": 20,
    "download:download_explanations": 20,
    "download:download_activations": 20,
    "merge_activations": 5,
    "extract_vectors": 35,
}
_STAGE_OFFSETS: dict[str, int] = {}
_offset = 0
for _k, _v in _STAGE_WEIGHTS.items():
    _STAGE_OFFSETS[_k] = _offset
    _offset += _v


def prepare_sae_data(
    layer: int,
    width: str = "16k",
    hook_type: str = "resid_post",
    skip_download: bool = False,
    include_activations: bool = False,
    job_id: str | None = None,
) -> dict:
    """Run the SAE download + extraction pipeline.

    This is a **synchronous** function intended to be called via
    ``asyncio.to_thread()`` from the GraphQL mutation layer.

    Returns output file paths — does NOT ingest into DuckDB.
    """
    start = time.time()

    ht = HOOK_TYPE_FROM_STR.get(hook_type)
    if ht is None:
        return {
            "model_id": "",
            "sae_id": "",
            "features_parquet": None,
            "activations_jsonl": None,
            "duration_seconds": 0.0,
            "status": "failed",
            "error": (f"Unknown hook_type '{hook_type}'. Valid: {list(HOOK_TYPE_FROM_STR.keys())}"),
        }

    sae_config = GemmaScopeSAEConfig(
        layer_index=layer,
        width=width,
        hook_type=ht,
        device="cpu",
    )

    model_id = sae_config.neuronpedia_model_id
    sae_id = neuronpedia_source_id(sae_config)

    result: dict = {
        "model_id": model_id,
        "sae_id": sae_id,
        "features_parquet": None,
        "activations_jsonl": None,
        "duration_seconds": 0.0,
        "status": "completed",
        "error": None,
    }

    # Check if parquet already exists on disk
    parquet_path = vectors_parquet_path(sae_config)
    if not skip_download and parquet_path.exists():
        result["features_parquet"] = str(parquet_path)
        result["status"] = "already_downloaded"
        result["duration_seconds"] = round(time.time() - start, 2)
        logger.info("SAE %s/%s parquet already exists at %s", model_id, sae_id, parquet_path)
        return result

    # Build progress callback
    def _progress(stage: str, done: int, total: int) -> None:
        if not job_id:
            return
        stage_offset = _STAGE_OFFSETS.get(stage, 0)
        stage_weight = _STAGE_WEIGHTS.get(stage, 10)
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

    result["duration_seconds"] = round(time.time() - start, 2)

    # Final progress
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
        model_id,
        sae_id,
        result["duration_seconds"],
        result["features_parquet"],
    )
    return result
