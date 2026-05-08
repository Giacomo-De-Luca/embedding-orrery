"""Bridge between the interpret/ SAE pipeline and the backend GraphQL API.

Runs the standalone pipeline (download → merge → extract decoder vectors),
ingests features + activations into DuckDB (sae_features / sae_activations
tables), and returns output file paths. Does NOT store vectors in ChromaDB
(no projections/topics) — the user can import the parquet via the Local
Files flow for visualization.
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
    "download:download_features": 15,
    "download:download_explanations": 15,
    "download:download_activations": 15,
    "merge_activations": 5,
    "extract_vectors": 25,
    "ingest_features": 15,
    "ingest_activations": 10,
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
    """Run the SAE pipeline: download, extract, and ingest into DuckDB.

    This is a **synchronous** function intended to be called via
    ``asyncio.to_thread()`` from the GraphQL mutation layer.

    Ingests features + activations into DuckDB sae_features/sae_activations
    tables (without storing vectors in ChromaDB). Returns output file paths
    so the user can import the parquet for visualization separately.
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
        "features_inserted": 0,
        "activations_inserted": 0,
        "duration_seconds": 0.0,
        "status": "completed",
        "error": None,
    }

    # Build progress callback
    # Note: no early-return for existing files — downloads have resume support
    # and DuckDB inserts use INSERT OR REPLACE, so re-runs are safe.
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

    # ── Ingest features into DuckDB (without ChromaDB vectors) ────────
    if pipeline_result.features_parquet and pipeline_result.features_parquet.exists():
        try:
            from ..embedding_functions.ingest_sae import ingest_sae_features

            _progress("ingest_features", 0, 1)
            feat_result = ingest_sae_features(
                parquet_path=str(pipeline_result.features_parquet),
                model_id=model_id,
                sae_id=sae_id,
                store_vectors=False,  # no ChromaDB — user imports parquet for viz
            )
            result["features_inserted"] = feat_result.get("records_inserted", 0)
            if feat_result.get("error"):
                result["error"] = feat_result["error"]
                result["status"] = "failed"
                return result
            _progress("ingest_features", 1, 1)
        except Exception as e:
            result["error"] = f"Feature ingestion failed: {e}"
            result["status"] = "failed"
            logger.exception("Feature ingestion failed for %s/%s", model_id, sae_id)
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
