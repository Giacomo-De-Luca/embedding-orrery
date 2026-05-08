"""Bridge between the interpret/ SAE pipeline and backend DuckDB ingestion.

Orchestrates: pipeline.run() -> ingest_sae_features() + ingest_sae_activations()

This module imports from both the standalone ``interpret.sae.pipeline`` (which
produces files on disk) and the backend ingestion functions (which load those
files into DuckDB / ChromaDB). It is the only module that crosses this boundary.
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

# Total progress is split across stages (100 units).
# Precompute cumulative offsets to avoid repeated list scans.
_STAGE_WEIGHTS = {
    "download": 40,
    "merge_activations": 10,
    "extract_vectors": 20,
    "ingest_features": 20,
    "ingest_activations": 10,
}
_STAGE_OFFSETS: dict[str, int] = {}
_offset = 0
for _k, _v in _STAGE_WEIGHTS.items():
    _STAGE_OFFSETS[_k] = _offset
    _offset += _v


def prepare_and_ingest(
    layer: int,
    width: str = "16k",
    hook_type: str = "resid_post",
    skip_download: bool = False,
    store_vectors: bool = True,
    include_activations: bool = False,
    job_id: str | None = None,
) -> dict:
    """Run the full SAE pipeline and ingest results into DuckDB.

    This is a **synchronous** function intended to be called via
    ``asyncio.to_thread()`` from the GraphQL mutation layer.

    Args:
        layer: Layer index (e.g. 9, 17, 22, 29).
        width: SAE width (e.g. "16k", "65k", "262k").
        hook_type: Hook type string ("resid_post", "mlp_out", "attn_out").
        skip_download: Skip the S3 download stage.
        store_vectors: Store explanation vectors in ChromaDB for semantic search.
        include_activations: Also download/merge/ingest activation examples.
        job_id: Optional job ID for progress emission.

    Returns:
        Dict with keys: model_id, sae_id, features_inserted,
        activations_inserted, duration_seconds, status, error.
    """
    start = time.time()

    ht = HOOK_TYPE_FROM_STR.get(hook_type)
    if ht is None:
        return {
            "model_id": "",
            "sae_id": "",
            "features_inserted": 0,
            "activations_inserted": 0,
            "duration_seconds": 0.0,
            "status": "failed",
            "error": (f"Unknown hook_type '{hook_type}'. Valid: {list(HOOK_TYPE_FROM_STR.keys())}"),
        }

    sae_config = GemmaScopeSAEConfig(
        layer_index=layer,
        width=width,
        hook_type=ht,
        device="cpu",  # extraction only, no GPU needed
    )

    model_id = sae_config.neuronpedia_model_id
    sae_id = neuronpedia_source_id(sae_config)

    result = {
        "model_id": model_id,
        "sae_id": sae_id,
        "features_inserted": 0,
        "activations_inserted": 0,
        "duration_seconds": 0.0,
        "status": "completed",
        "error": None,
    }

    # Check if already ingested
    from ..API.duckdb_instance import get_duckdb_client

    db = get_duckdb_client()
    existing = db.list_sae_models()
    if any(m["model_id"] == model_id and m["sae_id"] == sae_id for m in existing):
        parquet_path = vectors_parquet_path(sae_config)
        if parquet_path.exists():
            result["status"] = "already_ingested"
            result["duration_seconds"] = round(time.time() - start, 2)
            logger.info("SAE %s/%s already ingested, skipping", model_id, sae_id)
            return result

    # Build progress callback using precomputed offsets
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

    # ── Stage 1-3: Pipeline (download, merge, extract) ───────────────
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

    except Exception as e:
        result["error"] = f"Pipeline failed: {e}"
        result["status"] = "failed"
        logger.exception("SAE pipeline failed for %s/%s", model_id, sae_id)
        return result

    # ── Stage 4: Ingest features into DuckDB ─────────────────────────
    if pipeline_result.features_parquet and pipeline_result.features_parquet.exists():
        try:
            from ..embedding_functions.ingest_sae import ingest_sae_features

            _progress("ingest_features", 0, 1)
            feat_result = ingest_sae_features(
                parquet_path=str(pipeline_result.features_parquet),
                model_id=model_id,
                sae_id=sae_id,
                store_vectors=store_vectors,
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

    # ── Stage 5: Ingest activations into DuckDB ──────────────────────
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
        "SAE pipeline complete for %s/%s — %d features, %d activations in %.1fs",
        model_id,
        sae_id,
        result["features_inserted"],
        result["activations_inserted"],
        result["duration_seconds"],
    )
    return result
