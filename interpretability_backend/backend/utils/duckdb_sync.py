"""
DuckDB write helpers for the embedding pipelines.

Each function is a no-op if DuckDB is unavailable.
"""

import logging
from typing import Optional, List, Dict

logger = logging.getLogger("star_map." + __name__)


def _get_db():
    """Get the DuckDB client singleton. Returns None on failure."""
    try:
        from ..API.duckdb_instance import get_duckdb_client
        return get_duckdb_client()
    except Exception as e:
        logger.warning("DuckDB not available: %s", e)
        return None


def sync_dataset_and_collection(
    collection_name: str,
    collection_metadata: Dict,
    model_config=None,
    embedding_dim: int = None,
) -> Optional[str]:
    """Create dataset + register vector collection in DuckDB.

    Returns the dataset_name (= collection_name) or None on failure.
    """
    db = _get_db()
    if not db:
        return None

    try:
        # Check if dataset already exists (resume case)
        existing = db.get_dataset(collection_name)
        if not existing:
            db.create_dataset(
                collection_name,
                description=collection_metadata.get("description"),
                source_type="huggingface" if collection_metadata.get("source_dataset") else "local_file",
                source_dataset=collection_metadata.get("source_dataset"),
                source_config=collection_metadata.get("source_config"),
                source_split=collection_metadata.get("source_split"),
                source_file=collection_metadata.get("source_file"),
                embedded_columns=collection_metadata.get("embedded_columns"),
                data_type=collection_metadata.get("data_type"),
                total_in_source=collection_metadata.get("total_in_split")
                    or collection_metadata.get("total_in_file"),
            )

        # Check if vector collection already registered
        existing_vc = db.get_vector_collection(collection_name)
        if not existing_vc:
            provider = None
            model = None
            dim = embedding_dim
            task = None
            task_type = None
            prompt = None

            if model_config:
                provider = model_config.provider.value if hasattr(model_config.provider, "value") else str(model_config.provider)
                model = model_config.model_name
                task = model_config.task
                task_type = model_config.task_type
                prompt = model_config.prompt

            if not provider:
                provider = collection_metadata.get("embedding_provider")
            if not model:
                model = collection_metadata.get("embedding_model")
            if not dim:
                dim = collection_metadata.get("embedding_dim")

            db.register_vector_collection(
                collection_name, "chromadb", collection_name, "dense",
                embedding_provider=provider,
                embedding_model=model,
                embedding_dim=dim,
                embedding_task=task,
                embedding_task_type=task_type,
                embedding_prompt=prompt,
            )

        logger.info("DuckDB sync: dataset+vc for %s", collection_name)
        return collection_name

    except Exception as e:
        logger.error("DuckDB sync_dataset_and_collection failed: %s", e)
        return None


def sync_items(
    dataset_name: str,
    ids: List[str],
    documents: List[Optional[str]],
    metadatas: List[Optional[Dict]],
) -> int:
    """Insert items into DuckDB. Returns count inserted, or 0 on failure."""
    db = _get_db()
    if not db or not ids:
        return 0

    try:
        return db.insert_items_batch(dataset_name, ids, documents, metadatas)
    except Exception as e:
        logger.error("DuckDB sync_items failed: %s", e)
        return 0
