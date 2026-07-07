"""
Re-embed an existing DuckDB dataset with a new embedding model.

Reads documents from a DuckDB items table and embeds them into a new ChromaDB
vector collection. Does NOT create a new dataset or insert items into DuckDB —
only ChromaDB gets new vectors and a new vector_collections row is registered.
"""

import time
from collections.abc import Callable

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from ..services.job_state import get_job_state_service
from ..services.progress_emitter import emit_progress_sync
from ..utils.batch_utils import sort_items_by_length
from ..utils.text_processing import format_text_for_embedding
from .config import (
    DB_PATH,
    EMBEDDING_BATCH_SIZE,
    EmbeddingModelConfig,
    EmbeddingResult,
    ReEmbedConfig,
)
from .create_embedding_function import create_embedding_function, get_device

PROGRESS_UPDATE_FREQUENCY = 1


def embed_existing_dataset(
    config: ReEmbedConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EmbeddingResult:
    """
    Re-embed an existing DuckDB dataset with a new embedding model.

    Reads documents from the source dataset's items table, embeds them with
    the configured model, and writes vectors to a new ChromaDB collection.
    Registers the new vector_collection against the existing dataset.

    Args:
        config: Re-embedding configuration (source dataset + new model)
        progress_callback: Optional callback for progress updates

    Returns:
        EmbeddingResult with statistics
    """
    from ..API.duckdb_instance import get_duckdb_client

    start_time = time.time()
    job_state = get_job_state_service()
    device = get_device()
    model_config = config.embedding_model or EmbeddingModelConfig()
    batch_size = config.batch_size or EMBEDDING_BATCH_SIZE

    # Step 1: Resolve source name (may be a collection_name, not a dataset_name)
    db = get_duckdb_client()
    vc = db.get_vector_collection(config.source_dataset_name)
    dataset_name = vc["dataset_name"] if vc else config.source_dataset_name

    dataset = db.get_dataset(dataset_name)
    if not dataset:
        return EmbeddingResult(
            collection_name=config.collection_name,
            total_embedded=0,
            embedding_dim=0,
            device=device,
            duration_seconds=time.time() - start_time,
            error=f"Dataset '{dataset_name}' not found in DuckDB",
            embedding_provider=model_config.provider.value,
            embedding_model=model_config.model_name,
        )

    all_items = db.get_filtered_items(dataset_name, filters=[], limit=1_000_000)
    if not all_items:
        return EmbeddingResult(
            collection_name=config.collection_name,
            total_embedded=0,
            embedding_dim=0,
            device=device,
            duration_seconds=time.time() - start_time,
            error=f"Dataset '{dataset_name}' has no items",
            embedding_provider=model_config.provider.value,
            embedding_model=model_config.model_name,
        )

    print(f"Re-embedding {len(all_items)} items from dataset '{dataset_name}'")

    # Calculate total batches
    total_batches = (len(all_items) + batch_size - 1) // batch_size

    # Emit status: sorting
    emit_progress_sync(
        job_id=config.collection_name,
        status="running",
        items_processed=0,
        total_items=len(all_items),
        current_batch=0,
        total_batches=total_batches,
        message="Sorting by document length...",
    )

    # Determine text composition strategy
    use_columns = bool(config.columns)
    if use_columns:
        print(f"Composing text from metadata columns: {config.columns}")

    def _get_text(item: dict) -> str:
        """Get the text to embed from an item."""
        if use_columns:
            # Compose text from metadata columns using format_text_for_embedding
            meta = item.get("metadata") or {}
            # Build a row-like dict from metadata for format_text_for_embedding
            row = dict(meta)
            # Also include document as a possible field
            row["__document__"] = item.get("document") or ""
            return format_text_for_embedding(row, config.columns, config.text_template)
        return item.get("document") or ""

    # Step 2: Sort by text length for efficient batching
    all_items = sort_items_by_length(all_items, _get_text)
    print(f"Sorted {len(all_items)} items by text length")

    # Emit status: loading model
    emit_progress_sync(
        job_id=config.collection_name,
        status="running",
        items_processed=0,
        total_items=len(all_items),
        current_batch=0,
        total_batches=total_batches,
        message=f"Loading embedding model ({model_config.model_name})...",
    )

    # Step 3: Create embedding function
    embedding_func, embedding_dim = create_embedding_function(model_config, device)
    print(f"Using embedding model: {model_config.provider.value} / {model_config.model_name}")

    # Step 4: Handle resume — check for existing IDs in new ChromaDB collection
    existing_ids: set[str] = set()
    collection = None
    db_path = str(DB_PATH.resolve())
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))

    try:
        existing_collection = client.get_collection(
            name=config.collection_name, embedding_function=embedding_func
        )
        if config.resume:
            result = existing_collection.get(include=[])
            existing_ids = set(result["ids"])
            collection = existing_collection
            print(f"Resuming: found {len(existing_ids)} existing embeddings")
        else:
            client.delete_collection(name=config.collection_name)
            print(f"Deleted existing collection: {config.collection_name}")
    except Exception:
        pass

    # Step 5: Create new ChromaDB collection
    if collection is None:
        collection_metadata = {
            "description": f"Re-embedding of dataset '{config.source_dataset_name}'",
            "source_dataset": config.source_dataset_name,
            "data_type": "text",
            "embedding_provider": model_config.provider.value,
            "embedding_model": model_config.model_name,
            "embedding_dim": embedding_dim,
            "embedding_task": model_config.task,
            "embedding_task_type": model_config.task_type,
            "embedding_prompt": model_config.prompt,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        collection_metadata = {k: v for k, v in collection_metadata.items() if v is not None}

        collection = client.create_collection(
            name=config.collection_name,
            embedding_function=embedding_func,
            metadata=collection_metadata,
        )

    # Step 6: Register vector collection in DuckDB (dataset_name ≠ collection_name)
    # Check if already registered (resume case)
    existing_vc = db.get_vector_collection(config.collection_name)
    if not existing_vc:
        db.register_vector_collection(
            dataset_name,  # FK to existing dataset
            "chromadb",
            config.collection_name,  # New collection name (PK)
            "dense",
            embedding_provider=model_config.provider.value,
            embedding_model=model_config.model_name,
            embedding_dim=embedding_dim,
            embedding_task=model_config.task,
            embedding_task_type=model_config.task_type,
            embedding_prompt=model_config.prompt,
        )

    # Register job start
    if not config.resume or len(existing_ids) == 0:
        job_state.start_job(
            collection_name=config.collection_name,
            job_type="re_embed",
            total_expected=len(all_items),
            total_batches=total_batches,
            config={
                "source_dataset_name": config.source_dataset_name,
                "collection_name": config.collection_name,
                "embedding_provider": model_config.provider.value,
                "embedding_model": model_config.model_name,
            },
        )

    # Emit initial progress
    emit_progress_sync(
        job_id=config.collection_name,
        status="running",
        items_processed=len(existing_ids),
        total_items=len(all_items),
        current_batch=0,
        total_batches=total_batches,
        message="Starting embedding...",
    )

    # Step 7: Batch-embed and write to ChromaDB only
    total_embedded = len(existing_ids)
    new_embedded = 0
    batches_completed = 0

    for batch_start in tqdm(
        range(0, len(all_items), batch_size), desc="Re-embedding batches", unit="batch"
    ):
        batch = all_items[batch_start : batch_start + batch_size]

        ids = []
        documents = []

        for item in batch:
            doc_id = item["id"]

            # Skip if already embedded (resume mode)
            if doc_id in existing_ids:
                continue

            text = _get_text(item)
            if not text.strip():
                continue

            ids.append(doc_id)
            documents.append(text)

        if ids:
            # Compute embeddings
            batch_embeddings = embedding_func(documents)

            # Write to ChromaDB only (items already exist in DuckDB)
            collection.add(ids=ids, embeddings=batch_embeddings)

            total_embedded += len(ids)
            new_embedded += len(ids)

        batches_completed += 1

        # Emit progress
        if batches_completed % PROGRESS_UPDATE_FREQUENCY == 0:
            job_state.update_progress(
                collection_name=config.collection_name,
                items_embedded=total_embedded,
                batches_completed=batches_completed,
            )
            emit_progress_sync(
                job_id=config.collection_name,
                status="running",
                items_processed=total_embedded,
                total_items=len(all_items),
                current_batch=batches_completed,
                total_batches=total_batches,
            )

        if progress_callback:
            progress_callback(min(batch_start + batch_size, len(all_items)), len(all_items))

    # Emit pre-completion
    emit_progress_sync(
        job_id=config.collection_name,
        status="progress",
        items_processed=total_embedded,
        total_items=len(all_items),
        current_batch=total_batches,
        total_batches=total_batches,
    )

    duration = time.time() - start_time
    if config.resume and len(existing_ids) > 0:
        print(f"Re-embedded {new_embedded} new items ({total_embedded} total) in {duration:.2f}s")
    else:
        print(f"Re-embedded {total_embedded} items in {duration:.2f}s")

    return EmbeddingResult(
        collection_name=config.collection_name,
        total_embedded=total_embedded,
        embedding_dim=embedding_dim,
        device=device,
        duration_seconds=duration,
        embedding_provider=model_config.provider.value,
        embedding_model=model_config.model_name,
    )
