"""
Local file text embedding using SentenceTransformers and other providers.

This module handles embedding text from local files (parquet, JSON, CSV, TSV)
into ChromaDB using various embedding providers.
Supports resume capability for interrupted jobs.
"""

import chromadb
from chromadb.config import Settings
import time
import json
from typing import Optional, List, Dict, Callable, Set
from tqdm import tqdm

from .config import (
    DB_PATH,
    TEXT_EMBEDDING_DIMENSIONS,
    DataType,
    EmbeddingModelConfig,
    EmbeddingResult,
    LocalFileEmbeddingConfig,
)
from .create_embedding_function import create_embedding_function, get_device
from ..utils.text_processing import format_text_for_embedding, extract_metadata
from ..utils.color_preprocessing import preprocess_color_metadata
from ..utils.id_utils import IDDeduplicator
from ..utils.batch_utils import sort_items_by_length
from .embed_images import embed_images
from .embed_vectors import embed_vectors
from ..services.job_state import get_job_state_service, JobStatus
from ..services.progress_emitter import emit_progress_sync
from ..utils.duckdb_sync import sync_dataset_and_collection, sync_items


# Progress update frequency (every N batches)
PROGRESS_UPDATE_FREQUENCY = 1


def _config_to_dict(config: LocalFileEmbeddingConfig) -> dict:
    """Convert LocalFileEmbeddingConfig to a JSON-serializable dict for job state."""
    result = {
        "file_path": config.file_path,
        "collection_name": config.collection_name,
        "data_type": config.data_type.value,
        "columns": config.columns,
        "text_template": config.text_template,
        "image_column": config.image_column,
        "vector_column": config.vector_column,
        "id_column": config.id_column,
        "metadata_columns": config.metadata_columns,
        "n_rows": config.n_rows,
        "sample_n": config.sample_n,
        "sample_seed": config.sample_seed,
        "batch_size": config.batch_size,
    }

    if config.embedding_model:
        result["embedding_model"] = {
            "provider": config.embedding_model.provider.value,
            "model_name": config.embedding_model.model_name,
            "ollama_url": config.embedding_model.ollama_url,
            "task": config.embedding_model.task,
            "task_type": config.embedding_model.task_type,
        }

    return result


def embed_local_file(
    config: LocalFileEmbeddingConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> EmbeddingResult:
    """
    Embed a local file (parquet/json/csv/tsv) into ChromaDB.

    Supports text, image, and pre-computed vector embeddings.

    Args:
        config: Embedding configuration
        progress_callback: Optional callback for progress updates

    Returns:
        EmbeddingResult with statistics
    """
    from ..clients.local_data_client import load_local_file_portion

    start_time = time.time()
    job_state = get_job_state_service()

    try:
        device = get_device()
        print(f"Using device: {device}")

        # Load data from local file
        print(f"Loading file: {config.file_path}")
        rows, total = load_local_file_portion(
            file_path=config.file_path,
            n_rows=config.n_rows,
            sample_n=config.sample_n,
            sample_seed=config.sample_seed
        )
        print(f"Loaded {len(rows)} rows (total in file: {total})")

        if not rows:
            return EmbeddingResult(
                collection_name=config.collection_name,
                total_embedded=0,
                embedding_dim=TEXT_EMBEDDING_DIMENSIONS,
                device=device,
                duration_seconds=time.time() - start_time,
                error="No rows loaded from file"
            )

        # Initialize ChromaDB
        db_path = str(DB_PATH.resolve())
        DB_PATH.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Handle different data types
        if config.data_type == DataType.VECTOR:
            # Vector embedding doesn't support resume yet (pass through to existing)
            try:
                if not config.resume:
                    client.delete_collection(name=config.collection_name)
                    print(f"Deleted existing collection: {config.collection_name}")
            except Exception:
                pass
            return embed_vectors(
                client, config, rows, total, device, start_time, progress_callback
            )
        elif config.data_type == DataType.IMAGE:
            # Image embedding doesn't support resume yet (pass through to existing)
            try:
                if not config.resume:
                    client.delete_collection(name=config.collection_name)
                    print(f"Deleted existing collection: {config.collection_name}")
            except Exception:
                pass
            return embed_images(
                client, config, rows, total, device, start_time, progress_callback
            )
        else:  # TEXT
            return embed_text_from_local(
                client, config, rows, total, device, start_time, progress_callback
            )

    except Exception as e:
        # Mark job as failed
        job_state.fail_job(config.collection_name, str(e))

        # Emit failure event to WebSocket subscribers
        emit_progress_sync(
            job_id=config.collection_name,
            status="failed",
            items_processed=0,
            total_items=0,
            current_batch=0,
            total_batches=0,
            error=str(e)
        )

        return EmbeddingResult(
            collection_name=config.collection_name,
            total_embedded=0,
            embedding_dim=TEXT_EMBEDDING_DIMENSIONS,
            device=get_device(),
            duration_seconds=time.time() - start_time,
            error=str(e)
        )


def embed_text_from_local(
    client,
    config: LocalFileEmbeddingConfig,
    rows: List[Dict],
    total: int,
    device: str,
    start_time: float,
    progress_callback: Optional[Callable] = None
) -> EmbeddingResult:
    """
    Embed text data from local file.

    Args:
        client: ChromaDB client
        config: Embedding configuration
        rows: List of row dictionaries
        total: Total number of rows in the source file
        device: Device string (cpu, cuda, mps)
        start_time: Start time for duration calculation
        progress_callback: Optional progress callback

    Returns:
        EmbeddingResult with statistics
    """
    job_state = get_job_state_service()

    # Get model config early
    model_config = config.embedding_model or EmbeddingModelConfig()

    # Determine columns
    columns = config.columns
    if not columns:
        first_row = rows[0]
        columns = [k for k, v in first_row.items() if isinstance(v, str)]
        if not columns:
            return EmbeddingResult(
                collection_name=config.collection_name,
                total_embedded=0,
                embedding_dim=0,
                device=device,
                duration_seconds=time.time() - start_time,
                error="No text columns found",
                embedding_provider=model_config.provider.value,
                embedding_model=model_config.model_name
            )

    print(f"Embedding columns: {columns}")

    # Calculate total batches early for progress messages
    total_batches = (len(rows) + config.batch_size - 1) // config.batch_size

    # Emit status: sorting batches
    emit_progress_sync(
        job_id=config.collection_name,
        status="running",
        items_processed=0,
        total_items=len(rows),
        current_batch=0,
        total_batches=total_batches,
        message="Sorting batches by length..."
    )

    # Sort rows by text length for efficient batching (reduces padding waste)
    rows = sort_items_by_length(
        rows,
        lambda row: format_text_for_embedding(row, columns, config.text_template)
    )
    print(f"Sorted {len(rows)} rows by text length for efficient batching")

    # Emit status: loading model
    emit_progress_sync(
        job_id=config.collection_name,
        status="running",
        items_processed=0,
        total_items=len(rows),
        current_batch=0,
        total_batches=total_batches,
        message=f"Loading embedding model ({model_config.model_name})..."
    )

    # Create embedding function using factory
    embedding_func, embedding_dim = create_embedding_function(model_config, device)
    print(f"Using embedding model: {model_config.provider.value} / {model_config.model_name}")

    # Handle existing collection (resume vs overwrite)
    existing_ids: Set[str] = set()
    collection = None

    try:
        existing_collection = client.get_collection(
            name=config.collection_name,
            embedding_function=embedding_func
        )

        if config.resume:
            # Resume mode: get existing IDs to skip
            result = existing_collection.get(include=[])
            existing_ids = set(result['ids'])
            collection = existing_collection
            print(f"Resuming: found {len(existing_ids)} existing embeddings")
        else:
            # Overwrite mode: delete existing collection
            client.delete_collection(name=config.collection_name)
            print(f"Deleted existing collection: {config.collection_name}")
    except Exception:
        # Collection doesn't exist
        pass

    # Create new collection if needed
    if collection is None:
        collection_metadata = {
            "description": f"Embeddings from local file: {config.file_path}",
            "source_file": config.file_path,
            "embedded_columns": json.dumps(columns),
            "data_type": "text",
            "embedding_provider": model_config.provider.value,
            "embedding_model": model_config.model_name,
            "embedding_dim": embedding_dim,
            "embedding_task": model_config.task,
            "embedding_task_type": model_config.task_type,
            "embedding_prompt": model_config.prompt,
            "total_in_file": total,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Filter out None values from metadata (ChromaDB doesn't like them)
        collection_metadata = {k: v for k, v in collection_metadata.items() if v is not None}

        collection = client.create_collection(
            name=config.collection_name,
            embedding_function=embedding_func,
            metadata=collection_metadata
        )

    # DuckDB dual-write: create dataset + register vector collection
    _duckdb_dataset_name = sync_dataset_and_collection(
        config.collection_name, collection_metadata,
        model_config=model_config, embedding_dim=embedding_dim,
    )

    # Determine metadata columns (exclude embedded columns and id column)
    metadata_columns = config.metadata_columns
    if metadata_columns is None:
        first_row = rows[0]
        exclude_cols = set(columns)
        if config.id_column:
            exclude_cols.add(config.id_column)
        metadata_columns = [k for k in first_row.keys() if k not in exclude_cols]

    # Register job start (only if not resuming, or if resuming from scratch)
    if not config.resume or len(existing_ids) == 0:
        job_state.start_job(
            collection_name=config.collection_name,
            job_type="local_file",
            total_expected=len(rows),
            total_batches=total_batches,
            config=_config_to_dict(config)
        )

    # Emit initial progress so subscribers know the job has started
    emit_progress_sync(
        job_id=config.collection_name,
        status="running",
        items_processed=len(existing_ids),
        total_items=len(rows),
        current_batch=0,
        total_batches=total_batches,
        message="Starting embedding..."
    )

    # Process in batches
    total_embedded = len(existing_ids)  # Start from existing count if resuming
    new_embedded = 0
    id_deduplicator = IDDeduplicator()
    batches_completed = 0

    for batch_start in tqdm(range(0, len(rows), config.batch_size),
                            desc="Embedding batches", unit="batch"):
        batch = rows[batch_start:batch_start + config.batch_size]

        ids = []
        documents = []
        metadatas = []

        for i, row in enumerate(batch):
            row_idx = batch_start + i

            # Generate ID
            if config.id_column and config.id_column in row:
                base_id = str(row[config.id_column])
                doc_id = id_deduplicator.get_unique_id(base_id)
            else:
                doc_id = f"{config.collection_name}_{row_idx}"

            # Skip if already embedded (resume mode)
            if doc_id in existing_ids:
                continue

            # Format text
            text = format_text_for_embedding(row, columns, config.text_template)
            if not text.strip():
                continue

            # Extract metadata using shared function
            metadata = extract_metadata(row, metadata_columns)
            metadata = preprocess_color_metadata(metadata, row)
            metadata["row_index"] = row_idx

            ids.append(doc_id)
            documents.append(text)
            metadatas.append(metadata)

        if ids:
            # Compute embeddings explicitly
            batch_embeddings = embedding_func(documents)

            # ChromaDB: vectors only
            collection.add(ids=ids, embeddings=batch_embeddings)

            # DuckDB: documents + metadata
            if _duckdb_dataset_name:
                sync_items(_duckdb_dataset_name, ids, documents, metadatas)

            total_embedded += len(ids)
            new_embedded += len(ids)

        batches_completed += 1

        # Update job state and emit progress periodically
        if batches_completed % PROGRESS_UPDATE_FREQUENCY == 0:
            job_state.update_progress(
                collection_name=config.collection_name,
                items_embedded=total_embedded,
                batches_completed=batches_completed
            )
            # Emit progress to WebSocket subscribers
            emit_progress_sync(
                job_id=config.collection_name,
                status="running",
                items_processed=total_embedded,
                total_items=len(rows),
                current_batch=batches_completed,
                total_batches=total_batches
            )

        if progress_callback:
            progress_callback(min(batch_start + config.batch_size, len(rows)), len(rows))

  

    # Emit completion event to WebSocket subscribers
    emit_progress_sync(
        job_id=config.collection_name,
        status="progress",
        items_processed=total_embedded,
        total_items=len(rows),
        current_batch=total_batches,
        total_batches=total_batches
    )

    duration = time.time() - start_time
    if config.resume and len(existing_ids) > 0:
        print(f"Embedded {new_embedded} new items ({total_embedded} total) in {duration:.2f}s")
    else:
        print(f"Embedded {total_embedded} items in {duration:.2f}s")

    return EmbeddingResult(
        collection_name=config.collection_name,
        total_embedded=total_embedded,
        embedding_dim=embedding_dim,
        device=device,
        duration_seconds=duration,
        embedding_provider=model_config.provider.value,
        embedding_model=model_config.model_name
    )
