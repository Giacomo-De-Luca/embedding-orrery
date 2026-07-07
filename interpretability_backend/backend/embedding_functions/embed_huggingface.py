"""
HuggingFace dataset embedding.

This module handles embedding HuggingFace datasets into ChromaDB.
Supports resume capability for interrupted jobs.
"""

import json
import threading
import time
from collections.abc import Callable

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from ..clients.huggingface_client import (
    PortionStrategy,
    load_dataset_portion,
)
from ..services.job_state import get_job_state_service
from ..services.progress_emitter import emit_progress_sync
from ..utils.batch_utils import sort_items_by_length
from ..utils.color_preprocessing import preprocess_color_metadata
from ..utils.duckdb_sync import sync_dataset_and_collection, sync_items
from ..utils.id_utils import IDDeduplicator
from ..utils.text_processing import extract_metadata, format_text_for_embedding
from .config import (
    DB_PATH,
    EmbeddingConfig,
    EmbeddingModelConfig,
    EmbeddingResult,
)
from .create_embedding_function import create_embedding_function, get_device

# Progress update frequency (every N batches)
PROGRESS_UPDATE_FREQUENCY = 1

# Reserved row key holding the source split during multi-split embedding.
# Injected at load time (survives length-sorting), consumed when building
# metadata, and never embedded or stored as a normal column.
_SPLIT_KEY = "__source_split__"


def _load_rows_for_splits(
    splits: list[str],
    load_fn: Callable[[str], tuple[list[dict], int]],
) -> tuple[list[dict], int]:
    """Load and concatenate rows for several splits, tagging each with its split.

    Args:
        splits: Split names to load, in order.
        load_fn: Callable returning ``(rows, total_in_split)`` for one split.

    Returns:
        ``(combined_rows, combined_total)`` where every row carries ``_SPLIT_KEY``.
    """
    combined: list[dict] = []
    combined_total = 0
    for split in splits:
        rows, split_total = load_fn(split)
        for row in rows:
            row[_SPLIT_KEY] = split
        combined.extend(rows)
        combined_total += split_total or len(rows)
    return combined, combined_total


def _config_to_dict(config: EmbeddingConfig) -> dict:
    """Convert EmbeddingConfig to a JSON-serializable dict for job state."""
    result = {
        "dataset_id": config.dataset_id,
        "collection_name": config.collection_name,
        "config": config.config,
        "split": config.split,
        "splits": config.splits,
        "columns": config.columns,
        "text_template": config.text_template,
        "id_column": config.id_column,
        "metadata_columns": config.metadata_columns,
        "batch_size": config.batch_size,
    }

    if config.portion:
        result["portion"] = {
            "strategy": config.portion.strategy.value,
            "n": config.portion.n,
            "start": config.portion.start,
            "end": config.portion.end,
            "seed": config.portion.seed,
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


def embed_huggingface_dataset(
    config: EmbeddingConfig,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> EmbeddingResult:
    """
    Embed a HuggingFace dataset into ChromaDB.

    Args:
        config: Embedding configuration
        progress_callback: Optional callback for progress updates (current, total)

    Returns:
        EmbeddingResult with statistics
    """
    start_time = time.time()
    job_state = get_job_state_service()

    try:
        device = get_device()
        print(f"Using device: {device}")

        # Load dataset portion. When multiple splits are requested they are
        # embedded into one collection in a single pass; each row is tagged with
        # its source split so metadata and IDs stay correct after length-sorting.
        # Note: a non-ALL portion (FIRST_N/RANDOM_SAMPLE/ROW_RANGE) is applied
        # independently to EACH split (e.g. FIRST_N=100 over 3 splits -> up to 300).
        splits = config.splits or [config.split]
        print(f"Loading dataset: {config.dataset_id} (splits: {', '.join(splits)})")
        rows, total_in_split = _load_rows_for_splits(
            splits,
            lambda s: load_dataset_portion(
                dataset_id=config.dataset_id,
                config=config.config,
                split=s,
                portion=config.portion,
            ),
        )
        print(f"Loaded {len(rows)} rows (total across splits: {total_in_split})")

        # Get model config early for error responses
        model_config = config.embedding_model or EmbeddingModelConfig()

        if not rows:
            return EmbeddingResult(
                collection_name=config.collection_name,
                total_embedded=0,
                embedding_dim=0,
                device=device,
                duration_seconds=time.time() - start_time,
                error="No rows loaded from dataset",
                embedding_provider=model_config.provider.value,
                embedding_model=model_config.model_name,
            )

        # Determine columns to embed
        columns = config.columns
        if not columns:
            # Auto-detect text columns (never embed the internal split tag)
            first_row = rows[0]
            columns = [k for k, v in first_row.items() if isinstance(v, str) and k != _SPLIT_KEY]
            if not columns:
                return EmbeddingResult(
                    collection_name=config.collection_name,
                    total_embedded=0,
                    embedding_dim=0,
                    device=device,
                    duration_seconds=time.time() - start_time,
                    error="No text columns found in dataset",
                    embedding_provider=model_config.provider.value,
                    embedding_model=model_config.model_name,
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
            message="Sorting batches by length...",
        )

        # Sort rows by text length for efficient batching (reduces padding waste)
        rows = sort_items_by_length(
            rows, lambda row: format_text_for_embedding(row, columns, config.text_template)
        )
        print(f"Sorted {len(rows)} rows by text length for efficient batching")

        # Initialize ChromaDB
        db_path = str(DB_PATH.resolve())
        DB_PATH.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        # Emit status: loading model
        emit_progress_sync(
            job_id=config.collection_name,
            status="running",
            items_processed=0,
            total_items=len(rows),
            current_batch=0,
            total_batches=total_batches,
            message=f"Loading embedding model ({model_config.model_name})...",
        )

        # Create embedding function using factory
        embedding_func, embedding_dim = create_embedding_function(model_config, device)
        print(f"Using embedding model: {model_config.provider.value} / {model_config.model_name}")

        # Handle existing collection (resume vs overwrite)
        existing_ids: set[str] = set()
        collection = None

        try:
            existing_collection = client.get_collection(
                name=config.collection_name, embedding_function=embedding_func
            )

            if config.resume:
                # Resume mode: get existing IDs to skip
                result = existing_collection.get(include=[])
                existing_ids = set(result["ids"])
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
            # Build collection metadata
            portion_info = "all"
            if config.portion:
                if config.portion.strategy == PortionStrategy.FIRST_N:
                    portion_info = f"first_{config.portion.n}"
                elif config.portion.strategy == PortionStrategy.RANDOM_SAMPLE:
                    portion_info = f"random_{config.portion.n}_seed{config.portion.seed}"
                elif config.portion.strategy == PortionStrategy.ROW_RANGE:
                    portion_info = f"range_{config.portion.start}_{config.portion.end}"

            collection_metadata = {
                "description": f"Embeddings from HuggingFace dataset: {config.dataset_id}",
                "source_dataset": config.dataset_id,
                "source_config": config.config or "",
                "source_split": ", ".join(splits),
                "embedded_columns": json.dumps(columns),
                "portion_strategy": portion_info,
                "total_in_split": total_in_split,
                "embedding_provider": model_config.provider.value,
                "embedding_model": model_config.model_name,
                "embedding_dim": embedding_dim,
                "embedding_task": model_config.task,
                "embedding_task_type": model_config.task_type,
                "embedding_prompt": model_config.prompt,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Filter out None values from metadata (ChromaDB doesn't like them)
            collection_metadata = {k: v for k, v in collection_metadata.items() if v is not None}

            collection = client.create_collection(
                name=config.collection_name,
                embedding_function=embedding_func,
                metadata=collection_metadata,
            )
            print(f"Created collection: {config.collection_name}")

        # DuckDB dual-write: create dataset + register vector collection
        _duckdb_dataset_name = sync_dataset_and_collection(
            config.collection_name,
            collection_metadata,
            model_config=model_config,
            embedding_dim=embedding_dim,
        )

        # Determine metadata columns (default to all non-embedded columns)
        metadata_columns = config.metadata_columns
        if metadata_columns is None:
            first_row = rows[0]
            # Exclude embedded columns, the id column (if used as ChromaDB id),
            # and the internal split tag (stored separately as source_split).
            exclude_cols = set(columns)
            exclude_cols.add(_SPLIT_KEY)
            if config.id_column:
                exclude_cols.add(config.id_column)
            metadata_columns = [k for k in first_row.keys() if k not in exclude_cols]

        # Register job start (only if not resuming, or if resuming from scratch)
        if not config.resume or len(existing_ids) == 0:
            job_state.start_job(
                collection_name=config.collection_name,
                job_type="huggingface",
                total_expected=len(rows),
                total_batches=total_batches,
                config=_config_to_dict(config),
            )

        # Emit initial progress so subscribers know the job has started
        emit_progress_sync(
            job_id=config.collection_name,
            status="running",
            items_processed=len(existing_ids),
            total_items=len(rows),
            current_batch=0,
            total_batches=total_batches,
            message="Starting embedding...",
        )

        # Process in batches
        total_embedded = len(existing_ids)  # Start from existing count if resuming
        new_embedded = 0
        id_deduplicator = IDDeduplicator()
        batches_completed = 0

        for batch_start in tqdm(
            range(0, len(rows), config.batch_size), desc="Embedding batches", unit="batch"
        ):
            # Cooperative cancellation check between batches
            if cancel_event is not None and cancel_event.is_set():
                job_state.fail_job(config.collection_name, "Cancelled by user")
                emit_progress_sync(
                    job_id=config.collection_name,
                    status="failed",
                    items_processed=total_embedded,
                    total_items=len(rows),
                    current_batch=batches_completed,
                    total_batches=total_batches,
                    error="Cancelled by user",
                )
                duration = time.time() - start_time
                return EmbeddingResult(
                    collection_name=config.collection_name,
                    total_embedded=total_embedded,
                    embedding_dim=embedding_dim,
                    device=device,
                    duration_seconds=duration,
                    embedding_provider=model_config.provider.value,
                    embedding_model=model_config.model_name,
                )

            batch = rows[batch_start : batch_start + config.batch_size]

            ids = []
            documents = []
            metadatas = []

            for i, row in enumerate(batch):
                row_idx = batch_start + i
                row_split = row.get(_SPLIT_KEY, config.split)

                # Generate ID (per-row split keeps auto-ids unique across splits)
                if config.id_column and config.id_column in row:
                    base_id = str(row[config.id_column])
                    doc_id = id_deduplicator.get_unique_id(base_id)
                else:
                    doc_id = f"{config.collection_name}_{row_split}_{row_idx}"

                # Skip if already embedded (resume mode)
                if doc_id in existing_ids:
                    continue

                # Format text for embedding
                text = format_text_for_embedding(row, columns, config.text_template)

                # Skip empty texts
                if not text.strip():
                    continue

                # Extract metadata (only add source_split, no duplicated source info)
                metadata = extract_metadata(row, metadata_columns, source_split=row_split)
                metadata = preprocess_color_metadata(metadata, row)
                metadata["row_index"] = row_idx

                ids.append(doc_id)
                documents.append(text)
                metadatas.append(metadata)

            if ids:
                # Compute embeddings explicitly
                batch_embeddings = embedding_func(documents)

                # ChromaDB: vectors only
                collection.add(
                    ids=ids,
                    embeddings=batch_embeddings,
                )

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
                    batches_completed=batches_completed,
                )
                # Emit progress to WebSocket subscribers
                emit_progress_sync(
                    job_id=config.collection_name,
                    status="running",
                    items_processed=total_embedded,
                    total_items=len(rows),
                    current_batch=batches_completed,
                    total_batches=total_batches,
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
            total_batches=total_batches,
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
            embedding_model=model_config.model_name,
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
            error=str(e),
        )

        # Get model info for error response
        model_config = config.embedding_model or EmbeddingModelConfig()
        return EmbeddingResult(
            collection_name=config.collection_name,
            total_embedded=0,
            embedding_dim=0,
            device=get_device(),
            duration_seconds=time.time() - start_time,
            error=str(e),
            embedding_provider=model_config.provider.value,
            embedding_model=model_config.model_name,
        )
