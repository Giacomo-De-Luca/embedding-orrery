"""
Re-embed an existing DuckDB dataset with a new embedding model.

Reads documents from a DuckDB items table and embeds them into a new ChromaDB
vector collection. Does NOT create a new dataset or insert items into DuckDB —
only ChromaDB gets new vectors and a new vector_collections row is registered.
The batch loop's job state, progress, cancellation and result come from
``embedding_job``.
"""

import threading
import time
from collections.abc import Callable

import chromadb
from chromadb.config import Settings

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
from .embedding_job import EmbeddingBatchJob, open_embedding_collection, report_embedding_failure


def embed_existing_dataset(
    config: ReEmbedConfig,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> EmbeddingResult:
    """
    Re-embed an existing DuckDB dataset with a new embedding model.

    Reads documents from the source dataset's items table, embeds them with
    the configured model, and writes vectors to a new ChromaDB collection.
    Registers the new vector_collection against the existing dataset.

    Args:
        config: Re-embedding configuration (source dataset + new model)
        progress_callback: Optional callback for progress updates
        cancel_event: Checked between batches; when set, stops and leaves
            the job resumable

    Returns:
        EmbeddingResult with statistics
    """
    # Deferred: API.duckdb_instance imports the API package, which imports
    # this module through the embedding pipeline (circular at module level).
    from ..API.duckdb_instance import get_duckdb_client

    start_time = time.time()
    device = get_device()
    model_config = config.embedding_model or EmbeddingModelConfig()
    batch_size = config.batch_size or EMBEDDING_BATCH_SIZE

    try:
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

        # Step 4: Open (or resume) the new ChromaDB collection
        db_path = str(DB_PATH.resolve())
        DB_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )
        collection, existing_ids = open_embedding_collection(
            client,
            config.collection_name,
            resume=config.resume,
            metadata={
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
            },
            embedding_function=embedding_func,
        )

        # Step 5: Register vector collection in DuckDB (dataset_name ≠ collection_name)
        # Check if already registered (resume case)
        if not db.get_vector_collection(config.collection_name):
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

        # Step 6: Batch-embed and write to ChromaDB only
        job = EmbeddingBatchJob(
            collection_name=config.collection_name,
            job_type="re_embed",
            job_config=config.to_job_config(),
            n_rows=len(all_items),
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            device=device,
            start_time=start_time,
            model_config=model_config,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            description="Re-embedding batches",
        )
        job.start(already_embedded=len(existing_ids), resume=config.resume)

        for batch_start, batch_end in job.batches():
            ids = []
            documents = []

            for item in all_items[batch_start:batch_end]:
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
                # Write to ChromaDB only (items already exist in DuckDB)
                collection.add(ids=ids, embeddings=embedding_func(documents))

            job.record(added=len(ids), end=batch_end)

        return job.result()

    except Exception as e:
        return report_embedding_failure(
            config.collection_name,
            e,
            start_time=start_time,
            device=device,
            model_config=model_config,
        )
