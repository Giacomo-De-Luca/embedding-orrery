"""
Local file text embedding using SentenceTransformers and other providers.

This module handles embedding text from local files (parquet, JSON, CSV, TSV)
into ChromaDB using various embedding providers, and dispatches multimodal
(image / audio / video / mixed) and pre-computed vector files to their own
embedders. Every data type supports resume and cancellation through
``embedding_job``.
"""

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import chromadb
from chromadb.config import Settings

from ..clients.local_data_client import load_local_file_portion
from ..services.progress_emitter import emit_progress_sync
from ..utils.batch_utils import sort_items_by_length
from ..utils.color_preprocessing import preprocess_color_metadata
from ..utils.duckdb_sync import sync_dataset_and_collection, sync_items
from ..utils.embedding_tasks import EmbeddingTexts
from ..utils.id_utils import IDDeduplicator
from ..utils.text_processing import extract_metadata
from .config import (
    DB_PATH,
    TEXT_EMBEDDING_DIMENSIONS,
    DataType,
    EmbeddingModelConfig,
    EmbeddingResult,
    LocalFileEmbeddingConfig,
)
from .create_embedding_function import create_embedding_function, get_device
from .embed_media import MediaSource, MultimodalEmbeddingRun
from .embed_vectors import embed_vectors
from .embedding_job import EmbeddingBatchJob, open_embedding_collection, report_embedding_failure


def embed_local_file(
    config: LocalFileEmbeddingConfig,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> EmbeddingResult:
    """
    Embed a local file (parquet/json/csv/tsv) into ChromaDB.

    Supports text, multimodal (image / audio / video / mixed) and pre-computed
    vector embeddings.

    Args:
        config: Embedding configuration
        progress_callback: Optional callback for progress updates
        cancel_event: Checked between batches; when set, stops and leaves the
            job resumable

    Returns:
        EmbeddingResult with statistics
    """
    start_time = time.time()

    try:
        device = get_device()
        print(f"Using device: {device}")

        # Load data from local file
        print(f"Loading file: {config.file_path}")
        rows, total = load_local_file_portion(
            file_path=config.file_path,
            n_rows=config.n_rows,
            sample_n=config.sample_n,
            sample_seed=config.sample_seed,
            raw_columns=config.media_columns or ()
            if config.data_type == DataType.MULTIMODAL
            else (),
        )
        print(f"Loaded {len(rows)} rows (total in file: {total})")

        if not rows:
            return EmbeddingResult.failed(
                config.collection_name,
                "No rows loaded from file",
                device=device,
                start_time=start_time,
                embedding_dim=TEXT_EMBEDDING_DIMENSIONS,
            )

        # Initialize ChromaDB
        db_path = str(DB_PATH.resolve())
        DB_PATH.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        embedder = {
            DataType.VECTOR: embed_vectors,
            DataType.MULTIMODAL: embed_media_from_local,
        }.get(config.data_type, embed_text_from_local)
        return embedder(
            client,
            config,
            rows,
            total,
            device,
            start_time,
            progress_callback,
            cancel_event=cancel_event,
        )

    except Exception as e:
        return report_embedding_failure(
            config.collection_name,
            e,
            start_time=start_time,
            device=get_device(),
            embedding_dim=TEXT_EMBEDDING_DIMENSIONS,
        )


def embed_media_from_local(
    client,
    config: LocalFileEmbeddingConfig,
    rows: list[dict],
    total: int,
    device: str,
    start_time: float,
    progress_callback: Callable | None = None,
    cancel_event: threading.Event | None = None,
) -> EmbeddingResult:
    """Embed the images / audio / video (and optional text) columns of a local file.

    Relative media paths resolve against the data file's folder.
    """
    source = MediaSource(
        job_type="local_file",
        collection_metadata={
            "description": f"Media embeddings from local file: {config.file_path}",
            "source_file": config.file_path,
            "total_in_file": total,
        },
        auto_id=lambda row, row_idx: f"{config.collection_name}_{row_idx}",
        base_dir=Path(config.file_path).expanduser().parent,
    )
    return MultimodalEmbeddingRun(
        client=client,
        config=config,
        rows=rows,
        total=total,
        device=device,
        start_time=start_time,
        source=source,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    ).run()


def embed_text_from_local(
    client,
    config: LocalFileEmbeddingConfig,
    rows: list[dict],
    total: int,
    device: str,
    start_time: float,
    progress_callback: Callable | None = None,
    cancel_event: threading.Event | None = None,
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
        cancel_event: Checked between batches

    Returns:
        EmbeddingResult with statistics
    """
    # Get model config early
    model_config = config.embedding_model or EmbeddingModelConfig()

    # Determine columns
    columns = config.columns
    if not columns:
        first_row = rows[0]
        columns = [k for k, v in first_row.items() if isinstance(v, str)]
        if not columns:
            return EmbeddingResult.failed(
                config.collection_name,
                "No text columns found",
                device=device,
                start_time=start_time,
                model_config=model_config,
            )

    print(f"Embedding columns: {columns}")

    # Stored documents vs model inputs (task prefix); fails on unknown
    # placeholders before the model loads or an old collection is replaced.
    texts = EmbeddingTexts(
        columns, config.text_template, model_config.prompt_format, model_config.prompt
    )
    texts.validate(rows)

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
    rows = sort_items_by_length(rows, texts.document)
    print(f"Sorted {len(rows)} rows by text length for efficient batching")

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

    collection_metadata = {
        "description": f"Embeddings from local file: {config.file_path}",
        "source_file": config.file_path,
        "embedded_columns": json.dumps(columns),
        "data_type": "text",
        **model_config.collection_metadata(embedding_dim),
        "total_in_file": total,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    collection, existing_ids = open_embedding_collection(
        client,
        config.collection_name,
        resume=config.resume,
        metadata=collection_metadata,
    )

    # DuckDB dual-write: create dataset + register vector collection
    _duckdb_dataset_name = sync_dataset_and_collection(
        config.collection_name,
        collection_metadata,
        model_config=model_config,
        embedding_dim=embedding_dim,
    )

    # Determine metadata columns (exclude embedded columns and id column)
    metadata_columns = config.metadata_columns
    if metadata_columns is None:
        first_row = rows[0]
        exclude_cols = set(columns)
        if config.id_column:
            exclude_cols.add(config.id_column)
        metadata_columns = [k for k in first_row.keys() if k not in exclude_cols]

    job = EmbeddingBatchJob(
        collection_name=config.collection_name,
        job_type="local_file",
        job_config=config.to_job_config(),
        n_rows=len(rows),
        batch_size=config.batch_size,
        embedding_dim=embedding_dim,
        device=device,
        start_time=start_time,
        model_config=model_config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        truncation_source=embedding_func,
    )
    job.start(already_embedded=len(existing_ids), resume=config.resume)
    id_deduplicator = IDDeduplicator()

    for batch_start, batch_end in job.batches():
        ids = []
        documents = []
        model_inputs = []
        metadatas = []

        for row_idx in range(batch_start, batch_end):
            row = rows[row_idx]

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
            text = texts.document(row)
            if not text.strip():
                continue

            # Extract metadata using shared function
            metadata = extract_metadata(row, metadata_columns)
            metadata = preprocess_color_metadata(metadata, row)
            metadata["row_index"] = row_idx

            ids.append(doc_id)
            documents.append(text)
            model_inputs.append(texts.model_input(row, text))
            metadatas.append(metadata)

        if ids:
            # Compute embeddings explicitly; ChromaDB stores vectors only
            collection.add(ids=ids, embeddings=embedding_func(model_inputs))

            # DuckDB: documents + metadata
            if _duckdb_dataset_name:
                sync_items(_duckdb_dataset_name, ids, documents, metadatas)

        job.record(added=len(ids), end=batch_end)

    return job.result()
