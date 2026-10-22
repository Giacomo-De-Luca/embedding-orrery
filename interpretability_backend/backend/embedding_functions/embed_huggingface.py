"""
HuggingFace dataset embedding.

This module handles embedding HuggingFace datasets into ChromaDB.
Supports resume capability for interrupted jobs. The batch loop's job state,
progress, cancellation and result come from ``embedding_job``.
"""

import json
import threading
import time
from collections.abc import Callable

import chromadb
from chromadb.config import Settings

from ..clients.huggingface_client import (
    HFMediaFeatures,
    PortionConfig,
    PortionStrategy,
    get_media_column_kinds,
    load_dataset_portion,
)
from ..services.progress_emitter import emit_progress_sync
from ..utils.batch_utils import sort_items_by_length
from ..utils.color_preprocessing import preprocess_color_metadata
from ..utils.duckdb_sync import sync_dataset_and_collection, sync_items
from ..utils.embedding_tasks import EmbeddingTexts
from ..utils.id_utils import IDDeduplicator
from ..utils.text_processing import extract_metadata
from .config import (
    DB_PATH,
    DataType,
    EmbeddingConfig,
    EmbeddingModelConfig,
    EmbeddingResult,
)
from .create_embedding_function import create_embedding_function, get_device
from .embed_media import MediaSource, MultimodalEmbeddingRun
from .embedding_job import EmbeddingBatchJob, open_embedding_collection, report_embedding_failure

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


def _portion_label(portion: PortionConfig | None) -> str:
    """Short description of the embedded portion for the collection metadata."""
    if portion is None:
        return "all"
    if portion.strategy == PortionStrategy.FIRST_N:
        return f"first_{portion.n}"
    if portion.strategy == PortionStrategy.RANDOM_SAMPLE:
        return f"random_{portion.n}_seed{portion.seed}"
    if portion.strategy == PortionStrategy.ROW_RANGE:
        return f"range_{portion.start}_{portion.end}"
    return "all"


def _chroma_client():
    db_path = str(DB_PATH.resolve())
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))


def _embed_media(
    config: EmbeddingConfig,
    splits: list[str],
    rows: list[dict],
    total_in_split: int,
    device: str,
    start_time: float,
    progress_callback: Callable[[int, int], None] | None,
    cancel_event: threading.Event | None,
) -> EmbeddingResult:
    """Embed the Image / Audio / Video (and optional text) columns of the loaded rows."""
    source = MediaSource(
        job_type="huggingface",
        collection_metadata={
            "description": f"Media embeddings from HuggingFace dataset: {config.dataset_id}",
            "source_dataset": config.dataset_id,
            "source_config": config.config or "",
            "source_split": ", ".join(splits),
            "portion_strategy": _portion_label(config.portion),
            "total_in_split": total_in_split,
        },
        auto_id=lambda row, row_idx: (
            f"{config.collection_name}_{row.get(_SPLIT_KEY, config.split)}_{row_idx}"
        ),
        # The dataset's own Image / Audio / Video types: sniffing alone reads
        # audio in a generic MP4 container as video
        column_hints=get_media_column_kinds(config.dataset_id, config.config),
        excluded_columns=frozenset({_SPLIT_KEY}),
        row_metadata=lambda row: {"source_split": row.get(_SPLIT_KEY, config.split)},
        open_path=HFMediaFeatures.read_file,
    )
    return MultimodalEmbeddingRun(
        client=_chroma_client(),
        config=config,
        rows=rows,
        total=total_in_split,
        device=device,
        start_time=start_time,
        source=source,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    ).run()


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
        cancel_event: Checked between batches; when set, stops and leaves the
            job resumable

    Returns:
        EmbeddingResult with statistics
    """
    start_time = time.time()
    model_config = config.embedding_model or EmbeddingModelConfig()

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

        if config.data_type == DataType.MULTIMODAL:
            return _embed_media(
                config,
                splits,
                rows,
                total_in_split,
                device,
                start_time,
                progress_callback,
                cancel_event,
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

        client = _chroma_client()

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
            "description": f"Embeddings from HuggingFace dataset: {config.dataset_id}",
            "source_dataset": config.dataset_id,
            "source_config": config.config or "",
            "source_split": ", ".join(splits),
            "embedded_columns": json.dumps(columns),
            "portion_strategy": _portion_label(config.portion),
            "total_in_split": total_in_split,
            **model_config.collection_metadata(embedding_dim),
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

        job = EmbeddingBatchJob(
            collection_name=config.collection_name,
            job_type="huggingface",
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
                text = texts.document(row)

                # Skip empty texts
                if not text.strip():
                    continue

                # Extract metadata (only add source_split, no duplicated source info)
                metadata = extract_metadata(row, metadata_columns, source_split=row_split)
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

    except Exception as e:
        return report_embedding_failure(
            config.collection_name,
            e,
            start_time=start_time,
            device=get_device(),
            model_config=model_config,
        )
