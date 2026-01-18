"""
Local file text embedding using SentenceTransformers and other providers.

This module handles embedding text from local files (parquet, JSON, CSV, TSV)
into ChromaDB using various embedding providers.
"""

import chromadb
from chromadb.config import Settings
import time
import json
from typing import Optional, List, Dict, Callable
from tqdm import tqdm

from .config import (
    DB_PATH,
    TEXT_EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
    DataType,
    EmbeddingModelConfig,
    EmbeddingResult,
    LocalFileEmbeddingConfig,
)
from .create_embedding_function import create_embedding_function, get_device
from ..utils.text_processing import format_text_for_embedding, extract_metadata
from ..utils.id_utils import IDDeduplicator
from .embed_images import embed_images
from .embed_vectors import embed_vectors


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

        # Delete existing collection
        try:
            client.delete_collection(name=config.collection_name)
            print(f"Deleted existing collection: {config.collection_name}")
        except Exception:
            pass

        # Handle different data types
        if config.data_type == DataType.VECTOR:
            return embed_vectors(
                client, config, rows, total, device, start_time, progress_callback
            )
        elif config.data_type == DataType.IMAGE:
            return embed_images(
                client, config, rows, total, device, start_time, progress_callback
            )
        else:  # TEXT
            return embed_text_from_local(
                client, config, rows, total, device, start_time, progress_callback
            )

    except Exception as e:
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

    # Create embedding function using factory
    embedding_func, embedding_dim = create_embedding_function(model_config, device)
    print(f"Using embedding model: {model_config.provider.value} / {model_config.model_name}")

    # Create collection
    collection_metadata = {
        "description": f"Embeddings from local file: {config.file_path}",
        "source_file": config.file_path,
        "embedded_columns": json.dumps(columns),
        "data_type": "text",
        "embedding_provider": model_config.provider.value,
        "embedding_model": model_config.model_name,
        "embedding_dim": embedding_dim,
        "total_in_file": total,
        "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
    }

    collection = client.create_collection(
        name=config.collection_name,
        embedding_function=embedding_func,
        metadata=collection_metadata
    )

    # Determine metadata columns (exclude embedded columns and id column)
    metadata_columns = config.metadata_columns
    if metadata_columns is None:
        first_row = rows[0]
        exclude_cols = set(columns)
        if config.id_column:
            exclude_cols.add(config.id_column)
        metadata_columns = [k for k in first_row.keys() if k not in exclude_cols]

    # Process in batches
    total_embedded = 0
    id_deduplicator = IDDeduplicator()

    for batch_start in tqdm(range(0, len(rows), EMBEDDING_BATCH_SIZE),
                            desc="Embedding batches", unit="batch"):
        batch = rows[batch_start:batch_start + EMBEDDING_BATCH_SIZE]

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

            # Format text
            text = format_text_for_embedding(row, columns, config.text_template)
            if not text.strip():
                continue

            # Extract metadata using shared function
            metadata = extract_metadata(row, metadata_columns)
            metadata["row_index"] = row_idx

            ids.append(doc_id)
            documents.append(text)
            metadatas.append(metadata)

        if ids:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            total_embedded += len(ids)

        if progress_callback:
            progress_callback(min(batch_start + EMBEDDING_BATCH_SIZE, len(rows)), len(rows))

    duration = time.time() - start_time
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


# Alias for backward compatibility
_embed_text_from_local = embed_text_from_local
