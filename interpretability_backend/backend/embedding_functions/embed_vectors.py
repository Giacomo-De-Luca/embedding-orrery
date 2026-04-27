"""
Pre-computed vector embedding support.

This module handles loading and storing pre-computed vector embeddings in ChromaDB.
"""

import time
import json
import numpy as np
from typing import Optional, List, Dict, Callable
from tqdm import tqdm

from .config import (
    EMBEDDING_BATCH_SIZE,
    EmbeddingResult,
    LocalFileEmbeddingConfig,
)
from ..utils.text_processing import format_text_for_embedding, extract_metadata
from ..utils.color_preprocessing import preprocess_color_metadata
from ..utils.id_utils import IDDeduplicator
from ..utils.duckdb_sync import sync_dataset_and_collection, sync_items


def _parse_vector(value) -> Optional[List[float]]:
    """Parse a vector value from various formats into a list of floats.

    Handles: list, numpy array, or string representation of a JSON array
    (common when loading from TSV/CSV where arrays are stored as text).
    Returns None if the value cannot be parsed.
    """
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('['):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
            except (json.JSONDecodeError, ValueError, TypeError):
                return None
    return None


def _is_vector_like(value) -> bool:
    """Check if a value looks like it could be a vector column."""
    if isinstance(value, (list, np.ndarray)) and len(value) >= 3:
        return True
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('[') and len(value) > 10:
            try:
                parsed = json.loads(value)
                return isinstance(parsed, list) and len(parsed) >= 3
            except (json.JSONDecodeError, ValueError):
                return False
    return False


def embed_vectors(
    client,
    config: LocalFileEmbeddingConfig,
    rows: List[Dict],
    total: int,
    device: str,
    start_time: float,
    progress_callback: Optional[Callable] = None
) -> EmbeddingResult:
    """
    Embed using pre-computed vector embeddings.

    Args:
        client: ChromaDB client
        config: Embedding configuration
        rows: List of row dictionaries
        total: Total number of rows in the source file
        device: Device string (for result metadata)
        start_time: Start time for duration calculation
        progress_callback: Optional progress callback

    Returns:
        EmbeddingResult with statistics
    """
    vector_column = config.vector_column
    if not vector_column:
        # Try to find a vector column (handles lists, numpy arrays, AND string-encoded arrays from TSV/CSV)
        first_row = rows[0]
        for k, v in first_row.items():
            if _is_vector_like(v):
                vector_column = k
                break

    if not vector_column or vector_column not in rows[0]:
        return EmbeddingResult(
            collection_name=config.collection_name,
            total_embedded=0,
            embedding_dim=0,
            device=device,
            duration_seconds=time.time() - start_time,
            error="No vector column found or specified"
        )

    # Get embedding dimension from first row (parse string vectors if needed)
    first_vector = _parse_vector(rows[0][vector_column])
    if first_vector is None:
        return EmbeddingResult(
            collection_name=config.collection_name,
            total_embedded=0,
            embedding_dim=0,
            device=device,
            duration_seconds=time.time() - start_time,
            error=f"Could not parse vector from column '{vector_column}'. Value: {str(rows[0][vector_column])[:100]}"
        )
    embedding_dim = len(first_vector)

    print(f"Using pre-computed vectors from column '{vector_column}' (dim={embedding_dim})")

    # Create collection without embedding function (we provide embeddings)
    collection_metadata = {
        "description": f"Pre-computed embeddings from: {config.file_path}",
        "source_file": config.file_path,
        "vector_column": vector_column,
        "data_type": "vector",
        "embedding_dim": embedding_dim,
        "total_in_file": total,
        "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
    }

    collection = client.create_collection(
        name=config.collection_name,
        metadata=collection_metadata
    )

    # DuckDB dual-write: create dataset + register vector collection
    _duckdb_dataset_name = sync_dataset_and_collection(
        config.collection_name, collection_metadata, embedding_dim=embedding_dim,
    )

    # Determine text columns for document text
    text_columns = config.columns or []
    if not text_columns:
        first_row = rows[0]
        text_columns = [k for k, v in first_row.items()
                        if isinstance(v, str) and k != vector_column]

    # Determine metadata columns (exclude vector column and id column)
    metadata_columns = config.metadata_columns
    if metadata_columns is None:
        first_row = rows[0]
        exclude_cols = {vector_column}
        if config.id_column:
            exclude_cols.add(config.id_column)
        # Also exclude text columns used for document
        exclude_cols.update(text_columns)
        metadata_columns = [k for k in first_row.keys() if k not in exclude_cols]

    # Process in batches
    total_embedded = 0
    id_deduplicator = IDDeduplicator()

    for batch_start in tqdm(range(0, len(rows), EMBEDDING_BATCH_SIZE),
                            desc="Adding vectors", unit="batch"):
        batch = rows[batch_start:batch_start + EMBEDDING_BATCH_SIZE]

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, row in enumerate(batch):
            row_idx = batch_start + i

            # Get vector (parse string-encoded arrays from TSV/CSV)
            vector = _parse_vector(row[vector_column])
            if vector is None:
                continue

            # Generate ID (deduplicate to avoid ChromaDB rejection)
            if config.id_column and config.id_column in row:
                base_id = str(row[config.id_column])
                doc_id = id_deduplicator.get_unique_id(base_id)
            else:
                doc_id = f"{config.collection_name}_{row_idx}"

            # Create document text
            doc_text = format_text_for_embedding(row, text_columns, config.text_template) if text_columns else ""

            # Extract metadata using shared function
            metadata = extract_metadata(row, metadata_columns)
            metadata = preprocess_color_metadata(metadata, row)
            metadata["row_index"] = row_idx

            ids.append(doc_id)
            embeddings.append(vector)
            documents.append(doc_text or f"Item {row_idx}")
            metadatas.append(metadata)

        if ids:
            # ChromaDB: vectors only
            collection.add(ids=ids, embeddings=embeddings)

            # DuckDB: documents + metadata
            if _duckdb_dataset_name:
                sync_items(_duckdb_dataset_name, ids, documents, metadatas)

            total_embedded += len(ids)

        if progress_callback:
            progress_callback(min(batch_start + EMBEDDING_BATCH_SIZE, len(rows)), len(rows))

    duration = time.time() - start_time
    print(f"Added {total_embedded} pre-computed vectors in {duration:.2f}s")

    return EmbeddingResult(
        collection_name=config.collection_name,
        total_embedded=total_embedded,
        embedding_dim=embedding_dim,
        device=device,
        duration_seconds=duration
    )
