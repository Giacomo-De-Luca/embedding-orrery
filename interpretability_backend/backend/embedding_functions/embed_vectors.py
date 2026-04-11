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
        # Try to find a vector column
        first_row = rows[0]
        for k, v in first_row.items():
            if isinstance(v, (list, np.ndarray)) and len(v) > 10:
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

    # Get embedding dimension from first row
    first_vector = rows[0][vector_column]
    if isinstance(first_vector, list):
        embedding_dim = len(first_vector)
    elif isinstance(first_vector, np.ndarray):
        embedding_dim = first_vector.shape[0]
    else:
        embedding_dim = len(list(first_vector))

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

    for batch_start in tqdm(range(0, len(rows), EMBEDDING_BATCH_SIZE),
                            desc="Adding vectors", unit="batch"):
        batch = rows[batch_start:batch_start + EMBEDDING_BATCH_SIZE]

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, row in enumerate(batch):
            row_idx = batch_start + i

            # Get vector
            vector = row[vector_column]
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()

            # Generate ID
            if config.id_column and config.id_column in row:
                doc_id = str(row[config.id_column])
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
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
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
