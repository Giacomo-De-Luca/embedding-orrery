"""
Image embedding using ViT models.

This module handles embedding images into vector representations using
Vision Transformer (ViT) models from HuggingFace transformers.
"""

import time
from typing import Optional, List, Dict, Callable
from tqdm import tqdm

from .config import (
    IMAGE_MODEL_NAME,
    IMAGE_EMBEDDING_DIMENSIONS,
    IMAGE_BATCH_SIZE,
    EmbeddingResult,
    LocalFileEmbeddingConfig,
)
from ..utils.text_processing import format_text_for_embedding, extract_metadata
from ..utils.color_preprocessing import preprocess_color_metadata
from ..utils.duckdb_sync import sync_dataset_and_collection, sync_items


def embed_images(
    client,
    config: LocalFileEmbeddingConfig,
    rows: List[Dict],
    total: int,
    device: str,
    start_time: float,
    progress_callback: Optional[Callable] = None
) -> EmbeddingResult:
    """
    Embed images using a ViT model.

    Args:
        client: ChromaDB client
        config: Embedding configuration
        rows: List of row dictionaries containing image data
        total: Total number of rows in the source file
        device: Device string (for result metadata)
        start_time: Start time for duration calculation
        progress_callback: Optional progress callback

    Returns:
        EmbeddingResult with statistics
    """
    image_column = config.image_column
    if not image_column:
        # Try to find an image column
        first_row = rows[0]
        for k, v in first_row.items():
            if isinstance(v, (bytes, dict)) or (isinstance(v, str) and
                any(ext in v.lower() for ext in ['.jpg', '.png', '.jpeg', '.gif', '.webp'])):
                image_column = k
                break

    if not image_column or image_column not in rows[0]:
        return EmbeddingResult(
            collection_name=config.collection_name,
            total_embedded=0,
            embedding_dim=IMAGE_EMBEDDING_DIMENSIONS,
            device=device,
            duration_seconds=time.time() - start_time,
            error="No image column found or specified"
        )

    print(f"Embedding images from column '{image_column}'")

    try:
        from io import BytesIO
        import torch
        from PIL import Image
        from transformers import pipeline
    except ImportError as e:
        return EmbeddingResult(
            collection_name=config.collection_name,
            total_embedded=0,
            embedding_dim=IMAGE_EMBEDDING_DIMENSIONS,
            device=device,
            duration_seconds=time.time() - start_time,
            error=f"Missing dependencies for image embedding: {e}"
        )

    # Load image embedding model
    print(f"Loading image model: {IMAGE_MODEL_NAME}")
    pipe = pipeline("image-feature-extraction", model=IMAGE_MODEL_NAME, device_map="auto")

    def load_image(value):
        """Load image from various formats."""
        if isinstance(value, bytes):
            return Image.open(BytesIO(value)).convert("RGB")
        elif isinstance(value, dict) and "bytes" in value:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        elif isinstance(value, str):
            # Assume it's a file path
            return Image.open(value).convert("RGB")
        else:
            raise ValueError(f"Cannot load image from {type(value)}")

    # Create collection
    collection_metadata = {
        "description": f"Image embeddings from: {config.file_path}",
        "source_file": config.file_path,
        "image_column": image_column,
        "data_type": "image",
        "embedding_model": IMAGE_MODEL_NAME,
        "embedding_dim": IMAGE_EMBEDDING_DIMENSIONS,
        "total_in_file": total,
        "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
    }

    collection = client.create_collection(
        name=config.collection_name,
        metadata=collection_metadata
    )

    # DuckDB dual-write: create dataset + register vector collection
    _duckdb_ids = sync_dataset_and_collection(
        config.collection_name, collection_metadata,
        embedding_dim=IMAGE_EMBEDDING_DIMENSIONS,
    )
    _duckdb_dataset_id = _duckdb_ids[0] if _duckdb_ids else None

    # Determine text columns for document text
    text_columns = config.columns or []
    if not text_columns:
        first_row = rows[0]
        text_columns = [k for k, v in first_row.items()
                        if isinstance(v, str) and k != image_column]

    # Determine metadata columns (exclude image column and id column)
    metadata_columns = config.metadata_columns
    if metadata_columns is None:
        first_row = rows[0]
        exclude_cols = {image_column}
        if config.id_column:
            exclude_cols.add(config.id_column)
        exclude_cols.update(text_columns)
        metadata_columns = [k for k in first_row.keys() if k not in exclude_cols]

    # Process in batches
    total_embedded = 0

    for batch_start in tqdm(range(0, len(rows), IMAGE_BATCH_SIZE),
                            desc="Embedding images", unit="batch"):
        batch = rows[batch_start:batch_start + IMAGE_BATCH_SIZE]

        # Load images for batch
        images = []
        valid_indices = []
        for i, row in enumerate(batch):
            try:
                img = load_image(row[image_column])
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"Warning: Could not load image at row {batch_start + i}: {e}")

        if not images:
            continue

        # Get embeddings
        with torch.no_grad():
            features = pipe(images, return_tensors=True)
            # Average pool if needed
            embeddings_batch = []
            for feat in features:
                if len(feat.shape) == 3:
                    feat = feat.mean(1)
                embeddings_batch.append(feat.squeeze().cpu().numpy().tolist())

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for idx, emb in zip(valid_indices, embeddings_batch):
            row = batch[idx]
            row_idx = batch_start + idx

            # Generate ID
            if config.id_column and config.id_column in row:
                doc_id = str(row[config.id_column])
            else:
                doc_id = f"{config.collection_name}_{row_idx}"

            # Create document text
            doc_text = format_text_for_embedding(row, text_columns, None) if text_columns else f"Image {row_idx}"

            # Extract metadata using shared function
            metadata = extract_metadata(row, metadata_columns)
            metadata = preprocess_color_metadata(metadata, row)
            metadata["row_index"] = row_idx

            ids.append(doc_id)
            embeddings.append(emb)
            documents.append(doc_text)
            metadatas.append(metadata)

        if ids:
            # ChromaDB: vectors only
            collection.add(ids=ids, embeddings=embeddings)

            # DuckDB: documents + metadata
            if _duckdb_dataset_id:
                sync_items(_duckdb_dataset_id, ids, documents, metadatas)

            total_embedded += len(ids)

        if progress_callback:
            progress_callback(min(batch_start + IMAGE_BATCH_SIZE, len(rows)), len(rows))

    duration = time.time() - start_time
    print(f"Embedded {total_embedded} images in {duration:.2f}s")

    return EmbeddingResult(
        collection_name=config.collection_name,
        total_embedded=total_embedded,
        embedding_dim=IMAGE_EMBEDDING_DIMENSIONS,
        device=device,
        duration_seconds=duration
    )
