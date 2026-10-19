"""
Image embedding using ViT models.

This module handles embedding images into vector representations using
Vision Transformer (ViT) models from HuggingFace transformers.
"""

import threading
import time
from collections.abc import Callable

from ..utils.color_preprocessing import preprocess_color_metadata
from ..utils.duckdb_sync import sync_dataset_and_collection, sync_items
from ..utils.text_processing import extract_metadata, format_text_for_embedding
from .config import (
    IMAGE_BATCH_SIZE,
    IMAGE_EMBEDDING_DIMENSIONS,
    IMAGE_MODEL_NAME,
    EmbeddingResult,
    LocalFileEmbeddingConfig,
)
from .embedding_job import EmbeddingBatchJob, open_embedding_collection


def embed_images(
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
    Embed images using a ViT model.

    Args:
        client: ChromaDB client
        config: Embedding configuration
        rows: List of row dictionaries containing image data
        total: Total number of rows in the source file
        device: Device string (for result metadata)
        start_time: Start time for duration calculation
        progress_callback: Optional progress callback
        cancel_event: Checked between batches; when set, stops and leaves the
            job resumable

    Returns:
        EmbeddingResult with statistics
    """
    image_column = config.image_column
    if not image_column:
        # Try to find an image column
        first_row = rows[0]
        for k, v in first_row.items():
            if isinstance(v, (bytes, dict)) or (
                isinstance(v, str)
                and any(ext in v.lower() for ext in [".jpg", ".png", ".jpeg", ".gif", ".webp"])
            ):
                image_column = k
                break

    if not image_column or image_column not in rows[0]:
        return EmbeddingResult(
            collection_name=config.collection_name,
            total_embedded=0,
            embedding_dim=IMAGE_EMBEDDING_DIMENSIONS,
            device=device,
            duration_seconds=time.time() - start_time,
            error="No image column found or specified",
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
            error=f"Missing dependencies for image embedding: {e}",
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
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    collection, existing_ids = open_embedding_collection(
        client, config.collection_name, resume=config.resume, metadata=collection_metadata
    )

    # DuckDB dual-write: create dataset + register vector collection
    _duckdb_dataset_name = sync_dataset_and_collection(
        config.collection_name,
        collection_metadata,
        embedding_dim=IMAGE_EMBEDDING_DIMENSIONS,
    )

    # Determine text columns for document text
    text_columns = config.columns or []
    if not text_columns:
        first_row = rows[0]
        text_columns = [k for k, v in first_row.items() if isinstance(v, str) and k != image_column]

    # Determine metadata columns (exclude image column and id column)
    metadata_columns = config.metadata_columns
    if metadata_columns is None:
        first_row = rows[0]
        exclude_cols = {image_column}
        if config.id_column:
            exclude_cols.add(config.id_column)
        exclude_cols.update(text_columns)
        metadata_columns = [k for k in first_row.keys() if k not in exclude_cols]

    job = EmbeddingBatchJob(
        collection_name=config.collection_name,
        job_type="local_file",
        job_config=config.to_job_config(),
        n_rows=len(rows),
        batch_size=IMAGE_BATCH_SIZE,
        embedding_dim=IMAGE_EMBEDDING_DIMENSIONS,
        device=device,
        start_time=start_time,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        description="Embedding images",
    )
    job.start(already_embedded=len(existing_ids), resume=config.resume)

    for batch_start, batch_end in job.batches():
        # Load the batch's images, skipping rows a resumed job already embedded
        loaded = []  # (row_idx, doc_id, image)
        for row_idx in range(batch_start, batch_end):
            row = rows[row_idx]
            if config.id_column and config.id_column in row:
                doc_id = str(row[config.id_column])
            else:
                doc_id = f"{config.collection_name}_{row_idx}"
            if doc_id in existing_ids:
                continue
            try:
                loaded.append((row_idx, doc_id, load_image(row[image_column])))
            except Exception as e:
                print(f"Warning: Could not load image at row {row_idx}: {e}")

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        if loaded:
            with torch.no_grad():
                features = pipe([image for _, _, image in loaded], return_tensors=True)

            for (row_idx, doc_id, _), feat in zip(loaded, features, strict=True):
                row = rows[row_idx]
                # Average pool if needed
                if len(feat.shape) == 3:
                    feat = feat.mean(1)

                # Create document text
                doc_text = (
                    format_text_for_embedding(row, text_columns, None)
                    if text_columns
                    else f"Image {row_idx}"
                )

                # Extract metadata using shared function
                metadata = extract_metadata(row, metadata_columns)
                metadata = preprocess_color_metadata(metadata, row)
                metadata["row_index"] = row_idx

                ids.append(doc_id)
                embeddings.append(feat.squeeze().cpu().numpy().tolist())
                documents.append(doc_text)
                metadatas.append(metadata)

            # ChromaDB: vectors only
            collection.add(ids=ids, embeddings=embeddings)

            # DuckDB: documents + metadata
            if _duckdb_dataset_name:
                sync_items(_duckdb_dataset_name, ids, documents, metadatas)

        job.record(added=len(ids), end=batch_end)

    return job.result()
