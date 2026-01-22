"""
HuggingFace dataset embedding.

This module handles embedding HuggingFace datasets into ChromaDB.
"""

import chromadb
from chromadb.config import Settings
import time
import json
from typing import Optional, Callable
from tqdm import tqdm

from .config import (
    DB_PATH,
    DB_PATH,
    EmbeddingConfig,
    EmbeddingModelConfig,
    EmbeddingResult,
)
from .create_embedding_function import create_embedding_function, get_device
from ..clients.huggingface_client import (
    PortionStrategy,
    load_dataset_portion,
)
from ..utils.text_processing import format_text_for_embedding, extract_metadata
from ..utils.id_utils import IDDeduplicator
from ..utils.batch_utils import sort_items_by_length


def embed_huggingface_dataset(
    config: EmbeddingConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None
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

    try:
        device = get_device()
        print(f"Using device: {device}")

        # Load dataset portion
        print(f"Loading dataset: {config.dataset_id}")
        rows, total_in_split = load_dataset_portion(
            dataset_id=config.dataset_id,
            config=config.config,
            split=config.split,
            portion=config.portion
        )
        print(f"Loaded {len(rows)} rows (total in split: {total_in_split})")

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
                embedding_model=model_config.model_name
            )

        # Determine columns to embed
        columns = config.columns
        if not columns:
            # Auto-detect text columns
            first_row = rows[0]
            columns = [k for k, v in first_row.items() if isinstance(v, str)]
            if not columns:
                return EmbeddingResult(
                    collection_name=config.collection_name,
                    total_embedded=0,
                    embedding_dim=0,
                    device=device,
                    duration_seconds=time.time() - start_time,
                    error="No text columns found in dataset",
                    embedding_provider=model_config.provider.value,
                    embedding_model=model_config.model_name
                )
        print(f"Embedding columns: {columns}")

        # Sort rows by text length for efficient batching (reduces padding waste)
        rows = sort_items_by_length(
            rows,
            lambda row: format_text_for_embedding(row, columns, config.text_template)
        )
        print(f"Sorted {len(rows)} rows by text length for efficient batching")

        # Initialize ChromaDB
        db_path = str(DB_PATH.resolve())
        DB_PATH.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create embedding function using factory (model_config already set above)
        embedding_func, embedding_dim = create_embedding_function(model_config, device)
        print(f"Using embedding model: {model_config.provider.value} / {model_config.model_name}")

        # Delete existing collection if it exists
        try:
            client.delete_collection(name=config.collection_name)
            print(f"Deleted existing collection: {config.collection_name}")
        except Exception:
            pass

        # Create collection with metadata
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
            "source_split": config.split,
            "embedded_columns": json.dumps(columns),
            "portion_strategy": portion_info,
            "total_in_split": total_in_split,
            "embedding_provider": model_config.provider.value,
            "embedding_model": model_config.model_name,
            "embedding_dim": embedding_dim,
            "embedding_task": model_config.task,  # QWEN: query instruction (used at query time)
            "embedding_task_type": model_config.task_type,  # Gemini: optimization type
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Filter out None values from metadata (ChromaDB doesn't like them)
        collection_metadata = {k: v for k, v in collection_metadata.items() if v is not None}

        collection = client.create_collection(
            name=config.collection_name,
            embedding_function=embedding_func,
            metadata=collection_metadata
        )
        print(f"Created collection: {config.collection_name}")

        # Determine metadata columns (default to all non-embedded columns)
        metadata_columns = config.metadata_columns
        if metadata_columns is None:
            first_row = rows[0]
            # Exclude embedded columns and the id column (if used as ChromaDB id)
            exclude_cols = set(columns)
            if config.id_column:
                exclude_cols.add(config.id_column)
            metadata_columns = [k for k in first_row.keys() if k not in exclude_cols]

        # Process in batches
        total_embedded = 0
        id_deduplicator = IDDeduplicator()

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
                    doc_id = f"{config.collection_name}_{config.split}_{row_idx}"

                # Format text for embedding
                text = format_text_for_embedding(row, columns, config.text_template)

                # Skip empty texts
                if not text.strip():
                    continue

                # Extract metadata (only add source_split, no duplicated source info)
                metadata = extract_metadata(row, metadata_columns, source_split=config.split)
                metadata["row_index"] = row_idx

                ids.append(doc_id)
                documents.append(text)
                metadatas.append(metadata)

            if ids:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                total_embedded += len(ids)

            if progress_callback:
                progress_callback(min(batch_start + config.batch_size, len(rows)), len(rows))

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

    except Exception as e:
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
            embedding_model=model_config.model_name
        )
