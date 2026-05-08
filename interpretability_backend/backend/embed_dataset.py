"""
Generalized embedding function for datasets.

This module is a facade that re-exports the embedding infrastructure:
- Configuration dataclasses from embedding_functions.config
- Embedding provider factory from embedding_functions.create_embedding_function
- Embedding implementations from embedding_functions package

Projection utilities are in utils/compute_projections.
"""

# ========== Re-exports from embedding_functions.config ==========
# Re-export PortionConfig and PortionStrategy for convenience
from .clients.huggingface_client import PortionConfig, PortionStrategy
from .embedding_functions.config import (
    # Constants
    DB_PATH,
    EMBEDDING_BATCH_SIZE,
    IMAGE_BATCH_SIZE,
    IMAGE_EMBEDDING_DIMENSIONS,
    IMAGE_MODEL_NAME,
    TEXT_EMBEDDING_DIMENSIONS,
    TEXT_MODEL_NAME,
    # Enums
    DataType,
    EmbeddingConfig,
    # Dataclasses
    EmbeddingModelConfig,
    EmbeddingProvider,
    EmbeddingResult,
    LocalFileEmbeddingConfig,
)

# ========== Re-exports from embedding_functions.create_embedding_function ==========
from .embedding_functions.create_embedding_function import (
    create_embedding_function,
    get_device,
)

# ========== Re-exports from embedding implementations ==========
from .embedding_functions.embed_huggingface import embed_huggingface_dataset
from .embedding_functions.embed_local_file import embed_local_file

# ========== Re-exports from utils ==========
from .utils.compute_projections import compute_projections_for_collection

# Internal functions for backward compatibility

__all__ = [
    # Constants
    "DB_PATH",
    "TEXT_MODEL_NAME",
    "IMAGE_MODEL_NAME",
    "TEXT_EMBEDDING_DIMENSIONS",
    "IMAGE_EMBEDDING_DIMENSIONS",
    "EMBEDDING_BATCH_SIZE",
    "IMAGE_BATCH_SIZE",
    # Enums
    "DataType",
    "EmbeddingProvider",
    # Dataclasses
    "EmbeddingModelConfig",
    "EmbeddingResult",
    "EmbeddingConfig",
    "LocalFileEmbeddingConfig",
    # Functions
    "create_embedding_function",
    "get_device",
    # Embedding implementations
    "embed_huggingface_dataset",
    "embed_local_file",
    "compute_projections_for_collection",
    # HuggingFace client types
    "PortionConfig",
    "PortionStrategy",
]
