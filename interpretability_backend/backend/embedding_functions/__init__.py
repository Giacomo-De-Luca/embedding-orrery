"""
Embedding function implementations.

This package contains specialized embedding implementations:
- embed_huggingface: For HuggingFace datasets
- embed_local_file: For local text files (formerly embed_sentence_transformers)
- embed_images: For image embedding with ViT
- embed_vectors: For pre-computed vector embeddings
"""

from .embed_huggingface import embed_huggingface_dataset
from .embed_images import embed_images
from .embed_local_file import embed_local_file, embed_text_from_local
from .embed_vectors import embed_vectors

__all__ = [
    "embed_huggingface_dataset",
    "embed_local_file",
    "embed_text_from_local",
    "embed_images",
    "embed_vectors",
]
