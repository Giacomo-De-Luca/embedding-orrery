"""Utility modules for embedding operations."""

from .batch_utils import sort_items_by_length
from .compute_projections import compute_projections_for_collection
from .id_utils import IDDeduplicator
from .text_processing import extract_metadata, format_text_for_embedding

__all__ = [
    "format_text_for_embedding",
    "extract_metadata",
    "compute_projections_for_collection",
    "IDDeduplicator",
    "sort_items_by_length",
]
