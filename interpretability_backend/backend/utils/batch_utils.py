"""Batch processing utilities for embedding."""

from collections.abc import Callable
from typing import Any


def sort_items_by_length(
    items: list[dict[str, Any]],
    text_key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    """
    Sort items by text length (descending) for efficient batching.

    Sorting by length reduces padding waste when embedding with
    transformer models, as sequences are padded to the longest
    in each batch.

    Args:
        items: List of row dictionaries
        text_key_fn: Function that extracts the text to measure from each item
                     e.g., lambda row: format_text_for_embedding(row, columns, template)

    Returns:
        Sorted list (longest first)
    """
    return sorted(items, key=lambda x: len(text_key_fn(x)), reverse=True)
