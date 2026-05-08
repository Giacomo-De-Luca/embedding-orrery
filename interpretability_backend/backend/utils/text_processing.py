"""
Text processing utilities for embedding operations.

Provides functions for formatting text and extracting metadata from rows.
"""

import json
from typing import Any


def format_text_for_embedding(
    row: dict[str, Any], columns: list[str], template: str | None = None
) -> str:
    """
    Format row data into text for embedding.

    Args:
        row: Dictionary of column values
        columns: Columns to include in text
        template: Optional template string with {column_name} placeholders

    Returns:
        Formatted text string
    """
    if template:
        # Use template with column placeholders
        try:
            return template.format(**{col: row.get(col, "") for col in columns})
        except KeyError:
            pass

    # Default: concatenate column values
    parts = []
    for col in columns:
        value = row.get(col)
        if value is not None:
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.append(" ".join(str(v) for v in value))
            else:
                parts.append(str(value))

    return " ".join(parts)


def extract_metadata(
    row: dict[str, Any], metadata_columns: list[str] | None, source_split: str | None = None
) -> dict[str, Any]:
    """
    Extract metadata from row for storage in ChromaDB.

    Args:
        row: Dictionary of column values
        metadata_columns: Columns to include in metadata
        source_split: Optional split name (train/test/validation) - useful when mixed

    Returns:
        Metadata dictionary (ChromaDB compatible: str, int, float, bool only)
    """
    metadata = {}

    # Only add source_split if provided and non-empty (useful for train/test distinction)
    if source_split:
        metadata["source_split"] = source_split

    # Add specified columns as metadata
    if metadata_columns:
        for col in metadata_columns:
            value = row.get(col)
            if value is None:
                continue
            # ChromaDB only supports str, int, float, bool
            if isinstance(value, (str, int, float, bool)):
                metadata[col] = value
            elif isinstance(value, (list, dict)):
                # Store lists and dicts as JSON strings
                metadata[col] = json.dumps(value)
            else:
                metadata[col] = str(value)[:1000]  # Truncate long values

    return metadata


# Aliases for backward compatibility (internal use)
_format_text_for_embedding = format_text_for_embedding
_extract_metadata = extract_metadata
