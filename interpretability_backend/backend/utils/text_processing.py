"""
Text processing utilities for embedding operations.

Provides ``ColumnTemplate`` (``{column}`` templates over one row: the text
template of an embed job and the task prefix of ``utils/embedding_tasks.py``),
and functions for formatting text and extracting metadata from rows.
"""

import json
import math
from collections.abc import Iterable, Mapping
from functools import lru_cache
from string import Formatter
from typing import Any

from .media_kinds import MediaReference


def _value_text(value: Any) -> str:
    """A column value as text; lists join with spaces."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return value if isinstance(value, str) else str(value)


def _is_missing(value: Any) -> bool:
    """None, NaN, or a blank string."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return isinstance(value, str) and not value.strip()


class ColumnTemplate:
    """A template with ``{column}`` placeholders, rendered over one row.

    ``str.format`` syntax (``{{`` and ``}}`` are literal braces), but a
    placeholder names a column literally: ``{Conc.M}`` is the column
    ``Conc.M``, not an attribute lookup. Empty placeholders, conversions
    (``!r``) and format specs (``:>4``) are rejected.
    """

    def __init__(self, template: str):
        self.template = template
        self._parts: list[tuple[str, str | None]] = []
        try:
            parsed = list(Formatter().parse(template))
        except ValueError as exc:
            raise ValueError(f"Invalid template {template!r}: {exc}") from exc
        for literal, name, format_spec, conversion in parsed:
            if name is not None:
                if not name:
                    raise ValueError(f"Template {template!r} has an empty {{}}: name a column")
                if format_spec or conversion:
                    raise ValueError(
                        f"Template {template!r}: placeholders take a column name only, "
                        f"got {{{name}{'!' + conversion if conversion else ''}"
                        f"{':' + format_spec if format_spec else ''}}}"
                    )
            self._parts.append((literal, name))
        self.fields: tuple[str, ...] = tuple(
            dict.fromkeys(name for _, name in self._parts if name is not None)
        )

    @staticmethod
    @lru_cache(maxsize=64)
    def parse(template: str) -> "ColumnTemplate":
        """A cached ``ColumnTemplate`` (templates render once per row)."""
        return ColumnTemplate(template)

    def missing_from(self, columns: Iterable[str]) -> list[str]:
        """The placeholders that name none of ``columns``."""
        available = set(columns)
        return [name for name in self.fields if name not in available]

    def require_columns(self, columns: Iterable[str], what: str = "Template") -> None:
        """Raise ``ValueError`` naming the placeholders that are not in ``columns``."""
        unknown = self.missing_from(columns)
        if unknown:
            raise ValueError(
                f"{what} {self.template!r} references unknown column(s): {', '.join(unknown)}"
            )

    def render(self, row: Mapping[str, Any], missing: str = "") -> str:
        """The template with each placeholder replaced by its column's value.

        Absent, None, NaN and blank values render as ``missing``.
        """
        out = []
        for literal, name in self._parts:
            out.append(literal)
            if name is not None:
                value = row.get(name)
                out.append(missing if _is_missing(value) else _value_text(value))
        return "".join(out)


def format_text_for_embedding(
    row: dict[str, Any], columns: list[str], template: str | None = None
) -> str:
    """
    Format row data into text for embedding.

    Args:
        row: Dictionary of column values
        columns: Columns to include in text
        template: Optional template string with {column_name} placeholders
            (``ColumnTemplate``); each must name one of ``columns``

    Returns:
        Formatted text string

    Raises:
        ValueError: the template is malformed or names a column outside ``columns``
    """
    if template:
        parsed = ColumnTemplate.parse(template)
        parsed.require_columns(columns, "Text template")
        return parsed.render(row)

    # Default: concatenate column values
    parts = []
    for col in columns:
        value = row.get(col)
        if value is not None:
            parts.append(_value_text(value))

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
            elif (reference := MediaReference.for_metadata(value)) is not None:
                # Its path or a placeholder: bytes never reach json.dumps
                metadata[col] = reference
            elif isinstance(value, (list, dict)):
                # Store lists and dicts as JSON strings
                metadata[col] = json.dumps(value)
            else:
                metadata[col] = str(value)[:1000]  # Truncate long values

    return metadata


# Aliases for backward compatibility (internal use)
_format_text_for_embedding = format_text_for_embedding
_extract_metadata = extract_metadata
