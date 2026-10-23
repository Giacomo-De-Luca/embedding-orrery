"""One multimodal model input per dataset row.

``MediaRowBuilder.build(row)`` decodes the row's media cells and combines them
with its text into one input dict for sentence-transformers, in a fixed key
order (text, image, video, audio; sentence-transformers follows dict key
order). It also gives what the embedder stores: the document (the row's text,
or labels such as ``image: cat.jpg`` for media-only rows) and a small
reference per media cell (its path or URL, or a placeholder such as
``[image 34 KB]``) that replaces the raw value in the item metadata.

Task prefixes go on the text only: the input's ``text`` is
``EmbeddingTexts.model_input`` (the prefix rendered when the config has one),
media never get one, and the stored document never contains it.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from .embedding_tasks import EmbeddingTexts
from .media_kinds import MediaKind, MediaKindDetector, MediaReference

INPUT_KEYS = ("image", "video", "audio")  # after "text"


class MediaCellDecoder(Protocol):
    def decode(self, value: Any, kind: MediaKind) -> Any: ...


@dataclass(frozen=True)
class MediaColumn:
    """A selected media column and the kind of media it holds."""

    name: str
    kind: MediaKind

    @classmethod
    def detect(
        cls,
        rows: Sequence[Mapping[str, Any]],
        names: Iterable[str],
        hints: Mapping[str, str] | None = None,
        detector: MediaKindDetector | None = None,
        sample_size: int = 20,
    ) -> list[Self]:
        """The kind of each named column, from dataset dtype ``hints`` or the values.

        Raises ``ValueError`` for no columns, a column no row has, or one whose
        values are not recognisable media.
        """
        names = list(dict.fromkeys(names))
        if not names:
            raise ValueError("Select at least one media column (images, audio or video)")
        detector = detector or MediaKindDetector()
        hints = hints or {}
        present = set().union(*(row.keys() for row in rows)) if rows else set()
        columns = []
        for name in names:
            if name not in present:
                raise ValueError(f"Media column {name!r} is not in the data")
            values = (row.get(name) for row in rows)
            kind = detector.kind_of_column(values, hint=hints.get(name), sample_size=sample_size)
            if kind is None:
                raise ValueError(
                    f"Column {name!r} holds no recognisable images, audio or video "
                    "(expected bytes, file paths or URLs with a media extension)"
                )
            columns.append(cls(name, kind))
        return columns


@dataclass
class MediaRow:
    """The model input of one row and what the embedder stores for it."""

    embed_input: dict[str, Any]
    document: str
    references: dict[str, str]
    has_text: bool


class MediaRowBuilder:
    """Build ``MediaRow``s for one job (see module docstring)."""

    def __init__(
        self,
        columns: Sequence[MediaColumn],
        decoder: MediaCellDecoder,
        texts: EmbeddingTexts | None = None,
    ):
        self.columns = list(columns)
        self.decoder = decoder
        self.texts = texts

    @property
    def kinds(self) -> tuple[MediaKind, ...]:
        """The media kinds of the job, in input order."""
        present = {column.kind for column in self.columns}
        return tuple(MediaKind(key) for key in INPUT_KEYS if key in present)

    def build(self, row: Mapping[str, Any]) -> MediaRow | None:
        """The row's input, or None when it has neither text nor media.

        Raises ``MediaDecodeError`` when a media cell cannot be decoded.
        """
        document = self.texts.document(row) if self.texts is not None else ""
        has_text = bool(document.strip())
        decoded: dict[str, list[Any]] = {key: [] for key in INPUT_KEYS}
        references: dict[str, str] = {}
        labels: list[str] = []
        for column in self.columns:
            value = row.get(column.name)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            decoded[column.kind.value].append(self.decoder.decode(value, column.kind))
            references[column.name] = MediaReference.describe(value, column.kind)
            labels.append(MediaReference.label(value, column.kind))
        if not has_text and not references:
            return None

        embed_input: dict[str, Any] = {}
        if has_text:
            embed_input["text"] = self.texts.model_input(row, document)
        for key in INPUT_KEYS:
            values = decoded[key]
            if values:
                embed_input[key] = values[0] if len(values) == 1 else values
        return MediaRow(
            embed_input=embed_input,
            document=document if has_text else "; ".join(labels),
            references=references,
            has_text=has_text,
        )
