"""Tell which dataset cells hold media (image / audio / video) and describe them safely.

Cells arrive in many shapes: raw bytes (parquet), file paths or URLs (CSV),
HuggingFace ``Image/Audio/Video(decode=False)`` dicts (``{"bytes", "path"}``),
decoded HF audio (``{"array", "sampling_rate"}``) and PIL images.
``MediaKindDetector`` classifies them by magic bytes, then extension, without
importing PIL or any audio/video library (torch-free import boundary).
``MediaReference`` turns a media cell into a small JSON-safe string for item
metadata, previews and document labels, so bytes never reach ``json.dumps``.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from itertools import islice
from pathlib import PurePosixPath
from urllib.parse import urlsplit

URL_SCHEMES = ("http://", "https://")
MAX_PATH_LENGTH = 2048


class MediaKind(StrEnum):
    """Media modalities, named like the catalog's ``MEDIA_MODALITIES``."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


DEFAULT_EXTENSIONS: Mapping[MediaKind, tuple[str, ...]] = {
    MediaKind.IMAGE: (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"),
    MediaKind.AUDIO: (".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac"),
    MediaKind.VIDEO: (".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi"),
}

# ISO base-media brands (bytes 8-12 after "ftyp") that are not video.
_AUDIO_BRANDS = (b"M4A ", b"M4B ", b"M4P ")
_IMAGE_BRANDS = (b"heic", b"heix", b"mif1", b"msf1", b"avif")
# Generic brands that audio-only (.m4a) and video files both use.
_GENERIC_BRANDS = (b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"dash")


def _has_generic_brand(data: bytes) -> bool:
    return data[4:8] == b"ftyp" and data[8:12] in _GENERIC_BRANDS


def _sniff(data: bytes) -> MediaKind | None:
    """Classify encoded media by its leading magic bytes (a generic ISO brand reads as video)."""
    if data.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM")):
        return MediaKind.IMAGE
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return MediaKind.IMAGE
    if data.startswith(b"RIFF") and len(data) >= 12:
        return {b"WEBP": MediaKind.IMAGE, b"WAVE": MediaKind.AUDIO, b"AVI ": MediaKind.VIDEO}.get(
            data[8:12]
        )
    if data.startswith((b"ID3", b"fLaC", b"OggS")):
        return MediaKind.AUDIO
    if len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0:  # MP3 / ADTS frame sync
        return MediaKind.AUDIO
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in _AUDIO_BRANDS:
            return MediaKind.AUDIO
        return MediaKind.IMAGE if brand in _IMAGE_BRANDS else MediaKind.VIDEO
    if data.startswith(b"\x1a\x45\xdf\xa3"):  # EBML: WebM / Matroska
        return MediaKind.VIDEO
    return None


def _path_part(value: str) -> str | None:
    """The path of a file path or http(s) URL string, or None for other text."""
    if not value or len(value) > MAX_PATH_LENGTH or "\n" in value:
        return None
    return urlsplit(value).path if value.startswith(URL_SCHEMES) else value


def _is_pil_image(value: object) -> bool:
    return type(value).__module__.startswith("PIL.") and hasattr(value, "size")


def _is_decoded_audio(value: object) -> bool:
    return isinstance(value, Mapping) and "array" in value and "sampling_rate" in value


def _as_bytes(value: object) -> bytes | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return None


class MediaKindDetector:
    """Classify single cells and whole columns as image / audio / video media."""

    def __init__(self, extensions: Mapping[MediaKind, Iterable[str]] | None = None):
        extensions = DEFAULT_EXTENSIONS if extensions is None else extensions
        self._by_extension = {
            ext.lower(): MediaKind(kind) for kind, exts in extensions.items() for ext in exts
        }

    def kind_of(self, value: object) -> MediaKind | None:
        """The media kind of one cell, or None when it is not (recognisable) media."""
        data = _as_bytes(value)
        if data is not None:
            return _sniff(data)
        if isinstance(value, str):
            return self._kind_of_path(value)
        if _is_decoded_audio(value):
            return MediaKind.AUDIO
        if isinstance(value, Mapping) and ("bytes" in value or "path" in value):
            data = _as_bytes(value.get("bytes"))
            path = value.get("path")
            by_path = self._kind_of_path(path) if isinstance(path, str) else None
            if data and by_path is not None and _has_generic_brand(data):
                return by_path  # the extension tells an .m4a from an .mp4
            return (_sniff(data) if data else None) or by_path
        if _is_pil_image(value):
            return MediaKind.IMAGE
        return None

    def kind_of_column(
        self, values: Iterable[object], hint: str | None = None, sample_size: int = 20
    ) -> MediaKind | None:
        """The kind most sampled non-null cells share, or None if most are not media.

        ``hint`` is a dataset dtype label (HF ``"image"`` / ``"audio"`` /
        ``"video"``); a media hint wins over sniffing.
        """
        if hint in MediaKind.__members__.values():
            return MediaKind(hint)
        sample = [v for v in islice(values, sample_size) if v is not None]
        kinds = Counter(kind for kind in map(self.kind_of, sample) if kind is not None)
        if not kinds:
            return None
        kind, count = kinds.most_common(1)[0]
        return kind if count * 2 > len(sample) else None

    def _kind_of_path(self, value: str) -> MediaKind | None:
        path = _path_part(value)
        if path is None:
            return None
        return self._by_extension.get(PurePosixPath(path).suffix.lower())


class MediaReference:
    """JSON-safe strings describing a media cell (paths kept, bytes summarised)."""

    @staticmethod
    def of(value: object) -> str | None:
        """The path or URL a cell points at, if it carries one."""
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and isinstance(value.get("path"), str) and value["path"]:
            return value["path"]
        return None

    @staticmethod
    def holds_bytes(value: object) -> bool:
        """Raw bytes, or a ``{"bytes", "path"}`` dict carrying them."""
        if isinstance(value, Mapping):
            value = value.get("bytes")
        return _as_bytes(value) is not None

    @classmethod
    def placeholder(cls, value: object, kind: MediaKind | None) -> str:
        """A short description such as ``[image 34 KB]`` or ``[audio 2.0 s]``
        (``[binary 2 KB]`` for bytes of no known kind)."""
        name = kind or "binary"
        reference = cls.of(value)
        if isinstance(value, str):
            return f"[{name} {reference}]"
        data = _as_bytes(value)
        if data is None and isinstance(value, Mapping):
            data = _as_bytes(value.get("bytes"))
        if data is not None:
            return f"[{name} {cls._size(len(data))}]"
        if _is_decoded_audio(value) and value["sampling_rate"]:
            seconds = len(value["array"]) / value["sampling_rate"]
            return f"[{name} {seconds:.1f} s]"
        if _is_pil_image(value):
            width, height = value.size
            return f"[{name} {width}x{height}]"
        return f"[{name} {reference}]" if reference else f"[{name}]"

    @classmethod
    def describe(cls, value: object, kind: MediaKind | None) -> str:
        """The cell's path, else its placeholder: what metadata stores for a media cell."""
        return cls.of(value) or cls.placeholder(value, kind)

    @classmethod
    def for_metadata(cls, value: object, detector: "MediaKindDetector | None" = None) -> str | None:
        """What metadata stores for a media cell (recognisable media, or bytes of
        any kind), or None for any other value. The local loader keeps such
        cells as read, and ``extract_metadata`` stores this instead of them."""
        kind = (detector or _DEFAULT_DETECTOR).kind_of(value)
        if kind is None and not cls.holds_bytes(value):
            return None
        return cls.describe(value, kind)

    @classmethod
    def label(cls, value: object, kind: MediaKind) -> str:
        """A document label for a media-only row: ``image: cat.jpg``, or just the kind."""
        reference = cls.of(value)
        path = _path_part(reference) if reference else None
        name = PurePosixPath(path).name if path else ""
        return f"{kind}: {name}" if name else str(kind)

    @staticmethod
    def _size(n_bytes: int) -> str:
        if n_bytes < 1024:
            return f"{n_bytes} B"
        if n_bytes < 1024 * 1024:
            return f"{n_bytes / 1024:.0f} KB"
        return f"{n_bytes / (1024 * 1024):.1f} MB"


# Detects with the default extensions (MediaReference.for_metadata)
_DEFAULT_DETECTOR = MediaKindDetector()
