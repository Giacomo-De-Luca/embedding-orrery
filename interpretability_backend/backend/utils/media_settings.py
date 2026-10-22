"""Media embedding settings: the per-job media options, their defaults and limits.

``config/media_embedding.json`` holds:

- ``defaults``: the ``MediaOptions`` a user can change per job from the embed
  form (image / video soft-token budgets, video sampling, audio length);
- the limits that bound those overrides;
- fixed technical settings (batching, remote-URL policy, extension lists).

The GraphQL ``mediaEmbeddingSettings`` query serves the defaults and limits to
the embed forms; overrides come back in the embed input and are validated by
``with_overrides``. EmbeddingGemma 2 shares one context window (8192 tokens)
per input across all modalities, so ``check_budget`` rejects options whose
worst case per row would not fit.
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from functools import cache
from typing import ClassVar, Self

from .json_settings import JsonFileSettings
from .media_kinds import DEFAULT_EXTENSIONS, MediaKind
from .resource_paths import PACKAGE_ROOT

MEDIA_SETTINGS_FILE = PACKAGE_ROOT / "config" / "media_embedding.json"
SOFT_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


@dataclass(frozen=True)
class MediaOptions:
    """Per-job media options; ``MediaEmbeddingSettings`` validates them against its limits."""

    image_max_soft_tokens: int = 280
    video_fps: float = 1.0
    video_max_frames: int = 32
    video_max_soft_tokens: int = 140
    max_audio_seconds: float = 120.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def processing_kwargs(self, kinds: Iterable[MediaKind | str]) -> dict[str, dict]:
        """``encode(processing_kwargs=...)`` for a job embedding these media kinds.

        Video frames arrive pre-sampled (``MediaDecoder``), so the processor
        must not sample them again.
        """
        kinds = {MediaKind(kind) for kind in kinds}
        kwargs: dict[str, dict] = {}
        if MediaKind.IMAGE in kinds:
            kwargs["image"] = {"max_soft_tokens": self.image_max_soft_tokens}
        if MediaKind.VIDEO in kinds:
            kwargs["video"] = {
                "max_soft_tokens": self.video_max_soft_tokens,
                "do_sample_frames": False,
            }
        return kwargs


OPTION_NAMES = tuple(f.name for f in fields(MediaOptions))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class MediaEmbeddingSettings(JsonFileSettings):
    """Defaults, limits and technical settings of media embedding (see module docstring)."""

    SETTINGS_LABEL: ClassVar[str] = "media embedding settings"
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = (
        "soft_token_budgets",
        "image_extensions",
        "audio_extensions",
        "video_extensions",
    )

    defaults: MediaOptions = field(default_factory=MediaOptions)
    # Limits for per-job overrides.
    soft_token_budgets: tuple[int, ...] = SOFT_TOKEN_BUDGETS
    video_fps_limit: float = 4.0
    video_frames_limit: int = 128
    audio_seconds_limit: float = 327.0
    # Token budget of one input.
    context_tokens: int = 8192
    reserved_tokens: int = 512  # text, task prompt and special tokens
    audio_tokens_per_second: int = 25
    # Technical settings.
    audio_sample_rate: int = 16000
    video_frame_max_side: int = 768
    rows_per_batch: int = 16
    encode_batch_size: int = 8
    detection_sample_rows: int = 20
    allow_remote_urls: bool = False
    remote_timeout_seconds: float = 10.0
    # Empty = the detector's defaults (media_kinds.DEFAULT_EXTENSIONS).
    image_extensions: tuple[str, ...] = ()
    audio_extensions: tuple[str, ...] = ()
    video_extensions: tuple[str, ...] = ()

    def __post_init__(self):
        budgets = self.soft_token_budgets
        if (
            not isinstance(budgets, tuple)
            or not budgets
            or not all(_is_integer(b) and b > 0 for b in budgets)
        ):
            raise ValueError("soft_token_budgets must be a non-empty list of positive integers")
        for name in ("video_fps_limit", "audio_seconds_limit", "remote_timeout_seconds"):
            if not _is_number(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive number, got {getattr(self, name)!r}")
        for name in (
            "video_frames_limit",
            "context_tokens",
            "audio_tokens_per_second",
            "audio_sample_rate",
            "video_frame_max_side",
            "rows_per_batch",
            "encode_batch_size",
            "detection_sample_rows",
        ):
            self.require_int(name, minimum=1)
        self.require_int("reserved_tokens", minimum=0)
        self.require_bool("allow_remote_urls")
        for kind in MediaKind:
            extensions = getattr(self, f"{kind}_extensions")
            if not isinstance(extensions, tuple) or not all(
                isinstance(e, str) and len(e) > 1 and e.startswith(".") for e in extensions
            ):
                raise ValueError(f"{kind}_extensions must list dotted extensions like '.png'")
            object.__setattr__(self, f"{kind}_extensions", tuple(e.lower() for e in extensions))
        defaults = self.defaults
        if isinstance(defaults, Mapping):
            defaults = self._options_from(defaults, MediaOptions())
        object.__setattr__(self, "defaults", self._validated(defaults))

    @classmethod
    @cache
    def shipped(cls) -> Self:
        """The settings in ``config/media_embedding.json``, loaded once per process."""
        return cls.from_file(MEDIA_SETTINGS_FILE)

    def with_overrides(self, overrides: Mapping[str, object] | None) -> MediaOptions:
        """The defaults with the given options replaced; ``None`` values keep the default."""
        if not overrides:
            return self.defaults
        return self._validated(self._options_from(overrides, self.defaults))

    def extensions(self) -> dict[MediaKind, tuple[str, ...]]:
        """File extensions per media kind (configured, else the detector defaults)."""
        return {
            kind: getattr(self, f"{kind}_extensions") or DEFAULT_EXTENSIONS[kind]
            for kind in MediaKind
        }

    def estimate_tokens(self, options: MediaOptions, kinds: Mapping[MediaKind | str, int]) -> int:
        """Worst-case tokens of one row with ``kinds[kind]`` media columns of each kind."""
        counts = {MediaKind(kind): count for kind, count in kinds.items()}
        audio_tokens = math.ceil(options.max_audio_seconds * self.audio_tokens_per_second)
        return (
            self.reserved_tokens
            + counts.get(MediaKind.IMAGE, 0) * options.image_max_soft_tokens
            + counts.get(MediaKind.VIDEO, 0)
            * options.video_max_frames
            * options.video_max_soft_tokens
            + counts.get(MediaKind.AUDIO, 0) * audio_tokens
        )

    def check_budget(self, options: MediaOptions, kinds: Mapping[MediaKind | str, int]) -> None:
        """Raise ``ValueError`` when a row could exceed the model's context window."""
        estimate = self.estimate_tokens(options, kinds)
        if estimate > self.context_tokens:
            raise ValueError(
                f"A row could need about {estimate} tokens, more than the model's "
                f"{self.context_tokens}-token context. Lower the video frames, the tokens "
                "per image or frame, or the audio seconds."
            )

    @staticmethod
    def _options_from(raw: Mapping[str, object], base: MediaOptions) -> MediaOptions:
        unknown = set(raw) - set(OPTION_NAMES)
        if unknown:
            raise ValueError(f"Unknown media options: {sorted(unknown)}")
        values = {name: value for name, value in raw.items() if value is not None}
        return replace(base, **values)

    def _validated(self, options: MediaOptions) -> MediaOptions:
        """Check ``options`` against the limits; floats are normalised to ``float``."""
        for name in ("image_max_soft_tokens", "video_max_soft_tokens"):
            value = getattr(options, name)
            if not _is_integer(value) or value not in self.soft_token_budgets:
                raise ValueError(
                    f"{name} must be one of {list(self.soft_token_budgets)}, got {value!r}"
                )
        if not _is_integer(options.video_max_frames) or not (
            1 <= options.video_max_frames <= self.video_frames_limit
        ):
            raise ValueError(
                f"video_max_frames must be an integer in 1..{self.video_frames_limit}, "
                f"got {options.video_max_frames!r}"
            )
        for name, limit in (
            ("video_fps", self.video_fps_limit),
            ("max_audio_seconds", self.audio_seconds_limit),
        ):
            value = getattr(options, name)
            if not _is_number(value) or not (0 < value <= limit):
                raise ValueError(f"{name} must be a number in (0, {limit}], got {value!r}")
        return replace(
            options,
            video_fps=float(options.video_fps),
            max_audio_seconds=float(options.max_audio_seconds),
        )
