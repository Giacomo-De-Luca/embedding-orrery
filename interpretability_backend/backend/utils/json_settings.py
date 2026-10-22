"""Load frozen settings dataclasses from JSON files in ``config/``.

Backend tunables live in ``interpretability_backend/config/*.json`` (config
files are preferred to CLI flags). ``JsonFileSettings`` gives a settings
dataclass a ``from_file`` constructor with the shared rules:

- a missing file means the dataclass defaults;
- unknown keys are rejected, so a typo cannot silently fall back to a default;
- JSON lists become tuples for the fields named in ``TUPLE_FIELDS``;
- wrong value types surface as ``ValueError``, like the range checks the
  dataclass runs in ``__post_init__`` (``require_bool`` / ``require_int`` /
  ``require_number`` are the shared checks for those).
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import ClassVar, Self


class JsonFileSettings:
    """Mixin for frozen settings dataclasses loaded from a JSON object file."""

    SETTINGS_LABEL: ClassVar[str] = "settings"
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Build the settings from ``path``; a missing file means defaults."""
        if not path.exists():
            return cls()
        return cls.from_mapping(json.loads(path.read_text()), source=path)

    @classmethod
    def from_mapping(cls, raw: object, source: object = "input") -> Self:
        """Build the settings from a parsed JSON object, with the same rules as ``from_file``.

        Also used for nested objects (e.g. catalog entries); ``source`` only
        names the object in the "must hold a JSON object" error.
        """
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid {cls.SETTINGS_LABEL}: {source} must hold a JSON object")
        raw = dict(raw)
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {cls.SETTINGS_LABEL}: {sorted(unknown)}")
        for name in cls.TUPLE_FIELDS:
            if isinstance(raw.get(name), list):
                raw[name] = tuple(raw[name])
        try:
            return cls(**raw)
        except TypeError as exc:  # wrong JSON types reaching the range checks
            raise ValueError(f"Invalid {cls.SETTINGS_LABEL}: {exc}") from exc

    def require_bool(self, name: str) -> None:
        """Raise ``ValueError`` unless field ``name`` is a bool."""
        value = getattr(self, name)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false, got {value!r}")

    def require_int(self, name: str, minimum: int, maximum: int | None = None) -> None:
        """Raise ``ValueError`` unless field ``name`` is an int (not a bool) in range."""
        self._require_range(name, int, "an integer", minimum, maximum)

    def require_number(self, name: str, minimum: float, maximum: float | None = None) -> None:
        """Raise ``ValueError`` unless field ``name`` is a number (not a bool) in range."""
        self._require_range(name, int | float, "a number", minimum, maximum)

    def _require_range(self, name: str, kind, description: str, minimum, maximum) -> None:
        value = getattr(self, name)
        in_range = isinstance(value, kind) and not isinstance(value, bool) and value >= minimum
        if maximum is not None:
            in_range = in_range and value <= maximum
        if not in_range:
            bound = f"in {minimum}..{maximum}" if maximum is not None else f">= {minimum}"
            raise ValueError(f"{name} must be {description} {bound}, got {value!r}")
