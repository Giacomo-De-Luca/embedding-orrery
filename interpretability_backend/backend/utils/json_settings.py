"""Load frozen settings dataclasses from JSON files in ``config/``.

Backend tunables live in ``interpretability_backend/config/*.json`` (config
files are preferred to CLI flags). ``JsonFileSettings`` gives a settings
dataclass a ``from_file`` constructor with the shared rules:

- a missing file means the dataclass defaults;
- unknown keys are rejected, so a typo cannot silently fall back to a default;
- JSON lists become tuples for the fields named in ``TUPLE_FIELDS``;
- wrong value types surface as ``ValueError``, like the range checks the
  dataclass runs in ``__post_init__``.
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
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid {cls.SETTINGS_LABEL}: {path} must hold a JSON object")
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
