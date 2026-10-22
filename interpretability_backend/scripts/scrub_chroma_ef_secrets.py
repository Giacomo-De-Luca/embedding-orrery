"""Remove secrets (an HF token) from the embedding-function configs ChromaDB persisted.

The sentence-transformers wrapper used to save its load kwargs, the HF token
among them, into every new collection's config in ``chroma.sqlite3``. New
collections no longer store them; this cleans the existing ones (see
``backend/utils/chroma_config_scrubber.py``). Config:
``scrub_chroma_ef_secrets_config.toml`` next to this file (override with
``ORRERY_SCRUB_EF_SECRETS_CONFIG``). It defaults to a dry run with a backup.

Run with the backend stopped:
    uv run python -m interpretability_backend.scripts.scrub_chroma_ef_secrets
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import duckdb

from interpretability_backend.backend.utils.chroma_config_scrubber import (
    DEFAULT_SECRET_KEYS,
    ChromaConfigScrubber,
)
from interpretability_backend.backend.utils.json_settings import JsonFileSettings
from interpretability_backend.backend.utils.resource_paths import CHROMA_DB_PATH, DUCKDB_PATH
from interpretability_backend.evaluation.utils.runner_common import (
    load_config,
    resolve_config_path,
)

CONFIG_ENV_VAR = "ORRERY_SCRUB_EF_SECRETS_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("scrub_chroma_ef_secrets_config.toml")


@dataclass(frozen=True)
class ScrubSettings(JsonFileSettings):
    """The script's TOML config, type-checked: a stray ``dry_run = 0`` must not write."""

    SETTINGS_LABEL: ClassVar[str] = "scrub_chroma_ef_secrets config"
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = ("secret_keys",)

    dry_run: bool = True
    backup: bool = True
    chroma_path: str = ""  # empty = the live store
    secret_keys: tuple[str, ...] = DEFAULT_SECRET_KEYS

    def __post_init__(self) -> None:
        self.require_bool("dry_run")
        self.require_bool("backup")
        if not isinstance(self.chroma_path, str):
            raise ValueError(f"chroma_path must be a string, got {self.chroma_path!r}")
        if not self.secret_keys or not all(isinstance(k, str) and k for k in self.secret_keys):
            raise ValueError("secret_keys must be a non-empty list of key names")

    @classmethod
    def from_toml(cls, path: Path) -> "ScrubSettings":
        return cls.from_mapping(load_config(path), source=path)


def _backend_running() -> bool:
    """True when another process holds the live DuckDB store (the backend).

    Read-only, so the probe never replays or checkpoints a WAL left on a
    stopped store (replaying ALTERs on FK tables has wedged DuckDB before);
    it still fails while the backend holds the write lock.
    """
    if not DUCKDB_PATH.exists():
        return False
    try:
        duckdb.connect(str(DUCKDB_PATH), read_only=True).close()
    except duckdb.IOException:
        return True
    return False


def main() -> int:
    settings = ScrubSettings.from_toml(resolve_config_path(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))
    live_store = not settings.chroma_path
    sqlite_path = Path(settings.chroma_path or CHROMA_DB_PATH) / "chroma.sqlite3"

    if live_store and not settings.dry_run and _backend_running():
        print("ERROR: the backend is running; stop it before scrubbing.", file=sys.stderr)
        return 1

    scrubber = ChromaConfigScrubber(sqlite_path, secret_keys=settings.secret_keys)
    try:
        report = scrubber.scan() if settings.dry_run else scrubber.apply(backup=settings.backup)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    action = "would scrub" if settings.dry_run else "scrubbed"
    for name in report.affected_collections:
        print(f"{action}  {name}")
    print(f"{report.summary()} in {sqlite_path}")
    if report.backup_path is not None:
        print(f"Backup: {report.backup_path}")
    if report.applied and report.affected_collections:
        print("Restart the backend, then spot-check a semantic search.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
