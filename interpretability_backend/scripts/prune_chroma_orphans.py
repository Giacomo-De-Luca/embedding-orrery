"""Remove the HNSW index directories chromadb left behind for deleted collections.

chromadb never removes a deleted collection's index directory (see
``backend/utils/chroma_segments.py``). Deletions made by the backend now clean
up after themselves; this removes the directories leaked before. Config:
``prune_chroma_orphans_config.toml`` next to this file (override with
``ORRERY_CHROMA_PRUNE_CONFIG``). It defaults to a dry run that only lists them.

Run with the backend stopped:
    uv run python -m interpretability_backend.scripts.prune_chroma_orphans
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from interpretability_backend.backend.utils.chroma_segments import ChromaSegmentJanitor
from interpretability_backend.backend.utils.json_settings import JsonFileSettings
from interpretability_backend.backend.utils.resource_paths import CHROMA_DB_PATH
from interpretability_backend.evaluation.utils.runner_common import (
    load_config,
    resolve_config_path,
)

CONFIG_ENV_VAR = "ORRERY_CHROMA_PRUNE_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("prune_chroma_orphans_config.toml")


@dataclass(frozen=True)
class PruneSettings(JsonFileSettings):
    """The script's TOML config, type-checked: a stray ``dry_run = 0`` must not delete."""

    SETTINGS_LABEL: ClassVar[str] = "prune_chroma_orphans config"

    dry_run: bool = True
    min_age_minutes: int = 10
    chroma_path: str = ""  # empty = the live store

    def __post_init__(self) -> None:
        self.require_bool("dry_run")
        self.require_int("min_age_minutes", minimum=0)
        if not isinstance(self.chroma_path, str):
            raise ValueError(f"chroma_path must be a string, got {self.chroma_path!r}")

    @classmethod
    def from_toml(cls, path: Path) -> "PruneSettings":
        try:
            return cls(**load_config(path))
        except TypeError as error:  # unknown key
            raise ValueError(f"Invalid {cls.SETTINGS_LABEL}: {error}") from error


def main() -> int:
    settings = PruneSettings.from_toml(resolve_config_path(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))
    chroma_path = Path(settings.chroma_path or CHROMA_DB_PATH)

    result = ChromaSegmentJanitor(chroma_path).prune_orphans(
        min_age_seconds=settings.min_age_minutes * 60, dry_run=settings.dry_run
    )

    failed = set(result.failed)
    for orphan in result.orphans:
        action = (
            "would remove" if settings.dry_run else "FAILED" if orphan.path in failed else "removed"
        )
        modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(orphan.modified))
        print(
            f"{action}  {orphan.path.name}  {orphan.size_bytes / 1e6:9.1f} MB  modified {modified}"
        )
    total_gb = sum(orphan.size_bytes for orphan in result.orphans) / 1e9
    found = f"{len(result.orphans)} orphan index directories ({total_gb:.2f} GB) in {chroma_path}"
    if settings.dry_run:
        print(f"Found {found}")
    else:
        print(f"Removed {len(result.orphans) - len(result.failed)} of {found}")
    if result.failed:
        print(f"{len(result.failed)} could not be removed (see the log above)", file=sys.stderr)
        return 1
    if settings.dry_run and result.orphans:
        print("Dry run: set dry_run = false in the config to delete them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
