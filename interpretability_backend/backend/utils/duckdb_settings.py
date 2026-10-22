"""DuckDB connection settings from ``config/duckdb.json``.

Without a cap DuckDB's buffer pool may grow to 80% of RAM and keep cached
table blocks for the life of the backend. ``memory_limit`` takes DuckDB's
absolute sizes ("4GB", "2GiB") or a share of the memory available to this
process ("25%"), which DuckDB itself does not accept and is resolved here:
physical RAM, capped by a container (cgroup) limit, so a Docker deployment
never resolves against the host's memory. Large queries spill to
``<db>.tmp`` beyond the cap instead of growing the process.
"""

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .json_settings import JsonFileSettings
from .resource_paths import PACKAGE_ROOT

_ABSOLUTE = re.compile(r"^(\d+(\.\d+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)$", re.IGNORECASE)
_PERCENT = re.compile(r"^(\d+(\.\d+)?)%$")
# cgroup v2, then v1; "max" (v2) and huge sentinels (v1) mean "no limit".
_CGROUP_LIMIT_FILES = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)
# DuckDB fails to connect with a 0 limit; a resolved percentage never goes below this.
_MIN_LIMIT_MIB = 64
_MIB = 1024 * 1024


class MachineMemory:
    """Memory available to this process: physical RAM, capped by a container limit."""

    @staticmethod
    def available_bytes(
        physical_bytes: int | None = None,
        cgroup_files: Iterable[Path] = _CGROUP_LIMIT_FILES,
    ) -> int:
        total = (
            physical_bytes
            if physical_bytes is not None
            else os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        )
        for path in cgroup_files:
            try:
                raw = path.read_text().strip()
            except OSError:
                continue
            if raw.isdigit() and 0 < int(raw) < total:
                total = int(raw)
        return total


@dataclass(frozen=True)
class DuckDBSettings(JsonFileSettings):
    """Connection options for the backend's DuckDB database."""

    SETTINGS_LABEL: ClassVar[str] = "DuckDB settings"

    # Absolute size, a percentage of available memory, or null for DuckDB's default (80%).
    memory_limit: str | None = "25%"
    # Worker threads; null keeps DuckDB's default (one per core).
    threads: int | None = None

    def __post_init__(self):
        if self.memory_limit is not None and not self._valid_limit(self.memory_limit):
            raise ValueError(
                "memory_limit must be a non-zero size like '4GB' or a percentage in (0, 100], "
                f"got {self.memory_limit!r}"
            )
        if self.threads is not None:
            self.require_int("threads", minimum=1)

    @staticmethod
    def _valid_limit(limit: object) -> bool:
        if not isinstance(limit, str):
            return False
        absolute = _ABSOLUTE.match(limit)
        if absolute:
            return float(absolute.group(1)) > 0
        percent = _PERCENT.match(limit)
        return bool(percent) and 0 < float(percent.group(1)) <= 100

    def connect_config(self, memory_bytes: int | None = None) -> dict[str, str | int]:
        """``duckdb.connect(config=...)`` options.

        A percentage is resolved against ``memory_bytes`` (default: what
        ``MachineMemory`` reports) and never below 64 MiB.
        """
        config: dict[str, str | int] = {}
        if self.memory_limit is not None:
            percent = _PERCENT.match(self.memory_limit)
            if percent:
                total = (
                    memory_bytes if memory_bytes is not None else MachineMemory.available_bytes()
                )
                mib = int(total * float(percent.group(1)) / 100) // _MIB
                config["memory_limit"] = f"{max(mib, _MIN_LIMIT_MIB)}MiB"
            else:
                config["memory_limit"] = self.memory_limit
        if self.threads is not None:
            config["threads"] = self.threads
        return config


SETTINGS = DuckDBSettings.from_file(PACKAGE_ROOT / "config" / "duckdb.json")
