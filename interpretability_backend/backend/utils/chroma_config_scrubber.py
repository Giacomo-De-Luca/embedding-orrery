"""Remove secrets from the embedding-function configs ChromaDB persisted.

The sentence-transformers wrapper used to return its load kwargs (the HF token
among them) from ``get_config()``, and Chroma stores that config for every
collection in ``chroma.sqlite3``: ``collections.config_json_str`` and the Rust
sysdb's ``schema_str`` (twice there: under ``defaults`` and under the
``#embedding`` key). New collections no longer persist a real embedding
function; this cleans the configs written before.

Only keys inside an ``embedding_function`` object are removed, so an unrelated
``token`` field elsewhere in a config survives. Used by
``scripts/scrub_chroma_ef_secrets.py``, which runs with the backend stopped.
"""

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SECRET_KEYS = ("token", "use_auth_token", "hf_token", "api_key")
CONFIG_COLUMNS = ("config_json_str", "schema_str")
EMBEDDING_FUNCTION_KEY = "embedding_function"


def scrub_json(text: str, secret_keys: Iterable[str]) -> tuple[str, int]:
    """Drop ``secret_keys`` (case-insensitive) inside every embedding-function object.

    Returns ``(text, removed)``. ``text`` comes back unchanged when nothing was
    removed or it is not JSON; otherwise it is re-serialised compactly, the way
    Chroma writes it.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return text, 0
    removed = _scrub(data, {key.casefold() for key in secret_keys}, inside=False)
    if not removed:
        return text, 0
    return json.dumps(data, separators=(",", ":")), removed


def _scrub(node: object, secret_keys: set[str], inside: bool) -> int:
    """Remove secret keys from dicts nested in an embedding-function object."""
    removed = 0
    if isinstance(node, dict):
        if inside:
            for key in [k for k in node if isinstance(k, str) and k.casefold() in secret_keys]:
                del node[key]
                removed += 1
        for key, value in node.items():
            removed += _scrub(value, secret_keys, inside or key == EMBEDDING_FUNCTION_KEY)
    elif isinstance(node, list):
        for item in node:
            removed += _scrub(item, secret_keys, inside)
    return removed


@dataclass
class ScrubReport:
    """What a scan found or an apply removed. Never holds secret values."""

    secrets_removed: int = 0
    affected_collections: list[str] = field(default_factory=list)
    applied: bool = False
    backup_path: Path | None = None

    def summary(self) -> str:
        action = "Removed" if self.applied else "Would remove"
        return (
            f"{action} {self.secrets_removed} secret value(s) from "
            f"{len(self.affected_collections)} collection config(s)"
        )


class ChromaConfigScrubber:
    """Scan or scrub the embedding-function configs of one ``chroma.sqlite3``."""

    def __init__(
        self,
        sqlite_path: str | Path,
        secret_keys: Iterable[str] = DEFAULT_SECRET_KEYS,
        timeout_seconds: float = 5.0,
    ):
        self.sqlite_path = Path(sqlite_path)
        self.secret_keys = tuple(secret_keys)
        self.timeout_seconds = timeout_seconds

    def scan(self) -> ScrubReport:
        """Report the secrets ``apply`` would remove, without writing anything."""
        con = self._connect(read_only=True)
        try:
            report, _ = self._plan(con)
        finally:
            con.close()
        return report

    def apply(self, backup: bool = True) -> ScrubReport:
        """Remove the secrets in one transaction, after an optional timestamped backup.

        Raises ``RuntimeError`` when another connection holds the write lock
        (the backend is running) or when a re-scan still finds secrets.
        """
        con = self._connect(read_only=False)
        try:
            try:
                con.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as error:
                raise RuntimeError(
                    f"{self.sqlite_path} is locked; stop the backend first ({error})"
                ) from error
            backup_path = self._backup() if backup else None
            report, updates = self._plan(con)
            for (row_id, column), text in updates.items():
                # column comes from CONFIG_COLUMNS, never from the data
                con.execute(f"UPDATE collections SET {column} = ? WHERE id = ?", (text, row_id))
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()
        remaining = self.scan().secrets_removed
        if remaining:
            raise RuntimeError(f"{remaining} secret value(s) remain after scrubbing")
        report.applied = True
        report.backup_path = backup_path
        return report

    def _connect(self, read_only: bool) -> sqlite3.Connection:
        if not self.sqlite_path.exists():
            raise FileNotFoundError(f"ChromaDB catalog not found: {self.sqlite_path}")
        if read_only:
            uri = f"{self.sqlite_path.resolve().as_uri()}?mode=ro"
            return sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
        # isolation_level=None: transactions are opened explicitly (BEGIN IMMEDIATE)
        return sqlite3.connect(self.sqlite_path, timeout=self.timeout_seconds, isolation_level=None)

    def _plan(self, con: sqlite3.Connection) -> tuple[ScrubReport, dict[tuple[str, str], str]]:
        """The scrubbed text per (row id, column), and the report describing it."""
        report = ScrubReport()
        updates: dict[tuple[str, str], str] = {}
        columns = ", ".join(CONFIG_COLUMNS)
        for row_id, name, *texts in con.execute(f"SELECT id, name, {columns} FROM collections"):
            for column, text in zip(CONFIG_COLUMNS, texts, strict=True):
                if text is None:
                    continue
                cleaned, removed = scrub_json(text, self.secret_keys)
                if removed:
                    updates[(row_id, column)] = cleaned
                    report.secrets_removed += removed
                    if name not in report.affected_collections:
                        report.affected_collections.append(name)
        return report, updates

    def _backup(self) -> Path:
        """Copy the store (WAL content included) next to it with a timestamp suffix."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.sqlite_path.with_name(f"{self.sqlite_path.name}.{stamp}.bak")
        source = sqlite3.connect(self.sqlite_path, timeout=self.timeout_seconds)
        destination = sqlite3.connect(target)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        return target
