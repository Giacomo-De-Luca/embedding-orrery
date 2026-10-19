"""HNSW index directories of a persistent ChromaDB store.

chromadb (up to 1.5.9, the latest release) deletes a collection's catalog rows
but leaves its persisted HNSW index, ``<persist_dir>/<vector segment id>/``, on
disk forever (upstream chroma-core/chroma #5259 and #7853; the fix, #7767, is
on chromadb main but unreleased, and only cleans up deletions it recorded
itself). Every collection deleted or overwritten here leaked its index this
way.

``ChromaSegmentJanitor.delete_collection`` deletes a collection together with
its index directory; ``orphan_dirs`` / ``prune_orphans`` find and remove the
directories leaked earlier (``scripts/prune_chroma_orphans.py``). chromadb has
no public API for segment ids, so they are read from the store's own catalog
(table ``segments`` of ``chroma.sqlite3``), read-only.
"""

import logging
import shutil
import sqlite3
import time
import uuid
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from chromadb.api import ClientAPI

logger = logging.getLogger("orrery")

CATALOG_FILE = "chroma.sqlite3"
# Files chromadb's local HNSW segment writes. A directory holding anything
# else is never treated as an index directory, so never removed.
HNSW_FILES = frozenset(
    {"header.bin", "data_level0.bin", "length.bin", "link_lists.bin", "index_metadata.pickle"}
)
DEFAULT_MIN_ORPHAN_AGE_SECONDS = 600


@dataclass(frozen=True)
class SegmentDir:
    """An HNSW index directory on disk."""

    path: Path
    size_bytes: int
    modified: float  # newest mtime of the directory and its files (epoch seconds)


@dataclass(frozen=True)
class PruneResult:
    """The orphan directories found, and those that could not be removed."""

    orphans: list[SegmentDir]
    failed: list[Path] = field(default_factory=list)


class ChromaSegmentJanitor:
    """Removes the HNSW index directories chromadb leaves behind."""

    def __init__(self, persist_dir: str | Path):
        self.persist_dir = Path(persist_dir).resolve()

    @classmethod
    def delete_collection(cls, client: ClientAPI, name: str) -> list[Path]:
        """Delete collection ``name`` and its HNSW index directory.

        Raises chromadb's ``NotFoundError`` when the collection does not exist.
        Returns the directories removed: none for an in-memory client or a
        collection that never stored a vector (chromadb writes the directory
        on the first add). When the catalog cannot be read or a directory
        cannot be removed, the problem is logged and the directory is left
        for ``prune_orphans``; the collection itself is deleted either way.
        """
        collection_id = str(client.get_collection(name=name, embedding_function=None).id)
        settings = client.get_settings()
        if not settings.is_persistent:
            client.delete_collection(name=name)
            return []

        janitor = cls(settings.persist_directory)
        try:
            segment_ids = janitor._vector_segment_ids(collection_id)
        except sqlite3.Error as error:
            logger.warning("Could not look up the HNSW directory of '%s': %s", name, error)
            segment_ids = []
        client.delete_collection(name=name)

        paths = []
        for segment_id in segment_ids:
            path = janitor.persist_dir / segment_id
            if janitor._is_segment_dir(path):
                paths.append(path)
            elif path.exists():
                logger.warning("Left %s of '%s': not a plain HNSW index directory", path, name)
        removed, _ = cls._remove(paths)
        return removed

    def orphan_dirs(
        self, min_age_seconds: float = DEFAULT_MIN_ORPHAN_AGE_SECONDS
    ) -> list[SegmentDir]:
        """Index directories no segment of the store references, oldest first.

        Directories are listed before the catalog is read: chromadb registers a
        collection's segments when it creates the collection and writes the
        directory later, so a collection created meanwhile is never taken for
        an orphan. Directories modified within ``min_age_seconds`` are skipped
        as a further margin.

        Raises when the catalog is missing, and when it has vector segments but
        not one directory matches them: chromadb's on-disk layout would then
        have changed, and every live index would look orphaned.
        """
        candidates = [path for path in self.persist_dir.iterdir() if self._is_segment_dir(path)]
        rows = self._catalog_rows("SELECT id, scope FROM segments")
        referenced = {self._canonical(segment_id) for segment_id, _ in rows}
        vector_ids = {
            self._canonical(segment_id) for segment_id, scope in rows if scope == "VECTOR"
        }
        if candidates and vector_ids and not vector_ids & {path.name for path in candidates}:
            raise RuntimeError(
                f"No index directory in {self.persist_dir} matches a vector segment of its "
                "catalog; chromadb's on-disk layout may have changed, so nothing is treated "
                "as an orphan"
            )

        cutoff = time.time() - min_age_seconds
        orphans = []
        for path in candidates:
            if path.name in referenced:
                continue
            try:
                info = self._describe(path)
            except FileNotFoundError:  # removed while listing
                continue
            if info.modified <= cutoff:
                orphans.append(info)
        return sorted(orphans, key=lambda info: info.modified)

    def prune_orphans(
        self, min_age_seconds: float = DEFAULT_MIN_ORPHAN_AGE_SECONDS, dry_run: bool = True
    ) -> PruneResult:
        """Remove the ``orphan_dirs`` (only list them when ``dry_run``)."""
        orphans = self.orphan_dirs(min_age_seconds)
        if dry_run:
            return PruneResult(orphans)
        _, failed = self._remove(orphan.path for orphan in orphans)
        return PruneResult(orphans, failed)

    @staticmethod
    def _remove(paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
        """Remove each directory; returns ``(removed, failed)``, failures logged."""
        removed, failed = [], []
        for path in paths:
            try:
                shutil.rmtree(path)
            except OSError as error:
                logger.warning("Could not remove HNSW directory %s: %s", path, error)
                failed.append(path)
            else:
                removed.append(path)
        return removed, failed

    def _vector_segment_ids(self, collection_id: str) -> list[str]:
        rows = self._catalog_rows(
            "SELECT id FROM segments WHERE collection = ? AND scope = 'VECTOR'", (collection_id,)
        )
        return [self._canonical(row[0]) for row in rows]

    def _catalog_rows(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Rows from the store's catalog, opened read-only (raises if it is missing)."""
        catalog = (self.persist_dir / CATALOG_FILE).as_uri()
        with closing(sqlite3.connect(f"{catalog}?mode=ro", uri=True)) as connection:
            return connection.execute(sql, params).fetchall()

    @staticmethod
    def _canonical(segment_id: str) -> str:
        """The canonical (lowercase, hyphenated) form of a segment UUID."""
        return str(uuid.UUID(segment_id))

    @classmethod
    def _is_segment_dir(cls, path: Path) -> bool:
        """A real directory named by a canonical segment id, holding only HNSW files."""
        if path.is_symlink() or not path.is_dir():
            return False
        try:
            if cls._canonical(path.name) != path.name:
                return False
        except ValueError:
            return False
        return all(
            child.is_file() and not child.is_symlink() and child.name in HNSW_FILES
            for child in path.iterdir()
        )

    @staticmethod
    def _describe(path: Path) -> SegmentDir:
        stats = [path.stat()] + [child.stat() for child in path.iterdir()]
        return SegmentDir(
            path=path,
            size_bytes=sum(stat.st_size for stat in stats[1:]),
            modified=max(stat.st_mtime for stat in stats),
        )
