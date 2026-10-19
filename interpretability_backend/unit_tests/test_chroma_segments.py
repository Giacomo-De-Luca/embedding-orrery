"""HNSW segment directories: removed with their collection, orphans found and pruned.

chromadb deletes a collection's catalog rows but leaves its persisted HNSW
index directory (``<persist_dir>/<vector segment id>/``) on disk. These tests
pin that library behaviour on real temporary stores and check that
``ChromaSegmentJanitor`` cleans up after it without ever touching a live index.
"""

import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import NamedTuple

import chromadb
import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from interpretability_backend.backend.clients.chromadb_client import ChromaDBClient
from interpretability_backend.backend.utils import chroma_segments
from interpretability_backend.backend.utils.chroma_segments import ChromaSegmentJanitor
from interpretability_backend.scripts import prune_chroma_orphans

# Persist the HNSW index after a couple of adds so tiny collections get a directory.
SMALL_INDEX = {"hnsw": {"sync_threshold": 2, "batch_size": 2}}
AN_HOUR_AGO = time.time() - 3600


class Store(NamedTuple):
    path: Path
    client: chromadb.ClientAPI
    keep_dir: str  # index directory of the "keep" collection
    drop_dir: str  # index directory of the "drop" collection


def _client(path: Path):
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def _add_collection(client, name: str, n: int = 10, configuration=SMALL_INDEX):
    collection = client.create_collection(
        name, configuration=configuration, embedding_function=None
    )
    collection.add(
        ids=[f"{name}-{i}" for i in range(n)],
        embeddings=[[float(i), 1.0, 0.0] for i in range(n)],
    )
    return collection


def _dirs(path: Path) -> set[str]:
    return {p.name for p in path.iterdir() if p.is_dir()}


def _age(path: Path) -> None:
    """Backdate a directory and its files past the janitor's minimum orphan age."""
    for entry in [*path.iterdir(), path]:
        os.utime(entry, (AN_HOUR_AGO, AN_HOUR_AGO))


def _failing_rmtree(path, *args, **kwargs):
    raise PermissionError(f"denied: {path}")


def _assert_keep_searchable(store: Store) -> None:
    """The "keep" index is intact: its directory exists and, reopened from
    disk, its nearest neighbour of [0, 1, 0] is keep-0 (a lost index directory
    still reports the right count but answers queries wrongly)."""
    assert (store.path / store.keep_dir).is_dir()
    SharedSystemClient.clear_system_cache()
    keep = _client(store.path).get_collection("keep")
    assert keep.query(query_embeddings=[[0.0, 1.0, 0.0]], n_results=1)["ids"] == [["keep-0"]]


@pytest.fixture
def store(tmp_path: Path):
    """A store with two collections whose HNSW indexes are persisted."""
    client = _client(tmp_path)
    _add_collection(client, "keep")
    [keep_dir] = _dirs(tmp_path)
    _add_collection(client, "drop")
    [drop_dir] = _dirs(tmp_path) - {keep_dir}
    yield Store(tmp_path, client, keep_dir, drop_dir)
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def leaked(store: Store) -> Store:
    """``store`` after a plain chromadb delete of "drop", which leaks its directory."""
    store.client.delete_collection("drop")
    return store


def test_library_delete_collection_leaves_hnsw_dir(leaked: Store):
    """Upstream bug (chroma #5259, #7853). If this starts failing after a
    chromadb upgrade, chromadb removes the directory itself and only the
    orphan prune for older stores is still needed."""
    assert _dirs(leaked.path) == {leaked.keep_dir, leaked.drop_dir}


# --- delete_collection -------------------------------------------------------


def test_delete_collection_removes_its_hnsw_dir(store: Store):
    removed = ChromaSegmentJanitor.delete_collection(store.client, "drop")

    assert [path.name for path in removed] == [store.drop_dir]
    assert _dirs(store.path) == {store.keep_dir}
    with pytest.raises(NotFoundError):
        store.client.get_collection("drop")
    _assert_keep_searchable(store)


def test_removed_dir_stays_gone_after_reuse_and_reopen(store: Store):
    ChromaSegmentJanitor.delete_collection(store.client, "drop")
    _add_collection(store.client, "drop", n=6)

    SharedSystemClient.clear_system_cache()
    reopened = _client(store.path)

    assert reopened.get_collection("drop").count() == 6
    assert store.drop_dir not in _dirs(store.path)
    assert len(_dirs(store.path)) == 2
    _assert_keep_searchable(store)


def test_delete_collection_with_default_index_settings(tmp_path: Path):
    """chromadb writes the index directory on the first add, whatever its settings."""
    client = _client(tmp_path)
    _add_collection(client, "tiny", n=3, configuration=None)
    [index_dir] = _dirs(tmp_path)

    removed = ChromaSegmentJanitor.delete_collection(client, "tiny")

    assert [path.name for path in removed] == [index_dir]
    assert _dirs(tmp_path) == set()
    SharedSystemClient.clear_system_cache()


def test_delete_empty_collection_has_no_dir_to_remove(tmp_path: Path):
    client = _client(tmp_path)
    client.create_collection("empty", embedding_function=None)
    assert _dirs(tmp_path) == set()

    assert ChromaSegmentJanitor.delete_collection(client, "empty") == []
    with pytest.raises(NotFoundError):
        client.get_collection("empty")
    SharedSystemClient.clear_system_cache()


def test_delete_missing_collection_raises_not_found(store: Store):
    with pytest.raises(NotFoundError):
        ChromaSegmentJanitor.delete_collection(store.client, "never-created")


def test_delete_on_in_memory_client():
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    name = f"mem-{uuid.uuid4().hex[:8]}"
    client.create_collection(name, embedding_function=None).add(ids=["a"], embeddings=[[1.0]])

    assert ChromaSegmentJanitor.delete_collection(client, name) == []
    with pytest.raises(NotFoundError):
        client.get_collection(name)


def test_delete_still_deletes_when_catalog_unreadable(store: Store, monkeypatch):
    def unreadable(self, sql, params=()):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ChromaSegmentJanitor, "_catalog_rows", unreadable)

    assert ChromaSegmentJanitor.delete_collection(store.client, "drop") == []
    with pytest.raises(NotFoundError):
        store.client.get_collection("drop")
    assert store.drop_dir in _dirs(store.path)  # left for prune_orphans


def test_delete_still_deletes_when_dir_removal_fails(store: Store, monkeypatch):
    monkeypatch.setattr(chroma_segments.shutil, "rmtree", _failing_rmtree)

    assert ChromaSegmentJanitor.delete_collection(store.client, "drop") == []
    with pytest.raises(NotFoundError):
        store.client.get_collection("drop")
    assert store.drop_dir in _dirs(store.path)


def test_chromadb_client_delete_collection_removes_dir(store: Store):
    wrapper = ChromaDBClient(db_path=str(store.path))

    wrapper.delete_collection("drop")

    assert _dirs(store.path) == {store.keep_dir}
    assert [c.name for c in wrapper.client.list_collections()] == ["keep"]
    _assert_keep_searchable(store)


# --- orphan_dirs / prune_orphans ---------------------------------------------


def test_orphan_dirs_lists_exactly_the_leaked_dir(leaked: Store):
    [orphan] = ChromaSegmentJanitor(leaked.path).orphan_dirs(min_age_seconds=0)

    assert orphan.path == leaked.path / leaked.drop_dir
    assert orphan.size_bytes > 0


def test_orphan_dirs_skips_anything_but_plain_segment_dirs(leaked: Store):
    (leaked.path / "notes").mkdir()  # not a segment id
    foreign = leaked.path / str(uuid.uuid4())  # segment-shaped, foreign file
    foreign.mkdir()
    (foreign / "unexpected.txt").write_text("keep me")
    nested = leaked.path / str(uuid.uuid4())  # segment-shaped, holds a directory
    (nested / "inner").mkdir(parents=True)
    upper = leaked.path / str(uuid.uuid4()).upper()  # non-canonical id
    upper.mkdir()
    (upper / "header.bin").write_bytes(b"x")
    linked = leaked.path / str(uuid.uuid4())  # symlink to a live index
    linked.symlink_to(leaked.path / leaked.keep_dir, target_is_directory=True)

    orphans = ChromaSegmentJanitor(leaked.path).orphan_dirs(min_age_seconds=0)

    assert [orphan.path.name for orphan in orphans] == [leaked.drop_dir]


def test_orphan_dirs_skips_recently_modified(leaked: Store):
    janitor = ChromaSegmentJanitor(leaked.path)

    assert janitor.orphan_dirs(min_age_seconds=600) == []
    _age(leaked.path / leaked.drop_dir)
    assert [o.path.name for o in janitor.orphan_dirs(min_age_seconds=600)] == [leaked.drop_dir]


def test_orphan_dirs_requires_the_catalog(tmp_path: Path):
    index = tmp_path / str(uuid.uuid4())
    index.mkdir()
    (index / "header.bin").write_bytes(b"x")

    with pytest.raises(sqlite3.Error):
        ChromaSegmentJanitor(tmp_path).orphan_dirs(min_age_seconds=0)


def test_orphan_dirs_refuses_when_no_dir_matches_the_catalog(leaked: Store):
    """A changed on-disk layout would make every live index look orphaned."""
    (leaked.path / leaked.keep_dir).rename(leaked.path / str(uuid.uuid4()))

    with pytest.raises(RuntimeError, match="layout"):
        ChromaSegmentJanitor(leaked.path).orphan_dirs(min_age_seconds=0)


def test_prune_orphans_dry_run_keeps_then_removes(leaked: Store):
    janitor = ChromaSegmentJanitor(leaked.path)

    listed = janitor.prune_orphans(min_age_seconds=0, dry_run=True)
    assert [o.path.name for o in listed.orphans] == [leaked.drop_dir]
    assert (leaked.path / leaked.drop_dir).is_dir()

    pruned = janitor.prune_orphans(min_age_seconds=0, dry_run=False)
    assert [o.path.name for o in pruned.orphans] == [leaked.drop_dir]
    assert pruned.failed == []
    assert _dirs(leaked.path) == {leaked.keep_dir}
    assert janitor.orphan_dirs(min_age_seconds=0) == []
    _assert_keep_searchable(leaked)


def test_prune_orphans_reports_failed_removals(leaked: Store, monkeypatch):
    monkeypatch.setattr(chroma_segments.shutil, "rmtree", _failing_rmtree)

    result = ChromaSegmentJanitor(leaked.path).prune_orphans(min_age_seconds=0, dry_run=False)

    assert result.failed == [leaked.path / leaked.drop_dir]
    assert leaked.drop_dir in _dirs(leaked.path)


# --- prune script ------------------------------------------------------------


def _write_config(directory: Path, body: str) -> Path:
    config = directory / "prune.toml"
    config.write_text(body)
    return config


@pytest.mark.parametrize("dry_run", [True, False])
def test_prune_script_follows_config(leaked: Store, tmp_path_factory, monkeypatch, capsys, dry_run):
    config = _write_config(
        tmp_path_factory.mktemp("cfg"),
        f'chroma_path = "{leaked.path}"\ndry_run = {str(dry_run).lower()}\nmin_age_minutes = 0\n',
    )
    monkeypatch.setenv(prune_chroma_orphans.CONFIG_ENV_VAR, str(config))

    assert prune_chroma_orphans.main() == 0

    assert leaked.drop_dir in capsys.readouterr().out
    assert (leaked.path / leaked.drop_dir).exists() == dry_run
    _assert_keep_searchable(leaked)


def test_prune_script_exits_nonzero_on_failed_removal(leaked: Store, tmp_path_factory, monkeypatch):
    config = _write_config(
        tmp_path_factory.mktemp("cfg"),
        f'chroma_path = "{leaked.path}"\ndry_run = false\nmin_age_minutes = 0\n',
    )
    monkeypatch.setenv(prune_chroma_orphans.CONFIG_ENV_VAR, str(config))
    monkeypatch.setattr(chroma_segments.shutil, "rmtree", _failing_rmtree)

    assert prune_chroma_orphans.main() == 1


@pytest.mark.parametrize(
    "body",
    [
        "dry_run = 0\n",  # an int must not be read as false (and delete)
        'dry_run = "no"\n',
        "min_age_minutes = -5\n",
        "min_age_minutes = 1.5\n",
        "chroma_path = 3\n",
        "dryrun = false\n",  # unknown key
    ],
)
def test_prune_settings_reject_invalid_config(tmp_path: Path, body: str):
    with pytest.raises(ValueError):
        prune_chroma_orphans.PruneSettings.from_toml(_write_config(tmp_path, body))


def test_prune_settings_defaults_to_a_dry_run(tmp_path: Path):
    settings = prune_chroma_orphans.PruneSettings.from_toml(_write_config(tmp_path, ""))
    assert settings.dry_run is True
    assert settings.min_age_minutes == 10


def test_shipped_config_is_a_dry_run():
    settings = prune_chroma_orphans.PruneSettings.from_toml(
        prune_chroma_orphans.DEFAULT_CONFIG_PATH
    )
    assert settings.dry_run is True
