"""CollectionDeletionService: a deleted collection leaves nothing behind.

Real in-memory DuckDB (conftest ``db``), a real temporary ChromaDB store, and
temporary job-state file, probe-artifact root and uploads folder.
"""

import os
import time
from pathlib import Path
from typing import NamedTuple

import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from backend.clients.chromadb_client import ChromaDBClient
from backend.clients.duckdb_client import DuckDBClient
from backend.services import cancel_registry
from backend.services.collection_deletion import CollectionBusyError, CollectionDeletionService
from backend.services.job_ids import JobIds
from backend.services.job_state import JobStateService
from backend.services.probing_types import probe_collection_dir

AN_HOUR_AGO = time.time() - 3600


class World(NamedTuple):
    db: DuckDBClient
    chroma: ChromaDBClient
    job_states: JobStateService
    service: CollectionDeletionService
    probing_root: Path
    uploads: Path


@pytest.fixture
def world(db: DuckDBClient, tmp_path: Path):
    chroma = ChromaDBClient(db_path=str(tmp_path / "chroma"))
    job_states = JobStateService(state_file=tmp_path / "job_state.json")
    probing_root = tmp_path / "probing_results"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    service = CollectionDeletionService(
        db, chroma, job_states, probing_dir=probing_root, uploads_dir=uploads
    )
    yield World(db, chroma, job_states, service, probing_root, uploads)
    SharedSystemClient.clear_system_cache()


def _upload(world: World, name: str = "poems.csv") -> Path:
    """An upload made before its dataset was embedded (an hour ago)."""
    path = world.uploads / name
    path.write_text("text\nhello\n")
    os.utime(path, (AN_HOUR_AGO, AN_HOUR_AGO))
    return path


def _record_job(
    world: World, job_id: str, owner: str, config: dict | None = None, running: bool = False
):
    """A job record as the pipelines write it: the owner in config["collection_name"]."""
    world.job_states.start_job(
        job_id,
        job_type="local_file",
        total_expected=2,
        total_batches=1,
        config={"collection_name": owner, **(config or {})},
    )
    if not running:
        world.job_states.fail_job(job_id, "interrupted")


def _probe_dir(world: World, collection: str) -> Path:
    return probe_collection_dir(collection, world.probing_root)


def _add(
    world: World,
    dataset: str,
    collection: str,
    source_file: Path | None = None,
    jobs: bool = True,
) -> None:
    """A collection of ``dataset`` with vectors, a probe artifact and job records."""
    if world.db.get_dataset(dataset) is None:
        world.db.create_dataset(dataset, source_file=str(source_file) if source_file else None)
        world.db.insert_items_batch(dataset, ["i1", "i2"], ["a", "b"], [{}, {}])
    world.db.register_vector_collection(dataset, "chromadb", collection, "dense")
    metadata = {"source_file": str(source_file)} if source_file else {"kind": "test"}
    chroma_collection = world.chroma.client.create_collection(
        collection, metadata=metadata, embedding_function=None
    )
    chroma_collection.add(ids=["i1", "i2"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
    artifact = _probe_dir(world, collection) / "label" / "ridge"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "probe.npz").write_bytes(b"weights")
    if jobs:
        file_config = {"file_path": str(source_file)} if source_file else {}
        _record_job(world, JobIds.embedding(collection), collection, file_config)
        _record_job(world, JobIds.llm_labeling(collection), collection)


def _chroma_names(world: World) -> set[str]:
    return {collection.name for collection in world.chroma.client.list_collections()}


# --- what goes ---------------------------------------------------------------


def test_deletes_everything_the_collection_owns(world: World):
    upload = _upload(world)
    _add(world, "poems", "poems", source_file=upload)
    _add(world, "other", "other")

    assert world.service.delete("poems") is True

    assert _chroma_names(world) == {"other"}
    assert world.db.get_dataset("poems") is None
    assert not _probe_dir(world, "poems").exists()
    assert world.job_states.get_job("poems") is None
    assert world.job_states.get_job("poems_llm_labeling") is None
    assert not upload.exists()
    # the unrelated collection keeps everything
    assert world.db.get_vector_collection("other") is not None
    assert _probe_dir(world, "other").is_dir()
    assert world.job_states.get_job("other") is not None


def test_upload_known_only_from_chroma_metadata_is_removed(world: World):
    """Collections predating the DuckDB registry record the file in ChromaDB only."""
    upload = _upload(world)
    collection = world.chroma.client.create_collection(
        "legacy", metadata={"source_file": str(upload)}, embedding_function=None
    )
    collection.add(ids=["i1"], embeddings=[[1.0, 0.0]])

    assert world.service.delete("legacy") is True
    assert not upload.exists()


def test_cleanup_failure_still_reports_the_deletion(world: World, monkeypatch):
    """Both stores are cleared: a retry would find nothing, so report success."""
    _add(world, "poems", "poems")

    def broken(job_id):
        raise OSError("disk full")

    monkeypatch.setattr(world.job_states, "remove_job", broken)

    assert world.service.delete("poems") is True
    assert "poems" not in _chroma_names(world)
    assert world.db.get_vector_collection("poems") is None


# --- what stays --------------------------------------------------------------


def test_reembed_keeps_what_its_dataset_still_uses(world: World):
    upload = _upload(world)
    _add(world, "poems", "poems", source_file=upload)
    _add(world, "poems", "poems_gemini", source_file=upload)

    assert world.service.delete("poems_gemini") is True

    assert _chroma_names(world) == {"poems"}
    assert not _probe_dir(world, "poems_gemini").exists()
    assert world.job_states.get_job("poems_gemini") is None
    assert world.db.get_dataset("poems") is not None
    assert _probe_dir(world, "poems").is_dir()
    assert world.job_states.get_job("poems") is not None
    assert upload.exists()  # the dataset behind "poems" was embedded from it


def test_upload_used_by_another_dataset_is_kept(world: World):
    upload = _upload(world)
    _add(world, "poems", "poems", source_file=upload)
    _add(world, "poems_copy", "poems_copy", source_file=upload)

    assert world.service.delete("poems") is True
    assert upload.exists()


@pytest.mark.parametrize("name", ["poems.csv", "città_poesie.csv"])
def test_upload_referenced_by_another_job_is_kept(world: World, name: str):
    """An interrupted embed of a collection not created yet still needs its file."""
    upload = _upload(world, name)
    _add(world, "poems", "poems", source_file=upload)
    _record_job(world, "poems_resumable", "poems_resumable", {"input": {"filePath": str(upload)}})

    assert world.service.delete("poems") is True
    assert upload.exists()


def test_upload_replaced_after_the_dataset_was_created_is_kept(world: World):
    """Uploads overwrite same-named files: a newer file is a re-upload waiting
    to be embedded, not the one the dataset came from."""
    upload = _upload(world)
    _add(world, "poems", "poems", source_file=upload)
    future = time.time() + 60
    os.utime(upload, (future, future))

    assert world.service.delete("poems") is True
    assert upload.exists()


def test_files_outside_the_uploads_folder_are_kept(world: World, tmp_path: Path):
    elsewhere = tmp_path / "data" / "poems.csv"
    elsewhere.parent.mkdir()
    elsewhere.write_text("text\n")
    os.utime(elsewhere, (AN_HOUR_AGO, AN_HOUR_AGO))
    _add(world, "poems", "poems", source_file=elsewhere)

    assert world.service.delete("poems") is True
    assert elsewhere.exists()


def test_probe_folder_shared_through_a_sanitised_name_is_kept(world: World):
    """``my-poems`` and ``my_poems`` both write probes under ``my_poems/``."""
    _add(world, "my-poems", "my-poems")
    _add(world, "my_poems", "my_poems")

    assert world.service.delete("my-poems") is True
    assert _probe_dir(world, "my_poems").is_dir()


def test_job_records_are_matched_by_their_owner(world: World):
    """``poems_llm_labeling`` is the LLM-labeling record of ``poems``, not the
    collection that happens to have that name."""
    _add(world, "poems", "poems")
    _add(world, "poems_llm_labeling", "poems_llm_labeling", jobs=False)

    assert world.service.delete("poems_llm_labeling") is True
    assert world.job_states.get_job("poems_llm_labeling") is not None

    assert world.service.delete("poems") is True
    assert world.job_states.get_job("poems_llm_labeling") is None


def test_unknown_collection_changes_nothing(world: World):
    _add(world, "poems", "poems")

    assert world.service.delete("nope") is False
    assert _chroma_names(world) == {"poems"}
    assert _probe_dir(world, "poems").is_dir()
    assert world.job_states.get_job("poems") is not None


# --- running jobs ------------------------------------------------------------


def test_running_job_blocks_deletion(world: World):
    _add(world, "poems", "poems")
    job_id = JobIds.llm_labeling("poems")
    event = cancel_registry.register_cancel_event(job_id)
    try:
        with pytest.raises(CollectionBusyError, match=job_id):
            world.service.delete("poems")
    finally:
        cancel_registry.unregister_cancel_event(job_id, event)

    assert "poems" in _chroma_names(world)
    assert world.db.get_vector_collection("poems") is not None


def test_running_job_record_blocks_deletion(world: World):
    _add(world, "poems", "poems", jobs=False)
    _record_job(world, JobIds.embedding("poems"), "poems", running=True)

    with pytest.raises(CollectionBusyError):
        world.service.delete("poems")
    assert world.db.get_vector_collection("poems") is not None


def test_no_job_of_the_collection_can_start_while_it_is_deleted(world: World, monkeypatch):
    _add(world, "poems", "poems")
    attempts = []
    real_delete = world.chroma.delete_collection

    def delete_while_a_job_starts(name):
        try:
            cancel_registry.register_cancel_event(JobIds.probe(name))
        except cancel_registry.JobIdReservedError as error:
            attempts.append(str(error))
        return real_delete(name)

    monkeypatch.setattr(world.chroma, "delete_collection", delete_while_a_job_starts)

    assert world.service.delete("poems") is True
    assert len(attempts) == 1 and "poems" in attempts[0]
    assert not cancel_registry.is_active(JobIds.probe("poems"))
    # the reservation ends with the deletion
    event = cancel_registry.register_cancel_event(JobIds.probe("poems"))
    cancel_registry.unregister_cancel_event(JobIds.probe("poems"), event)
