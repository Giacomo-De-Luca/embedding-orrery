"""deleteCollection removes one vector collection from both stores.

Runs the real resolver against an in-memory DuckDB (conftest ``db``) and a real
temporary ChromaDB store. The dataset ``poems`` backs two collections: ``poems``
itself and the re-embed ``poems_gemini``.
"""

import asyncio
from pathlib import Path

import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from backend.API import mutations, schema
from backend.clients.chromadb_client import ChromaDBClient
from backend.clients.duckdb_client import DuckDBClient
from backend.services import cancel_registry, collection_deletion, job_state
from backend.services.job_ids import JobIds
from backend.services.job_state import JobStateService

DELETE = "mutation($c: String!) { deleteCollection(collectionName: $c) }"


@pytest.fixture(autouse=True)
def isolated_resources(tmp_path: Path, monkeypatch):
    """Job state, probe artifacts and uploads in temporary folders, never resources/."""
    monkeypatch.setattr(
        job_state, "_job_state_service", JobStateService(state_file=tmp_path / "job_state.json")
    )
    monkeypatch.setattr(collection_deletion, "PROBING_RESULTS_DIR", tmp_path / "probing_results")
    monkeypatch.setattr(collection_deletion, "UPLOADS_DIR", tmp_path / "uploads")


@pytest.fixture
def stores(db: DuckDBClient, tmp_path: Path, monkeypatch):
    chroma = ChromaDBClient(db_path=str(tmp_path / "chroma"))
    db.create_dataset("poems")
    db.insert_items_batch("poems", ["i1", "i2"], ["a", "b"], [{}, {}])
    for name in ("poems", "poems_gemini"):
        db.register_vector_collection("poems", "chromadb", name, "dense")
        collection = chroma.client.create_collection(name, embedding_function=None)
        collection.add(ids=["i1", "i2"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(mutations, "get_duckdb_client", lambda: db)
    monkeypatch.setattr(mutations, "get_chromadb_client", lambda: chroma)
    yield db, chroma, tmp_path / "chroma"
    SharedSystemClient.clear_system_cache()


def _delete(name: str) -> bool:
    result = asyncio.run(schema.execute(DELETE, variable_values={"c": name}))
    assert result.errors is None, result.errors
    return result.data["deleteCollection"]


def _chroma_names(chroma: ChromaDBClient) -> set[str]:
    return {collection.name for collection in chroma.client.list_collections()}


def _index_dirs(path: Path) -> set[str]:
    return {entry.name for entry in path.iterdir() if entry.is_dir()}


def test_deletes_one_collection_from_both_stores(stores):
    db, chroma, path = stores
    dirs_before = _index_dirs(path)

    assert _delete("poems_gemini") is True

    assert _chroma_names(chroma) == {"poems"}
    assert [vc["collection_name"] for vc in db.get_vector_collections("poems")] == ["poems"]
    assert db.get_dataset("poems") is not None
    assert len(_index_dirs(path)) == len(dirs_before) - 1  # its HNSW directory too


def test_deleting_the_source_keeps_the_reembed(stores):
    db, chroma, _ = stores

    assert _delete("poems") is True

    assert _chroma_names(chroma) == {"poems_gemini"}
    assert [vc["collection_name"] for vc in db.get_vector_collections("poems")] == ["poems_gemini"]
    assert db.get_items_by_ids("poems", ["i1"])  # the shared items stay


def test_unknown_collection_reports_failure(stores):
    _, chroma, _ = stores

    assert _delete("nope") is False
    assert _chroma_names(chroma) == {"poems", "poems_gemini"}


def test_vectors_only_collection_is_deleted(stores):
    _, chroma, _ = stores
    chroma.client.create_collection("stray", embedding_function=None)

    assert _delete("stray") is True
    assert "stray" not in _chroma_names(chroma)


def test_running_job_is_reported_as_an_error(stores):
    db, chroma, _ = stores
    job_id = JobIds.probe("poems_gemini")
    event = cancel_registry.register_cancel_event(job_id)
    try:
        result = asyncio.run(schema.execute(DELETE, variable_values={"c": "poems_gemini"}))
    finally:
        cancel_registry.unregister_cancel_event(job_id, event)

    assert result.errors and "cancel it" in str(result.errors[0])
    assert "poems_gemini" in _chroma_names(chroma)


def test_duckdb_failure_reports_false_and_a_retry_finishes(stores, monkeypatch):
    db, chroma, _ = stores
    real_delete = db.delete_collection

    def broken(name):
        raise RuntimeError("disk full")

    monkeypatch.setattr(db, "delete_collection", broken)

    assert _delete("poems_gemini") is False
    assert "poems_gemini" not in _chroma_names(chroma)  # vectors are deleted first
    assert db.get_vector_collection("poems_gemini") is not None  # still listed: retryable

    monkeypatch.setattr(db, "delete_collection", real_delete)
    assert _delete("poems_gemini") is True
    assert db.get_vector_collection("poems_gemini") is None
