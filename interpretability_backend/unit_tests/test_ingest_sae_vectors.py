"""SAE explanation vectors stored in ChromaDB by ``ingest_sae``.

Re-ingesting replaces the collection (and its HNSW index directory), and the
collection records no embedding function: with the argument omitted, chromadb
records its default 384-d embedder ({"type": "known", "name": "default"}).
"""

from pathlib import Path

import pandas as pd
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from backend.API import chromadb_instance
from backend.clients.chromadb_client import ChromaDBClient
from backend.embedding_functions import ingest_sae
from backend.utils import duckdb_sync

COLLECTION = "sae_test-explanations"
FEATURES = pd.DataFrame(
    {
        "index": [0, 1, 2],
        "vector": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "label": ["first", "second", None],
        "density": [0.1, 0.2, 0.3],
    }
)


@pytest.fixture
def chroma(tmp_path: Path, monkeypatch) -> ChromaDBClient:
    client = ChromaDBClient(db_path=str(tmp_path))
    monkeypatch.setattr(chromadb_instance, "get_chromadb_client", lambda: client)
    monkeypatch.setattr(duckdb_sync, "sync_dataset_and_collection", lambda *a, **k: None)
    monkeypatch.setattr(duckdb_sync, "sync_items", lambda *a, **k: None)
    yield client
    SharedSystemClient.clear_system_cache()


def _store() -> None:
    ingest_sae._store_feature_vectors(
        None, FEATURES, "model", "sae", COLLECTION, "features.parquet", None, len(FEATURES)
    )


def _index_dirs(chroma: ChromaDBClient) -> set[str]:
    return {entry.name for entry in Path(chroma.db_path).iterdir() if entry.is_dir()}


def test_collection_records_no_embedding_function(chroma: ChromaDBClient):
    _store()

    collection = chroma.client.get_collection(COLLECTION, embedding_function=None)
    assert collection.count() == 3
    assert collection.configuration_json.get("embedding_function") is None


def test_reingest_replaces_the_collection_and_its_index(chroma: ChromaDBClient):
    _store()
    [first_dir] = _index_dirs(chroma)

    _store()

    assert chroma.client.get_collection(COLLECTION, embedding_function=None).count() == 3
    [second_dir] = _index_dirs(chroma)
    assert second_dir != first_dir
