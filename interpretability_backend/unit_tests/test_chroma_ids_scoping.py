"""Vector search scoped to an ID subset (chromadb ``query(ids=...)``).

Filtered semantic search relies on chromadb's ``ids`` argument restricting the
nearest-neighbour search to an allowed subset. This pins that library
behaviour on a real (temporary) persistent client, plus the wrapper's handling
of ``n_results`` larger than the subset and of an empty subset.
"""

from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings

from backend.clients.chromadb_client import ChromaDBClient

IDS = [f"item-{i}" for i in range(20)]


@pytest.fixture
def chroma_path(tmp_path: Path) -> Path:
    client = chromadb.PersistentClient(
        path=str(tmp_path), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.create_collection("vectors")
    # item-i sits at distance i from the query point [0, 0].
    collection.add(ids=IDS, embeddings=[[float(i), 0.0] for i in range(20)])
    return tmp_path


def test_library_query_restricts_results_to_ids(chroma_path: Path):
    client = chromadb.PersistentClient(
        path=str(chroma_path), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection("vectors")
    allowed = ["item-15", "item-3", "item-9", "item-12"]

    result = collection.query(query_embeddings=[[0.0, 0.0]], n_results=3, ids=allowed)

    assert result["ids"][0] == ["item-3", "item-9", "item-12"]


def test_wrapper_caps_n_results_to_the_scope(chroma_path: Path):
    client = ChromaDBClient(db_path=str(chroma_path))
    allowed = ["item-7", "item-2"]

    result = client.semantic_search(
        "vectors", query_embeddings=[[0.0, 0.0]], n_results=10, ids=allowed
    )

    assert result["ids"][0] == ["item-2", "item-7"]
    assert len(result["similarities"][0]) == 2


def test_wrapper_ignores_scope_ids_without_vectors(chroma_path: Path):
    # A dataset can hold items its vector collection lacks (empty texts skipped
    # at embed time, partial or resumed jobs); chromadb raises on such ids.
    client = ChromaDBClient(db_path=str(chroma_path))
    allowed = ["item-7", "never-embedded", "item-2"]

    result = client.semantic_search(
        "vectors", query_embeddings=[[0.0, 0.0]], n_results=5, ids=allowed
    )

    assert result["ids"][0] == ["item-2", "item-7"]


def test_wrapper_scope_of_only_missing_ids_returns_no_hits(chroma_path: Path):
    client = ChromaDBClient(db_path=str(chroma_path))

    result = client.semantic_search("vectors", query_embeddings=[[0.0, 0.0]], ids=["gone"])

    assert result["ids"] == [[]]


def test_wrapper_empty_scope_returns_no_hits(chroma_path: Path):
    client = ChromaDBClient(db_path=str(chroma_path))

    result = client.semantic_search("vectors", query_embeddings=[[0.0, 0.0]], ids=[])

    assert result["ids"] == [[]]
    assert result["similarities"] == [[]]
