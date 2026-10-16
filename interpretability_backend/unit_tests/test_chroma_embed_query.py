"""ChromaDBClient.embed_query: query vectors from the collection's own model.

The embedding-function factory is stubbed (no model download); the collection
is a real temporary persistent client so the metadata path is exercised.
"""

from pathlib import Path

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings

from backend.clients import chromadb_client
from backend.clients.chromadb_client import ChromaDBClient


class StubEmbeddingFunction:
    def __call__(self, texts):
        return [np.array([float(len(t)), 1.0], dtype=np.float32) for t in texts]


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> ChromaDBClient:
    raw = chromadb.PersistentClient(
        path=str(tmp_path), settings=Settings(anonymized_telemetry=False)
    )
    raw.create_collection(
        "gemini_docs",
        metadata={
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-001",
            "embedding_task_type": "RETRIEVAL_DOCUMENT",
            "embedding_prompt": "doc-prompt",
            "embedding_dim": 2,
        },
    )
    raw.create_collection("bare", metadata={"kind": "vectors-only"})

    created = []

    def fake_factory(config, device, known_dimension=None, is_query=False):
        created.append({"config": config, "is_query": is_query, "dim": known_dimension})
        return StubEmbeddingFunction(), 2

    monkeypatch.setattr(chromadb_client, "create_embedding_function", fake_factory)
    monkeypatch.setattr(chromadb_client, "get_device", lambda: "cpu")
    wrapper = ChromaDBClient(db_path=str(tmp_path))
    wrapper.created = created
    return wrapper


def test_embeds_with_the_query_mode_model(client):
    vector = client.embed_query("gemini_docs", "cats")

    assert vector.dtype == np.float64
    np.testing.assert_allclose(vector, [4.0, 1.0])
    call = client.created[-1]
    assert call["is_query"] is True
    assert call["dim"] == 2
    # Document task type maps to its query counterpart; the stored prompt applies.
    assert call["config"].task_type == "RETRIEVAL_QUERY"
    assert call["config"].prompt == "doc-prompt"


def test_explicit_query_prompt_overrides_the_stored_one(client):
    client.embed_query("gemini_docs", "cats", query_prompt="Retrieval-query")
    assert client.created[-1]["config"].prompt == "Retrieval-query"


def test_collection_without_model_raises_instead_of_using_a_default(client):
    with pytest.raises(ValueError, match="no embedding model"):
        client.embed_query("bare", "cats")
