"""ChromaDBClient.embed_query / semantic_search: query vectors from the collection's own model.

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
from backend.utils.embedding_model_catalog import TASK_PREFIX_FORMAT

SEARCH_QUERY = "task: search result | query: "


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
    marked = raw.create_collection(
        "gemini_2_docs",
        metadata={
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_task": SEARCH_QUERY,
            "embedding_prompt": "title: {title} | text: ",
            "embedding_prompt_format": TASK_PREFIX_FORMAT,
            "embedding_dim": 2,
        },
        embedding_function=None,
    )
    marked.add(ids=["near", "far"], embeddings=[[4.0, 1.0], [-1.0, 0.2]])
    raw.create_collection("bare", metadata={"kind": "vectors-only"})

    created = []

    def fake_factory(config, device, known_dimension=None, is_query=False, query_prompt=None):
        created.append(
            {
                "config": config,
                "is_query": is_query,
                "dim": known_dimension,
                "query_prompt": query_prompt,
            }
        )
        return StubEmbeddingFunction(), 2

    monkeypatch.setattr(chromadb_client, "create_embedding_function", fake_factory)
    monkeypatch.setattr(chromadb_client, "device_for", lambda provider: "cpu")
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
    assert call["query_prompt"] is None
    assert call["config"].prompt_format is None


def test_explicit_query_prompt_reaches_the_factory_on_its_own(client):
    client.embed_query("gemini_docs", "cats", query_prompt="Retrieval-query")
    call = client.created[-1]
    assert call["query_prompt"] == "Retrieval-query"
    assert call["config"].prompt == "doc-prompt"


def test_a_task_prefix_collection_carries_its_marker_and_query_prefix(client):
    client.embed_query("gemini_2_docs", "cats")
    config = client.created[-1]["config"]
    assert config.prompt_format == TASK_PREFIX_FORMAT
    assert config.task == SEARCH_QUERY


def test_collection_without_model_raises_instead_of_using_a_default(client):
    with pytest.raises(ValueError, match="no embedding model"):
        client.embed_query("bare", "cats")


class TestSemanticSearch:
    def test_text_queries_search_by_the_query_vector(self, client):
        results = client.semantic_search("gemini_2_docs", query_texts=["cats"], n_results=2)
        assert results["ids"][0] == ["near", "far"]
        assert results["similarities"][0][0] == pytest.approx(1.0)
        assert client.created[-1]["is_query"] is True

    def test_a_failing_embedding_function_raises_instead_of_falling_back(self, client, monkeypatch):
        def broken_factory(*args, **kwargs):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(chromadb_client, "create_embedding_function", broken_factory)
        with pytest.raises(RuntimeError, match="model unavailable"):
            client.semantic_search("gemini_2_docs", query_texts=["cats"])

    def test_collection_without_model_cannot_take_text_queries(self, client):
        with pytest.raises(ValueError, match="no embedding model"):
            client.semantic_search("bare", query_texts=["cats"])

    def test_vector_queries_need_no_model(self, client):
        results = client.semantic_search(
            "gemini_2_docs", query_embeddings=[[-1.0, 0.2]], n_results=1
        )
        assert results["ids"][0] == ["far"]
        assert client.created == []
