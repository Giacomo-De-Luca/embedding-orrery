"""GraphQL ``embeddingModelCatalog`` and ``mediaEmbeddingSettings`` (the embed forms' data)."""

import asyncio

import pytest

from backend.API import queries, schema

CATALOG_QUERY = """
query {
  embeddingModelCatalog {
    provider modelName description modalities prompts dimension defaultFor suggested collectionCount
  }
}
"""

SETTINGS_QUERY = """
query {
  mediaEmbeddingSettings {
    defaults { imageMaxSoftTokens videoFps videoMaxFrames videoMaxSoftTokens maxAudioSeconds }
    softTokenBudgets videoFpsLimit videoFramesLimit audioSecondsLimit
    contextTokens reservedTokens audioTokensPerSecond defaultModel
  }
}
"""

LOCAL_CHECKPOINT = "/Users/me/models/gemma-finetuned"


def _execute(query: str) -> dict:
    result = asyncio.run(schema.execute(query))
    assert result.errors is None, result.errors
    return result.data


@pytest.fixture
def used_models(db, monkeypatch):
    db.create_dataset("ds")
    used = [
        ("a", "sentence_transformers", "all-MiniLM-L6-v2"),
        ("b", "sentence_transformers", "all-MiniLM-L6-v2"),
        ("c", "sentence_transformers", LOCAL_CHECKPOINT),
        ("d", "neuronpedia", "sae_explanations"),
        ("e", None, None),
    ]
    for name, provider, model in used:
        db.register_vector_collection(
            "ds", "chromadb", name, "dense", embedding_provider=provider, embedding_model=model
        )
    monkeypatch.setattr(queries, "get_duckdb_client", lambda: db)
    monkeypatch.setattr(queries, "is_read_only", lambda: False)
    return db


def _entries(data):
    return {(e["provider"], e["modelName"]): e for e in data["embeddingModelCatalog"]}


class TestCatalogQuery:
    def test_catalog_models_with_dimensions_and_prompts(self, used_models):
        entries = _entries(_execute(CATALOG_QUERY))

        eg2 = entries[("SENTENCE_TRANSFORMERS", "google/embeddinggemma-2")]
        assert eg2["suggested"] and eg2["defaultFor"] == ["multimodal"]
        assert set(eg2["modalities"]) == {"text", "image", "audio", "video"}
        assert "Retrieval-query" in eg2["prompts"] and eg2["dimension"] == 768
        assert entries[("OPENAI", "text-embedding-3-small")]["dimension"] == 1536

    def test_used_models_are_counted(self, used_models):
        entries = _entries(_execute(CATALOG_QUERY))
        assert entries[("SENTENCE_TRANSFORMERS", "all-MiniLM-L6-v2")]["collectionCount"] == 2

    def test_models_only_used_appear_unsuggested(self, used_models):
        local = _entries(_execute(CATALOG_QUERY))[("SENTENCE_TRANSFORMERS", LOCAL_CHECKPOINT)]
        assert local["suggested"] is False and local["collectionCount"] == 1
        assert local["modalities"] == ["text"]

    def test_unknown_providers_are_skipped(self, used_models):
        providers = {e["provider"] for e in _execute(CATALOG_QUERY)["embeddingModelCatalog"]}
        assert "neuronpedia" not in {p.lower() for p in providers}

    def test_read_only_instances_hide_used_models(self, used_models, monkeypatch):
        monkeypatch.setattr(queries, "is_read_only", lambda: True)
        entries = _entries(_execute(CATALOG_QUERY))
        assert ("SENTENCE_TRANSFORMERS", LOCAL_CHECKPOINT) not in entries
        assert entries[("SENTENCE_TRANSFORMERS", "all-MiniLM-L6-v2")]["collectionCount"] == 0


def test_media_settings_query():
    settings = _execute(SETTINGS_QUERY)["mediaEmbeddingSettings"]
    assert settings["defaults"]["imageMaxSoftTokens"] == 280
    assert settings["defaults"]["videoMaxFrames"] == 32
    assert settings["softTokenBudgets"] == [70, 140, 280, 560, 1120]
    assert settings["contextTokens"] == 8192 and settings["audioTokensPerSecond"] == 25
    assert settings["defaultModel"] == "google/embeddinggemma-2"
