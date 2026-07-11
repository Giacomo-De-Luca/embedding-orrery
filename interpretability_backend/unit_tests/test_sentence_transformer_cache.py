"""Tests for the single-slot SentenceTransformer model cache.

The embedding function keeps at most one model resident: requesting a
different model evicts the cached one before loading. A fake
``sentence_transformers`` module is injected so no real model is downloaded.
"""

import sys
import types

import pytest


class FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer."""

    instances = 0

    def __init__(self, model_name_or_path=None, device=None, **kwargs):
        FakeSentenceTransformer.instances += 1
        self.model_name = model_name_or_path
        self.device = device


@pytest.fixture
def ef_class(monkeypatch):
    """Import the EF class against a stubbed sentence_transformers module."""
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    from backend.embedding_functions.specific_functions.embed_sentence_transformer import (
        SentenceTransformerEmbeddingFunction,
    )

    # Isolate cache state between tests
    monkeypatch.setattr(SentenceTransformerEmbeddingFunction, "models", {})
    FakeSentenceTransformer.instances = 0
    return SentenceTransformerEmbeddingFunction


def test_same_model_is_reused(ef_class):
    ef_a = ef_class(model_name="model-a")
    ef_b = ef_class(model_name="model-a")

    assert FakeSentenceTransformer.instances == 1
    assert ef_a._model is ef_b._model
    assert list(ef_class.models) == ["model-a"]


def test_new_model_evicts_previous(ef_class):
    ef_class(model_name="model-a")
    ef_class(model_name="model-b")

    assert list(ef_class.models) == ["model-b"]
    assert FakeSentenceTransformer.instances == 2


def test_existing_instance_keeps_working_after_eviction(ef_class):
    ef_a = ef_class(model_name="model-a")
    ef_class(model_name="model-b")

    # The old instance still holds its own reference even though the
    # class-level cache dropped it.
    assert ef_a._model.model_name == "model-a"


def test_reloading_evicted_model_creates_fresh_instance(ef_class):
    ef_class(model_name="model-a")
    ef_class(model_name="model-b")
    ef_class(model_name="model-a")

    assert list(ef_class.models) == ["model-a"]
    assert FakeSentenceTransformer.instances == 3
