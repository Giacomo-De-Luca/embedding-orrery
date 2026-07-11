"""HuggingFace authentication wiring for local SentenceTransformer models."""

from __future__ import annotations

from backend.embedding_functions import create_embedding_function as factory
from backend.embedding_functions.config import (
    EmbeddingModelConfig,
    EmbeddingProvider,
)


class DummySentenceTransformerEmbeddingFunction:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)

    def __call__(self, input):
        return [[0.0, 0.0, 0.0] for _ in input]


def test_sentence_transformer_receives_huggingface_api_key(monkeypatch):
    DummySentenceTransformerEmbeddingFunction.calls = []
    login_calls: list[str] = []

    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_test_token")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(factory, "_hf_logged_in", False)
    monkeypatch.setattr(factory, "login", lambda token, add_to_git_credential: login_calls.append(token))
    monkeypatch.setattr(
        "backend.embedding_functions.specific_functions.embed_sentence_transformer.SentenceTransformerEmbeddingFunction",
        DummySentenceTransformerEmbeddingFunction,
    )

    ef, dim = factory.create_embedding_function(
        EmbeddingModelConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            model_name="google/embeddinggemma-300m",
        ),
        device="cpu",
        known_dimension=768,
    )

    assert isinstance(ef, DummySentenceTransformerEmbeddingFunction)
    assert dim == 768
    assert login_calls == ["hf_test_token"]
    assert DummySentenceTransformerEmbeddingFunction.calls[0]["token"] == "hf_test_token"


def test_sentence_transformer_accepts_hf_token_alias(monkeypatch):
    DummySentenceTransformerEmbeddingFunction.calls = []
    login_calls: list[str] = []

    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_alias_token")
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(factory, "_hf_logged_in", False)
    monkeypatch.setattr(factory, "login", lambda token, add_to_git_credential: login_calls.append(token))
    monkeypatch.setattr(
        "backend.embedding_functions.specific_functions.embed_sentence_transformer.SentenceTransformerEmbeddingFunction",
        DummySentenceTransformerEmbeddingFunction,
    )

    factory.create_embedding_function(
        EmbeddingModelConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            model_name="google/embeddinggemma-300m",
        ),
        device="cpu",
        known_dimension=768,
    )

    assert login_calls == ["hf_alias_token"]
    assert DummySentenceTransformerEmbeddingFunction.calls[0]["token"] == "hf_alias_token"
