"""Modalities and query prompts in the embedding-function factory."""

import pytest

from backend.embedding_functions import create_embedding_function as factory
from backend.embedding_functions.config import EmbeddingModelConfig, EmbeddingProvider

ST_CLASS = (
    "backend.embedding_functions.specific_functions.embed_sentence_transformer."
    "SentenceTransformerEmbeddingFunction"
)


class RecordingEF:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        RecordingEF.calls.append(kwargs)

    def __call__(self, input):
        return [[0.0, 0.0] for _ in input]


@pytest.fixture(autouse=True)
def recording_ef(monkeypatch):
    RecordingEF.calls = []
    monkeypatch.setattr(ST_CLASS, RecordingEF)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)


def _st_config(prompt=None) -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        model_name="google/embeddinggemma-2",
        prompt=prompt,
    )


def test_text_is_the_default_modality():
    factory.create_embedding_function(_st_config(), device="cpu", known_dimension=768)
    call = RecordingEF.calls[0]
    assert call["modalities"] == ("text",)
    assert call["is_query"] is False and call["query_prompt"] is None
    assert "token" not in call


def test_media_modalities_reach_the_sentence_transformer():
    factory.create_embedding_function(
        _st_config("Retrieval-document"),
        device="cpu",
        known_dimension=768,
        modalities=["text", "image", "audio"],
    )
    call = RecordingEF.calls[0]
    assert call["modalities"] == ("text", "image", "audio")
    assert call["prompt"] == "Retrieval-document"


def test_query_settings_reach_the_sentence_transformer():
    factory.create_embedding_function(
        _st_config("Retrieval-document"),
        device="cpu",
        known_dimension=768,
        is_query=True,
        query_prompt="STS",
    )
    call = RecordingEF.calls[0]
    assert call["is_query"] is True and call["query_prompt"] == "STS"


@pytest.mark.parametrize("provider", [EmbeddingProvider.OPENAI, EmbeddingProvider.GEMINI])
def test_text_only_providers_reject_media(provider):
    config = EmbeddingModelConfig(provider=provider, model_name="m")
    with pytest.raises(ValueError, match="sentence-transformers"):
        factory.create_embedding_function(
            config, device="cpu", known_dimension=8, modalities=["image"]
        )


def test_unknown_dimension_is_probed_and_recorded(monkeypatch, tmp_path):
    recorded = {}
    monkeypatch.setattr(factory.KnownDimensions, "get", lambda self, name: None)
    monkeypatch.setattr(
        factory.KnownDimensions, "record", lambda self, name, dim: recorded.update({name: dim})
    )

    _, dim = factory.create_embedding_function(_st_config(), device="cpu")

    assert dim == 2 and recorded == {"google/embeddinggemma-2": 2}
