"""Modalities and query prompts in the embedding-function factory."""

import pytest

from backend.embedding_functions import create_embedding_function as factory
from backend.embedding_functions.config import EmbeddingModelConfig, EmbeddingProvider
from backend.utils.embedding_model_catalog import TASK_PREFIX_FORMAT

EG2_MODEL = "google/embeddinggemma-2"
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


class FakeLimits:
    """Stands in for ModelMaxLengthResolver: a fixed limit, the real request checks."""

    def __init__(self, limit: int | None):
        self.limit = limit

    def effective(self, provider: str, model_name: str, requested: int | None) -> int | None:
        if requested is None:
            return self.limit
        if self.limit is not None and requested > self.limit:
            raise ValueError(f"max_length {requested} exceeds the limit of {self.limit:,} tokens")
        return requested


def test_model_limit_and_batch_size_reach_the_sentence_transformer():
    config = _st_config()
    config.batch_size = 16

    factory.create_embedding_function(config, device="cpu", known_dimension=768)

    call = RecordingEF.calls[0]
    assert call["max_length"] == 8192  # EmbeddingGemma 2's limit, from the shipped catalog
    assert call["batch_size"] == 16


def test_requested_cap_below_the_limit_is_used():
    config = _st_config()
    config.max_length = 1024

    factory.create_embedding_function(config, device="cpu", known_dimension=768)

    assert RecordingEF.calls[0]["max_length"] == 1024


def test_cap_above_the_model_limit_is_rejected():
    config = _st_config()
    config.max_length = 9000

    with pytest.raises(ValueError, match="8,192"):
        factory.create_embedding_function(config, device="cpu", known_dimension=768)


def test_limits_reach_the_qwen_embedder(monkeypatch):
    from backend.embedding_functions.specific_functions import embed_qwen

    calls = []

    class RecordingQwen:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(embed_qwen, "EmbedTextQWEN", RecordingQwen)
    config = EmbeddingModelConfig(
        provider=EmbeddingProvider.QWEN, model_name="Qwen/Qwen3-Embedding-0.6B", batch_size=8
    )

    factory.create_embedding_function(
        config, device="cpu", known_dimension=1024, max_length_resolver=FakeLimits(32768)
    )

    assert calls[0]["max_length"] == 32768 and calls[0]["batch_size"] == 8


def test_unknown_limit_leaves_the_embedder_default(monkeypatch):
    from backend.embedding_functions.specific_functions import embed_qwen

    calls = []
    monkeypatch.setattr(embed_qwen, "EmbedTextQWEN", lambda **kwargs: calls.append(kwargs))
    config = EmbeddingModelConfig(provider=EmbeddingProvider.QWEN, model_name="q")

    factory.create_embedding_function(
        config, device="cpu", known_dimension=1024, max_length_resolver=FakeLimits(None)
    )

    assert "max_length" not in calls[0] and "batch_size" not in calls[0]


class TestDeviceFor:
    @pytest.mark.parametrize(
        "provider",
        [EmbeddingProvider.GEMINI, EmbeddingProvider.OPENAI, EmbeddingProvider.COHERE],
    )
    def test_api_providers_never_probe_the_device(self, monkeypatch, provider):
        def probe():
            raise AssertionError("an API provider must not import torch")

        monkeypatch.setattr(factory, "get_device", probe)
        assert factory.device_for(provider) == "cpu"

    @pytest.mark.parametrize(
        "provider",
        [EmbeddingProvider.SENTENCE_TRANSFORMERS, EmbeddingProvider.QWEN, EmbeddingProvider.BGE],
    )
    def test_local_providers_use_the_detected_device(self, monkeypatch, provider):
        monkeypatch.setattr(factory, "get_device", lambda: "mps")
        assert factory.device_for(provider) == "mps"


SEARCH_QUERY = "task: search result | query: "
TITLED = "title: {title} | text: "


class RecordingGemini:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        RecordingGemini.calls.append(kwargs)


class TestTaskPrefixes:
    """Task-prefix models: documents arrive prefixed by the pipeline, queries
    get the collection's query prefix (utils/embedding_tasks.py)."""

    @pytest.fixture(autouse=True)
    def recording_gemini(self, monkeypatch):
        from backend.embedding_functions.specific_functions import embed_gemini

        RecordingGemini.calls = []
        monkeypatch.setattr(embed_gemini, "EmbedTextGemini", RecordingGemini)

    @staticmethod
    def _marked(provider=EmbeddingProvider.SENTENCE_TRANSFORMERS, model=EG2_MODEL, **fields):
        return EmbeddingModelConfig(
            provider=provider,
            model_name=model,
            prompt=TITLED,
            task=SEARCH_QUERY,
            prompt_format=TASK_PREFIX_FORMAT,
            **fields,
        )

    def test_marked_documents_reach_the_model_without_a_prompt(self):
        factory.create_embedding_function(self._marked(), "cpu", known_dimension=768)
        call = RecordingEF.calls[0]
        assert call["prompt"] is None and call["query_prompt"] is None

    def test_marked_queries_get_the_stored_query_prefix(self):
        factory.create_embedding_function(self._marked(), "cpu", known_dimension=768, is_query=True)
        call = RecordingEF.calls[0]
        assert call["prompt"] is None and call["query_prompt"] == SEARCH_QUERY

    def test_an_explicit_prompt_name_maps_through_its_task(self):
        factory.create_embedding_function(
            self._marked(), "cpu", known_dimension=768, is_query=True, query_prompt="STS"
        )
        assert RecordingEF.calls[0]["query_prompt"] == "task: sentence similarity | query: "

    def test_prompt_names_keep_the_sentence_transformers_path(self):
        factory.create_embedding_function(
            _st_config("Retrieval-document"), "cpu", known_dimension=768, is_query=True
        )
        call = RecordingEF.calls[0]
        assert call["prompt"] == "Retrieval-document" and call["query_prompt"] is None

    def test_gemini_2_never_sends_a_task_type(self):
        """Also on the query path, which rebuilds a stored CLUSTERING."""
        config = EmbeddingModelConfig(
            provider=EmbeddingProvider.GEMINI,
            model_name="gemini-embedding-2",
            task_type="CLUSTERING",
        )
        factory.create_embedding_function(config, "cpu", known_dimension=3072, is_query=True)
        assert RecordingGemini.calls[0]["task_type"] is None
        assert RecordingGemini.calls[0]["text_prefix"] is None

    def test_gemini_2_queries_get_the_query_prefix_and_documents_none(self):
        config = self._marked(EmbeddingProvider.GEMINI, "gemini-embedding-2")
        factory.create_embedding_function(config, "cpu", known_dimension=3072)
        factory.create_embedding_function(config, "cpu", known_dimension=3072, is_query=True)
        assert [c["text_prefix"] for c in RecordingGemini.calls] == [None, SEARCH_QUERY]

    def test_gemini_2_resolves_explicit_query_prompts(self):
        config = EmbeddingModelConfig(
            provider=EmbeddingProvider.GEMINI, model_name="gemini-embedding-2"
        )
        factory.create_embedding_function(
            config, "cpu", known_dimension=3072, is_query=True, query_prompt="Retrieval-query"
        )
        assert RecordingGemini.calls[0]["text_prefix"] == SEARCH_QUERY
        with pytest.raises(ValueError, match="Retrieval-document"):
            factory.create_embedding_function(
                config,
                "cpu",
                known_dimension=3072,
                is_query=True,
                query_prompt="Retrieval-document",
            )

    def test_gemini_001_keeps_its_task_type(self):
        config = EmbeddingModelConfig(
            provider=EmbeddingProvider.GEMINI,
            model_name="gemini-embedding-001",
            task_type="RETRIEVAL_QUERY",
        )
        factory.create_embedding_function(
            config, "cpu", known_dimension=3072, is_query=True, query_prompt="Retrieval-query"
        )
        default = EmbeddingModelConfig(
            provider=EmbeddingProvider.GEMINI, model_name="gemini-embedding-001"
        )
        factory.create_embedding_function(default, "cpu", known_dimension=3072)
        assert [(c["task_type"], c["text_prefix"]) for c in RecordingGemini.calls] == [
            ("RETRIEVAL_QUERY", None),
            ("SEMANTIC_SIMILARITY", None),
        ]
