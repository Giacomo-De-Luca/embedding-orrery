"""GraphQL EmbeddingModelInput -> EmbeddingModelConfig (API/converters.py).

Task-prefix models (EmbeddingGemma, Gemini Embedding 2) get the
``prompt_format`` marker and lose the Gemini ``task_type`` they don't support;
everything else converts as before.
"""

from backend.API.converters import build_embedding_model_config
from backend.API.types import EmbeddingModelInput, EmbeddingProviderEnum
from backend.embedding_functions.config import EmbeddingModelConfig, EmbeddingProvider
from backend.utils.embedding_model_catalog import TASK_PREFIX_FORMAT

SEARCH_QUERY = "task: search result | query: "
TITLED = "title: {title} | text: "


def _convert(**fields) -> EmbeddingModelConfig:
    return build_embedding_model_config(EmbeddingModelInput(**fields), batch_size=8)


def test_gemini_2_gets_the_marker_and_no_task_type():
    config = _convert(
        provider=EmbeddingProviderEnum.GEMINI,
        model_name="gemini-embedding-2",
        task_type="CLUSTERING",
        prompt=TITLED,
        task=SEARCH_QUERY,
    )
    assert config.prompt_format == TASK_PREFIX_FORMAT
    assert config.task_type is None
    assert (config.prompt, config.task) == (TITLED, SEARCH_QUERY)


def test_embeddinggemma_with_a_prefix_template_gets_the_marker():
    config = _convert(
        provider=EmbeddingProviderEnum.SENTENCE_TRANSFORMERS,
        model_name="google/embeddinggemma-2",
        prompt=TITLED,
        task=SEARCH_QUERY,
    )
    assert config.prompt_format == TASK_PREFIX_FORMAT
    assert config.batch_size == 8


def test_prompt_names_keep_the_old_path():
    config = _convert(
        provider=EmbeddingProviderEnum.SENTENCE_TRANSFORMERS,
        model_name="google/embeddinggemma-2",
        prompt="Retrieval-document",
    )
    assert config.prompt_format is None
    assert config.prompt == "Retrieval-document"


def test_gemini_001_keeps_its_task_type():
    config = _convert(
        provider=EmbeddingProviderEnum.GEMINI,
        model_name="gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
    )
    assert config.prompt_format is None
    assert config.task_type == "RETRIEVAL_DOCUMENT"


def test_none_input_is_none():
    assert build_embedding_model_config(None) is None


class TestEmbeddingModelConfig:
    def _config(self) -> EmbeddingModelConfig:
        return EmbeddingModelConfig(
            provider=EmbeddingProvider.GEMINI,
            model_name="gemini-embedding-2",
            prompt=TITLED,
            task=SEARCH_QUERY,
            prompt_format=TASK_PREFIX_FORMAT,
        )

    def test_job_config_keeps_the_marker(self):
        assert self._config().to_job_config()["prompt_format"] == TASK_PREFIX_FORMAT

    def test_collection_metadata_records_the_model_and_its_prefixes(self):
        assert self._config().collection_metadata(3072) == {
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 3072,
            "embedding_task": SEARCH_QUERY,
            "embedding_task_type": None,
            "embedding_prompt": TITLED,
            "embedding_prompt_format": TASK_PREFIX_FORMAT,
        }


def test_model_names_are_trimmed_like_the_form_matches_them():
    """The embed form matches models after trimming; the marker must agree."""
    config = _convert(
        provider=EmbeddingProviderEnum.SENTENCE_TRANSFORMERS,
        model_name=" google/embeddinggemma-300m ",
        prompt=TITLED,
        task=SEARCH_QUERY,
    )
    assert config.model_name == "google/embeddinggemma-300m"
    assert config.prompt_format == TASK_PREFIX_FORMAT
