"""Known-model catalog (config/embedding_models.json, utils/embedding_model_catalog.py).

The catalog feeds the embed forms' model suggestions, the prompt-name
vocabulary, the document -> query prompt map, the multimodal default model and
the encoder towers a multimodal model can skip.
"""

import json
import re
from pathlib import Path

import pytest

from backend.utils.embedding_model_catalog import (
    CATALOG_FILE,
    CatalogModel,
    EmbeddingModelCatalog,
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "embedding_visualization"
EG2 = "google/embeddinggemma-2"


def _catalog(**overrides) -> EmbeddingModelCatalog:
    raw = {
        "query_prompt_map": {"Retrieval-document": "Retrieval-query"},
        "extra_prompt_names": ["s2p"],
        "providers": {
            "sentence_transformers": [
                {"model": "all-MiniLM-L6-v2"},
                {
                    "model": "org/media-model",
                    "modalities": ["text", "image", "audio"],
                    "prompts": ["query", "document"],
                    "default_for": ["multimodal"],
                    "encoders": {"image": "vision_config", "audio": "audio_config"},
                },
            ],
            "openai": [{"model": "text-embedding-3-small"}],
        },
    }
    raw.update(overrides)
    return EmbeddingModelCatalog.from_mapping(raw)


class TestShippedCatalog:
    @pytest.fixture(scope="class")
    def catalog(self) -> EmbeddingModelCatalog:
        return EmbeddingModelCatalog.from_file(CATALOG_FILE)

    def test_embeddinggemma_2_is_the_multimodal_default(self, catalog):
        model = catalog.default_model_for("multimodal")
        assert (model.provider, model.model) == ("sentence_transformers", EG2)
        assert set(model.modalities) == {"text", "image", "audio", "video"}

    def test_video_runs_on_the_vision_tower(self, catalog):
        encoders = catalog.find("sentence_transformers", EG2).encoders
        assert encoders == {
            "image": "vision_config",
            "video": "vision_config",
            "audio": "audio_config",
        }

    def test_embeddinggemma_prompt_names_are_known(self, catalog):
        for name in ("Retrieval-query", "Retrieval-document", "STS", "query", "SearchQuery"):
            assert name in catalog.prompt_vocabulary
        assert {"s2p", "s2s"} <= catalog.prompt_vocabulary

    def test_query_prompt_map(self, catalog):
        assert catalog.query_prompt_for("Retrieval-document") == "Retrieval-query"
        assert catalog.query_prompt_for("Document") == "SearchQuery"
        assert catalog.query_prompt_for("document") == "query"
        assert catalog.query_prompt_for("Clustering") is None
        assert catalog.query_prompt_for(None) is None

    def test_gemini_query_task_type_map(self, catalog):
        assert catalog.query_task_type_for("RETRIEVAL_DOCUMENT") == "RETRIEVAL_QUERY"
        assert catalog.query_task_type_for("CLUSTERING") == "CLUSTERING"
        assert catalog.query_task_type_for(None) is None

    def test_frontend_default_models_are_in_the_catalog(self, catalog):
        """lib/utils/embeddingProviders.ts defaults must stay suggestible."""
        ts_source = (FRONTEND_DIR / "lib" / "utils" / "embeddingProviders.ts").read_text()
        defaults = re.findall(r"(\w+):\s*\{\s*defaultModel:\s*'([^']+)'", ts_source)
        assert len(defaults) == 8
        for provider_name, model_name in defaults:
            assert catalog.find(provider_name.lower(), model_name) is not None, (
                provider_name,
                model_name,
            )

    def test_every_entry_is_a_catalog_model(self, catalog):
        models = catalog.models()
        assert models and all(isinstance(m, CatalogModel) for m in models)
        assert {m.provider for m in models} == set(catalog.providers)


class TestLookups:
    def test_find_ignores_case(self):
        catalog = _catalog()
        assert catalog.find("sentence_transformers", "ALL-minilm-L6-v2").model == "all-MiniLM-L6-v2"
        assert catalog.find("openai", "all-MiniLM-L6-v2") is None
        assert catalog.find("sentence_transformers", "missing") is None

    def test_models_filters_by_provider(self):
        catalog = _catalog()
        assert [m.model for m in catalog.models("openai")] == ["text-embedding-3-small"]
        assert len(catalog.models()) == 3

    def test_text_is_the_default_modality(self):
        assert _catalog().find("openai", "text-embedding-3-small").modalities == ("text",)

    def test_prompt_vocabulary_is_the_union_of_prompts_and_extras(self):
        assert _catalog().prompt_vocabulary == frozenset({"query", "document", "s2p"})

    def test_missing_default_raises(self):
        catalog = _catalog(providers={"sentence_transformers": [{"model": "all-MiniLM-L6-v2"}]})
        with pytest.raises(ValueError, match="multimodal"):
            catalog.default_model_for("multimodal")

    def test_missing_file_means_an_empty_catalog(self, tmp_path):
        catalog = EmbeddingModelCatalog.from_file(tmp_path / "absent.json")
        assert catalog.models() == ()

    def test_loads_from_a_file(self, tmp_path):
        path = tmp_path / "catalog.json"
        path.write_text(
            json.dumps({"providers": {"qwen": [{"model": "Qwen/Qwen3-Embedding-0.6B"}]}})
        )
        assert EmbeddingModelCatalog.from_file(path).find("qwen", "qwen/qwen3-embedding-0.6b")


class TestValidation:
    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"typo": 1}, "Unknown"),
            ({"providers": {"not_a_provider": [{"model": "m"}]}}, "not_a_provider"),
            ({"providers": {"openai": [{"model": "m", "colour": "red"}]}}, "colour"),
            ({"providers": {"openai": [{"model": ""}]}}, "model"),
            ({"providers": {"openai": [{"model": "m"}, {"model": "M"}]}}, "twice"),
            ({"providers": {"openai": [{"model": "m", "modalities": ["image"]}]}}, "text"),
            ({"providers": {"openai": [{"model": "m", "modalities": ["text", "smell"]}]}}, "smell"),
            (
                {"providers": {"openai": [{"model": "m", "encoders": {"image": "vision_config"}}]}},
                "encoders",
            ),
            ({"providers": {"openai": [{"model": "m", "default_for": ["vector"]}]}}, "default_for"),
            ({"providers": {"openai": [{"model": "m", "default_for": ["multimodal"]}]}}, "media"),
            ({"providers": {"openai": [{"model": "m", "provider": "qwen"}]}}, "provider"),
            ({"query_prompt_map": {"Retrieval-document": 3}}, "query_prompt_map"),
        ],
    )
    def test_rejects_invalid_catalogs(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            _catalog(**overrides)

    def test_rejects_two_defaults_for_one_data_type(self):
        second = {
            "model": "org/other-media-model",
            "modalities": ["text", "image"],
            "default_for": ["multimodal"],
        }
        raw_models = [
            {"model": "a", "modalities": ["text", "image"], "default_for": ["multimodal"]}
        ]
        with pytest.raises(ValueError, match="default"):
            _catalog(providers={"sentence_transformers": raw_models + [second]})
