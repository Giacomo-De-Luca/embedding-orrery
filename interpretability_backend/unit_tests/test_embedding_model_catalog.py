"""Known-model catalog (config/embedding_models.json, utils/embedding_model_catalog.py).

The catalog feeds the embed forms' model suggestions, the prompt-name
vocabulary, the document -> query prompt map, the multimodal default model,
the encoder towers a multimodal model can skip, and the task-prefix table of
the models that take Google's ``task: ... | query: `` prefixes.
"""

import json
import re
from pathlib import Path

import pytest

from backend.utils.embedding_model_catalog import (
    CATALOG_FILE,
    CatalogModel,
    EmbeddingModelCatalog,
    TaskPrefixTable,
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "embedding_visualization"
EG2 = "google/embeddinggemma-2"
SEARCH_QUERY = "task: search result | query: "
TITLE_DOCUMENT = "title: none | text: "

TASKS = {
    "default_task": "search",
    "tasks": [
        {
            "id": "search",
            "label": "Search",
            "query_prefix": SEARCH_QUERY,
            "document_prefix": TITLE_DOCUMENT,
            "prompt_names": ["Retrieval-query"],
        },
        {
            "id": "clustering",
            "label": "Clustering",
            "query_prefix": "task: clustering | query: ",
            "prompt_names": ["Clustering"],
        },
    ],
}


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

    @pytest.mark.parametrize(
        "provider, model",
        [
            ("sentence_transformers", EG2),
            ("sentence_transformers", "google/EmbeddingGemma-300M"),
            ("gemini", "gemini-embedding-2"),
        ],
    )
    def test_google_format_models_take_task_prefixes(self, catalog, provider, model):
        assert catalog.uses_task_prefixes(provider, model)

    @pytest.mark.parametrize(
        "provider, model",
        [
            ("sentence_transformers", "all-MiniLM-L6-v2"),
            ("gemini", "gemini-embedding-001"),
            ("sentence_transformers", "/models/local-gemma-checkpoint"),
            ("openai", "gemini-embedding-2"),
        ],
    )
    def test_other_models_take_none(self, catalog, provider, model):
        assert not catalog.uses_task_prefixes(provider, model)

    def test_shipped_task_table(self, catalog):
        table = catalog.task_prefixes
        assert [t.id for t in table.tasks] == [
            "search",
            "question_answering",
            "fact_checking",
            "code_retrieval",
            "classification",
            "clustering",
            "sentence_similarity",
        ]
        search = table.default
        assert search.id == "search"
        assert (search.query_prefix, search.document_prefix) == (SEARCH_QUERY, TITLE_DOCUMENT)
        assert not search.symmetric
        clustering = table.get("clustering")
        assert clustering.symmetric
        assert clustering.document_prefix == clustering.query_prefix == "task: clustering | query: "

    @pytest.mark.parametrize(
        "name, task_id",
        [
            ("Retrieval-query", "search"),
            ("SearchQuery", "search"),
            ("query", "search"),
            ("STS", "sentence_similarity"),
            ("InstructionRetrieval", "code_retrieval"),
            ("Clustering", "clustering"),
        ],
    )
    def test_prompt_names_alias_their_task(self, catalog, name, task_id):
        assert catalog.task_prefixes.task_for_prompt_name(name).id == task_id

    def test_document_prompt_names_alias_no_task(self, catalog):
        for name in ("Retrieval-document", "Document", "document", "Summarization"):
            assert catalog.task_prefixes.task_for_prompt_name(name) is None

    def test_task_aliases_are_prompt_names(self, catalog):
        aliases = {name for task in catalog.task_prefixes.tasks for name in task.prompt_names}
        assert aliases <= catalog.prompt_vocabulary

    def test_task_prefix_models_list(self, catalog):
        listed = {(m.provider, m.model) for m in catalog.task_prefix_models()}
        assert listed == {
            ("sentence_transformers", "google/embeddinggemma-300m"),
            ("sentence_transformers", EG2),
            ("gemini", "gemini-embedding-2"),
        }


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

    def test_rejects_an_unknown_prompt_format(self):
        with pytest.raises(ValueError, match="prompt_format"):
            _catalog(providers={"gemini": [{"model": "m", "prompt_format": "chatml"}]})

    def test_task_prefix_models_need_a_task_table(self):
        with pytest.raises(ValueError, match="task_prefixes"):
            _catalog(providers={"gemini": [{"model": "m", "prompt_format": "task_prefix"}]})

    def test_accepts_task_prefix_models_with_a_table(self):
        catalog = _catalog(
            task_prefixes=TASKS,
            providers={"gemini": [{"model": "Gemini-M", "prompt_format": "task_prefix"}]},
        )
        assert catalog.uses_task_prefixes("gemini", "gemini-m")
        assert catalog.task_prefixes.default.id == "search"

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


class TestTaskPrefixTable:
    def test_tasks_keep_their_order_and_resolve_by_id(self):
        table = TaskPrefixTable.from_mapping(TASKS)
        assert [t.id for t in table.tasks] == ["search", "clustering"]
        assert table.get("clustering").label == "Clustering"
        assert table.get("missing") is None
        assert table.task_for_prompt_name("Retrieval-query").id == "search"
        assert table.task_for_prompt_name("Retrieval-document") is None

    def test_a_task_without_a_document_prefix_is_symmetric(self):
        clustering = TaskPrefixTable.from_mapping(TASKS).get("clustering")
        assert clustering.symmetric
        assert clustering.document_prefix == clustering.query_prefix

    def test_an_empty_table_has_no_default(self):
        table = TaskPrefixTable()
        assert table.tasks == ()
        with pytest.raises(ValueError, match="no tasks"):
            _ = table.default

    @pytest.mark.parametrize(
        "change, message",
        [
            ({"default_task": "missing"}, "default_task"),
            ({"colour": "red"}, "Unknown"),
            ({"tasks": [{**TASKS["tasks"][0], "id": ""}]}, "id"),
            ({"tasks": [{**TASKS["tasks"][0], "label": ""}]}, "label"),
            ({"tasks": [{**TASKS["tasks"][0], "query_prefix": ""}]}, "query_prefix"),
            ({"tasks": [{**TASKS["tasks"][0], "document_prefix": 3}]}, "document_prefix"),
            ({"tasks": [{**TASKS["tasks"][0], "extra": 1}]}, "Unknown"),
            ({"tasks": [TASKS["tasks"][0], TASKS["tasks"][0]]}, "twice"),
            (
                {
                    "tasks": [
                        TASKS["tasks"][0],
                        {**TASKS["tasks"][1], "prompt_names": ["Retrieval-query"]},
                    ]
                },
                "Retrieval-query",
            ),
            ({"tasks": "search"}, "tasks"),
        ],
    )
    def test_rejects_invalid_tables(self, change, message):
        with pytest.raises(ValueError, match=message):
            TaskPrefixTable.from_mapping({**TASKS, **change})
