"""Prompt resolution for task-prompt embedding models (backend/utils/embedding_prompts.py)."""

import pytest

from backend.utils.embedding_model_catalog import EmbeddingModelCatalog
from backend.utils.embedding_prompts import (
    DocumentPromptResolver,
    QueryPromptResolver,
    ResolvedPrompt,
)

GEMMA_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
    "Clustering": "task: clustering | query: ",
}
# A fine-tuned checkpoint that kept only its document prompt.
FINE_TUNED_PROMPTS = {"Retrieval-document": "title: none | text: "}


@pytest.fixture(scope="module")
def catalog() -> EmbeddingModelCatalog:
    return EmbeddingModelCatalog.from_mapping(
        {
            "query_prompt_map": {"Retrieval-document": "Retrieval-query", "document": "query"},
            "query_task_type_map": {"RETRIEVAL_DOCUMENT": "RETRIEVAL_QUERY"},
            "extra_prompt_names": ["s2p"],
            "providers": {
                "sentence_transformers": [
                    {"model": "gemma", "prompts": list(GEMMA_PROMPTS) + ["SearchQuery"]}
                ]
            },
        }
    )


class TestResolvedPrompt:
    def test_encode_kwargs(self):
        assert ResolvedPrompt(name="STS").encode_kwargs() == {"prompt_name": "STS"}
        assert ResolvedPrompt(text="Classify: ").encode_kwargs() == {"prompt": "Classify: "}
        assert ResolvedPrompt().encode_kwargs() == {}


class TestDocumentPromptResolver:
    def test_names_the_model_defines_are_prompt_names(self, catalog):
        resolved = DocumentPromptResolver(catalog).resolve("Retrieval-document", GEMMA_PROMPTS)
        assert resolved == ResolvedPrompt(name="Retrieval-document")

    def test_lowercase_embeddinggemma_names_are_names_too(self, catalog):
        """Regression: "query"/"document" used to be sent as literal prompt text."""
        assert DocumentPromptResolver(catalog).resolve("document", GEMMA_PROMPTS).name == "document"

    def test_other_text_is_a_literal_prompt(self, catalog):
        resolved = DocumentPromptResolver(catalog).resolve("Represent this: ", GEMMA_PROMPTS)
        assert resolved == ResolvedPrompt(text="Represent this: ")

    def test_known_name_missing_from_the_model_raises(self, catalog):
        with pytest.raises(ValueError, match=r"'SearchQuery'.*available.*Clustering"):
            DocumentPromptResolver(catalog).resolve("SearchQuery", GEMMA_PROMPTS)

    def test_known_name_on_a_model_without_prompts_raises(self, catalog):
        with pytest.raises(ValueError, match="defines no prompts"):
            DocumentPromptResolver(catalog).resolve("STS", {})

    @pytest.mark.parametrize("prompt", [None, ""])
    def test_no_prompt(self, catalog, prompt):
        assert DocumentPromptResolver(catalog).resolve(prompt, GEMMA_PROMPTS) == ResolvedPrompt()


class TestQueryPromptResolver:
    def test_explicit_query_prompt_wins(self, catalog):
        resolver = QueryPromptResolver(catalog)
        assert resolver.for_model("STS", "Retrieval-document", GEMMA_PROMPTS) == "STS"
        # even a name the model lacks: the document-side resolver reports it
        assert (
            resolver.for_model("SearchQuery", "Retrieval-document", GEMMA_PROMPTS) == "SearchQuery"
        )

    def test_document_prompt_maps_to_its_query_counterpart(self, catalog):
        resolver = QueryPromptResolver(catalog)
        assert resolver.for_model(None, "Retrieval-document", GEMMA_PROMPTS) == "Retrieval-query"
        assert resolver.for_model("", "document", GEMMA_PROMPTS) == "query"

    def test_symmetric_prompts_are_reused(self, catalog):
        assert (
            QueryPromptResolver(catalog).for_model(None, "Clustering", GEMMA_PROMPTS)
            == "Clustering"
        )

    def test_falls_back_to_the_stored_prompt_when_the_counterpart_is_missing(self, catalog):
        resolver = QueryPromptResolver(catalog)
        assert (
            resolver.for_model(None, "Retrieval-document", FINE_TUNED_PROMPTS)
            == "Retrieval-document"
        )

    def test_literal_document_prompts_are_reused(self, catalog):
        assert QueryPromptResolver(catalog).for_model(None, "Represent: ", {}) == "Represent: "

    def test_no_prompt_anywhere(self, catalog):
        assert QueryPromptResolver(catalog).for_model(None, None, GEMMA_PROMPTS) is None

    def test_candidates_are_ordered_and_unique(self, catalog):
        resolver = QueryPromptResolver(catalog)
        assert resolver.candidates(None, "Retrieval-document") == (
            "Retrieval-query",
            "Retrieval-document",
        )
        assert resolver.candidates(None, "Clustering") == ("Clustering",)
        assert resolver.candidates("STS", "Clustering") == ("STS",)
        assert resolver.candidates(None, None) == ()

    def test_gemini_task_types(self, catalog):
        resolver = QueryPromptResolver(catalog)
        assert resolver.task_type("RETRIEVAL_DOCUMENT") == "RETRIEVAL_QUERY"
        assert resolver.task_type("SEMANTIC_SIMILARITY") == "SEMANTIC_SIMILARITY"
        assert resolver.task_type(None) is None

    def test_defaults_to_the_shipped_catalog(self):
        assert (
            QueryPromptResolver().for_model(None, "Document", {"SearchQuery": "q: "})
            == "SearchQuery"
        )
