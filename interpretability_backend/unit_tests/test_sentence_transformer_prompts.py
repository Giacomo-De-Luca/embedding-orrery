"""Prompt handling of the forked SentenceTransformer embedding function.

A prompt is resolved once, against the loaded model's own ``prompts``: a name
it defines becomes ``prompt_name``, other text a literal ``prompt``, and a
known prompt name the model lacks is an error. Queries map the collection's
document prompt to its query counterpart. A fake ``sentence_transformers``
module is injected, so no model is downloaded.
"""

import sys
import types

import numpy as np
import pytest

GEMMA_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
}


class PromptedModel:
    """Fake SentenceTransformer with EmbeddingGemma-style prompts and a recording encode."""

    prompts_by_model = {"gemma": GEMMA_PROMPTS, "fine-tuned": {"Retrieval-document": "doc: "}}

    def __init__(self, model_name_or_path=None, device=None, **kwargs):
        self.prompts = dict(self.prompts_by_model.get(model_name_or_path, {}))
        self.calls: list[dict] = []

    def encode(self, inputs, **kwargs):
        self.calls.append({"inputs": list(inputs), **kwargs})
        return np.zeros((len(inputs), 3), dtype=np.float32)


@pytest.fixture
def ef_class(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = PromptedModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    from backend.embedding_functions.specific_functions.embed_sentence_transformer import (
        SentenceTransformerEmbeddingFunction,
    )

    monkeypatch.setattr(SentenceTransformerEmbeddingFunction, "models", {})
    monkeypatch.setattr(SentenceTransformerEmbeddingFunction, "_loaded_options", {})
    return SentenceTransformerEmbeddingFunction


def _encode_kwargs(ef) -> dict:
    ef(["some text"])
    call = ef._model.calls[-1]
    return {k: call[k] for k in ("prompt_name", "prompt") if k in call}


class TestDocumentPrompts:
    def test_a_name_the_model_defines_is_a_prompt_name(self, ef_class):
        assert _encode_kwargs(ef_class(model_name="gemma", prompt="STS")) == {"prompt_name": "STS"}

    def test_lowercase_embeddinggemma_names_are_names(self, ef_class):
        """Regression: "document" was sent as literal text before."""
        ef = ef_class(model_name="gemma", prompt="document")
        assert _encode_kwargs(ef) == {"prompt_name": "document"}

    def test_other_text_is_a_literal_prompt(self, ef_class):
        ef = ef_class(model_name="gemma", prompt="Represent this sentence: ")
        assert _encode_kwargs(ef) == {"prompt": "Represent this sentence: "}

    def test_known_name_missing_from_the_model_fails_at_construction(self, ef_class):
        with pytest.raises(ValueError, match="SearchQuery"):
            ef_class(model_name="gemma", prompt="SearchQuery")

    def test_no_prompt(self, ef_class):
        assert _encode_kwargs(ef_class(model_name="gemma")) == {}

    def test_models_without_prompts_attribute(self, ef_class, monkeypatch):
        monkeypatch.delattr(PromptedModel, "prompts_by_model")
        monkeypatch.setattr(
            PromptedModel, "__init__", lambda self, *a, **k: setattr(self, "calls", [])
        )
        ef = ef_class(model_name="bare", prompt="Classify: ")
        assert _encode_kwargs(ef) == {"prompt": "Classify: "}


class TestQueryPrompts:
    def test_document_prompt_maps_to_the_query_prompt(self, ef_class):
        ef = ef_class(model_name="gemma", prompt="Retrieval-document", is_query=True)
        assert _encode_kwargs(ef) == {"prompt_name": "Retrieval-query"}

    def test_explicit_query_prompt_wins(self, ef_class):
        ef = ef_class(
            model_name="gemma", prompt="Retrieval-document", is_query=True, query_prompt="STS"
        )
        assert _encode_kwargs(ef) == {"prompt_name": "STS"}

    def test_fine_tuned_checkpoint_reuses_its_document_prompt(self, ef_class):
        ef = ef_class(model_name="fine-tuned", prompt="Retrieval-document", is_query=True)
        assert _encode_kwargs(ef) == {"prompt_name": "Retrieval-document"}

    def test_document_side_ignores_the_query_prompt(self, ef_class):
        ef = ef_class(model_name="gemma", prompt="Retrieval-document", query_prompt="STS")
        assert _encode_kwargs(ef) == {"prompt_name": "Retrieval-document"}

    def test_config_keeps_the_document_prompt(self, ef_class):
        ef = ef_class(model_name="gemma", prompt="Retrieval-document", is_query=True)
        assert ef.get_config()["prompt"] == "Retrieval-document"
