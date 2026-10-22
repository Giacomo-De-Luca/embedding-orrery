"""Batch size, input-length cap and truncation counting in the SentenceTransformer EF.

The cap is passed per ``encode()`` call (``processing_kwargs``), never set on
the shared cached model, so a query embedder built from the same cache is not
affected. A fake ``sentence_transformers`` module stands in for the library.
"""

import sys
import types

import numpy as np
import pytest

DOC_PROMPT = "title: none | text: "  # 4 whitespace tokens


class FakeTokenizer:
    def __init__(self):
        self.texts: list[str] = []

    def __call__(self, texts, **kwargs):
        self.texts.extend(texts)
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


class FakeSentenceTransformer:
    def __init__(self, model_name_or_path=None, device=None, **kwargs):
        self.prompts = {"Retrieval-document": DOC_PROMPT}
        self.tokenizer = FakeTokenizer()
        self.encode_calls: list[tuple[list, dict]] = []

    def encode(self, inputs, **kwargs):
        self.encode_calls.append((list(inputs), kwargs))
        return np.zeros((len(inputs), 2), dtype=np.float32)


@pytest.fixture
def ef_class(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    from backend.embedding_functions.specific_functions.embed_sentence_transformer import (
        SentenceTransformerEmbeddingFunction,
    )

    monkeypatch.setattr(SentenceTransformerEmbeddingFunction, "models", {})
    monkeypatch.setattr(SentenceTransformerEmbeddingFunction, "_loaded_options", {})
    return SentenceTransformerEmbeddingFunction


def test_batch_size_and_length_cap_reach_encode(ef_class):
    ef = ef_class(model_name="m", max_length=8, batch_size=4)

    vectors = ef(["one two", "three"])

    _, kwargs = ef._model.encode_calls[0]
    assert kwargs["batch_size"] == 4
    assert kwargs["processing_kwargs"] == {"text": {"max_length": 8}}
    assert len(vectors) == 2 and vectors[0].dtype == np.float32


def test_without_limits_encode_keeps_its_defaults(ef_class):
    ef = ef_class(model_name="m")

    ef(["a"])

    _, kwargs = ef._model.encode_calls[0]
    assert "batch_size" not in kwargs
    assert "processing_kwargs" not in kwargs
    assert ef.truncation.active is False
    assert ef._model.tokenizer.texts == []  # no extra tokenization without a cap


def test_counts_documents_over_the_cap(ef_class):
    ef = ef_class(model_name="m", max_length=3)

    ef(["a b", "a b c d", "a b c"])

    assert (ef.truncation.truncated, ef.truncation.seen) == (1, 3)


def test_prompt_name_tokens_count_towards_the_cap(ef_class):
    ef = ef_class(model_name="m", prompt="Retrieval-document", max_length=4)

    ef(["word"])

    assert ef._model.tokenizer.texts == [DOC_PROMPT + "word"]
    assert ef.truncation.truncated == 1  # 4 prompt tokens + 1 document token


def test_literal_prompt_tokens_count_too(ef_class):
    # encode() prepends a literal prompt as is, so it carries its own separator.
    ef = ef_class(model_name="m", prompt="Classify this text: ", max_length=4)

    ef(["a b"])

    assert ef.truncation.truncated == 1  # 3 prompt tokens + 2 document tokens
