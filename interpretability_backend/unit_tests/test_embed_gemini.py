"""Gemini embedding function (specific_functions/embed_gemini.py), on a fake client.

google-genai merges a plain list of strings into ONE Content for
gemini-embedding-2, which the API embeds as one aggregated vector; each text
must travel as its own Content. Gemini Embedding 2 takes its task as a text
prefix, not the ``task_type`` field.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import types

from backend.embedding_functions.specific_functions.embed_gemini import EmbedTextGemini


class FakeModels:
    def __init__(self, drop_one: bool = False):
        self.calls: list[dict] = []
        self.drop_one = drop_one

    def embed_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        vectors = [
            SimpleNamespace(values=[float(i), 1.0])
            for i, _ in enumerate(contents[1:] if self.drop_one else contents)
        ]
        return SimpleNamespace(embeddings=vectors)


def _ef(models: FakeModels, **kwargs) -> EmbedTextGemini:
    return EmbedTextGemini(client=SimpleNamespace(models=models), **kwargs)


def test_each_text_is_its_own_content():
    models = FakeModels()
    vectors = _ef(models, model="gemini-embedding-2")(["first doc", "second doc"])

    contents = models.calls[0]["contents"]
    assert len(contents) == 2
    assert all(isinstance(c, types.Content) and len(c.parts) == 1 for c in contents)
    assert [c.parts[0].text for c in contents] == ["first doc", "second doc"]
    assert len(vectors) == 2 and vectors[0].dtype == np.float32
    np.testing.assert_allclose(vectors[1], [1.0, 1.0])


def test_text_prefix_is_prepended():
    models = FakeModels()
    _ef(models, model="gemini-embedding-2", text_prefix="task: search result | query: ")(["cats"])
    assert models.calls[0]["contents"][0].parts[0].text == "task: search result | query: cats"


def test_no_task_type_means_no_config():
    models = FakeModels()
    _ef(models, model="gemini-embedding-2")(["doc"])
    assert models.calls[0]["config"] is None


def test_task_type_is_sent_when_given():
    models = FakeModels()
    _ef(models, model="gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT")(["doc"])
    assert models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_a_short_response_raises_without_retrying():
    models = FakeModels(drop_one=True)
    with pytest.raises(ValueError, match="2 inputs"):
        _ef(models, model="gemini-embedding-2")(["a", "b"])
    assert len(models.calls) == 1


def test_empty_input_calls_nothing():
    """chromadb's EmbeddingFunction wrapper rejects the empty result itself."""
    models = FakeModels()
    with pytest.raises(ValueError, match="non-empty"):
        _ef(models, model="gemini-embedding-2")([])
    assert models.calls == []
