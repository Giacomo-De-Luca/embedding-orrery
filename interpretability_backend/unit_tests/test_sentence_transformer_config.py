"""Load options, media inputs and persisted config of the forked SentenceTransformer EF.

Covers: secrets never reaching ``get_config()`` (Chroma persists it), loading
only the encoder towers the requested modalities need, reusing a cached load
that already covers a request, ``require_modalities`` and ``embed_inputs``
(prompt only on rows with text, order restored). A fake ``sentence_transformers``
module is injected, so no model is downloaded.
"""

import sys
import types

import numpy as np
import pytest

EG2 = "google/embeddinggemma-2"


class MultimodalModel:
    """Fake SentenceTransformer recording its constructor kwargs and encode calls."""

    loads: list[dict] = []

    def __init__(self, model_name_or_path=None, device=None, **kwargs):
        MultimodalModel.loads.append({"model": model_name_or_path, "device": device, **kwargs})
        self.prompts = {"Retrieval-document": "title: none | text: "}
        self.calls: list[dict] = []
        self.supported = {
            "text",
            "image",
            "audio",
            "video",
            ("image", "text"),
            ("audio", "image", "text"),
        }

    def supports(self, modality) -> bool:
        return modality in self.supported

    def encode(self, inputs, **kwargs):
        inputs = list(inputs)
        self.calls.append({"inputs": inputs, **kwargs})
        # one distinct row per input so the order can be checked
        return np.array([[float(len(self.calls)), float(i), 0.0] for i in range(len(inputs))])


class WordTokenizer:
    """Fake tokenizer: one token per whitespace-separated word."""

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": texts.split()}
        return {"input_ids": [text.split() for text in texts]}

    def decode(self, ids, **kwargs):
        return " ".join(ids)


@pytest.fixture
def ef_class(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = MultimodalModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    from backend.embedding_functions.specific_functions.embed_sentence_transformer import (
        SentenceTransformerEmbeddingFunction,
    )

    monkeypatch.setattr(SentenceTransformerEmbeddingFunction, "models", {})
    monkeypatch.setattr(SentenceTransformerEmbeddingFunction, "_loaded_options", {})
    MultimodalModel.loads = []
    return SentenceTransformerEmbeddingFunction


class TestSecrets:
    def test_token_reaches_the_model_but_never_the_config(self, ef_class):
        ef = ef_class(model_name="all-MiniLM-L6-v2", token="hf_secret_value")

        assert MultimodalModel.loads[0]["token"] == "hf_secret_value"
        config = ef.get_config()
        assert "hf_secret_value" not in repr(config)
        assert "token" not in config["kwargs"]

    def test_non_secret_kwargs_are_kept(self, ef_class):
        ef = ef_class(model_name="all-MiniLM-L6-v2", token="t", trust_remote_code=False)
        assert ef.get_config()["kwargs"] == {"trust_remote_code": False}


class TestLoadOptions:
    def test_text_only_loads_skip_the_media_towers(self, ef_class):
        ef_class(model_name=EG2)
        load = MultimodalModel.loads[0]
        assert load["config_kwargs"] == {"audio_config": None, "vision_config": None}
        assert load["model_kwargs"] == {"dtype": "float32"}

    def test_image_jobs_keep_the_vision_tower(self, ef_class):
        ef_class(model_name=EG2, modalities=("text", "image"))
        assert MultimodalModel.loads[0]["config_kwargs"] == {"audio_config": None}

    def test_audio_jobs_skip_the_vision_tower(self, ef_class):
        ef_class(model_name=EG2, modalities=("text", "audio"))
        assert MultimodalModel.loads[0]["config_kwargs"] == {"vision_config": None}

    def test_full_media_loads_have_no_config_kwargs(self, ef_class):
        ef_class(model_name=EG2, modalities=("text", "video", "audio"))
        assert "config_kwargs" not in MultimodalModel.loads[0]

    def test_a_covering_cached_load_is_reused(self, ef_class):
        ef_class(model_name=EG2, modalities=("text", "image", "audio"))
        ef_class(model_name=EG2)  # a text query after a media job
        assert len(MultimodalModel.loads) == 1

    def test_a_narrower_cached_load_is_replaced(self, ef_class):
        ef_class(model_name=EG2)
        ef_class(model_name=EG2, modalities=("text", "image"))
        assert len(MultimodalModel.loads) == 2
        assert list(ef_class.models) == [EG2]

    def test_models_outside_the_catalog_load_whole(self, ef_class):
        ef_class(model_name="/local/checkpoint")
        assert "config_kwargs" not in MultimodalModel.loads[0]


class TestRequireModalities:
    def test_supported_kinds_pass(self, ef_class):
        ef = ef_class(model_name=EG2, modalities=("text", "image", "audio"))
        ef.require_modalities(["image"], with_text=True)
        ef.require_modalities(["image", "audio"], with_text=True)
        ef.require_modalities(["video"], with_text=False)

    def test_unsupported_kind_raises(self, ef_class):
        ef = ef_class(model_name=EG2)
        ef._model.supported = {"text", "image"}
        with pytest.raises(ValueError, match="audio"):
            ef.require_modalities(["audio"], with_text=False)

    def test_unsupported_combination_raises(self, ef_class):
        ef = ef_class(model_name=EG2)
        ef._model.supported = {"text", "audio"}
        with pytest.raises(ValueError, match="combine"):
            ef.require_modalities(["audio"], with_text=True)

    def test_text_only_models_raise(self, ef_class):
        ef = ef_class(model_name="all-MiniLM-L6-v2")
        del ef._model.supported
        ef._model.supports = lambda modality: modality == "text"
        with pytest.raises(ValueError, match="image"):
            ef.require_modalities(["image"], with_text=False)


class TestEmbedInputs:
    def test_prompt_applies_only_to_rows_with_text(self, ef_class):
        ef = ef_class(model_name=EG2, prompt="Retrieval-document", modalities=("text", "image"))
        inputs = [{"image": "a"}, {"text": "t", "image": "b"}, {"image": "c"}]

        vectors = ef.embed_inputs(inputs, has_text=[False, True, False])

        with_text, media_only = ef._model.calls
        assert (
            with_text["inputs"] == [inputs[1]] and with_text["prompt_name"] == "Retrieval-document"
        )
        assert media_only["inputs"] == [inputs[0], inputs[2]]
        assert "prompt_name" not in media_only and "prompt" not in media_only
        # order restored: input 0 and 2 come from the second call, input 1 from the first
        assert [v[0] for v in vectors] == [2.0, 1.0, 2.0]
        assert all(v.dtype == np.float32 for v in vectors)

    def test_passes_processing_kwargs_and_batch_size(self, ef_class):
        ef = ef_class(model_name=EG2, modalities=("text", "image"))
        processing = {"image": {"max_soft_tokens": 70}}

        ef.embed_inputs(
            [{"image": "a"}], has_text=[False], processing_kwargs=processing, batch_size=4
        )

        call = ef._model.calls[0]
        assert call["processing_kwargs"] == processing and call["batch_size"] == 4

    def test_text_is_cut_to_the_cap_prompt_included(self, ef_class):
        """The context holds text and media together: text past its share would
        overflow it and fail the whole batch."""
        ef = ef_class(model_name=EG2, prompt="Retrieval-document", modalities=("text", "image"))
        ef._model.tokenizer = WordTokenizer()
        ef.cap_text_tokens(8)  # the prompt "title: none | text: " takes 4 of them
        long_row = {"text": "one two three four five six", "image": "a"}
        short_row = {"text": "one two", "image": "b"}

        ef.embed_inputs([long_row, short_row, {"image": "c"}], has_text=[True, True, False])

        with_text, media_only = ef._model.calls
        assert with_text["inputs"] == [{"text": "one two three four", "image": "a"}, short_row]
        assert media_only["inputs"] == [{"image": "c"}]
        assert long_row["text"] == "one two three four five six"  # the caller's rows untouched
        assert ef.truncation.truncated == 1 and "8-token" in ef.truncation.warning()

    def test_plain_text_inputs_are_cut_too(self, ef_class):
        ef = ef_class(model_name=EG2, modalities=("text", "image"))
        ef._model.tokenizer = WordTokenizer()
        ef.cap_text_tokens(2)

        ef.embed_inputs(["one two three"], has_text=[True])

        assert ef._model.calls[0]["inputs"] == ["one two"]

    def test_a_lower_cap_already_set_wins(self, ef_class):
        ef = ef_class(model_name=EG2, modalities=("text", "image"), max_length=5)
        ef.cap_text_tokens(100)
        assert ef.max_length == 5 and ef.truncation.max_length == 5

    def test_without_a_cap_inputs_pass_unchanged(self, ef_class):
        ef = ef_class(model_name=EG2, modalities=("text", "image"))
        row = {"text": "one two three", "image": "a"}
        ef.embed_inputs([row], has_text=[True])
        assert ef._model.calls[0]["inputs"] == [row]

    def test_lengths_must_match(self, ef_class):
        ef = ef_class(model_name=EG2, modalities=("text", "image"))
        with pytest.raises(ValueError, match="has_text"):
            ef.embed_inputs([{"image": "a"}], has_text=[])

    def test_empty_batch(self, ef_class):
        assert ef_class(model_name=EG2).embed_inputs([], has_text=[]) == []
