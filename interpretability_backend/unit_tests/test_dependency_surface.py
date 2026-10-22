"""The library APIs EmbeddingGemma 2 and the media pipeline rely on (imports only, no downloads).

EmbeddingGemma 2 needs sentence-transformers >= 6.1 (ordered multimodal dict inputs)
and transformers >= 5.19 (the first release with ``embedding_gemma2``). Its
processor always builds a video sub-processor, which needs torchvision. PyAV
decodes audio and video with the FFmpeg it bundles.
"""

import inspect

import pytest
from packaging.version import Version

sentence_transformers = pytest.importorskip("sentence_transformers")
transformers = pytest.importorskip("transformers")


def test_minimum_versions():
    assert Version(sentence_transformers.__version__) >= Version("6.1.0")
    assert Version(transformers.__version__) >= Version("5.19.0")


def test_sentence_transformer_constructor_kwargs():
    parameters = inspect.signature(sentence_transformers.SentenceTransformer.__init__).parameters
    for name in ("model_kwargs", "config_kwargs", "processor_kwargs", "token"):
        assert name in parameters


def test_sentence_transformer_multimodal_api():
    model_class = sentence_transformers.SentenceTransformer
    for name in ("supports", "modalities", "get_embedding_dimension", "encode"):
        assert hasattr(model_class, name)
    encode = inspect.signature(model_class.encode).parameters
    assert "prompt_name" in encode and "prompt" in encode


def test_embeddinggemma_2_classes_exist():
    from transformers.models import embedding_gemma2

    for name in ("EmbeddingGemma2Model", "EmbeddingGemma2Processor"):
        assert hasattr(embedding_gemma2, name)


def test_torchvision_backs_the_video_processor():
    from transformers.utils import is_torchvision_available

    assert is_torchvision_available()


def test_pyav_decodes_and_resamples():
    av = pytest.importorskip("av")
    assert hasattr(av, "AudioResampler") and hasattr(av, "open")


def test_bge_provider_imports():
    """flagembedding 1.3.5 failed to import on transformers 5 (is_torch_fx_available)."""
    from FlagEmbedding import BGEM3FlagModel  # noqa: F401


def test_qwen_wrapper_imports_resolve():
    from transformers import StoppingCriteria, StoppingCriteriaList  # noqa: F401
    from transformers.generation.streamers import BaseStreamer  # noqa: F401
