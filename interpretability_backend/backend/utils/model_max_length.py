"""The input-length limit of a local embedding model, and the cap a job may use.

The embed form shows each model's real limit and embedding truncates at it.
The limit comes from the first source that states one:

1. the catalog's ``max_length`` (``config/embedding_models.json``), for models
   whose files are wrong or silent (EmbeddingGemma 2);
2. sentence-transformers' ``max_seq_length`` (``sentence_bert_config.json``);
3. the tokenizer's ``model_max_length``, unless it is a "no limit" sentinel
   (Hugging Face writes 1e30);
4. the model config's ``max_position_embeddings`` (top level or ``text_config``).

API providers (Gemini, OpenAI, ...) truncate remotely: they have no limit here.
Torch-free: model files are read without loading the model.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Self

from .embedding_model_catalog import EmbeddingModelCatalog
from .model_files import ModelConfigFiles
from .provider_list import LOCAL_PROVIDER_VALUES

# Tokenizer limits at or above this are "unbounded" markers, not real limits.
_SENTINEL_FLOOR = 10_000_000


@dataclass(frozen=True)
class ModelMaxLength:
    """A model's limit in tokens and where it came from (both None when unknown)."""

    max_length: int | None
    source: str | None


_UNKNOWN = ModelMaxLength(None, None)


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value < _SENTINEL_FLOOR:
        return value
    return None


class ModelMaxLengthResolver:
    """Look up model limits and check requested caps against them."""

    def __init__(
        self,
        catalog: EmbeddingModelCatalog | None = None,
        files: ModelConfigFiles | None = None,
    ):
        self.catalog = catalog or EmbeddingModelCatalog.shipped()
        self.files = files or ModelConfigFiles.shared()

    @classmethod
    @cache
    def shared(cls) -> Self:
        """The process-wide resolver (shipped catalog, shared file reader)."""
        return cls()

    @classmethod
    @cache
    def cache_only(cls) -> Self:
        """A resolver that never downloads model files (the read-only demo's)."""
        return cls(files=ModelConfigFiles.cache_only())

    def resolve(self, provider: str, model_name: str) -> ModelMaxLength:
        if provider not in LOCAL_PROVIDER_VALUES:
            return _UNKNOWN
        entry = self.catalog.find(provider, model_name)
        if entry is not None and entry.max_length is not None:
            return ModelMaxLength(entry.max_length, "catalog")
        seq_length = self._read(model_name, "sentence_bert_config.json", self._max_seq_length)
        if seq_length is not None:
            return ModelMaxLength(seq_length, "sentence_transformers_config")
        positions = self._read(model_name, "config.json", self._position_limit)
        tokenizer = self._read(model_name, "tokenizer_config.json", self._tokenizer_limit)
        # Tokenizer configs can claim more than the position table holds
        # (Qwen3-Embedding: 131,072 vs 32,768); sentence-transformers takes the min too.
        if tokenizer is not None and (positions is None or tokenizer <= positions):
            return ModelMaxLength(tokenizer, "tokenizer")
        if positions is not None:
            return ModelMaxLength(positions, "model_config")
        return _UNKNOWN

    def _read(self, model_name: str, filename: str, extract) -> int | None:
        parsed = self.files.read(model_name, filename)
        return extract(parsed) if parsed is not None else None

    def effective(self, provider: str, model_name: str, requested: int | None) -> int | None:
        """The cap to embed with: the request if it fits, else the model limit.

        Raises ValueError for a non-positive request or one above the model's
        known limit. None means "no known limit": the embedder keeps its default.
        """
        if provider not in LOCAL_PROVIDER_VALUES:
            return None
        if requested is not None and (
            not isinstance(requested, int) or isinstance(requested, bool) or requested < 1
        ):
            raise ValueError(f"max_length must be a positive integer, got {requested!r}")
        limit = self.resolve(provider, model_name).max_length
        if requested is None:
            return limit
        if limit is not None and requested > limit:
            raise ValueError(
                f"max_length {requested:,} exceeds the {limit:,}-token limit of {model_name}"
            )
        return requested

    @staticmethod
    def _max_seq_length(config: Mapping) -> int | None:
        return _positive_int(config.get("max_seq_length"))

    @staticmethod
    def _tokenizer_limit(config: Mapping) -> int | None:
        return _positive_int(config.get("model_max_length"))

    @staticmethod
    def _position_limit(config: Mapping) -> int | None:
        direct = _positive_int(config.get("max_position_embeddings"))
        if direct is not None:
            return direct
        text_config = config.get("text_config")
        if isinstance(text_config, Mapping):
            return _positive_int(text_config.get("max_position_embeddings"))
        return None
