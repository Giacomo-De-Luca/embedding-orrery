"""
Modified fork of ChromaDB's SentenceTransformerEmbeddingFunction.

Differences from ChromaDB's implementation:

- Task prompts (EmbeddingGemma and similar models). The prompt string is resolved
  once, against the loaded model's own ``prompts``: a name the model defines is
  passed as ``prompt_name``, other text as a literal ``prompt``, and a known
  prompt name the model lacks raises (``utils/embedding_prompts.py``). For
  queries (``is_query=True``) the collection's document prompt maps to its
  query counterpart (``Retrieval-document`` -> ``Retrieval-query``) unless an
  explicit ``query_prompt`` is given.
- Load options (``utils/model_load_options.py``): dtype per device (never
  float16), and only the encoder towers the requested ``modalities`` need
  (EmbeddingGemma 2 text queries load the text tower only).
- Media inputs: ``embed_inputs`` encodes image / audio / video / mixed inputs,
  applying the prompt only to inputs that contain text (the model card's rule).
- Secrets passed as load kwargs (the HF ``token``) are never returned by
  ``get_config()``, which ChromaDB persists.
- A class-level single-slot model cache (see ``models``).
- Text inputs honour the job's ``batch_size`` and input-length cap
  (``max_length``). The cap goes to each ``encode()`` call as
  ``processing_kwargs``, never onto the shared cached model, and documents
  longer than it (prompt included) are counted in ``truncation``.
"""

import gc
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Space
from chromadb.utils.embedding_functions.schemas import validate_config_schema

from ...utils.chroma_config_scrubber import DEFAULT_SECRET_KEYS
from ...utils.embedding_prompts import DocumentPromptResolver, QueryPromptResolver
from ...utils.model_load_options import SentenceTransformerLoadOptions
from ...utils.truncation import TruncationTracker

logger = logging.getLogger("orrery." + __name__)

_SECRET_KWARGS = frozenset(key.casefold() for key in DEFAULT_SECRET_KEYS)


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """Fork of ChromaDB's SentenceTransformerEmbeddingFunction (see module docstring)."""

    # Class-level single-slot model cache (shared across all instances), keyed by
    # model name. Unlike ChromaDB's unbounded dict, loading a different model (or
    # the same model with towers the cached load lacks) evicts the previous one,
    # so only one SentenceTransformer stays resident (each is 100 MB-1 GB+).
    models: dict[str, Any] = {}
    # Load options of each cached model; a cached load is reused when it covers a request.
    _loaded_options: dict[str, SentenceTransformerLoadOptions] = {}

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        prompt: str | None = None,
        modalities: Iterable[str] = ("text",),
        is_query: bool = False,
        query_prompt: str | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ):
        """Initialize SentenceTransformerEmbeddingFunction.

        Args:
            model_name: Identifier of the SentenceTransformer model
            device: Device used for computation (cpu, cuda, mps)
            normalize_embeddings: Whether to normalize returned vectors
            prompt: The collection's prompt: a prompt name of the model or literal text
            modalities: Input modalities this instance will embed ("text", "image", ...)
            is_query: Embed queries: map ``prompt`` to its query counterpart
            query_prompt: Explicit query prompt (only with ``is_query``), wins over the mapping
            max_length: Input-length cap in tokens for text (None: the model's own setting)
            batch_size: Texts per forward pass (None: sentence-transformers' default of 32)
            **kwargs: Extra SentenceTransformer constructor arguments (e.g. ``token``);
                secrets among them are never returned by ``get_config()``.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError(
                "The sentence_transformers python package is not installed. "
                "Please install it with `pip install sentence_transformers`"
            ) from None

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.prompt = prompt
        self.is_query = is_query
        self.query_prompt = query_prompt
        self.max_length = max_length
        self.batch_size = batch_size
        self.truncation = TruncationTracker(max_length)

        for key, value in kwargs.items():
            if not isinstance(value, (str, int, float, bool, list, dict, tuple)):
                raise ValueError(f"Keyword argument {key} is not a primitive type")
        self.kwargs = {k: v for k, v in kwargs.items() if k.casefold() not in _SECRET_KWARGS}

        options = SentenceTransformerLoadOptions.for_modalities(model_name, device, modalities)
        # Bind through a local: a concurrent constructor may evict between the
        # membership check and a dict re-read (embed jobs run in worker threads
        # while sync semantic-search resolvers construct EFs on the event loop).
        model = self.models.get(model_name)
        cached = self._loaded_options.get(model_name)
        if model is None or cached is None or not cached.covers(options):
            self._evict_cached_models()
            model = SentenceTransformer(
                model_name_or_path=model_name,
                device=device,
                **options.constructor_kwargs(),
                **kwargs,
            )
            self.models[model_name] = model
            self._loaded_options[model_name] = options
        self._model = model

        model_prompts = getattr(model, "prompts", None) or {}
        effective = (
            QueryPromptResolver().for_model(query_prompt, prompt, model_prompts)
            if is_query
            else prompt
        )
        self._resolved_prompt = DocumentPromptResolver().resolve(effective, model_prompts)
        # Prepended to every text input by encode(), so it counts towards the cap.
        resolved = self._resolved_prompt
        self._prompt_prefix = (
            model_prompts.get(resolved.name, "") if resolved.name else resolved.text or ""
        )

    @classmethod
    def _evict_cached_models(cls) -> None:
        """Drop all cached models before loading a different one.

        Evicting first (rather than after the new load) keeps peak RAM to one
        *cached* model. Instances created earlier keep working: they hold their
        own ``self._model`` reference, so an evicted model is only freed once
        those instances are garbage-collected too (until then it stays resident
        alongside the new one).
        """
        if not cls.models:
            cls._loaded_options.clear()
            return
        evicted = list(cls.models)
        cls.models.clear()
        cls._loaded_options.clear()
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # best-effort cleanup: never fail the new model's load
            logger.debug("torch cache cleanup after eviction failed", exc_info=True)
        logger.info("Evicted cached SentenceTransformer model(s): %s", ", ".join(evicted))

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the given text documents."""
        texts = list(input)
        if self.truncation.active:
            prefixed = [self._prompt_prefix + text for text in texts]
            self.truncation.count(getattr(self._model, "tokenizer", None), prefixed)
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            **self._resolved_prompt.encode_kwargs(),
            **self._text_limits(),
        )
        return [np.array(embedding, dtype=np.float32) for embedding in embeddings]

    def _text_limits(self) -> dict[str, Any]:
        """``encode()`` kwargs for the batch size and the per-call input-length cap."""
        limits: dict[str, Any] = {}
        if self.batch_size is not None:
            limits["batch_size"] = self.batch_size
        if self.max_length is not None:
            limits["processing_kwargs"] = {"text": {"max_length": self.max_length}}
        return limits

    def embed_inputs(
        self,
        inputs: Sequence[Any],
        has_text: Sequence[bool],
        processing_kwargs: Mapping[str, Any] | None = None,
        batch_size: int | None = None,
    ) -> list[np.ndarray]:
        """Embed media or mixed inputs (``{"text": ..., "image": ...}`` dicts and the like).

        The prompt applies only to inputs with text (``has_text``): media-only
        inputs are encoded without a prefix. Returns one float32 vector per
        input, in input order.
        """
        if len(inputs) != len(has_text):
            raise ValueError("embed_inputs needs one has_text flag per input")
        extra: dict[str, Any] = {}
        if processing_kwargs:
            extra["processing_kwargs"] = dict(processing_kwargs)
        if batch_size is not None:
            extra["batch_size"] = batch_size
        vectors: list[np.ndarray | None] = [None] * len(inputs)
        for with_text in (True, False):
            positions = [i for i, flag in enumerate(has_text) if flag is with_text]
            if not positions:
                continue
            prompt_kwargs = self._resolved_prompt.encode_kwargs() if with_text else {}
            encoded = self._model.encode(
                [inputs[i] for i in positions],
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                **prompt_kwargs,
                **extra,
            )
            for position, vector in zip(positions, encoded, strict=True):
                vectors[position] = np.asarray(vector, dtype=np.float32)
        return vectors

    def require_modalities(self, kinds: Iterable[str], with_text: bool) -> None:
        """Raise ``ValueError`` unless the loaded model embeds these media kinds.

        With ``with_text`` (or several kinds) the model must also combine them
        in one input.
        """
        kinds = tuple(dict.fromkeys(kinds))
        supports = getattr(self._model, "supports", None)
        if supports is None:
            raise ValueError(f"{self.model_name} embeds text only, not {', '.join(kinds)}")
        missing = [kind for kind in kinds if not supports(kind)]
        if missing:
            raise ValueError(f"{self.model_name} cannot embed {', '.join(missing)}")
        combination = tuple(sorted(set(kinds) | ({"text"} if with_text else set())))
        if len(combination) > 1 and not supports(combination):
            raise ValueError(
                f"{self.model_name} cannot combine {', '.join(combination)} in one input"
            )

    @staticmethod
    def name() -> str:
        return "sentence_transformer"

    def default_space(self) -> Space:
        # If normalize_embeddings is True, cosine is equivalent to dot product
        return "cosine"

    def supported_spaces(self) -> list[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "EmbeddingFunction[Documents]":
        model_name = config.get("model_name")
        device = config.get("device")
        normalize_embeddings = config.get("normalize_embeddings")
        prompt = config.get("prompt")
        kwargs = config.get("kwargs", {})

        if model_name is None or device is None or normalize_embeddings is None:
            raise AssertionError("This code should not be reached")

        return SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            prompt=prompt,
            **kwargs,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "prompt": self.prompt,
            "kwargs": self.kwargs,
        }

    def validate_config_update(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        # model_name is also used as the identifier for model path if stored locally.
        # Users should be able to change the path if needed, so we should not validate that.
        return

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        """Validate the configuration using the JSON schema.

        Args:
            config: Configuration to validate

        Raises:
            ValidationError: If the configuration does not match the schema
        """
        validate_config_schema(config, "sentence_transformer")
