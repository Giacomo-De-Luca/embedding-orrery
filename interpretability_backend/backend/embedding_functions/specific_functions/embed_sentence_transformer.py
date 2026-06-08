"""
Modified fork of ChromaDB's SentenceTransformerEmbeddingFunction.
Adds prompt support passed to encode() for EmbeddingGemma and similar models.

ChromaDB's original implementation only passes **kwargs to the SentenceTransformer constructor,
not to encode(). This fork adds prompt support to enable task-specific embeddings.

For EmbeddingGemma and similar models, pass a prompt that can be either:
- A known prompt name (e.g., "Retrieval-query", "STS") → passed as prompt_name
- A custom prompt string (e.g., "Classify: ") → passed as prompt

See: https://huggingface.co/google/gemma-embedding-001
"""

from typing import Any

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Space
from chromadb.utils.embedding_functions.schemas import validate_config_schema

# Known prompt names that SentenceTransformers recognizes for models like EmbeddingGemma
KNOWN_PROMPT_NAMES = {
    "Retrieval-query",
    "Retrieval-document",
    "STS",
    "Classification",
    "Clustering",
    "s2p",  # Sentence to passage
    "s2s",  # Sentence to sentence
}


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """Fork of ChromaDB's SentenceTransformerEmbeddingFunction with prompt support."""

    # Class-level model cache (shared across all instances, matches ChromaDB)
    models: dict[str, Any] = {}

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        prompt: str | None = None,
        **kwargs: Any,
    ):
        """Initialize SentenceTransformerEmbeddingFunction.

        Args:
            model_name: Identifier of the SentenceTransformer model
            device: Device used for computation (cpu, cuda, mps)
            normalize_embeddings: Whether to normalize returned vectors
            prompt: Prompt string - can be a known name (e.g., "Retrieval-query") or custom string
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError(
                "The sentence_transformers python package is not installed. "
                "Please install it with `pip install sentence_transformers`"
            )

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.prompt = prompt

        # Resolve: if it's a known name, use prompt_name; otherwise use prompt
        self._is_prompt_name = prompt in KNOWN_PROMPT_NAMES if prompt else False

        for key, value in kwargs.items():
            if not isinstance(value, (str, int, float, bool, list, dict, tuple)):
                raise ValueError(f"Keyword argument {key} is not a primitive type")
        self.kwargs = kwargs

        if model_name not in self.models:
            self.models[model_name] = SentenceTransformer(
                model_name_or_path=model_name, device=device, **kwargs
            )
        self._model = self.models[model_name]

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the given documents.

        Args:
            input: Documents to generate embeddings for.

        Returns:
            Embeddings for the documents.
        """
        # Build encode kwargs with prompt support
        encode_kwargs: dict[str, Any] = {
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize_embeddings,
        }

        # Add prompt - use prompt_name if it's a known name, otherwise use prompt directly
        if self.prompt is not None:
            if self._is_prompt_name:
                encode_kwargs["prompt_name"] = self.prompt
            else:
                encode_kwargs["prompt"] = self.prompt

        embeddings = self._model.encode(list(input), **encode_kwargs)
        return [np.array(embedding, dtype=np.float32) for embedding in embeddings]

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
            assert False, "This code should not be reached"

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
