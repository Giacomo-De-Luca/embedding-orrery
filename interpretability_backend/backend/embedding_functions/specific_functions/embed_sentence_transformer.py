"""
Modified fork of ChromaDB's SentenceTransformerEmbeddingFunction.
Adds prompt/prompt_name support passed to encode() for EmbeddingGemma and similar models.

ChromaDB's original implementation only passes **kwargs to the SentenceTransformer constructor,
not to encode(). This fork adds prompt/prompt_name support to enable task-specific embeddings.

For EmbeddingGemma and similar models, use:
- prompt: Direct prompt string (e.g., "task: search result | query: ")
- prompt_name: Predefined prompt name (e.g., "Retrieval-query", "Retrieval-document", "STS")

See: https://huggingface.co/google/gemma-embedding-001
"""
from chromadb.api.types import EmbeddingFunction, Space, Embeddings, Documents
from chromadb.utils.embedding_functions.schemas import validate_config_schema
from typing import List, Dict, Any, Optional
import numpy as np


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """Fork of ChromaDB's SentenceTransformerEmbeddingFunction with prompt support."""

    # Class-level model cache (shared across all instances, matches ChromaDB)
    models: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        prompt: Optional[str] = None,
        prompt_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize SentenceTransformerEmbeddingFunction.

        Args:
            model_name: Identifier of the SentenceTransformer model
            device: Device used for computation (cpu, cuda, mps)
            normalize_embeddings: Whether to normalize returned vectors
            prompt: Direct prompt string to prepend (e.g., "task: search result | query: ")
            prompt_name: Predefined prompt name (e.g., "Retrieval-query", "STS")
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
        self.prompt_name = prompt_name

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
        encode_kwargs: Dict[str, Any] = {
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize_embeddings,
        }

        # Add prompt or prompt_name if specified (prompt takes precedence)
        if self.prompt is not None:
            encode_kwargs["prompt"] = self.prompt
        elif self.prompt_name is not None:
            encode_kwargs["prompt_name"] = self.prompt_name

        embeddings = self._model.encode(list(input), **encode_kwargs)
        return [np.array(embedding, dtype=np.float32) for embedding in embeddings]

    @staticmethod
    def name() -> str:
        return "sentence_transformer"

    def default_space(self) -> Space:
        # If normalize_embeddings is True, cosine is equivalent to dot product
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction[Documents]":
        model_name = config.get("model_name")
        device = config.get("device")
        normalize_embeddings = config.get("normalize_embeddings")
        prompt = config.get("prompt")
        prompt_name = config.get("prompt_name")
        kwargs = config.get("kwargs", {})

        if model_name is None or device is None or normalize_embeddings is None:
            assert False, "This code should not be reached"

        return SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            prompt=prompt,
            prompt_name=prompt_name,
            **kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "prompt": self.prompt,
            "prompt_name": self.prompt_name,
            "kwargs": self.kwargs,
        }

    def validate_config_update(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        # model_name is also used as the identifier for model path if stored locally.
        # Users should be able to change the path if needed, so we should not validate that.
        return

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate the configuration using the JSON schema.

        Args:
            config: Configuration to validate

        Raises:
            ValidationError: If the configuration does not match the schema
        """
        validate_config_schema(config, "sentence_transformer")
