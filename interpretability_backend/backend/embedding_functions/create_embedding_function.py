"""
Factory for creating embedding functions.
"""

import json
import os
from pathlib import Path
from typing import Any

from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import EmbeddingFunction
from dotenv import load_dotenv
from huggingface_hub import login
from huggingface_hub.errors import GatedRepoError

from .config import EmbeddingModelConfig, EmbeddingProvider

load_dotenv()

DIMENSIONS_FILE = Path(__file__).parent.parent / "utils" / "known_dimensions.json"

_hf_logged_in = False


def _get_hf_token() -> str | None:
    """Return the first configured HuggingFace token alias."""
    return (
        os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


def _ensure_hf_login():
    """Login to HuggingFace Hub once per process, only if API key is available.

    Called eagerly at module load so the .env token overwrites any stale
    token cached at ~/.cache/huggingface/token.  Without this, the
    transformers library silently sends the cached (possibly expired) token
    on every Hub request — even for public models — causing them to fail.
    """
    global _hf_logged_in
    if _hf_logged_in:
        return
    hf_token = _get_hf_token()
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        _hf_logged_in = True


# Login eagerly so the .env token replaces any stale cached token before
# any model download is attempted.
_ensure_hf_login()


def _load_known_dimensions() -> dict[str, int]:
    """Load dimensions from the JSON file."""
    if not DIMENSIONS_FILE.exists():
        return {}
    try:
        with open(DIMENSIONS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_known_dimension(model_name: str, dimension: int, current_dims: dict[str, int]) -> None:
    """Update the JSON file with a new model dimension."""
    try:
        current_dims[model_name] = dimension
        with open(DIMENSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(current_dims, f, indent=1, sort_keys=True)
            print(f"Updated known dimensions file with {model_name}: {dimension}")
    except OSError as e:
        print(f"Warning: Could not save dimension to file: {e}")


def create_embedding_function(
    config: EmbeddingModelConfig | None,
    device: str,
    known_dimension: int | None = None,
    is_query: bool = False,
) -> tuple[EmbeddingFunction, int]:
    """
    Create an embedding function based on the configuration.

    Args:
        config: Embedding model configuration (None uses defaults)
        device: Device for local models (cpu, cuda, mps)
        known_dimension: Pre-computed dimension (from metadata or cache).
                        If provided, skips test embedding.
        is_query: If True, configures EF for query embedding (QWEN adds instruction prefix)

    Returns:
        Tuple of (embedding_function, embedding_dimension)

    Raises:
        ValueError: If provider is not supported or API key is missing
    """
    if config is None:
        config = EmbeddingModelConfig()

    provider = config.provider
    model_name = config.model_name

    # Helper to get dimension (with fallback chain)
    def get_dimension(ef_for_test: Any) -> int:
        # 1. Use provided dimension (from ChromaDB metadata)
        if known_dimension is not None:
            return known_dimension

        known_dimensions = _load_known_dimensions()

        # 2. Check local cache
        if model_name in known_dimensions:
            return known_dimensions[model_name]

        # 3. Fallback: Run test embedding (only if needed)
        print(f"Warning: Unknown dimension for {model_name}, running test embedding...")
        test_embedding = ef_for_test(["test"])
        _save_known_dimension(model_name, len(test_embedding[0]), known_dimensions)
        return len(test_embedding[0])

    if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        from .specific_functions.embed_sentence_transformer import (
            SentenceTransformerEmbeddingFunction,
        )

        hf_token = _get_hf_token()
        if hf_token:
            _ensure_hf_login()

        st_kwargs = {"token": hf_token} if hf_token else {}

        try:
            ef = SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                device=device,
                prompt=config.prompt,
                **st_kwargs,
            )
        except GatedRepoError:
            _ensure_hf_login()
            ef = SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                device=device,
                prompt=config.prompt,
                **st_kwargs,
            )
        dim = get_dimension(ef)
        return ef, dim

    elif provider == EmbeddingProvider.OPENAI:
        # OpenAI API - reads from CHROMA_OPENAI_API_KEY env var
        ef = embedding_functions.OpenAIEmbeddingFunction(
            model_name=model_name
            # api_key read from CHROMA_OPENAI_API_KEY by default
        )
        dim = get_dimension(ef)

        return ef, dim

    elif provider == EmbeddingProvider.COHERE:
        # Cohere API - reads from CHROMA_COHERE_API_KEY env var
        ef = embedding_functions.CohereEmbeddingFunction(
            model_name=model_name
            # api_key read from CHROMA_COHERE_API_KEY by default
        )
        # Cohere v3 default is 1024 dim, but respect provided dimension
        dim = get_dimension(ef)
        return ef, dim

    elif provider == EmbeddingProvider.OLLAMA:
        # Local Ollama server
        url = config.ollama_url or "http://localhost:11434"
        ef = embedding_functions.OllamaEmbeddingFunction(url=url, model_name=model_name)
        dim = get_dimension(ef)
        return ef, dim

    elif provider == EmbeddingProvider.HUGGINGFACE_API:
        try:
            ef = embedding_functions.HuggingFaceEmbeddingFunction(
                model_name=model_name
                # api_key read from CHROMA_HUGGINGFACE_API_KEY by default
            )
            dim = get_dimension(ef)  # Use helper instead of test embedding
        except GatedRepoError:
            _ensure_hf_login()
            ef = embedding_functions.HuggingFaceEmbeddingFunction(
                model_name=model_name
                # api_key read from CHROMA_HUGGINGFACE_API_KEY by default
            )
            dim = get_dimension(ef)  # Use helper instead of test embedding

        return ef, dim

    elif provider == EmbeddingProvider.GEMINI:
        from .specific_functions.embed_gemini import EmbedTextGemini

        ef = EmbedTextGemini(model=model_name, task_type=config.task_type or "SEMANTIC_SIMILARITY")
        dim = get_dimension(ef)
        return ef, dim

    elif provider == EmbeddingProvider.BGE:
        from .specific_functions.embed_bge import EmbedTextBGE

        ef = EmbedTextBGE(model=model_name)
        dim = get_dimension(ef)
        return ef, dim

    elif provider == EmbeddingProvider.QWEN:
        from .specific_functions.embed_qwen import EmbedTextQWEN

        # Build kwargs - only include task if explicitly set (to preserve default)
        qwen_kwargs = {
            "model": model_name,
            "device": device,
            "is_query": is_query,
        }
        if config.task is not None:
            qwen_kwargs["task"] = config.task
        ef = EmbedTextQWEN(**qwen_kwargs)
        dim = get_dimension(ef)
        return ef, dim

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def get_device() -> str:
    """Detect and return the best available device for computation."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
