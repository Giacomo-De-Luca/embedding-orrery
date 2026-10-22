"""
Factory for creating embedding functions.
"""

import os
from collections.abc import Iterable
from typing import Any

from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import EmbeddingFunction
from dotenv import load_dotenv
from huggingface_hub import login
from huggingface_hub.errors import GatedRepoError

from ..utils.embedding_model_catalog import TASK_PREFIX_FORMAT
from ..utils.embedding_tasks import TaskPrefixPolicy
from ..utils.known_dimensions import KnownDimensions
from ..utils.model_max_length import ModelMaxLengthResolver
from ..utils.provider_list import LOCAL_PROVIDER_VALUES
from .config import EmbeddingModelConfig, EmbeddingProvider

load_dotenv()

TEXT_ONLY = ("text",)

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

try:
    _ensure_hf_login()
except Exception:
    print("Internet not accessible or HF token not in env.")


def create_embedding_function(
    config: EmbeddingModelConfig | None,
    device: str,
    known_dimension: int | None = None,
    is_query: bool = False,
    modalities: Iterable[str] = TEXT_ONLY,
    query_prompt: str | None = None,
    max_length_resolver: ModelMaxLengthResolver | None = None,
) -> tuple[EmbeddingFunction, int]:
    """
    Create an embedding function based on the configuration.

    Args:
        config: Embedding model configuration (None uses defaults)
        device: Device for local models (cpu, cuda, mps)
        known_dimension: Pre-computed dimension (from metadata or cache).
                        If provided, skips test embedding.
        is_query: If True, configures EF for query embedding (QWEN adds its instruction,
                  SentenceTransformers maps the document prompt to its query prompt,
                  task-prefix models prepend the query prefix)
        modalities: Input modalities to embed ("text", "image", "audio", "video"). Only
                    SentenceTransformers models embed media; they load only the
                    encoder towers these need.
        query_prompt: Explicit query prompt (with ``is_query``; SentenceTransformers
                    and task-prefix models)
        max_length_resolver: Model-limit lookup for local providers (default: the
                    shared one). ``config.max_length`` falls back to the model's
                    limit and may not exceed it; ``config.batch_size`` is the
                    number of texts per forward pass.

    Task prefixes (``utils/embedding_tasks.py``): a config marked
    ``prompt_format`` embeds documents the pipeline already prefixed (no
    prompt here) and queries with the collection's query prefix, or the
    explicit ``query_prompt``. Gemini Embedding 2 never gets a ``task_type``.

    Returns:
        Tuple of (embedding_function, embedding_dimension)

    Raises:
        ValueError: If provider is not supported, API key is missing, the
            provider cannot embed the requested modalities, or an explicit
            query prompt is a prompt name without a query prefix
    """
    if config is None:
        config = EmbeddingModelConfig()

    provider = config.provider
    model_name = config.model_name
    modalities = tuple(modalities)
    task_prefixes = TaskPrefixPolicy()
    task_prefixed = config.prompt_format == TASK_PREFIX_FORMAT
    media = sorted(set(modalities) - set(TEXT_ONLY))
    if media and provider != EmbeddingProvider.SENTENCE_TRANSFORMERS:
        raise ValueError(
            f"{provider.value} embeds text only; {', '.join(media)} needs a "
            "sentence-transformers model such as google/embeddinggemma-2"
        )

    # Local providers: input-length cap (the model's limit unless a lower one is
    # requested) and texts per forward pass; unset values keep the embedder defaults.
    limits: dict[str, int] = {}
    if provider.value in LOCAL_PROVIDER_VALUES:
        resolver = max_length_resolver or ModelMaxLengthResolver.shared()
        max_length = resolver.effective(provider.value, model_name, config.max_length)
        if max_length is not None:
            limits["max_length"] = max_length
        if config.batch_size is not None:
            limits["batch_size"] = config.batch_size

    # Helper to get dimension (with fallback chain)
    def get_dimension(ef_for_test: Any) -> int:
        # 1. Use provided dimension (from ChromaDB metadata)
        if known_dimension is not None:
            return known_dimension

        known_dimensions = KnownDimensions()

        # 2. Check local cache
        cached = known_dimensions.get(model_name)
        if cached is not None:
            return cached

        # 3. Fallback: Run test embedding (only if needed)
        print(f"Warning: Unknown dimension for {model_name}, running test embedding...")
        test_embedding = ef_for_test(["test"])
        known_dimensions.record(model_name, len(test_embedding[0]))
        return len(test_embedding[0])

    if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        from .specific_functions.embed_sentence_transformer import (
            SentenceTransformerEmbeddingFunction,
        )

        hf_token = _get_hf_token()
        if hf_token:
            _ensure_hf_login()

        prompt, st_query_prompt = config.prompt, query_prompt
        if task_prefixed:
            # Documents arrive prefixed; a query prefix is literal prompt text.
            prompt = None
            st_query_prompt = (
                task_prefixes.query_prefix(config.task, query_prompt) if is_query else None
            )
        st_kwargs = {
            "model_name": model_name,
            "device": device,
            "prompt": prompt,
            "modalities": modalities,
            "is_query": is_query,
            "query_prompt": st_query_prompt,
            **limits,
        }
        if hf_token:
            st_kwargs["token"] = hf_token

        try:
            ef = SentenceTransformerEmbeddingFunction(**st_kwargs)
        except GatedRepoError:
            _ensure_hf_login()
            ef = SentenceTransformerEmbeddingFunction(**st_kwargs)
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

        if task_prefixes.uses_task_prefixes(provider.value, model_name):
            # Gemini Embedding 2: no task_type; documents arrive prefixed.
            task_type = None
            text_prefix = (
                task_prefixes.query_prefix(config.task, query_prompt) if is_query else None
            )
        else:
            task_type, text_prefix = config.task_type or "SEMANTIC_SIMILARITY", None
        ef = EmbedTextGemini(model=model_name, task_type=task_type, text_prefix=text_prefix)
        dim = get_dimension(ef)
        return ef, dim

    elif provider == EmbeddingProvider.BGE:
        from .specific_functions.embed_bge import EmbedTextBGE

        ef = EmbedTextBGE(model=model_name, device=device, **limits)
        dim = get_dimension(ef)
        return ef, dim

    elif provider == EmbeddingProvider.QWEN:
        from .specific_functions.embed_qwen import EmbedTextQWEN

        # Build kwargs - only include task if explicitly set (to preserve default)
        qwen_kwargs = {
            "model": model_name,
            "device": device,
            "is_query": is_query,
            **limits,
        }
        if config.task is not None:
            qwen_kwargs["task"] = config.task
        ef = EmbedTextQWEN(**qwen_kwargs)
        dim = get_dimension(ef)
        return ef, dim

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


API_DEVICE = "cpu"


def device_for(provider: EmbeddingProvider) -> str:
    """The device a provider's model runs on.

    Remote providers (Gemini, OpenAI, ...) get ``"cpu"`` without probing, so a
    search on their collections never imports torch into the API process.
    """
    if provider.value in LOCAL_PROVIDER_VALUES:
        return get_device()
    return API_DEVICE


def get_device() -> str:
    """Detect and return the best available device for computation."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
