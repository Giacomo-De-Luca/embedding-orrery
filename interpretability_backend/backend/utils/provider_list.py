"""
Single source of truth for embedding provider definitions.

This module contains the master list of all supported embedding providers.
All other modules import from here to avoid duplication.
"""
from enum import Enum
import strawberry


# Provider descriptions (used by GraphQL and frontend)
PROVIDER_DESCRIPTIONS = {
    "SENTENCE_TRANSFORMERS": "Local models via sentence-transformers library (no API key)",
    "OPENAI": "OpenAI API (requires CHROMA_OPENAI_API_KEY env var)",
    "COHERE": "Cohere API (requires CHROMA_COHERE_API_KEY env var)",
    "OLLAMA": "Local Ollama server (no API key required)",
    "HUGGINGFACE_API": "HuggingFace Inference API (requires CHROMA_HUGGINGFACE_API_KEY env var)",
    "GEMINI": "Google Gemini API (requires GEMINI_API_KEY env var)",
    "BGE": "Local BGE models (no API key required)",
    "QWEN": "Local Qwen models (no API key required)",
}


@strawberry.enum
class EmbeddingProviderEnum(Enum):
    """Embedding model provider.

    Supported providers:
    - SENTENCE_TRANSFORMERS: Local models via sentence-transformers library (no API key)
    - OPENAI: OpenAI API (requires CHROMA_OPENAI_API_KEY env var)
    - COHERE: Cohere API (requires CHROMA_COHERE_API_KEY env var)
    - OLLAMA: Local Ollama server (no API key required)
    - HUGGINGFACE_API: HuggingFace Inference API (requires CHROMA_HUGGINGFACE_API_KEY env var)
    - GEMINI: Google Gemini API (requires GEMINI_API_KEY env var)
    - BGE: Local BGE models (no API key required)
    - QWEN: Local Qwen models (no API key required)
    """
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    COHERE = "cohere"
    OLLAMA = "ollama"
    HUGGINGFACE_API = "huggingface_api"
    GEMINI = "gemini"
    BGE = "bge"
    QWEN = "qwen"

