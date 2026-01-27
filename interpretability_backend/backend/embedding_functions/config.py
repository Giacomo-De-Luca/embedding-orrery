"""
Configuration for embedding datasets.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

# ========== Configuration Constants ==========
DB_PATH = Path(__file__).parent.parent.parent / "resources" / "vector_db"
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "google/vit-base-patch16-384"


###TODO: this is terrible, it should be dynamic based on model used
## at the moment it's only used for errors
TEXT_EMBEDDING_DIMENSIONS = 384
IMAGE_EMBEDDING_DIMENSIONS = 768

# TODO: make these automatically adjust 
# first call for the amount of available memory
# second, check how much memory the model takes
# third, automatically set up batch size
EMBEDDING_BATCH_SIZE = 100
IMAGE_BATCH_SIZE = 16


class DataType(Enum):
    """Type of data to embed."""
    TEXT = "text"
    IMAGE = "image"
    VECTOR = "vector"  # Pre-computed embeddings


 
### HACK: Terrible replace it with the correct enum used also by strawberry
class EmbeddingProvider(Enum):
    """Embedding model provider."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    COHERE = "cohere"
    OLLAMA = "ollama"
    HUGGINGFACE_API = "huggingface_api"
    GEMINI = "gemini"
    BGE = "bge" 
    QWEN = "qwen"


### TODO why this is has only Sentence Transformers?
@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding model."""
    provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    model_name: str = "all-MiniLM-L6-v2"
    ollama_url: Optional[str] = None  # Default: http://localhost:11434
    task: Optional[str] = None  # QWEN: Query instruction prefix (used at query time only)
    task_type: Optional[str] = None  # Gemini: Embedding optimization type (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, etc.)
    # SentenceTransformers: Prompt support for models like EmbeddingGemma
    prompt: Optional[str] = None  # Direct prompt string (e.g., "task: search result | query: ")
    prompt_name: Optional[str] = None  # Predefined prompt name (e.g., "Retrieval-query", "STS")


@dataclass
class EmbeddingResult:
    """Result of embedding a dataset."""
    collection_name: str
    total_embedded: int
    embedding_dim: int
    device: str
    duration_seconds: float
    error: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None


# Import PortionConfig for EmbeddingConfig
# Note: PortionConfig is in huggingface_client. 
# We might need to handle this dependency or import it.
# Ideally configs shouldn't depend on clients, but let's import for now to match original struct.
from ..clients.huggingface_client import PortionConfig




@dataclass(kw_only=True)
class BaseConfig:
    """base configuration for embedding collections"""
    collection_name: str
    embedding_model: Optional[EmbeddingModelConfig] = None  # Embedding model config
    columns: Optional[List[str]] = None  # Columns to embed (combined into text)
    text_template: Optional[str] = None  # Template for combining columns, e.g., "{title}: {text}"
    id_column: Optional[str] = None  # Column to use as document ID (default: row index)
    metadata_columns: Optional[List[str]] = None  # Additional columns to store as metadata
    batch_size: int = 100  # Batch size for embedding
    resume: bool = False  # Resume an interrupted job instead of starting fresh

@dataclass(kw_only=True)
class EmbeddingConfig(BaseConfig):
    """Configuration for embedding a HuggingFace dataset."""
    dataset_id: str
    config: Optional[str] = None
    split: str = "train"
    portion: Optional[PortionConfig] = None  # Portion of dataset to embed

@dataclass(kw_only=True)
class LocalFileEmbeddingConfig(BaseConfig):
    """Configuration for embedding a local file."""
    file_path: str
    data_type: DataType = DataType.TEXT
    image_column: Optional[str] = None  # Column containing image data/paths
    vector_column: Optional[str] = None  # Column containing pre-computed vectors
    n_rows: Optional[int] = None  # Limit rows
    sample_n: Optional[int] = None  # Random sample
    sample_seed: int = 42
