"""
Configuration for embedding datasets.
"""

from dataclasses import dataclass
from enum import Enum

from ..clients.huggingface_client import PortionConfig
from ..utils.resource_paths import CHROMA_DB_PATH as DB_PATH, DUCKDB_PATH  # noqa: F401

# ========== Configuration Constants ==========
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_url: str | None = None  # Default: http://localhost:11434
    task: str | None = None  # QWEN: Query instruction prefix (used at query time only)
    task_type: str | None = (
        None  # Gemini: Embedding optimization type (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, etc.)
    )
    # SentenceTransformers: Prompt support for models like EmbeddingGemma
    prompt: str | None = None  # Single field - can be predefined name or custom string


@dataclass
class EmbeddingResult:
    """Result of embedding a dataset."""

    collection_name: str
    total_embedded: int
    embedding_dim: int
    device: str
    duration_seconds: float
    error: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None


@dataclass(kw_only=True)
class BaseConfig:
    """base configuration for embedding collections"""

    collection_name: str
    embedding_model: EmbeddingModelConfig | None = None  # Embedding model config
    columns: list[str] | None = None  # Columns to embed (combined into text)
    text_template: str | None = None  # Template for combining columns, e.g., "{title}: {text}"
    id_column: str | None = None  # Column to use as document ID (default: row index)
    metadata_columns: list[str] | None = None  # Additional columns to store as metadata
    batch_size: int = 100  # Batch size for embedding
    resume: bool = False  # Resume an interrupted job instead of starting fresh


@dataclass(kw_only=True)
class EmbeddingConfig(BaseConfig):
    """Configuration for embedding a HuggingFace dataset."""

    dataset_id: str
    config: str | None = None
    split: str = "train"
    # When set (multiple splits), all listed splits are embedded into one
    # collection in a single pass, each row tagged with its `source_split`.
    # Falls back to `[split]` when None.
    splits: list[str] | None = None
    portion: PortionConfig | None = None  # Portion of dataset to embed


@dataclass(kw_only=True)
class LocalFileEmbeddingConfig(BaseConfig):
    """Configuration for embedding a local file."""

    file_path: str
    data_type: DataType = DataType.TEXT
    image_column: str | None = None  # Column containing image data/paths
    vector_column: str | None = None  # Column containing pre-computed vectors
    n_rows: int | None = None  # Limit rows
    sample_n: int | None = None  # Random sample
    sample_seed: int = 42


@dataclass(kw_only=True)
class ReEmbedConfig(BaseConfig):
    """Configuration for re-embedding an existing DuckDB dataset with a new model.

    Reads documents from an existing dataset in DuckDB and embeds them with a
    different embedding model into a new ChromaDB vector collection.
    No new items are written to DuckDB — only ChromaDB gets new vectors.
    """

    source_dataset_name: str  # Existing dataset in DuckDB to read documents from
