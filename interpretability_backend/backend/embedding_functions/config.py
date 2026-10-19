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

    def to_job_config(self) -> dict:
        """JSON form stored in job state (see BaseConfig.to_job_config)."""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "ollama_url": self.ollama_url,
            "task": self.task,
            "task_type": self.task_type,
            "prompt": self.prompt,
        }


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
    # The GraphQL input the job was started from (camelCase, as built by
    # API/converters.graphql_input_payload). Stored in job state so the
    # frontend resumes by replaying it with resume=true.
    job_input: dict | None = None

    def to_job_config(self) -> dict:
        """JSON summary stored in job state: Jobs-list display fields + the
        replayable input."""
        return {
            "collection_name": self.collection_name,
            "columns": self.columns,
            "text_template": self.text_template,
            "id_column": self.id_column,
            "metadata_columns": self.metadata_columns,
            "batch_size": self.batch_size,
            "embedding_model": self.embedding_model.to_job_config()
            if self.embedding_model
            else None,
            **self._source_job_config(),
            "input": self.job_input,
        }

    def _source_job_config(self) -> dict:
        """Source-specific fields of to_job_config (overridden per source)."""
        return {}


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

    def _source_job_config(self) -> dict:
        portion = None
        if self.portion:
            portion = {
                "strategy": self.portion.strategy.value,
                "n": self.portion.n,
                "start": self.portion.start,
                "end": self.portion.end,
                "seed": self.portion.seed,
            }
        return {
            "dataset_id": self.dataset_id,
            "config": self.config,
            "split": self.split,
            "splits": self.splits,
            "portion": portion,
        }


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

    def _source_job_config(self) -> dict:
        return {
            "file_path": self.file_path,
            "data_type": self.data_type.value,
            "image_column": self.image_column,
            "vector_column": self.vector_column,
            "n_rows": self.n_rows,
            "sample_n": self.sample_n,
            "sample_seed": self.sample_seed,
        }


@dataclass(kw_only=True)
class ReEmbedConfig(BaseConfig):
    """Configuration for re-embedding an existing DuckDB dataset with a new model.

    Reads documents from an existing dataset in DuckDB and embeds them with a
    different embedding model into a new ChromaDB vector collection.
    No new items are written to DuckDB — only ChromaDB gets new vectors.
    """

    source_dataset_name: str  # Existing dataset in DuckDB to read documents from

    def _source_job_config(self) -> dict:
        return {"source_dataset_name": self.source_dataset_name}
