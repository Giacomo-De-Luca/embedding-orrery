"""GraphQL type definitions for embedding visualization backend."""

import strawberry
from typing import List, Optional, Dict, Any
from enum import Enum
from ..utils.provider_list import EmbeddingProviderEnum


# ========== JSON Scalar ==========

@strawberry.scalar(
    serialize=lambda v: v,
    parse_value=lambda v: v,
)
class JSON:
    """JSON scalar type for flexible metadata."""
    __slots__ = ()


# ========== HuggingFace Dataset Types ==========

@strawberry.type
class HFSplitInfo:
    """Information about a dataset split."""
    name: str
    num_rows: Optional[int] = None
    num_bytes: Optional[int] = None


@strawberry.type
class HFFeatureInfo:
    """Information about a dataset feature/column."""
    name: str
    dtype: str
    description: Optional[str] = None


@strawberry.type
class HFConfigInfo:
    """Information about a dataset configuration."""
    name: str
    splits: List[HFSplitInfo]
    features: List[HFFeatureInfo]


@strawberry.type
class HFDatasetInfo:
    """Complete information about a HuggingFace dataset."""
    dataset_id: str
    description: Optional[str] = None
    license: Optional[str] = None
    configs: List[HFConfigInfo]
    default_config: Optional[str] = None
    error: Optional[str] = None


@strawberry.type
class HFDatasetPreview:
    """Preview rows from a dataset."""
    dataset_id: str
    config: Optional[str] = None
    split: str
    columns: List[str]
    rows: List[JSON]
    total_rows: Optional[int] = None
    error: Optional[str] = None


@strawberry.enum
class PortionStrategyEnum(Enum):
    """Strategy for selecting which rows to embed."""
    FIRST_N = "first_n"
    RANDOM_SAMPLE = "random_sample"
    ROW_RANGE = "row_range"
    ALL = "all"


@strawberry.input
class PortionInput:
    """Input for selecting dataset portion."""
    strategy: PortionStrategyEnum
    n: Optional[int] = None  # For FIRST_N and RANDOM_SAMPLE
    start: Optional[int] = None  # For ROW_RANGE
    end: Optional[int] = None  # For ROW_RANGE
    seed: int = 42  # For RANDOM_SAMPLE




@strawberry.input
class EmbeddingModelInput:
    """Configuration for embedding model.

    Model names are free-form strings - any valid model for the provider works.

    Examples:
    - SentenceTransformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"
    - OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
    - Cohere: "embed-english-v3.0", "embed-multilingual-v3.0"
    - Ollama: "nomic-embed-text", "mxbai-embed-large"
    - HuggingFace API: "sentence-transformers/all-MiniLM-L6-v2"
    - QWEN: "Qwen/Qwen3-Embedding-0.6B" (supports task instruction for queries)
    - Gemini: "gemini-embedding-001" (supports task_type for embedding optimization)
    """
    provider: EmbeddingProviderEnum
    model_name: str
    # Provider-specific options
    ollama_url: Optional[str] = None  # Ollama: server URL (default: http://localhost:11434)
    task: Optional[str] = None  # QWEN: Query instruction prefix (used at query time only)
    task_type: Optional[str] = None  # Gemini: Embedding optimization (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, etc.)


@strawberry.input
class EmbedDatasetInput:
    """Input for embedding a HuggingFace dataset."""
    dataset_id: str
    collection_name: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[List[str]] = None  # Columns to embed
    text_template: Optional[str] = None  # Template for combining columns
    id_column: Optional[str] = None  # Column to use as document ID
    portion: Optional[PortionInput] = None
    metadata_columns: Optional[List[str]] = None
    compute_projections: bool = True  # Whether to compute PCA/UMAP after embedding
    # Embedding model configuration (default: SentenceTransformers with all-MiniLM-L6-v2)
    embedding_model: Optional[EmbeddingModelInput] = None


@strawberry.type
class EmbedDatasetResult:
    """Result of embedding a dataset."""
    collection_name: str
    total_embedded: int
    embedding_dim: int
    device: str
    duration_seconds: float
    projections_computed: bool = False
    error: Optional[str] = None
    # Model information
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None


# ========== Local File Types ==========

@strawberry.type
class LocalFileInfo:
    """Information about a local data file."""
    file_path: str
    file_type: str
    columns: List[str]
    num_rows: int
    file_size_bytes: int
    error: Optional[str] = None


@strawberry.type
class LocalFilePreview:
    """Preview rows from a local file."""
    file_path: str
    columns: List[str]
    rows: List[JSON]
    total_rows: int
    error: Optional[str] = None


@strawberry.enum
class DataTypeEnum(Enum):
    """Type of data to embed."""
    TEXT = "text"
    IMAGE = "image"
    VECTOR = "vector"


@strawberry.input
class EmbedLocalFileInput:
    """Input for embedding a local file."""
    file_path: str
    collection_name: str
    data_type: DataTypeEnum = DataTypeEnum.TEXT
    columns: Optional[List[str]] = None  # Columns to embed (for text)
    text_template: Optional[str] = None
    image_column: Optional[str] = None  # Column containing image data
    vector_column: Optional[str] = None  # Column containing pre-computed vectors
    id_column: Optional[str] = None
    metadata_columns: Optional[List[str]] = None
    n_rows: Optional[int] = None  # Limit rows
    sample_n: Optional[int] = None  # Random sample
    sample_seed: int = 42
    compute_projections: bool = True
    # Embedding model configuration (default: SentenceTransformers with all-MiniLM-L6-v2)
    # Only used for TEXT data_type; IMAGE uses ViT, VECTOR uses pre-computed
    embedding_model: Optional[EmbeddingModelInput] = None


# ========== Search & Filter Types ==========

@strawberry.enum
class SimilarityMeasure(Enum):
    """Similarity/distance metrics supported by ChromaDB."""
    COSINE = "cosine"
    L2 = "l2"
    IP = "ip"  # Inner product


@strawberry.enum
class FilterOperator(Enum):
    """Filter operators for ChromaDB where clauses."""
    EQ = "$eq"
    NE = "$ne"
    GT = "$gt"
    GTE = "$gte"
    LT = "$lt"
    LTE = "$lte"
    IN = "$in"
    NIN = "$nin"


@strawberry.input
class FilterInput:
    """Input for filtering collections."""
    field: str
    operator: FilterOperator
    value: JSON


# ========== Collection Types ==========

@strawberry.type
class CollectionMetadata:
    """Metadata about a collection."""
    total_items: Optional[int] = None  # Generic item count
    total_words: Optional[int] = None  # Legacy: same as total_items
    embedding_dim: Optional[int] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    timestamp: Optional[str] = None
    pca_2d_variance: Optional[List[float]] = None
    pca_3d_variance: Optional[List[float]] = None
    # Source metadata (varies by data source)
    source_dataset: Optional[str] = None  # HuggingFace dataset ID
    source_split: Optional[str] = None
    source_file: Optional[str] = None  # Local file path
    embedded_columns: Optional[str] = None
    has_projections: Optional[bool] = None


@strawberry.type
class Collection:
    """Information about a collection."""
    name: str
    metadata: Optional[JSON] = None
    count: int


@strawberry.type
class UpdateCollectionMetadataResult:
    """Result of updating collection metadata."""
    name: str
    metadata: JSON
    error: Optional[str] = None


@strawberry.type
class EmbeddingItem:
    """Single embedding item with all associated data."""
    id: str
    word: Optional[str] = None
    definition: Optional[str] = None
    pos: Optional[str] = None
    embedding: Optional[List[float]] = None
    document: Optional[str] = None
    metadata: Optional[JSON] = None


@strawberry.type
class SemanticSearchResult:
    """Result from semantic search."""
    id: str
    document: Optional[str] = None
    metadata: Optional[JSON] = None  # All item metadata
    distance: float
    similarity: float
    embedding: Optional[List[float]] = None


@strawberry.type
class ProjectionData:
    """Complete projection data for visualization.

    Generic structure that works with any data source:
    - ids: unique identifiers for each item
    - documents: the main text content (what was embedded)
    - item_metadata: raw metadata per item (flexible schema)
    - available_fields: list of available metadata field names
    - Projections: PCA and UMAP coordinates
    """
    ids: List[str]
    documents: List[str]
    item_metadata: List[JSON]  # Raw metadata per item - flexible schema
    available_fields: List[str]  # What metadata fields are available
    # Projections
    pca_2d: List[List[float]]
    pca_3d: List[List[float]]
    umap_2d: List[List[float]]
    umap_3d: List[List[float]]
    # Collection-level metadata
    metadata: CollectionMetadata


# ========== Helper Functions ==========

def build_where_clause(filters: Optional[List[FilterInput]]) -> Optional[Dict[str, Any]]:
    """Build ChromaDB where clause from filter inputs.

    Args:
        filters: List of filter inputs

    Returns:
        ChromaDB where clause dictionary
    """
    if not filters:
        return None

    where = {}
    for f in filters:
        if f.operator == FilterOperator.EQ:
            where[f.field] = {"$eq": f.value}
        elif f.operator == FilterOperator.NE:
            where[f.field] = {"$ne": f.value}
        elif f.operator == FilterOperator.GT:
            where[f.field] = {"$gt": f.value}
        elif f.operator == FilterOperator.GTE:
            where[f.field] = {"$gte": f.value}
        elif f.operator == FilterOperator.LT:
            where[f.field] = {"$lt": f.value}
        elif f.operator == FilterOperator.LTE:
            where[f.field] = {"$lte": f.value}
        elif f.operator == FilterOperator.IN:
            where[f.field] = {"$in": f.value}
        elif f.operator == FilterOperator.NIN:
            where[f.field] = {"$nin": f.value}

    return where if where else None
