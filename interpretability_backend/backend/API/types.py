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
    - SentenceTransformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2", "google/gemma-embedding-001"
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
    # SentenceTransformers: Prompt support for models like EmbeddingGemma
    prompt: Optional[str] = None  # Single field - can be predefined name (e.g., "Retrieval-query") or custom string


# ========== Topic Extraction Types ==========

@strawberry.type
class TopicKeyword:
    """Keyword extracted for a topic with its c-TF-IDF score."""
    word: str
    score: float


@strawberry.type
class TopicInfo:
    """Information about an extracted topic."""
    topic_id: int
    keywords: List[TopicKeyword]
    label: Optional[str]
    count: int
    subtopics: Optional[List[str]] = None


@strawberry.input
class TopicReductionInput:
    """Configuration for topic reduction."""
    enabled: bool = False
    method: str = "auto"  # "auto" or "fixed_n"
    n_topics: Optional[int] = None  # Required when method="fixed_n"
    use_ctfidf: bool = True  # True=c-TF-IDF (fast), False=semantic (better quality)


@strawberry.input
class TopicConfigInput:
    """Topic extraction configuration (shared by standalone and embedded mutations)."""
    min_topic_size: int = 10
    n_keywords: int = 10
    use_llm_labels: bool = False
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3-flash-preview"
    projection_type: str = "umap_2d"

    # Topic reduction config
    reduction: Optional[TopicReductionInput] = None


@strawberry.input
class ExtractTopicsInput:
    """Input for standalone topic extraction from an existing collection."""
    collection_name: str
    config: Optional[TopicConfigInput] = None


@strawberry.type
class ExtractTopicsResult:
    """Result of topic extraction."""
    collection_name: str
    num_topics: int
    num_noise_points: int
    topics: List[TopicInfo]
    duration_seconds: float
    error: Optional[str] = None

    # Reduction tracking
    num_topics_before_reduction: Optional[int] = None
    reduction_applied: bool = False


@strawberry.input
class ReduceTopicsInput:
    """Input for standalone topic reduction mutation."""
    collection_name: str
    method: str = "auto"  # "auto" or "fixed_n"
    n_topics: Optional[int] = None  # Required when method="fixed_n"
    use_ctfidf: bool = True  # True=c-TF-IDF (fast), False=semantic (better quality)

    # Re-labeling after reduction
    regenerate_labels: bool = False
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3-flash-preview"


@strawberry.type
class ReduceTopicsResult:
    """Result of standalone topic reduction."""
    collection_name: str
    num_topics_before: int
    num_topics_after: int
    topics: List[TopicInfo]
    topic_mappings: JSON  # {old_id: new_id}
    duration_seconds: float
    error: Optional[str] = None


@strawberry.input
class GenerateLlmLabelsInput:
    """Input for standalone LLM label generation on existing topics."""
    collection_name: str
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3-flash-preview"
    label_scope: str = "both"  # "both" | "topics_only" | "subtopics_only"
    resume: bool = False


@strawberry.type
class GenerateLlmLabelsResult:
    """Result of standalone LLM label generation."""
    collection_name: str
    topics_labeled: int
    subtopics_labeled: int
    total_topics: int
    total_subtopics: int
    duration_seconds: float
    error: Optional[str] = None


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
    batch_size: Optional[int] = 100
    # Embedding model configuration (default: SentenceTransformers with all-MiniLM-L6-v2)
    embedding_model: Optional[EmbeddingModelInput] = None
    # Resume an interrupted job instead of starting fresh
    resume: bool = False
    # Topic extraction after embedding
    extract_topics: bool = False
    topic_config: Optional[TopicConfigInput] = None


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
    batch_size: Optional[int] = 100
    # Embedding model configuration (default: SentenceTransformers with all-MiniLM-L6-v2)
    # Only used for TEXT data_type; IMAGE uses ViT, VECTOR uses pre-computed
    embedding_model: Optional[EmbeddingModelInput] = None
    # Resume an interrupted job instead of starting fresh
    resume: bool = False
    # Topic extraction after embedding
    extract_topics: bool = False
    topic_config: Optional[TopicConfigInput] = None


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
    # Prompt info (for models like Gemma Embedding)
    embedding_prompt: Optional[str] = None  # Single field - can be predefined name or custom string
    # Topic extraction metadata
    has_topics: Optional[bool] = None
    topic_count: Optional[int] = None
    topics_extracted_at: Optional[str] = None
    topics: Optional[List[TopicInfo]] = None  # Topic summary with keywords
    topic_hierarchy: Optional[JSON] = None  # {reduced_label: [subtopic_labels]}
    # Pre-computed field analysis (cached for fast frontend loading)
    field_analysis: Optional[JSON] = None


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
    # Projections (Optional — only requested types are returned, others are null)
    pca_2d: Optional[List[List[float]]] = None
    pca_3d: Optional[List[List[float]]] = None
    umap_2d: Optional[List[List[float]]] = None
    umap_3d: Optional[List[List[float]]] = None
    # Collection-level metadata
    metadata: CollectionMetadata


# ========== Helper Functions ==========

def build_where_clause(filters: Optional[List[FilterInput]]) -> Optional[Dict[str, Any]]:
    """Build ChromaDB where clause from filter inputs.

    Uses $and wrapping when there are multiple filters, which is required
    when the same field appears more than once (e.g. year >= 1956 AND year <= 1979).

    Args:
        filters: List of filter inputs

    Returns:
        ChromaDB where clause dictionary
    """
    if not filters:
        return None

    clauses = []
    for f in filters:
        clauses.append({f.field: {f.operator.value: f.value}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ========== Embedding Job Types ==========

@strawberry.enum
class JobStatusEnum(Enum):
    """Status of an embedding job."""
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"


@strawberry.type
class EmbeddingJob:
    """Information about an embedding job."""
    collection_name: str
    status: str  # running, interrupted, completed
    job_type: str  # huggingface, local_file

    # Progress
    items_embedded: int
    total_expected: int
    batches_completed: int
    total_batches: int
    percent_complete: float

    # Config summary
    source: str  # dataset_id or file_path
    columns: Optional[List[str]] = None
    embedding_model: Optional[str] = None
    batch_size: int = 100
    started_at: str = ""

    # Full config for resume verification
    config: Optional[JSON] = None


# JobProgress is defined in subscriptions.py to avoid circular imports
# when embedding functions import emit_progress_sync
