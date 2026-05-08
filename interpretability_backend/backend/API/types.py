"""GraphQL type definitions for embedding visualization backend."""

from enum import Enum
from typing import Any

import strawberry

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
    num_rows: int | None = None
    num_bytes: int | None = None


@strawberry.type
class HFFeatureInfo:
    """Information about a dataset feature/column."""

    name: str
    dtype: str
    description: str | None = None


@strawberry.type
class HFConfigInfo:
    """Information about a dataset configuration."""

    name: str
    splits: list[HFSplitInfo]
    features: list[HFFeatureInfo]


@strawberry.type
class HFDatasetInfo:
    """Complete information about a HuggingFace dataset."""

    dataset_id: str
    description: str | None = None
    license: str | None = None
    configs: list[HFConfigInfo]
    default_config: str | None = None
    error: str | None = None


@strawberry.type
class HFDatasetPreview:
    """Preview rows from a dataset."""

    dataset_id: str
    config: str | None = None
    split: str
    columns: list[str]
    rows: list[JSON]
    total_rows: int | None = None
    error: str | None = None


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
    n: int | None = None  # For FIRST_N and RANDOM_SAMPLE
    start: int | None = None  # For ROW_RANGE
    end: int | None = None  # For ROW_RANGE
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
    ollama_url: str | None = None  # Ollama: server URL (default: http://localhost:11434)
    task: str | None = None  # QWEN: Query instruction prefix (used at query time only)
    task_type: str | None = (
        None  # Gemini: Embedding optimization (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, etc.)
    )
    # SentenceTransformers: Prompt support for models like EmbeddingGemma
    prompt: str | None = (
        None  # Single field - can be predefined name (e.g., "Retrieval-query") or custom string
    )


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
    keywords: list[TopicKeyword]
    label: str | None
    count: int
    subtopics: list[str] | None = None


@strawberry.input
class TopicReductionInput:
    """Configuration for topic reduction."""

    enabled: bool = False
    method: str = "auto"  # "auto" or "fixed_n"
    n_topics: int | None = None  # Required when method="fixed_n"
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

    # Clustering method: "hdbscan" (default), "kmeans", "gmm", "spectral"
    clustering_method: str = "hdbscan"
    # Number of clusters (required for kmeans, gmm, spectral; ignored for hdbscan)
    n_clusters: int | None = None

    # Topic reduction config
    reduction: TopicReductionInput | None = None


@strawberry.input
class ExtractTopicsInput:
    """Input for standalone topic extraction from an existing collection."""

    collection_name: str
    config: TopicConfigInput | None = None


@strawberry.type
class ExtractTopicsResult:
    """Result of topic extraction."""

    collection_name: str
    num_topics: int
    num_noise_points: int
    topics: list[TopicInfo]
    duration_seconds: float
    error: str | None = None

    # Reduction tracking
    num_topics_before_reduction: int | None = None
    reduction_applied: bool = False


@strawberry.input
class ReduceTopicsInput:
    """Input for standalone topic reduction mutation."""

    collection_name: str
    method: str = "auto"  # "auto" or "fixed_n"
    n_topics: int | None = None  # Required when method="fixed_n"
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
    topics: list[TopicInfo]
    topic_mappings: JSON  # {old_id: new_id}
    duration_seconds: float
    error: str | None = None


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
    error: str | None = None


@strawberry.input
class RenameTopicLabelInput:
    """Input for renaming a topic or subtopic label."""

    collection_name: str
    topic_id: int
    new_label: str
    is_subtopic: bool = False  # If True, topic_id is treated as subtopic_id


@strawberry.type
class RenameTopicLabelResult:
    """Result of renaming a topic label."""

    collection_name: str
    topic_id: int
    new_label: str
    error: str | None = None


@strawberry.input
class EmbedDatasetInput:
    """Input for embedding a HuggingFace dataset."""

    dataset_id: str
    collection_name: str
    config: str | None = None
    split: str = "train"
    columns: list[str] | None = None  # Columns to embed
    text_template: str | None = None  # Template for combining columns
    id_column: str | None = None  # Column to use as document ID
    portion: PortionInput | None = None
    metadata_columns: list[str] | None = None
    compute_projections: bool = True  # Whether to compute PCA/UMAP after embedding
    batch_size: int | None = 100
    # Embedding model configuration (default: SentenceTransformers with all-MiniLM-L6-v2)
    embedding_model: EmbeddingModelInput | None = None
    # Resume an interrupted job instead of starting fresh
    resume: bool = False
    # Topic extraction after embedding
    extract_topics: bool = False
    topic_config: TopicConfigInput | None = None


@strawberry.type
class EmbedDatasetResult:
    """Result of embedding a dataset."""

    collection_name: str
    total_embedded: int
    embedding_dim: int
    device: str
    duration_seconds: float
    projections_computed: bool = False
    error: str | None = None
    # Model information
    embedding_provider: str | None = None
    embedding_model: str | None = None


# ========== Local File Types ==========


@strawberry.type
class LocalFileInfo:
    """Information about a local data file."""

    file_path: str
    file_type: str
    columns: list[str]
    num_rows: int
    file_size_bytes: int
    error: str | None = None


@strawberry.type
class LocalFilePreview:
    """Preview rows from a local file."""

    file_path: str
    columns: list[str]
    rows: list[JSON]
    total_rows: int
    error: str | None = None


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
    columns: list[str] | None = None  # Columns to embed (for text)
    text_template: str | None = None
    image_column: str | None = None  # Column containing image data
    vector_column: str | None = None  # Column containing pre-computed vectors
    id_column: str | None = None
    metadata_columns: list[str] | None = None
    n_rows: int | None = None  # Limit rows
    sample_n: int | None = None  # Random sample
    sample_seed: int = 42
    compute_projections: bool = True
    batch_size: int | None = 100
    # Embedding model configuration (default: SentenceTransformers with all-MiniLM-L6-v2)
    # Only used for TEXT data_type; IMAGE uses ViT, VECTOR uses pre-computed
    embedding_model: EmbeddingModelInput | None = None
    # Resume an interrupted job instead of starting fresh
    resume: bool = False
    # Topic extraction after embedding
    extract_topics: bool = False
    topic_config: TopicConfigInput | None = None


# ========== Text Search Types ==========


@strawberry.enum
class TextSearchMode(Enum):
    """Mode for text search matching."""

    CONTAINS = "contains"
    EXACT = "exact"


@strawberry.type
class TextSearchMatch:
    """A single text search match."""

    id: str
    matched_field: str
    snippet: str | None = None


@strawberry.type
class TextSearchResponse:
    """Result of a text search query."""

    matches: list[TextSearchMatch]
    total_matches: int


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

    total_items: int | None = None  # Generic item count
    total_words: int | None = None  # Legacy: same as total_items
    embedding_dim: int | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    timestamp: str | None = None
    pca_2d_variance: list[float] | None = None
    pca_3d_variance: list[float] | None = None
    # Source metadata (varies by data source)
    source_dataset: str | None = None  # HuggingFace dataset ID
    source_split: str | None = None
    source_file: str | None = None  # Local file path
    embedded_columns: str | None = None
    has_projections: bool | None = None
    # Prompt info (for models like Gemma Embedding)
    embedding_prompt: str | None = None  # Single field - can be predefined name or custom string
    # Topic extraction metadata
    has_topics: bool | None = None
    topic_count: int | None = None
    topics_extracted_at: str | None = None
    topics: list[TopicInfo] | None = None  # Topic summary with keywords
    topic_hierarchy: JSON | None = None  # {reduced_label: [subtopic_labels]}
    # Pre-computed field analysis (cached for fast frontend loading)
    field_analysis: JSON | None = None


@strawberry.type
class Collection:
    """Information about a collection."""

    name: str
    metadata: JSON | None = None
    count: int


@strawberry.type
class UpdateCollectionMetadataResult:
    """Result of updating collection metadata."""

    name: str
    metadata: JSON
    error: str | None = None


@strawberry.type
class EmbeddingItem:
    """Single embedding item with all associated data."""

    id: str
    word: str | None = None
    definition: str | None = None
    pos: str | None = None
    embedding: list[float] | None = None
    document: str | None = None
    metadata: JSON | None = None


@strawberry.type
class SemanticSearchResult:
    """Result from semantic search."""

    id: str
    document: str | None = None
    metadata: JSON | None = None  # All item metadata
    distance: float
    similarity: float
    embedding: list[float] | None = None


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

    ids: list[str]
    documents: list[str]
    item_metadata: list[JSON]  # Raw metadata per item - flexible schema
    available_fields: list[str]  # What metadata fields are available
    # Projections (Optional — only requested types are returned, others are null)
    pca_2d: list[list[float]] | None = None
    pca_3d: list[list[float]] | None = None
    umap_2d: list[list[float]] | None = None
    umap_3d: list[list[float]] | None = None
    # Collection-level metadata
    metadata: CollectionMetadata


# ========== Helper Functions ==========


def build_where_clause(filters: list[FilterInput] | None) -> dict[str, Any] | None:
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
    columns: list[str] | None = None
    embedding_model: str | None = None
    batch_size: int = 100
    started_at: str = ""

    # Full config for resume verification
    config: JSON | None = None


# JobProgress is defined in subscriptions.py to avoid circular imports
# when embedding functions import emit_progress_sync


# ========== SAE (Sparse Autoencoder) Types ==========


@strawberry.type
class SaeLogitEntry:
    """A single token/score pair from top or bottom logits."""

    token: str
    score: float


@strawberry.type
class SaeFeature:
    """SAE feature with metadata, label, and logits."""

    model_id: str
    sae_id: str
    feature_index: int
    density: float | None = None
    label: str | None = None
    top_logits: list[SaeLogitEntry] | None = None
    bottom_logits: list[SaeLogitEntry] | None = None


@strawberry.type
class SaeActivation:
    """A single top-activating example for a feature."""

    id: str
    tokens: list[str]
    values: list[float]
    max_value: float
    max_value_token_index: int


@strawberry.type
class SaeModelInfo:
    """Available SAE model/layer combination with counts."""

    model_id: str
    sae_id: str
    feature_count: int
    activation_count: int


@strawberry.type
class SaeFeatureSearchResult:
    """Feature search result with optional activation count."""

    feature: SaeFeature
    activation_count: int | None = None


@strawberry.input
class IngestSaeFeaturesInput:
    """Input for ingesting SAE feature parquet."""

    parquet_path: str
    model_id: str
    sae_id: str
    store_vectors: bool = True


@strawberry.input
class IngestSaeActivationsInput:
    """Input for ingesting SAE activation JSONL."""

    jsonl_path: str
    model_id: str | None = None
    sae_id: str | None = None


@strawberry.type
class IngestSaeResult:
    """Result of SAE data ingestion."""

    model_id: str
    sae_id: str
    records_inserted: int
    duration_seconds: float
    error: str | None = None


@strawberry.type
class SaeActivationQuantileGroup:
    """A quantile bin of activations for a feature."""

    quantile: int
    bin_min: float
    bin_max: float
    activations: list[SaeActivation]
