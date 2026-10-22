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
    - SentenceTransformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2", "google/embeddinggemma-2"
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
    # Query time only: QWEN's query instruction, or the query prefix of a
    # task-prefix model (EmbeddingGemma, Gemini Embedding 2; utils/embedding_tasks.py).
    task: str | None = None
    task_type: str | None = (
        None  # Gemini 001: Embedding optimization (SEMANTIC_SIMILARITY, ...); dropped for Gemini 2
    )
    # SentenceTransformers: a predefined prompt name (e.g. "Retrieval-query") or
    # custom string; for task-prefix models the document prefix template
    # ("title: {title} | text: "), rendered per row into the model input only.
    prompt: str | None = None
    # Local providers: input-length cap in tokens. Omitted = the model's own limit
    # (see the embeddingModelMaxLength query); above it the job is rejected.
    max_length: int | None = None


@strawberry.type
class EmbeddingModelMaxLength:
    """A local model's input-length limit in tokens, and where it was found.

    ``source``: "catalog" (config/embedding_models.json),
    "sentence_transformers_config", "tokenizer" or "model_config"; both fields
    are null when the limit is unknown or the provider is a remote API.
    """

    max_length: int | None = None
    source: str | None = None


@strawberry.type
class EmbeddingModelCatalogEntry:
    """A model suggestion for the embed forms (``embeddingModelCatalog`` query).

    ``suggested``: listed in ``config/embedding_models.json``. ``collection_count``:
    how many existing collections use it (models only used, such as local
    checkpoint paths, appear with ``suggested`` false). ``default_for``: the
    data types whose default model this is (``"multimodal"``).
    """

    provider: EmbeddingProviderEnum
    model_name: str
    description: str | None = None
    modalities: list[str] = strawberry.field(default_factory=lambda: ["text"])
    prompts: list[str] = strawberry.field(default_factory=list)
    dimension: int | None = None
    default_for: list[str] = strawberry.field(default_factory=list)
    suggested: bool = True
    collection_count: int = 0


@strawberry.input
class MediaOptionsInput:
    """Per-job media options of a multimodal embed; omitted values keep the
    defaults of ``config/media_embedding.json`` (``mediaEmbeddingSettings``)."""

    image_max_soft_tokens: int | None = None
    video_fps: float | None = None
    video_max_frames: int | None = None
    video_max_soft_tokens: int | None = None
    max_audio_seconds: float | None = None


@strawberry.type
class MediaOptionsInfo:
    """Effective media options (a job's, or the defaults)."""

    image_max_soft_tokens: int
    video_fps: float
    video_max_frames: int
    video_max_soft_tokens: int
    max_audio_seconds: float


@strawberry.type
class MediaEmbeddingSettingsInfo:
    """Defaults and limits of the media options, and the token-budget constants
    the embed form uses to estimate a row's size (``mediaEmbeddingSettings``)."""

    defaults: MediaOptionsInfo
    soft_token_budgets: list[int]
    video_fps_limit: float
    video_frames_limit: int
    audio_seconds_limit: float
    context_tokens: int
    reserved_tokens: int
    audio_tokens_per_second: int
    default_model: str


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
    """Configuration for topic reduction (strategies: services/topic_reduction_types.py)."""

    enabled: bool = False
    method: str = "auto"  # "auto" or "fixed_n"
    n_topics: int | None = None  # Topics to keep (noise not counted); required for "fixed_n"
    # "average" | "ward" | "density" | "legacy_ctfidf" | "legacy_semantic";
    # null = the legacy strategy use_ctfidf picks.
    strategy: str | None = None
    use_ctfidf: bool = True  # Legacy: True=c-TF-IDF, False=raw semantic centroids
    distance_threshold: float | None = None  # "auto" + "average": centred cosine distance
    absorb_noise: bool = False  # "density": move noise into the topics that absorb it


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

    # Clustering space: "projection" (viz coords) | "cluster_umap" (BERTopic 5D UMAP) | "embedding"
    cluster_on: str = "cluster_umap"
    # BERTopic-style clustering UMAP params (used only when cluster_on == "cluster_umap")
    cluster_n_components: int = 5
    cluster_min_dist: float = 0.0
    cluster_n_neighbors: int = 15

    # Topic reduction config
    reduction: TopicReductionInput | None = None


@strawberry.input
class SaeLinkInput:
    """SAE to link a collection to once its embedding run succeeds (the SAE
    whose document activations ``computeDocumentActivations`` will compute)."""

    sae_model_id: str
    sae_id: str
    # The client follows a successful embed with computeDocumentActivations
    # (the post-embed SAE step); false for SAE feature collections, which are
    # only linked. Stored with the job's input, so a resumed run does the same.
    compute_activations: bool = False


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

    # Stored quality metrics, keyed by level: {"topic": {...}, "subtopic": {...}}
    quality_metrics: JSON | None = None


@strawberry.input
class EvaluateTopicsInput:
    """Input for scoring the quality of a collection's active topic extraction."""

    collection_name: str
    level: str = "topic"  # "topic" or "subtopic" (pre-reduction clusters)
    # Subset of {"dbcv","silhouette","diversity","coherence_cv","coherence_umass"};
    # None = all. DBCV is null for stored labels (needs the live fitted model).
    metrics: list[str] | None = None
    sample_size: int = 10000  # silhouette subsample cap


@strawberry.type
class EvaluateTopicsResult:
    """Result of topic-quality scoring (also persisted on the extraction row)."""

    collection_name: str
    level: str
    metrics: JSON | None = None
    duration_seconds: float = 0.0
    error: str | None = None


@strawberry.input
class ReduceTopicsInput:
    """Input for standalone topic reduction mutation.

    Always reduces the extraction's base topics (its original clusters), so
    reductions are reversible and never compound; see TopicReductionInput.
    """

    collection_name: str
    method: str = "auto"  # "auto" or "fixed_n"
    n_topics: int | None = None  # Topics to keep (noise not counted); required for "fixed_n"
    # "average" | "ward" | "density" | "legacy_ctfidf" | "legacy_semantic";
    # null = the legacy strategy use_ctfidf picks.
    strategy: str | None = None
    use_ctfidf: bool = True  # Legacy: True=c-TF-IDF, False=raw semantic centroids
    distance_threshold: float | None = None  # "auto" + "average": centred cosine distance
    absorb_noise: bool = False  # "density": move noise into the topics that absorb it

    # Re-labeling after reduction (merged topics only; unmerged ones keep their label)
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
    absorbed_noise: int = 0  # noise documents moved into topics (density)


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


@strawberry.enum
class DataTypeEnum(Enum):
    """Type of data to embed."""

    TEXT = "text"
    # Images, audio, video and rows mixing them with text (EmbeddingGemma 2)
    MULTIMODAL = "multimodal"
    VECTOR = "vector"


@strawberry.input
class EmbedDatasetInput:
    """Input for embedding a HuggingFace dataset."""

    dataset_id: str
    collection_name: str
    config: str | None = None
    split: str = "train"
    # Embed several splits into one collection in a single pass (each item gets
    # a `source_split` metadata value). Takes precedence over `split` when set.
    splits: list[str] | None = None
    # TEXT or MULTIMODAL (VECTOR is local files only)
    data_type: DataTypeEnum = DataTypeEnum.TEXT
    columns: list[str] | None = None  # Columns to embed (MULTIMODAL: optional text)
    # MULTIMODAL: the Image / Audio / Video columns, embedded with `columns`
    media_columns: list[str] | None = None
    media_options: MediaOptionsInput | None = None
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
    # Link the collection to this SAE after a successful run
    sae_link: SaeLinkInput | None = None


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
    # Non-cancelling issues, e.g. documents truncated at the length cap
    warnings: list[str] = strawberry.field(default_factory=list)


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
    # Media columns of the preview: {column: "image" | "audio" | "video"}
    column_kinds: JSON = strawberry.field(default_factory=dict)


@strawberry.input
class EmbedLocalFileInput:
    """Input for embedding a local file."""

    file_path: str
    collection_name: str
    data_type: DataTypeEnum = DataTypeEnum.TEXT
    columns: list[str] | None = None  # Columns to embed (MULTIMODAL: optional text)
    text_template: str | None = None
    # MULTIMODAL: the image / audio / video columns, embedded with `columns`
    media_columns: list[str] | None = None
    media_options: MediaOptionsInput | None = None
    vector_column: str | None = None  # Column containing pre-computed vectors
    id_column: str | None = None
    metadata_columns: list[str] | None = None
    n_rows: int | None = None  # Limit rows
    sample_n: int | None = None  # Random sample
    sample_seed: int = 42
    compute_projections: bool = True
    batch_size: int | None = 100
    # Embedding model configuration (default: SentenceTransformers with all-MiniLM-L6-v2;
    # MULTIMODAL: the catalog's multimodal default, EmbeddingGemma 2). VECTOR uses none.
    embedding_model: EmbeddingModelInput | None = None
    # Resume an interrupted job instead of starting fresh
    resume: bool = False
    # Topic extraction after embedding
    extract_topics: bool = False
    topic_config: TopicConfigInput | None = None
    # Link the collection to this SAE after a successful run
    sae_link: SaeLinkInput | None = None


@strawberry.input
class ReEmbedDatasetInput:
    """Input for re-embedding an existing dataset with a different model.

    Reads documents from an existing DuckDB dataset and embeds them with a new
    embedding model into a new ChromaDB vector collection.
    """

    source_dataset_name: str  # Existing dataset to read from
    collection_name: str  # New collection name for the re-embedded vectors
    embedding_model: EmbeddingModelInput  # Required: the new model to use
    columns: list[str] | None = (
        None  # Metadata fields to compose text from (None = use existing document)
    )
    text_template: str | None = None  # Template e.g. "{title}: {text}" (None = concatenate columns)
    batch_size: int | None = 100
    resume: bool = False
    compute_projections: bool = True
    extract_topics: bool = False
    topic_config: TopicConfigInput | None = None
    # Link the collection to this SAE after a successful run
    sae_link: SaeLinkInput | None = None


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
    """Filter operators, resolved in DuckDB (metadata JSON or topic assignments)."""

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
    # Reduced extractions: {topic_id: [base subtopic ids]} (older ones:
    # {reduced_label: [subtopic_labels]}); set means the topics have subtopics.
    topic_hierarchy: JSON | None = None
    # Pre-computed field analysis (cached for fast frontend loading)
    field_analysis: JSON | None = None
    # SAE linkage (set via updateCollectionMetadata to connect scatter plot to feature explorer)
    sae_model_id: str | None = None
    sae_id: str | None = None


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
class TopicSimilarityResult:
    """A topic ranked by similarity between a query and its embedding centroid.

    When one of the topic's subtopics matched better than the topic as a
    whole, ``matched_subtopic_*`` names it.
    """

    topic_id: int
    similarity: float
    matched_subtopic_id: int | None = None
    matched_subtopic_label: str | None = None


@strawberry.type
class TopicTreeLeaf:
    """A base topic of a merge tree (an original cluster of the extraction)."""

    topic_id: int
    label: str
    size: int  # documents


@strawberry.type
class TopicTreeMerge:
    """One merge of a topic tree, in scipy's linkage convention: ``left`` and
    ``right`` below the leaf count are leaves, ``leafCount + i`` is merge ``i``."""

    left: int
    right: int
    height: float
    size: int  # documents under the merged node


@strawberry.type(name="TopicMergeTree")
class TopicMergeTreeInfo:
    """Merge tree over a collection's base topics, for the live topic slider.

    Cutting at N topics applies the first ``leafCount - N`` merges. Base topic
    ids are in the items' ``topic_id`` field, or ``subtopic_id`` when
    ``baseLevel`` is "subtopic" (a reduced extraction). ``noiseItemIds`` /
    ``noiseMerges`` (density trees, on request): the merge whose cut absorbs
    each noise item.
    """

    extraction_id: str
    linkage: str  # "average" | "ward" | "density"
    base_level: str  # "topic" | "subtopic"
    leaves: list[TopicTreeLeaf]
    merges: list[TopicTreeMerge]
    noise_count: int
    available_linkages: list[str]
    noise_item_ids: list[str] | None = None
    noise_merges: list[int] | None = None


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
    # Ordered-item digests used to validate projection-only follow-up requests.
    item_signature: str | None = None
    projection_signatures: JSON | None = None


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


@strawberry.type
class DocumentActivationResult:
    """A document ranked by its SAE feature activation score."""

    item_id: str
    document: str | None = None
    metadata: JSON | None = None
    score: float
    matching_features: int
    row_index: int | None = None


@strawberry.type
class DocumentActivationSearchResponse:
    """Response from the two-hop feature-label → document search."""

    results: list[DocumentActivationResult]
    total_results: int
    matched_feature_count: int
    matched_features: list[SaeFeature] | None = None
    error: str | None = None


@strawberry.type
class PromptDocumentSearchResult:
    """A document found by sparse dot-product similarity to a prompt's SAE activations."""

    item_id: str
    document: str | None = None
    metadata: JSON | None = None
    score: float
    shared_features: int
    row_index: int | None = None


@strawberry.type
class PromptDocumentSearchResponse:
    """Response from prompt → SAE → document similarity search."""

    results: list[PromptDocumentSearchResult]
    prompt_feature_count: int
    error: str | None = None


@strawberry.input
class SearchDocumentsByPromptInput:
    """Input for prompt-based document similarity search via SAE activations."""

    collection_name: str
    prompt: str
    limit: int = 50
    top_k_features: int | None = None


@strawberry.input
class ComputeDocumentActivationsInput:
    """Input for batch SAE inference on all documents in a collection."""

    collection_name: str


@strawberry.type
class ComputeDocumentActivationsResult:
    """Result of batch SAE document activation computation."""

    collection_name: str
    items_processed: int
    total_items: int
    duration_seconds: float
    error: str | None = None


@strawberry.input
class TrainProbeInput:
    """Input for training an embedding-space probe on a metadata field."""

    collection_name: str
    target_field: str
    # ridge | lasso | massmean | massmean_cov | svr | mlp (regression)
    # logreg | svc | mlp_clf (classifiers: binary or multiclass targets)
    kind: str = "ridge"
    alpha: float | None = None  # ridge / lasso
    c: float | None = None  # svr / logreg / svc
    kernel: str | None = None  # svr / svc
    class_weight: str | None = None  # classifiers: null | "balanced"
    epochs: int | None = None  # mlp
    hidden_dims: list[int] | None = None  # mlp
    patience: int | None = None  # mlp early-stopping patience (dev-set based)
    dev_split: float | None = None  # mlp: train fraction held out for early stopping
    activation: str | None = None  # mlp: relu | gelu | tanh | silu
    seed: int | None = None
    train_split: float | None = None
    max_train_samples: int | None = None
    # Classifiers, 3+ class targets: classes with fewer than
    # max(5, ceil(fraction * labelled items)) items are excluded from training.
    min_class_fraction: float | None = None


@strawberry.type
class ProbeInfo:
    """A trained probe: target field, kind, metrics, derived field names."""

    target_field: str
    kind: str
    score_field: str
    residual_field: str | None
    metrics: JSON | None
    n_train: int
    n_val: int
    created_at: str
    # Binary categorical targets only: the applied value->0/1 mapping
    # (e.g. {"safe": 0, "unsafe": 1}); null for numeric targets.
    target_mapping: JSON | None = None
    # "regression" | "binary" | "multiclass" (derived for pre-multiclass rows).
    task: str = "regression"
    # Classifiers: kept classes in class-index order (predicted / classIndex
    # refer to positions here); null for regression and legacy binary probes.
    classes: list[str] | None = None
    # Classifiers: classes excluded by the min-class threshold -> item count.
    excluded_classes: JSON | None = None
    min_class_fraction: float | None = None


@strawberry.type
class TrainProbeResult:
    """Result of a probe training run."""

    collection_name: str
    probe: ProbeInfo | None
    duration_seconds: float
    error: str | None = None


@strawberry.type
class CollectionProbesResult:
    """All trained probes for a collection."""

    collection_name: str
    probes: list[ProbeInfo]


@strawberry.type
class ProbeScores:
    """Per-item probe scores, parallel arrays aligned by index."""

    item_ids: list[str]
    scores: list[float]
    residuals: list[float | None] | None
    # Multiclass probes: predicted class index (into ProbeInfo.classes).
    predicted: list[int | None] | None = None


@strawberry.type
class ProbeClassProbabilities:
    """One class's per-item probabilities for a multiclass probe (lazy fetch)."""

    item_ids: list[str]
    probabilities: list[float | None]


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


@strawberry.enum
class SaeCollectionMode(Enum):
    """Vector mode for SAE visualization collection creation."""

    DECODER_VECTORS = "decoder_vectors"
    LABEL_EMBEDDINGS = "label_embeddings"


@strawberry.enum
class DocumentRankingMode(Enum):
    """Ranking mode for document search over selected SAE features."""

    MAX = "max"
    SUM = "sum"
    SCALED_SUM = "scaled_sum"
    MATCHING_FEATURES = "matching_features"


@strawberry.input
class PrepareSaeInput:
    """Input for the SAE download + extraction pipeline."""

    layer: int
    width: str = "16k"
    hook_type: str = "resid_post"
    model_size: str = "4b"
    variant: str = "it"
    # SAE family: "gemma" (Neuronpedia download) or "qwen" (qwen-scope,
    # extract-only — no labels/activations; variant is ignored).
    family: str = "gemma"
    skip_download: bool = False
    include_activations: bool = False
    # Collection creation options
    create_collection: bool = False
    collection_mode: SaeCollectionMode | None = None
    embedding_model: EmbeddingModelInput | None = None
    extract_topics: bool = False
    topic_config: TopicConfigInput | None = None
    delete_source_files: bool = False


@strawberry.type
class PrepareSaeResult:
    """Result of the SAE pipeline — file paths + DuckDB ingestion counts."""

    model_id: str
    sae_id: str
    features_parquet: str | None = None
    activations_jsonl: str | None = None
    features_inserted: int = 0
    activations_inserted: int = 0
    duration_seconds: float = 0.0
    status: str = ""  # "completed", "already_downloaded", "failed"
    error: str | None = None
    # Collection creation results
    collection_name: str | None = None
    collection_items: int = 0


# ========== Interpret / SAE Inference Types ==========


@strawberry.enum
class HookTypeEnum(Enum):
    """Hook site within a decoder layer for SAE attachment."""

    RESID_POST = "resid_post"
    MLP_OUT = "mlp_out"
    ATTN_OUT = "attn_out"


@strawberry.enum
class ActivationFilterMode(Enum):
    """Filter mode for prompt activation results.

    NONE: All nonzero features, no filtering.
    NEURONPEDIA: Top-50 per token, no server-side density filter (Neuronpedia-style).
    COVERAGE_BOS: Remove features firing on >=80% of all positions (incl. BOS).
    COVERAGE_NO_BOS: Same but coverage on prompt tokens only (stricter).
    """

    NONE = "none"
    NEURONPEDIA = "neuronpedia"
    COVERAGE_BOS = "coverage_bos"
    COVERAGE_NO_BOS = "coverage_no_bos"


# --- Inputs ---


@strawberry.input
class SteeringInput:
    """A single steering specification.

    Two flavours, distinguished by ``direction_name``:
      - SAE feature (default): ``feature_index`` + ``layer`` + ``hook_type``
        + ``width`` resolve a direction from ``sae.w_dec``.
      - Pre-extracted direction: ``direction_name`` (e.g. "refusal", "poetry")
        looks up a 1-D vector from the backend's DIRECTION_REGISTRY. The
        other fields are ignored in that case (pass zeros / defaults).

    Multiple SteeringInput entries can be combined to steer on several
    features / directions simultaneously.
    """

    feature_index: int
    layer: int
    hook_type: HookTypeEnum = HookTypeEnum.RESID_POST
    width: str = "16k"
    strength: float = 800.0
    direction_name: str | None = None


@strawberry.input
class SaeLayerSpecInput:
    """One SAE to hook during prompt activations: layer, width and hook site."""

    layer: int
    width: str = "16k"
    hook_type: HookTypeEnum = HookTypeEnum.RESID_POST


@strawberry.input
class RunPromptActivationsInput:
    """Input for running a prompt through the model with SAE hooks.

    SAEs are given either as ``layers`` (residual-stream SAEs at the shared
    ``width``) or as explicit ``saes`` specs, which take precedence and support
    mixed widths and hook sites — including two widths or two sites at the
    same layer — in a single forward pass.
    """

    prompt: str
    layers: list[int] | None = None
    width: str = "16k"
    saes: list[SaeLayerSpecInput] | None = None
    top_k: int = 10
    model_id: str | None = None
    sae_id: str | None = None
    skip_chat_template: bool = False
    filter_mode: ActivationFilterMode = ActivationFilterMode.NEURONPEDIA


@strawberry.input
class GenerateSteeredInput:
    """Input for generating baseline vs steered text with one or more features."""

    prompt: str
    steering: list[SteeringInput]
    output_len: int = 128
    temperature: float | None = None


@strawberry.input
class RunPromptHighlightInput:
    """Input for running a prompt and returning max-pooled feature activations."""

    prompt: str
    layer: int
    width: str = "16k"
    hook_type: HookTypeEnum = HookTypeEnum.RESID_POST


# --- Outputs ---


@strawberry.type
class ModelStatus:
    """Status of the interpretability model."""

    loaded: bool
    model_name: str | None = None
    device: str | None = None
    variant: str | None = None  # "it" (instruction-tuned) or "pt" (pretrained/base)
    model_size: str | None = None  # "4b", "12b", etc.


@strawberry.type
class InterpretActiveFeature:
    """A single SAE feature active at a token position (inference result)."""

    index: int
    activation: float
    label: str
    density: float | None = None


@strawberry.type
class InterpretTokenFeatures:
    """Features active at one token position within a layer."""

    token: str
    position: int
    features: list[InterpretActiveFeature]


@strawberry.type
class InterpretLayerResult:
    """Per-token features for one hooked SAE (layer, width, hook site)."""

    layer: int
    width: str
    hook_type: HookTypeEnum
    tokens: list[InterpretTokenFeatures]


@strawberry.type
class PromptActivationsResponse:
    """Result of running a prompt through the model with SAE hooks."""

    prompt: str
    token_strings: list[str]
    layers: list[InterpretLayerResult]
    error: str | None = None


@strawberry.type
class AppliedSteering:
    """A steering feature that was applied during generation (output type)."""

    feature_index: int
    layer: int
    hook_type: str
    width: str
    strength: float
    direction_name: str | None = None


@strawberry.type
class SteeredGenerationResponse:
    """Result of baseline vs steered generation."""

    baseline_text: str
    steered_text: str
    steering: list[AppliedSteering]
    error: str | None = None


@strawberry.type
class PromptHighlightFeature:
    """A single feature activation from max-pooled prompt inference."""

    feature_index: int
    activation: float


@strawberry.type
class PromptHighlightResponse:
    """Max-pooled feature activations for scatter plot highlighting."""

    features: list[PromptHighlightFeature]
    error: str | None = None


# ---------------------------------------------------------------------------
# Streaming chat generation
# ---------------------------------------------------------------------------


@strawberry.input
class ChatTurnInput:
    """A single turn in a multi-turn conversation."""

    role: str  # "user" or "model"
    content: str


@strawberry.input
class GenerateStreamInput:
    """Input for streaming text generation."""

    turns: list[ChatTurnInput]
    output_len: int = 256
    temperature: float | None = None
    top_p: float = 0.95
    top_k: int = 64
    steering: list[SteeringInput] | None = None
    seed: int | None = None  # When set, torch.manual_seed() before sampling for reproducible output
    enable_thinking: bool = False  # Qwen only: stream raw <think> reasoning; ignored for Gemma


@strawberry.type
class TokenChunk:
    """A single token emitted during streaming generation."""

    stream_id: str
    token_index: int
    token_id: int
    text: str
    done: bool
    error: str | None = None


# ========== Chat History Types ==========


@strawberry.type
class ChatSessionInfo:
    """Summary of a chat session for listing."""

    id: str
    title: str
    config: JSON
    created_at: str
    updated_at: str


@strawberry.type
class ChatSessionMessage:
    """A single message within a chat session."""

    id: str
    session_id: str
    role: str
    content: str
    parts: JSON | None = None
    steering_snapshot: JSON | None = None
    created_at: str


@strawberry.type
class ChatSessionDetail:
    """A chat session with all its messages."""

    id: str
    title: str
    config: JSON
    created_at: str
    updated_at: str
    messages: list[ChatSessionMessage]


@strawberry.input
class CreateChatSessionInput:
    """Input for creating a new chat session."""

    id: str
    title: str
    config: JSON


@strawberry.input
class SaveChatMessageInput:
    """Input for saving a chat message."""

    id: str
    session_id: str
    role: str
    content: str
    parts: JSON | None = None
    steering_snapshot: JSON | None = None


@strawberry.type
class EmbeddingTaskPrefix:
    """One task of the task-prefix table (``task_prefixes`` in config/embedding_models.json).

    Symmetric tasks embed queries and documents with the same prefix.
    """

    id: str
    label: str
    query_prefix: str
    document_prefix: str
    symmetric: bool


@strawberry.type
class EmbeddingTaskPrefixModel:
    """A model that reads its task from a task prefix (EmbeddingGemma, Gemini Embedding 2)."""

    provider: EmbeddingProviderEnum
    model_name: str


@strawberry.type
class EmbeddingTaskPrefixTable:
    """The task prefixes of the embed forms, and the models that take them."""

    default_task: str
    tasks: list[EmbeddingTaskPrefix]
    models: list[EmbeddingTaskPrefixModel]
