"""GraphQL API module for embedding visualization backend."""

import strawberry

from .mutations import Mutation
from .queries import Query
from .subscriptions import JobProgress, Subscription
from .types import (
    # Scalars
    JSON,
    # Interpret / SAE inference types
    AppliedSteering,
    # Streaming chat generation
    ChatTurnInput,
    Collection,
    # Collection types
    CollectionMetadata,
    # Probe types
    CollectionProbesResult,
    DataTypeEnum,
    EmbedDatasetInput,
    EmbedDatasetResult,
    EmbeddingItem,
    EmbeddingJob,
    EmbeddingModelInput,
    # Embedding model types
    EmbeddingProviderEnum,
    EmbedLocalFileInput,
    FilterInput,
    FilterOperator,
    GenerateSteeredInput,
    GenerateStreamInput,
    HFConfigInfo,
    HFDatasetInfo,
    HFDatasetPreview,
    HFFeatureInfo,
    # HuggingFace types
    HFSplitInfo,
    HookTypeEnum,
    IngestSaeActivationsInput,
    IngestSaeFeaturesInput,
    IngestSaeResult,
    InterpretActiveFeature,
    InterpretLayerResult,
    InterpretTokenFeatures,
    # Job types
    JobStatusEnum,
    # Local file types
    LocalFileInfo,
    LocalFilePreview,
    ModelStatus,
    PortionInput,
    PortionStrategyEnum,
    ProbeInfo,
    ProbeScores,
    ProjectionData,
    PromptActivationsResponse,
    PromptHighlightFeature,
    PromptHighlightResponse,
    RunPromptActivationsInput,
    RunPromptHighlightInput,
    SaeActivation,
    SaeActivationQuantileGroup,
    SaeFeature,
    SaeFeatureSearchResult,
    SaeLayerSpecInput,
    # SAE types
    SaeLogitEntry,
    SaeModelInfo,
    SemanticSearchResult,
    # Search & filter types
    SimilarityMeasure,
    SteeredGenerationResponse,
    SteeringInput,
    TextSearchMatch,
    # Text search types
    TextSearchMode,
    TextSearchResponse,
    TokenChunk,
    TrainProbeInput,
    TrainProbeResult,
    # Note: JobProgress is imported from subscriptions (not types) to avoid circular imports
    # Helper functions
    build_where_clause,
)

# Create the schema with subscription support
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)

__all__ = [
    # Schema
    "schema",
    # Resolvers
    "Query",
    "Mutation",
    "Subscription",
    # All types
    "JSON",
    "HFSplitInfo",
    "HFFeatureInfo",
    "HFConfigInfo",
    "HFDatasetInfo",
    "HFDatasetPreview",
    "PortionStrategyEnum",
    "PortionInput",
    "EmbeddingProviderEnum",
    "EmbeddingModelInput",
    "EmbedDatasetInput",
    "EmbedDatasetResult",
    "LocalFileInfo",
    "LocalFilePreview",
    "DataTypeEnum",
    "EmbedLocalFileInput",
    "TextSearchMode",
    "TextSearchMatch",
    "TextSearchResponse",
    "SimilarityMeasure",
    "FilterOperator",
    "FilterInput",
    "CollectionMetadata",
    "Collection",
    "EmbeddingItem",
    "SemanticSearchResult",
    "ProjectionData",
    "JobStatusEnum",
    "EmbeddingJob",
    "JobProgress",
    "build_where_clause",
    # SAE types
    "SaeLogitEntry",
    "SaeFeature",
    "SaeActivation",
    "SaeModelInfo",
    "SaeFeatureSearchResult",
    "SaeActivationQuantileGroup",
    "IngestSaeFeaturesInput",
    "IngestSaeActivationsInput",
    "IngestSaeResult",
    # Interpret / SAE inference types
    "HookTypeEnum",
    "ModelStatus",
    "RunPromptActivationsInput",
    "SaeLayerSpecInput",
    "GenerateSteeredInput",
    "RunPromptHighlightInput",
    "InterpretActiveFeature",
    "InterpretTokenFeatures",
    "InterpretLayerResult",
    "PromptActivationsResponse",
    "SteeredGenerationResponse",
    "PromptHighlightFeature",
    "PromptHighlightResponse",
    # Streaming chat generation
    "ChatTurnInput",
    "SteeringInput",
    "GenerateStreamInput",
    "TokenChunk",
    # Probe types
    "TrainProbeInput",
    "TrainProbeResult",
    "ProbeInfo",
    "ProbeScores",
    "CollectionProbesResult",
]
