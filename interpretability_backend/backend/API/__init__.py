"""GraphQL API module for embedding visualization backend."""

import strawberry

from .mutations import Mutation
from .queries import Query
from .subscriptions import JobProgress, Subscription
from .types import (
    # Scalars
    JSON,
    Collection,
    # Collection types
    CollectionMetadata,
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
    HFConfigInfo,
    HFDatasetInfo,
    HFDatasetPreview,
    HFFeatureInfo,
    # HuggingFace types
    HFSplitInfo,
    IngestSaeActivationsInput,
    IngestSaeFeaturesInput,
    IngestSaeResult,
    # Job types
    JobStatusEnum,
    # Local file types
    LocalFileInfo,
    LocalFilePreview,
    PortionInput,
    PortionStrategyEnum,
    ProjectionData,
    SaeActivation,
    SaeActivationQuantileGroup,
    SaeFeature,
    SaeFeatureSearchResult,
    # SAE types
    SaeLogitEntry,
    SaeModelInfo,
    SemanticSearchResult,
    # Search & filter types
    SimilarityMeasure,
    TextSearchMatch,
    # Text search types
    TextSearchMode,
    TextSearchResponse,
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
]
