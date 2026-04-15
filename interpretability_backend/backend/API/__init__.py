"""GraphQL API module for embedding visualization backend."""

import strawberry

from .queries import Query
from .mutations import Mutation
from .subscriptions import Subscription, JobProgress
from .types import (
    # Scalars
    JSON,
    # HuggingFace types
    HFSplitInfo,
    HFFeatureInfo,
    HFConfigInfo,
    HFDatasetInfo,
    HFDatasetPreview,
    PortionStrategyEnum,
    PortionInput,
    # Embedding model types
    EmbeddingProviderEnum,
    EmbeddingModelInput,
    EmbedDatasetInput,
    EmbedDatasetResult,
    # Local file types
    LocalFileInfo,
    LocalFilePreview,
    DataTypeEnum,
    EmbedLocalFileInput,
    # Text search types
    TextSearchMode,
    TextSearchMatch,
    TextSearchResponse,
    # Search & filter types
    SimilarityMeasure,
    FilterOperator,
    FilterInput,
    # Collection types
    CollectionMetadata,
    Collection,
    EmbeddingItem,
    SemanticSearchResult,
    ProjectionData,
    # Job types
    JobStatusEnum,
    EmbeddingJob,
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
]
