"""GraphQL schema for embedding visualization backend.

This module re-exports the schema from the API package for backward compatibility.
The actual implementation is now in:
- API/types.py: All type definitions (enums, inputs, outputs)
- API/queries.py: Query resolvers
- API/mutations.py: Mutation resolvers
"""

from .API import Mutation, Query, schema

# Re-export all types for backward compatibility
from .API.types import (
    JSON,
    Collection,
    CollectionMetadata,
    DataTypeEnum,
    EmbedDatasetInput,
    EmbedDatasetResult,
    EmbeddingItem,
    EmbeddingModelInput,
    EmbeddingProviderEnum,
    EmbedLocalFileInput,
    FilterInput,
    FilterOperator,
    HFConfigInfo,
    HFDatasetInfo,
    HFDatasetPreview,
    HFFeatureInfo,
    HFSplitInfo,
    LocalFileInfo,
    LocalFilePreview,
    PortionInput,
    PortionStrategyEnum,
    ProjectionData,
    SemanticSearchResult,
    SimilarityMeasure,
    build_where_clause,
)

__all__ = [
    "schema",
    "Query",
    "Mutation",
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
    "SimilarityMeasure",
    "FilterOperator",
    "FilterInput",
    "CollectionMetadata",
    "Collection",
    "EmbeddingItem",
    "SemanticSearchResult",
    "ProjectionData",
    "build_where_clause",
]
