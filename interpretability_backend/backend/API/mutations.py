"""GraphQL mutation resolvers for embedding visualization backend."""

import strawberry
from typing import Optional

from .types import (
    DataTypeEnum,
    EmbedDatasetInput,
    EmbedDatasetResult,
    EmbedLocalFileInput,
    EmbeddingProviderEnum,
    JSON,
    PortionStrategyEnum,
    UpdateCollectionMetadataResult,
)

# Import embedding functions and types at module level
from ..embed_dataset import (
    embed_huggingface_dataset as do_hf_embed,
    embed_local_file as do_local_embed,
    compute_projections_for_collection,
    EmbeddingConfig,
    LocalFileEmbeddingConfig,
    DataType,
    EmbeddingModelConfig,
    EmbeddingProvider,
)
from ..clients.huggingface_client import PortionConfig, PortionStrategy
from .chromadb_instance import get_chromadb_client


# Mapping from GraphQL enums to internal enums
PORTION_STRATEGY_MAP = {
    PortionStrategyEnum.FIRST_N: PortionStrategy.FIRST_N,
    PortionStrategyEnum.RANDOM_SAMPLE: PortionStrategy.RANDOM_SAMPLE,
    PortionStrategyEnum.ROW_RANGE: PortionStrategy.ROW_RANGE,
    PortionStrategyEnum.ALL: PortionStrategy.ALL,
}

# Auto-generate provider mapping - no manual maintenance needed
# When a new provider is added to provider_list.py, it automatically appears here
EMBEDDING_PROVIDER_MAP = {
    getattr(EmbeddingProviderEnum, member.name): member
    for member in EmbeddingProvider
}

DATA_TYPE_MAP = {
    DataTypeEnum.TEXT: DataType.TEXT,
    DataTypeEnum.IMAGE: DataType.IMAGE,
    DataTypeEnum.VECTOR: DataType.VECTOR,
}


@strawberry.type
class Mutation:
    """GraphQL mutation root."""

    @strawberry.mutation
    def embed_huggingface_dataset(self, input: EmbedDatasetInput, info=None) -> EmbedDatasetResult:
        """Embed a HuggingFace dataset into a ChromaDB collection.

        Args:
            input: Configuration for embedding the dataset

        Returns:
            Result with statistics about the embedding operation
        """
        # Convert GraphQL input to EmbeddingConfig
        portion = None
        if input.portion:
            portion = PortionConfig(
                strategy=PORTION_STRATEGY_MAP[input.portion.strategy],
                n=input.portion.n,
                start=input.portion.start,
                end=input.portion.end,
                seed=input.portion.seed
            )

        # Convert embedding model input
        embedding_model = None
        if input.embedding_model:
            embedding_model = EmbeddingModelConfig(
                provider=EMBEDDING_PROVIDER_MAP[input.embedding_model.provider],
                model_name=input.embedding_model.model_name,
                ollama_url=input.embedding_model.ollama_url,
                task=input.embedding_model.task,  # QWEN: query instruction
                task_type=input.embedding_model.task_type  # Gemini: optimization type
            )

        config = EmbeddingConfig(
            dataset_id=input.dataset_id,
            collection_name=input.collection_name,
            config=input.config,
            split=input.split,
            columns=input.columns,
            text_template=input.text_template,
            id_column=input.id_column,
            portion=portion,
            metadata_columns=input.metadata_columns,
            embedding_model=embedding_model,
            batch_size=input.batch_size or 100
        )

        # Run embedding
        result = do_hf_embed(config)

        # Compute projections if requested and embedding succeeded
        projections_computed = False
        if input.compute_projections and result.error is None and result.total_embedded > 0:
            projections_computed = compute_projections_for_collection(input.collection_name)

        return EmbedDatasetResult(
            collection_name=result.collection_name,
            total_embedded=result.total_embedded,
            embedding_dim=result.embedding_dim,
            device=result.device,
            duration_seconds=result.duration_seconds,
            projections_computed=projections_computed,
            error=result.error,
            embedding_provider=result.embedding_provider,
            embedding_model=result.embedding_model
        )

    @strawberry.mutation
    def embed_local_file(self, input: EmbedLocalFileInput, info=None) -> EmbedDatasetResult:
        """Embed a local file (parquet/json/csv) into a ChromaDB collection.

        Supports text, image, and pre-computed vector embeddings.

        Args:
            input: Configuration for embedding the local file

        Returns:
            Result with statistics about the embedding operation
        """
        # Convert embedding model input
        embedding_model = None
        if input.embedding_model:
            embedding_model = EmbeddingModelConfig(
                provider=EMBEDDING_PROVIDER_MAP[input.embedding_model.provider],
                model_name=input.embedding_model.model_name,
                ollama_url=input.embedding_model.ollama_url,
                task=input.embedding_model.task,  # QWEN: query instruction
                task_type=input.embedding_model.task_type  # Gemini: optimization type
            )

        config = LocalFileEmbeddingConfig(
            file_path=input.file_path,
            collection_name=input.collection_name,
            data_type=DATA_TYPE_MAP[input.data_type],
            columns=input.columns,
            text_template=input.text_template,
            image_column=input.image_column,
            vector_column=input.vector_column,
            id_column=input.id_column,
            metadata_columns=input.metadata_columns,
            n_rows=input.n_rows,
            sample_n=input.sample_n,
            sample_seed=input.sample_seed,
            embedding_model=embedding_model,
            batch_size=input.batch_size or 100
        )

        # Run embedding
        result = do_local_embed(config)

        # Compute projections if requested and embedding succeeded
        projections_computed = False
        if input.compute_projections and result.error is None and result.total_embedded > 0:
            projections_computed = compute_projections_for_collection(input.collection_name)

        return EmbedDatasetResult(
            collection_name=result.collection_name,
            total_embedded=result.total_embedded,
            embedding_dim=result.embedding_dim,
            device=result.device,
            duration_seconds=result.duration_seconds,
            projections_computed=projections_computed,
            error=result.error,
            embedding_provider=result.embedding_provider,
            embedding_model=result.embedding_model
        )

    @strawberry.mutation
    def delete_collection(self, collection_name: str, info=None) -> bool:
        """Delete a collection from ChromaDB.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            client = get_chromadb_client()
            client.client.delete_collection(name=collection_name)
            return True
        except Exception:
            return False

    @strawberry.mutation
    def update_collection_metadata(
        self,
        collection_name: str,
        metadata: JSON,
        info=None
    ) -> UpdateCollectionMetadataResult:
        """Update metadata for a collection.

        For each key in the metadata argument:
        - If the key exists in current metadata, it will be overwritten
        - If the key doesn't exist, it will be added

        Args:
            collection_name: Name of the collection to update
            metadata: Dictionary of metadata key/value pairs to set or update

        Returns:
            Result with updated metadata
        """
        try:
            client = get_chromadb_client()
            result = client.update_collection_metadata(collection_name, metadata)
            return UpdateCollectionMetadataResult(
                name=result["name"],
                metadata=result["metadata"]
            )
        except Exception as e:
            return UpdateCollectionMetadataResult(
                name=collection_name,
                metadata={},
                error=str(e)
            )
