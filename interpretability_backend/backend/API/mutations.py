"""GraphQL mutation resolvers for embedding visualization backend."""

import asyncio
import strawberry

from .types import (
    EmbedDatasetInput,
    EmbedDatasetResult,
    EmbedLocalFileInput,
    ExtractTopicsInput,
    ExtractTopicsResult,
    GenerateLlmLabelsInput,
    GenerateLlmLabelsResult,
    JSON,
    ReduceTopicsInput,
    ReduceTopicsResult,
    UpdateCollectionMetadataResult,
)
from .converters import (
    build_hf_embedding_config,
    build_local_file_embedding_config,
    build_topic_extraction_config,
    convert_topic_infos,
)
from .chromadb_instance import get_chromadb_client
from .duckdb_instance import get_duckdb_client
from ..clients.duckdb_client import _sanitize_for_json
from ..services.embedding_pipeline import (
    HuggingFaceEmbeddingPipeline,
    LocalFileEmbeddingPipeline,
)
from ..services.topic_extraction_service import (
    extract_topics as do_extract_topics,
)


@strawberry.type
class Mutation:
    """GraphQL mutation root."""

    @strawberry.mutation
    async def embed_huggingface_dataset(self, input: EmbedDatasetInput, info=None) -> EmbedDatasetResult:
        """Embed a HuggingFace dataset into a ChromaDB collection."""
        config = build_hf_embedding_config(input)
        topic_config = build_topic_extraction_config(
            input.collection_name, input.topic_config
        ) if input.extract_topics else None

        pipeline = HuggingFaceEmbeddingPipeline(
            config=config,
            compute_projections=input.compute_projections,
            extract_topics=input.extract_topics,
            topic_config=topic_config,
        )
        pipeline_result = await pipeline.run()
        result = pipeline_result.embedding_result

        return EmbedDatasetResult(
            collection_name=result.collection_name,
            total_embedded=result.total_embedded,
            embedding_dim=result.embedding_dim,
            device=result.device,
            duration_seconds=result.duration_seconds,
            projections_computed=pipeline_result.projections_computed,
            error=result.error,
            embedding_provider=result.embedding_provider,
            embedding_model=result.embedding_model,
        )

    @strawberry.mutation
    async def embed_local_file(self, input: EmbedLocalFileInput, info=None) -> EmbedDatasetResult:
        """Embed a local file (parquet/json/csv) into a ChromaDB collection."""
        config = build_local_file_embedding_config(input)
        topic_config = build_topic_extraction_config(
            input.collection_name, input.topic_config
        ) if input.extract_topics else None

        pipeline = LocalFileEmbeddingPipeline(
            config=config,
            compute_projections=input.compute_projections,
            extract_topics=input.extract_topics,
            topic_config=topic_config,
        )
        pipeline_result = await pipeline.run()
        result = pipeline_result.embedding_result

        return EmbedDatasetResult(
            collection_name=result.collection_name,
            total_embedded=result.total_embedded,
            embedding_dim=result.embedding_dim,
            device=result.device,
            duration_seconds=result.duration_seconds,
            projections_computed=pipeline_result.projections_computed,
            error=result.error,
            embedding_provider=result.embedding_provider,
            embedding_model=result.embedding_model,
        )

    @strawberry.mutation
    def delete_collection(self, collection_name: str, info=None) -> bool:
        """Delete a collection from both DuckDB and ChromaDB."""
        try:
            # Delete from DuckDB (cascades items, projections, topics)
            db = get_duckdb_client()
            db.delete_dataset(collection_name)

            # Delete from ChromaDB (vector storage)
            client = get_chromadb_client()
            try:
                client.client.delete_collection(name=collection_name)
            except Exception:
                pass  # ChromaDB collection may not exist
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
        """Update metadata for a collection."""
        try:
            # Update DuckDB (primary)
            db = get_duckdb_client()
            db.update_dataset(collection_name, extra_metadata=metadata)
            ds = db.get_dataset(collection_name)

            return UpdateCollectionMetadataResult(
                name=collection_name,
                metadata=_sanitize_for_json(ds) if ds else {},
            )
        except Exception as e:
            return UpdateCollectionMetadataResult(
                name=collection_name,
                metadata={},
                error=str(e)
            )

    @strawberry.mutation
    async def extract_topics(self, input: ExtractTopicsInput, info=None) -> ExtractTopicsResult:
        """Extract topic clusters from an existing collection."""
        config = build_topic_extraction_config(input.collection_name, input.config)
        result = await asyncio.to_thread(do_extract_topics, config)

        return ExtractTopicsResult(
            collection_name=result.collection_name,
            num_topics=result.num_topics,
            num_noise_points=result.num_noise_points,
            topics=convert_topic_infos(result.topics),
            duration_seconds=result.duration_seconds,
            error=result.error,
            num_topics_before_reduction=result.num_topics_before_reduction,
            reduction_applied=result.reduction_applied,
        )

    @strawberry.mutation
    async def reduce_topics(self, input: ReduceTopicsInput, info=None) -> ReduceTopicsResult:
        """Reduce topics on an existing collection (standalone post-processing)."""
        from ..services.topic_extraction_service import reduce_existing_topics

        result = await asyncio.to_thread(
            reduce_existing_topics,
            collection_name=input.collection_name,
            method=input.method,
            n_topics=input.n_topics,
            use_ctfidf=input.use_ctfidf,
            regenerate_labels=input.regenerate_labels,
            llm_provider=input.llm_provider,
            llm_model=input.llm_model,
        )

        topic_mappings = {str(k): v for k, v in (result.topic_mappings or {}).items()}

        return ReduceTopicsResult(
            collection_name=result.collection_name,
            num_topics_before=result.num_topics_before_reduction or 0,
            num_topics_after=result.num_topics,
            topics=convert_topic_infos(result.topics),
            topic_mappings=topic_mappings,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )

    @strawberry.mutation
    async def generate_llm_labels(self, input: GenerateLlmLabelsInput, info=None) -> GenerateLlmLabelsResult:
        """Generate LLM labels for existing topics in a collection."""
        from ..services.topic_extraction_service import generate_llm_labels_for_collection

        result = await asyncio.to_thread(
            generate_llm_labels_for_collection,
            collection_name=input.collection_name,
            llm_provider=input.llm_provider,
            llm_model=input.llm_model,
            label_scope=input.label_scope,
            resume=input.resume,
        )

        return GenerateLlmLabelsResult(
            collection_name=result.collection_name,
            topics_labeled=result.topics_labeled,
            subtopics_labeled=result.subtopics_labeled,
            total_topics=result.total_topics,
            total_subtopics=result.total_subtopics,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )
