"""GraphQL mutation resolvers for embedding visualization backend."""

import asyncio
import strawberry
from typing import Optional

from interpretability_backend.backend.services import job_state

from .types import (
    DataTypeEnum,
    EmbedDatasetInput,
    EmbedDatasetResult,
    EmbedLocalFileInput,
    EmbeddingProviderEnum,
    ExtractTopicsInput,
    ExtractTopicsResult,
    GenerateLlmLabelsInput,
    GenerateLlmLabelsResult,
    JSON,
    PortionStrategyEnum,
    ReduceTopicsInput,
    ReduceTopicsResult,
    TopicInfo,
    TopicKeyword,
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
from ..services.progress_emitter import emit_progress_sync
from ..services.job_state import get_job_state_service, JobStatus
from ..services.topic_extraction_service import (
    extract_topics as do_extract_topics,
    TopicExtractionConfig,
)


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
    async def embed_huggingface_dataset(self, input: EmbedDatasetInput, info=None) -> EmbedDatasetResult:
        """Embed a HuggingFace dataset into a ChromaDB collection.

        Runs embedding in a background thread to allow WebSocket progress updates.

        Args:
            input: Configuration for embedding the dataset

        Returns:
            Result with statistics about the embedding operation
        """
        # Convert GraphQL input to EmbeddingConfig
        portion = None
        job_state = get_job_state_service()

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
                task_type=input.embedding_model.task_type,  # Gemini: optimization type
                prompt=input.embedding_model.prompt  # SentenceTransformers: can be known name or custom string
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
            batch_size=input.batch_size or 100,
            resume=input.resume
        )

        # Run embedding in background thread to allow event loop to process WebSocket updates
        result = await asyncio.to_thread(do_hf_embed, config)

        # Compute projections if requested and embedding succeeded
        projections_computed = False
        if input.compute_projections and result.error is None and result.total_embedded > 0:
            # Emit status: computing projections
            emit_progress_sync(
                job_id=input.collection_name,
                status="running",
                items_processed=result.total_embedded,
                total_items=result.total_embedded,
                current_batch=0,
                total_batches=0,
                message="Computing projections (PCA/UMAP)..."
            )
            projections_computed = await asyncio.to_thread(
                compute_projections_for_collection, input.collection_name
            )

        # Extract topics if requested and projections succeeded
        topics_extracted = False
        if input.extract_topics and projections_computed and result.error is None:
            topic_config = input.topic_config or {}
            topics_extracted = await _extract_topics_for_collection(
                input.collection_name,
                topic_config
            )

        # Mark job as complete
        job_state.complete_job(config.collection_name)

        # Emit final completion status
        emit_progress_sync(
            job_id=input.collection_name,
            status="completed",
            items_processed=result.total_embedded,
            total_items=result.total_embedded,
            current_batch=0,
            total_batches=0,
            message="Complete!"
        )

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
    async def embed_local_file(self, input: EmbedLocalFileInput, info=None) -> EmbedDatasetResult:
        """Embed a local file (parquet/json/csv) into a ChromaDB collection.

        Supports text, image, and pre-computed vector embeddings.
        Runs embedding in a background thread to allow WebSocket progress updates.

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
                task_type=input.embedding_model.task_type,  # Gemini: optimization type
                prompt=input.embedding_model.prompt  # SentenceTransformers: can be known name or custom string
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
            batch_size=input.batch_size or 100,
            resume=input.resume
        )

        # Run embedding in background thread to allow event loop to process WebSocket updates
        result = await asyncio.to_thread(do_local_embed, config)

        # Compute projections if requested and embedding succeeded
        projections_computed = False
        if input.compute_projections and result.error is None and result.total_embedded > 0:
            # Emit status: computing projections
            emit_progress_sync(
                job_id=input.collection_name,
                status="running",
                items_processed=result.total_embedded,
                total_items=result.total_embedded,
                current_batch=0,
                total_batches=0,
                message="Computing projections (PCA/UMAP)..."
            )
            projections_computed = await asyncio.to_thread(
                compute_projections_for_collection, input.collection_name
            )

        # Extract topics if requested and projections succeeded
        topics_extracted = False
        if input.extract_topics and projections_computed and result.error is None:
            topic_config = input.topic_config or {}
            topics_extracted = await _extract_topics_for_collection(
                input.collection_name,
                topic_config
            )

        # Emit final completion status
        emit_progress_sync(
            job_id=input.collection_name,
            status="completed",
            items_processed=result.total_embedded,
            total_items=result.total_embedded,
            current_batch=0,
            total_batches=0,
            message="Complete!"
        )

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

    @strawberry.mutation
    async def extract_topics(self, input: ExtractTopicsInput, info=None) -> ExtractTopicsResult:
        """Extract topic clusters from an existing collection.

        Uses HDBSCAN clustering on projection coordinates (UMAP preferred) and
        c-TF-IDF for keyword extraction. Optionally generates human-readable
        labels using an LLM.

        Topic data is stored in item metadata as 'topic_id' and 'topic_label'.

        Args:
            input: Configuration for topic extraction

        Returns:
            Result with topic information
        """
        # Build config from input — unpack nested TopicConfigInput
        tc = input.config

        # Extract reduction config if present
        reduce_topics = False
        reduction_method = "auto"
        nr_topics = None
        use_ctfidf_for_reduction = True

        if tc and tc.reduction and tc.reduction.enabled:
            reduce_topics = True
            reduction_method = tc.reduction.method
            nr_topics = tc.reduction.n_topics
            use_ctfidf_for_reduction = tc.reduction.use_ctfidf

        config = TopicExtractionConfig(
            collection_name=input.collection_name,
            min_topic_size=tc.min_topic_size if tc else 10,
            n_keywords=tc.n_keywords if tc else 10,
            use_llm_labels=tc.use_llm_labels if tc else False,
            llm_provider=tc.llm_provider if tc else 'gemini',
            llm_model=tc.llm_model if tc else 'gemini-3-flash-preview',
            projection_type=tc.projection_type if tc else 'umap_2d',
            reduce_topics=reduce_topics,
            reduction_method=reduction_method,
            nr_topics=nr_topics,
            use_ctfidf_for_reduction=use_ctfidf_for_reduction
        )

        # Run topic extraction in background thread
        result = await asyncio.to_thread(do_extract_topics, config)

        # Convert result to GraphQL types
        topics = [
            TopicInfo(
                topic_id=topic.topic_id,
                keywords=[TopicKeyword(word=w, score=s) for w, s in topic.keywords],
                label=topic.label,
                count=topic.count,
                subtopics=topic.subtopics
            )
            for topic in result.topics
        ]

        return ExtractTopicsResult(
            collection_name=result.collection_name,
            num_topics=result.num_topics,
            num_noise_points=result.num_noise_points,
            topics=topics,
            duration_seconds=result.duration_seconds,
            error=result.error,
            num_topics_before_reduction=result.num_topics_before_reduction,
            reduction_applied=result.reduction_applied
        )

    @strawberry.mutation
    async def reduce_topics(self, input: ReduceTopicsInput, info=None) -> ReduceTopicsResult:
        """Reduce topics on an existing collection (standalone post-processing).

        Args:
            input: Configuration for topic reduction

        Returns:
            Result with reduced topics and mappings
        """
        from ..services.topic_extraction_service import reduce_existing_topics

        def run_reduction():
            return reduce_existing_topics(
                collection_name=input.collection_name,
                method=input.method,
                n_topics=input.n_topics,
                use_ctfidf=input.use_ctfidf,
                regenerate_labels=input.regenerate_labels,
                llm_provider=input.llm_provider,
                llm_model=input.llm_model
            )

        # Run in background thread (same pattern as extract_topics)
        result = await asyncio.to_thread(run_reduction)

        # Convert to GraphQL types
        topics = [
            TopicInfo(
                topic_id=topic.topic_id,
                keywords=[TopicKeyword(word=w, score=s) for w, s in topic.keywords],
                label=topic.label,
                count=topic.count,
                subtopics=topic.subtopics
            )
            for topic in result.topics
        ]

        # Build topic mappings JSON from reduction result
        topic_mappings = {str(k): v for k, v in (result.topic_mappings or {}).items()}

        return ReduceTopicsResult(
            collection_name=result.collection_name,
            num_topics_before=result.num_topics_before_reduction or 0,
            num_topics_after=result.num_topics,
            topics=topics,
            topic_mappings=topic_mappings,
            duration_seconds=result.duration_seconds,
            error=result.error
        )


    @strawberry.mutation
    async def generate_llm_labels(self, input: GenerateLlmLabelsInput, info=None) -> GenerateLlmLabelsResult:
        """Generate LLM labels for existing topics in a collection.

        Incrementally saves labels after each LLM call. Supports resume
        by detecting already-labeled topics via keyword pattern matching.

        Args:
            input: Configuration for LLM label generation

        Returns:
            Result with labeling counts and timing
        """
        from ..services.topic_extraction_service import generate_llm_labels_for_collection

        def run_labeling():
            return generate_llm_labels_for_collection(
                collection_name=input.collection_name,
                llm_provider=input.llm_provider,
                llm_model=input.llm_model,
                label_scope=input.label_scope,
                resume=input.resume,
            )

        result = await asyncio.to_thread(run_labeling)

        return GenerateLlmLabelsResult(
            collection_name=result.collection_name,
            topics_labeled=result.topics_labeled,
            subtopics_labeled=result.subtopics_labeled,
            total_topics=result.total_topics,
            total_subtopics=result.total_subtopics,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )


async def _extract_topics_for_collection(
    collection_name: str,
    topic_config_input
) -> bool:
    """Helper to extract topics for a collection (used by embedding mutations).

    Args:
        collection_name: Name of collection
        topic_config_input: TopicConfigInput or dict with config

    Returns:
        True if topics were extracted successfully
    """
    try:
        # Handle both TopicConfigInput and dict
        if hasattr(topic_config_input, '__dict__'):
            tc = topic_config_input

            # Extract reduction config if present
            reduce_topics = False
            reduction_method = "auto"
            nr_topics = None
            use_ctfidf_for_reduction = True

            reduction = getattr(tc, 'reduction', None)
            if reduction and getattr(reduction, 'enabled', False):
                reduce_topics = True
                reduction_method = getattr(reduction, 'method', 'auto')
                nr_topics = getattr(reduction, 'n_topics', None)
                use_ctfidf_for_reduction = getattr(reduction, 'use_ctfidf', True)

            config = TopicExtractionConfig(
                collection_name=collection_name,
                min_topic_size=getattr(tc, 'min_topic_size', 10),
                n_keywords=getattr(tc, 'n_keywords', 10),
                use_llm_labels=getattr(tc, 'use_llm_labels', False),
                llm_provider=getattr(tc, 'llm_provider', 'gemini'),
                llm_model=getattr(tc, 'llm_model', 'gemini-3-flash-preview'),
                projection_type=getattr(tc, 'projection_type', 'umap_2d'),
                reduce_topics=reduce_topics,
                reduction_method=reduction_method,
                nr_topics=nr_topics,
                use_ctfidf_for_reduction=use_ctfidf_for_reduction
            )
        else:
            # Extract reduction config from dict
            reduce_topics = False
            reduction_method = "auto"
            nr_topics = None
            use_ctfidf_for_reduction = True

            reduction = topic_config_input.get('reduction', None)
            if reduction and reduction.get('enabled', False):
                reduce_topics = True
                reduction_method = reduction.get('method', 'auto')
                nr_topics = reduction.get('n_topics', None)
                use_ctfidf_for_reduction = reduction.get('use_ctfidf', True)

            config = TopicExtractionConfig(
                collection_name=collection_name,
                min_topic_size=topic_config_input.get('min_topic_size', 10),
                n_keywords=topic_config_input.get('n_keywords', 10),
                use_llm_labels=topic_config_input.get('use_llm_labels', False),
                llm_provider=topic_config_input.get('llm_provider', 'gemini'),
                llm_model=topic_config_input.get('llm_model', 'gemini-3-flash-preview'),
                projection_type=topic_config_input.get('projection_type', 'umap_2d'),
                reduce_topics=reduce_topics,
                reduction_method=reduction_method,
                nr_topics=nr_topics,
                use_ctfidf_for_reduction=use_ctfidf_for_reduction
            )

        result = await asyncio.to_thread(do_extract_topics, config)
        return result.error is None
    except Exception as e:
        print(f"Topic extraction failed: {e}")
        return False
