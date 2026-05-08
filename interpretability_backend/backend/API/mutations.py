"""GraphQL mutation resolvers for embedding visualization backend."""

import asyncio

import strawberry

from ..clients.duckdb_client import _sanitize_for_json
from ..services.embedding_pipeline import (
    HuggingFaceEmbeddingPipeline,
    LocalFileEmbeddingPipeline,
)
from ..services.interpret_service import SteeringSpec
from ..services.topic_extraction_service import (
    extract_topics as do_extract_topics,
)
from .chromadb_instance import get_chromadb_client
from .converters import (
    build_hf_embedding_config,
    build_local_file_embedding_config,
    build_topic_extraction_config,
    convert_topic_infos,
)
from .duckdb_instance import get_duckdb_client
from .interpret_instance import get_interpret_service
from .types import (
    JSON,
    AppliedSteering,
    EmbedDatasetInput,
    EmbedDatasetResult,
    EmbedLocalFileInput,
    ExtractTopicsInput,
    ExtractTopicsResult,
    GenerateLlmLabelsInput,
    GenerateLlmLabelsResult,
    GenerateSteeredInput,
    IngestSaeActivationsInput,
    IngestSaeFeaturesInput,
    IngestSaeResult,
    InterpretActiveFeature,
    InterpretLayerResult,
    InterpretTokenFeatures,
    ModelStatus,
    PrepareSaeInput,
    PrepareSaeResult,
    PromptActivationsResponse,
    PromptHighlightFeature,
    PromptHighlightResponse,
    ReduceTopicsInput,
    ReduceTopicsResult,
    RenameTopicLabelInput,
    RenameTopicLabelResult,
    RunPromptActivationsInput,
    RunPromptHighlightInput,
    SteeredGenerationResponse,
    UpdateCollectionMetadataResult,
)


@strawberry.type
class Mutation:
    """GraphQL mutation root."""

    @strawberry.mutation
    async def embed_huggingface_dataset(
        self, input: EmbedDatasetInput, info=None
    ) -> EmbedDatasetResult:
        """Embed a HuggingFace dataset into a ChromaDB collection."""
        config = build_hf_embedding_config(input)
        topic_config = (
            build_topic_extraction_config(input.collection_name, input.topic_config)
            if input.extract_topics
            else None
        )

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
        topic_config = (
            build_topic_extraction_config(input.collection_name, input.topic_config)
            if input.extract_topics
            else None
        )

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
        self, collection_name: str, metadata: JSON, info=None
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
            return UpdateCollectionMetadataResult(name=collection_name, metadata={}, error=str(e))

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
    async def generate_llm_labels(
        self, input: GenerateLlmLabelsInput, info=None
    ) -> GenerateLlmLabelsResult:
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

    # ------------------------------------------------------------------
    # Topic label renaming
    # ------------------------------------------------------------------

    @strawberry.mutation
    def rename_topic_label(self, input: RenameTopicLabelInput, info=None) -> RenameTopicLabelResult:
        """Rename a topic or subtopic label."""
        new_label = input.new_label.strip()
        if not new_label:
            return RenameTopicLabelResult(
                collection_name=input.collection_name,
                topic_id=input.topic_id,
                new_label="",
                error="Label cannot be empty",
            )
        if input.topic_id == -1 and not input.is_subtopic:
            return RenameTopicLabelResult(
                collection_name=input.collection_name,
                topic_id=input.topic_id,
                new_label=new_label,
                error="Cannot rename the noise/unclustered topic",
            )

        db = get_duckdb_client()
        topic_data = db.get_active_topics(input.collection_name)
        if not topic_data:
            return RenameTopicLabelResult(
                collection_name=input.collection_name,
                topic_id=input.topic_id,
                new_label=new_label,
                error="No active topic extraction found for this collection",
            )

        extraction_id = topic_data["id"]
        try:
            if input.is_subtopic:
                db.update_subtopic_label(extraction_id, input.topic_id, new_label)
            else:
                db.update_topic_label(extraction_id, input.topic_id, new_label)
        except Exception as e:
            return RenameTopicLabelResult(
                collection_name=input.collection_name,
                topic_id=input.topic_id,
                new_label=new_label,
                error=str(e),
            )

        return RenameTopicLabelResult(
            collection_name=input.collection_name,
            topic_id=input.topic_id,
            new_label=new_label,
        )

    @strawberry.mutation
    async def regenerate_topic_label(
        self, input: RenameTopicLabelInput, info=None
    ) -> RenameTopicLabelResult:
        """Regenerate an LLM label for a single topic using its keywords and sample documents."""
        import asyncio

        if input.topic_id == -1:
            return RenameTopicLabelResult(
                collection_name=input.collection_name,
                topic_id=input.topic_id,
                new_label="",
                error="Cannot label the noise/unclustered topic",
            )

        db = get_duckdb_client()
        topic_data = db.get_active_topics(input.collection_name)
        if not topic_data:
            return RenameTopicLabelResult(
                collection_name=input.collection_name,
                topic_id=input.topic_id,
                new_label="",
                error="No active topic extraction found for this collection",
            )

        extraction_id = topic_data["id"]

        # Find the topic's keywords
        topic_info = next(
            (t for t in topic_data["topics"] if t["topic_id"] == input.topic_id), None
        )
        if not topic_info:
            return RenameTopicLabelResult(
                collection_name=input.collection_name,
                topic_id=input.topic_id,
                new_label="",
                error=f"Topic {input.topic_id} not found",
            )

        keywords = [(kw["word"], kw["score"]) for kw in (topic_info.get("keywords") or [])]

        # Get sample documents for this topic
        item_ids = db.get_items_for_topic(extraction_id, input.topic_id)
        dataset_name = topic_data.get("dataset_name", input.collection_name)
        items_table = db._items_table(dataset_name)
        sample_ids = item_ids[:10]
        if sample_ids:
            placeholders = ", ".join(["?"] * len(sample_ids))
            docs_rows = db._conn.execute(
                f"SELECT document FROM {items_table} WHERE id IN ({placeholders})",
                sample_ids,
            ).fetchall()
            sample_docs = [r[0] for r in docs_rows if r[0]]
        else:
            sample_docs = []

        # Use the LLM provider from input.new_label as "provider:model" or default
        llm_provider = "gemini"
        llm_model = "gemini-2.5-flash"
        if input.new_label and ":" in input.new_label:
            parts = input.new_label.split(":", 1)
            llm_provider = parts[0]
            llm_model = parts[1]

        def _do_label():
            from ..topic_extraction.llm_labeling import (
                _create_labeler,
                generate_llm_label_for_topic,
            )

            labeler = _create_labeler(llm_provider, llm_model)
            return generate_llm_label_for_topic(
                topic_id=input.topic_id,
                keywords=keywords,
                sample_documents=sample_docs,
                labeler=labeler,
            )

        label = await asyncio.to_thread(_do_label)

        if not label:
            return RenameTopicLabelResult(
                collection_name=input.collection_name,
                topic_id=input.topic_id,
                new_label="",
                error="LLM failed to generate a label",
            )

        # Save the new label
        db.update_topic_label(extraction_id, input.topic_id, label)

        return RenameTopicLabelResult(
            collection_name=input.collection_name,
            topic_id=input.topic_id,
            new_label=label,
        )

    # ------------------------------------------------------------------
    # SAE ingestion
    # ------------------------------------------------------------------

    @strawberry.mutation
    async def ingest_sae_features(
        self, input: IngestSaeFeaturesInput, info=None
    ) -> IngestSaeResult:
        """Ingest SAE feature parquet into DuckDB (+ optional ChromaDB vectors)."""
        from ..embedding_functions.ingest_sae import ingest_sae_features

        result = await asyncio.to_thread(
            ingest_sae_features,
            parquet_path=input.parquet_path,
            model_id=input.model_id,
            sae_id=input.sae_id,
            store_vectors=input.store_vectors,
        )
        return IngestSaeResult(
            model_id=result["model_id"],
            sae_id=result["sae_id"],
            records_inserted=result["records_inserted"],
            duration_seconds=result["duration_seconds"],
            error=result.get("error"),
        )

    @strawberry.mutation
    async def ingest_sae_activations(
        self, input: IngestSaeActivationsInput, info=None
    ) -> IngestSaeResult:
        """Ingest SAE activation JSONL into DuckDB."""
        from ..embedding_functions.ingest_sae import ingest_sae_activations

        result = await asyncio.to_thread(
            ingest_sae_activations,
            jsonl_path=input.jsonl_path,
            model_id=input.model_id,
            sae_id=input.sae_id,
        )
        return IngestSaeResult(
            model_id=result["model_id"],
            sae_id=result["sae_id"],
            records_inserted=result["records_inserted"],
            duration_seconds=result["duration_seconds"],
            error=result.get("error"),
        )

    @strawberry.mutation
    async def prepare_sae_data(self, input: PrepareSaeInput, info=None) -> PrepareSaeResult:
        """Download, prepare, and ingest SAE data for a specific layer/hook/width.

        Runs the full pipeline: S3 download -> merge -> extract decoder vectors
        -> ingest into DuckDB. Skips if data is already ingested.
        """
        from ..services.sae_pipeline_service import prepare_and_ingest

        job_id = f"sae_prepare_{input.layer}_{input.hook_type}_{input.width}"
        result = await asyncio.to_thread(
            prepare_and_ingest,
            layer=input.layer,
            width=input.width,
            hook_type=input.hook_type,
            skip_download=input.skip_download,
            store_vectors=input.store_vectors,
            include_activations=input.include_activations,
            job_id=job_id,
        )
        return PrepareSaeResult(
            model_id=result["model_id"],
            sae_id=result["sae_id"],
            features_inserted=result["features_inserted"],
            activations_inserted=result["activations_inserted"],
            duration_seconds=result["duration_seconds"],
            status=result["status"],
            error=result.get("error"),
        )

    # ------------------------------------------------------------------
    # Interpret / SAE inference mutations
    # ------------------------------------------------------------------

    @strawberry.mutation
    async def load_model(self, checkpoint: str = "google/gemma-3-4b-it", info=None) -> ModelStatus:
        """Load the Gemma interpretability model into GPU memory."""
        service = get_interpret_service()
        try:
            async with service._lock:
                result = await asyncio.wait_for(
                    asyncio.to_thread(service.load_model, checkpoint),
                    timeout=300.0,
                )
            return ModelStatus(
                loaded=result.loaded,
                model_name=result.model_name,
                device=result.device,
            )
        except (TimeoutError, RuntimeError) as e:
            return ModelStatus(loaded=False, model_name=str(e))

    @strawberry.mutation
    async def unload_model(self, info=None) -> ModelStatus:
        """Unload the interpretability model and free GPU memory."""
        service = get_interpret_service()
        async with service._lock:
            result = await asyncio.to_thread(service.unload_model)
        return ModelStatus(
            loaded=result.loaded,
            model_name=result.model_name,
            device=result.device,
        )

    @strawberry.mutation
    async def run_prompt_activations(
        self, input: RunPromptActivationsInput, info=None
    ) -> PromptActivationsResponse:
        """Run a prompt through the model with SAE hooks and return per-token features."""
        service = get_interpret_service()
        try:
            async with service._lock:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        service.run_prompt_activations,
                        input.prompt,
                        input.layers,
                        input.width,
                        input.top_k,
                    ),
                    timeout=120.0,
                )
            return _convert_prompt_activations(result)
        except (RuntimeError, ValueError) as e:
            return PromptActivationsResponse(
                prompt=input.prompt,
                token_strings=[],
                layers=[],
                error=str(e),
            )
        except TimeoutError:
            return PromptActivationsResponse(
                prompt=input.prompt,
                token_strings=[],
                layers=[],
                error="Generation timed out after 120s",
            )

    @strawberry.mutation
    async def generate_steered_response(
        self, input: GenerateSteeredInput, info=None
    ) -> SteeredGenerationResponse:
        """Generate baseline vs steered text with one or more features."""
        service = get_interpret_service()
        specs = _steering_inputs_to_specs(input.steering)
        applied = _steering_inputs_to_applied(input.steering)
        try:
            async with service._lock:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        service.generate_steered,
                        input.prompt,
                        specs,
                        input.output_len,
                        input.temperature,
                    ),
                    timeout=180.0,
                )
            return SteeredGenerationResponse(
                baseline_text=result.baseline_text,
                steered_text=result.steered_text,
                steering=applied,
            )
        except (RuntimeError, ValueError) as e:
            return SteeredGenerationResponse(
                baseline_text="",
                steered_text="",
                steering=applied,
                error=str(e),
            )
        except TimeoutError:
            return SteeredGenerationResponse(
                baseline_text="",
                steered_text="",
                steering=applied,
                error="Generation timed out after 180s",
            )

    @strawberry.mutation
    async def run_prompt_highlight(
        self, input: RunPromptHighlightInput, info=None
    ) -> PromptHighlightResponse:
        """Run a prompt and return max-pooled feature activations for scatter plot highlighting."""
        service = get_interpret_service()
        hook_type_str = input.hook_type.value
        try:
            async with service._lock:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        service.run_prompt_highlight,
                        input.prompt,
                        input.layer,
                        input.width,
                        hook_type_str,
                    ),
                    timeout=120.0,
                )
            return PromptHighlightResponse(
                features=[
                    PromptHighlightFeature(
                        feature_index=f.feature_index,
                        activation=f.activation,
                    )
                    for f in result
                ],
            )
        except (RuntimeError, ValueError) as e:
            return PromptHighlightResponse(features=[], error=str(e))
        except TimeoutError:
            return PromptHighlightResponse(
                features=[],
                error="Generation timed out after 120s",
            )


def _steering_inputs_to_specs(inputs) -> list[SteeringSpec]:
    """Convert GraphQL SteeringInput list to service SteeringSpec list."""
    return [
        SteeringSpec(
            feature_index=s.feature_index,
            layer=s.layer,
            hook_type=s.hook_type.value,
            width=s.width,
            strength=s.strength,
        )
        for s in inputs
    ]


def _steering_inputs_to_applied(inputs) -> list[AppliedSteering]:
    """Convert GraphQL SteeringInput list to AppliedSteering output list."""
    return [
        AppliedSteering(
            feature_index=s.feature_index,
            layer=s.layer,
            hook_type=s.hook_type.value,
            width=s.width,
            strength=s.strength,
        )
        for s in inputs
    ]


def _convert_prompt_activations(result) -> PromptActivationsResponse:
    """Convert service PromptActivationsResult to GraphQL response type."""
    layers = [
        InterpretLayerResult(
            layer=lr.layer,
            width=lr.width,
            tokens=[
                InterpretTokenFeatures(
                    token=tf.token,
                    position=tf.position,
                    features=[
                        InterpretActiveFeature(
                            index=f.index,
                            activation=f.activation,
                            label=f.label,
                            density=f.density,
                        )
                        for f in tf.features
                    ],
                )
                for tf in lr.tokens
            ],
        )
        for lr in result.layers
    ]
    return PromptActivationsResponse(
        prompt=result.prompt,
        token_strings=result.token_strings,
        layers=layers,
    )
