"""GraphQL mutation resolvers for embedding visualization backend."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import strawberry

from ..clients.duckdb_client import _sanitize_for_json
from ..services.cancel_registry import (
    register_cancel_event,
    request_cancel,
    unregister_cancel_event,
)
from ..services.embedding_pipeline import (
    HuggingFaceEmbeddingPipeline,
    LocalFileEmbeddingPipeline,
    ReEmbeddingPipeline,
)
from ..services.interpret_service import SteeringSpec
from ..services.job_state import JobStatus, get_job_state_service
from ..services.progress_emitter import emit_progress_sync
from ..services.topic_extraction_service import (
    extract_topics as do_extract_topics,
)
from .chromadb_instance import get_chromadb_client
from .converters import (
    build_embedding_model_config,
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
    ChatSessionInfo,
    ChatSessionMessage,
    ComputeDocumentActivationsInput,
    ComputeDocumentActivationsResult,
    CreateChatSessionInput,
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
    PromptDocumentSearchResponse,
    PromptDocumentSearchResult,
    PromptHighlightFeature,
    PromptHighlightResponse,
    ReduceTopicsInput,
    ReduceTopicsResult,
    ReEmbedDatasetInput,
    RenameTopicLabelInput,
    RenameTopicLabelResult,
    RunPromptActivationsInput,
    RunPromptHighlightInput,
    SaeCollectionMode,
    SaveChatMessageInput,
    SearchDocumentsByPromptInput,
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

        cancel_event = register_cancel_event(input.collection_name)
        try:
            pipeline = HuggingFaceEmbeddingPipeline(
                config=config,
                compute_projections=input.compute_projections,
                extract_topics=input.extract_topics,
                topic_config=topic_config,
                cancel_event=cancel_event,
            )
            pipeline_result = await pipeline.run()
        finally:
            unregister_cancel_event(input.collection_name)

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

        cancel_event = register_cancel_event(input.collection_name)
        try:
            pipeline = LocalFileEmbeddingPipeline(
                config=config,
                compute_projections=input.compute_projections,
                extract_topics=input.extract_topics,
                topic_config=topic_config,
                cancel_event=cancel_event,
            )
            pipeline_result = await pipeline.run()
        finally:
            unregister_cancel_event(input.collection_name)

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
    async def re_embed_dataset(self, input: ReEmbedDatasetInput, info=None) -> EmbedDatasetResult:
        """Re-embed an existing dataset with a different embedding model."""
        from ..embedding_functions.config import (
            ReEmbedConfig,  # noqa: E402 - avoid circular at module level
        )

        config = ReEmbedConfig(
            source_dataset_name=input.source_dataset_name,
            collection_name=input.collection_name,
            embedding_model=build_embedding_model_config(input.embedding_model),
            columns=input.columns,
            text_template=input.text_template,
            batch_size=input.batch_size or 100,
            resume=input.resume,
        )

        topic_config = (
            build_topic_extraction_config(input.collection_name, input.topic_config)
            if input.extract_topics
            else None
        )

        pipeline = ReEmbeddingPipeline(
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
    def cancel_embedding_job(self, collection_name: str, info=None) -> bool:
        """Cancel a running embedding job.

        Sets the cancel event so the batch loop stops after the current batch.
        The job will be marked as interrupted with partial results preserved.
        Returns True if a running job was found and signalled, False otherwise.
        """
        return request_cancel(collection_name)

    @strawberry.mutation
    def remove_embedding_job(self, collection_name: str, info=None) -> bool:
        """Remove an interrupted job record from the job state.

        Only removes the job tracking entry. Does NOT delete partially
        embedded data (ChromaDB vectors, DuckDB documents).
        Returns True if a job was found and removed, False otherwise.
        """
        job_service = get_job_state_service()
        job = job_service.get_job(collection_name)
        if job is None:
            return False
        if job.status == JobStatus.RUNNING:
            return False
        job_service.remove_job(collection_name)
        return True

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
            # Update DuckDB (primary) — resolve dataset name from vector_collections
            db = get_duckdb_client()
            dataset_name = db.get_dataset_name_for_collection(collection_name) or collection_name
            db.update_dataset(dataset_name, extra_metadata=metadata)
            ds = db.get_dataset(dataset_name)

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
        sample_ids = item_ids[:10]
        if sample_ids:
            docs_rows = db.get_items_columns(dataset_name, ("document",), where_ids=sample_ids)
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
        """Download and extract SAE data for a specific layer/hook/width.

        Runs the pipeline: S3 download -> merge activations -> extract decoder
        vectors into parquet. Optionally creates a visualization collection
        with projections and topic extraction.
        """
        from ..embedding_functions.config import (  # noqa: E402 - avoid circular
            DataType,
            LocalFileEmbeddingConfig,
        )
        from ..services.sae_pipeline_service import prepare_sae_data as run_pipeline

        # Phase 1: SAE download + ingest into DuckDB
        job_id = f"sae_prepare_{input.model_size}_{input.variant}_{input.layer}_{input.hook_type}_{input.width}"
        result = await asyncio.to_thread(
            run_pipeline,
            layer=input.layer,
            width=input.width,
            hook_type=input.hook_type,
            model_size=input.model_size,
            variant=input.variant,
            skip_download=input.skip_download,
            include_activations=input.include_activations,
            job_id=job_id,
        )

        collection_name = None
        collection_items = 0

        # Phase 2: Create visualization collection (if requested and phase 1 succeeded)
        collection_error = None
        if (
            input.create_collection
            and result["status"] == "completed"
            and result.get("features_parquet")
        ):
            hook_abbrev = {"resid_post": "res", "mlp_out": "mlp", "attn_out": "att"}.get(
                input.hook_type, "res"
            )
            mode = input.collection_mode or SaeCollectionMode.DECODER_VECTORS
            mode_suffix = "vectors" if mode == SaeCollectionMode.DECODER_VECTORS else "labels"
            variant_suffix = f"_{input.variant}" if input.variant != "it" else ""
            collection_name = f"sae_{input.model_size}{variant_suffix}_{input.layer}_{hook_abbrev}_{input.width}_{mode_suffix}"

            try:
                emb_model_config = build_embedding_model_config(input.embedding_model)

                if mode == SaeCollectionMode.DECODER_VECTORS:
                    config = LocalFileEmbeddingConfig(
                        file_path=result["features_parquet"],
                        collection_name=collection_name,
                        data_type=DataType.VECTOR,
                        vector_column="vector",
                        metadata_columns=["density", "label", "index"],
                    )
                else:
                    config = LocalFileEmbeddingConfig(
                        file_path=result["features_parquet"],
                        collection_name=collection_name,
                        data_type=DataType.TEXT,
                        columns=["label"],
                        metadata_columns=["density", "index"],
                        embedding_model=emb_model_config,
                    )

                topic_config = (
                    build_topic_extraction_config(collection_name, input.topic_config)
                    if input.extract_topics
                    else None
                )

                pipeline = LocalFileEmbeddingPipeline(
                    config=config,
                    compute_projections=True,
                    extract_topics=input.extract_topics,
                    topic_config=topic_config,
                )
                pipeline_result = await pipeline.run()
                collection_items = pipeline_result.embedding_result.total_embedded

                # Set SAE metadata for cross-linking with feature explorer
                db = get_duckdb_client()
                db.update_dataset(
                    collection_name,
                    extra_metadata={
                        "sae_model_id": result["model_id"],
                        "sae_id": result["sae_id"],
                    },
                )

                # Cleanup source files if requested
                if input.delete_source_files:

                    def _cleanup():
                        Path(result["features_parquet"]).unlink(missing_ok=True)
                        if result.get("activations_jsonl"):
                            Path(result["activations_jsonl"]).unlink(missing_ok=True)

                    await asyncio.to_thread(_cleanup)

            except Exception as e:
                logger = logging.getLogger("orrery.mutations")
                logger.exception("SAE collection creation failed for %s", collection_name)
                collection_error = f"Collection creation failed: {e}"
                collection_name = None

        return PrepareSaeResult(
            model_id=result["model_id"],
            sae_id=result["sae_id"],
            features_parquet=result.get("features_parquet"),
            activations_jsonl=result.get("activations_jsonl"),
            features_inserted=result.get("features_inserted", 0),
            activations_inserted=result.get("activations_inserted", 0),
            duration_seconds=result["duration_seconds"],
            status=result["status"],
            error=collection_error or result.get("error"),
            collection_name=collection_name,
            collection_items=collection_items,
        )

    @strawberry.mutation
    async def update_sae_feature_label(
        self, model_id: str, sae_id: str, feature_index: int, label: str, info=None
    ) -> bool:
        """Rename the label of a single SAE feature."""
        try:
            db = get_duckdb_client()
            return await asyncio.to_thread(
                db.update_sae_feature_label, model_id, sae_id, feature_index, label
            )
        except Exception:
            return False

    @strawberry.mutation
    async def delete_sae_data(self, model_id: str, sae_id: str, info=None) -> bool:
        """Delete all SAE features and activations for a model/sae pair."""
        try:
            db = get_duckdb_client()
            return await asyncio.to_thread(db.delete_sae_data, model_id, sae_id)
        except Exception:
            return False

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
                variant=result.variant,
                model_size=result.model_size,
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
            variant=result.variant,
            model_size=result.model_size,
        )

    @strawberry.mutation
    async def run_prompt_activations(
        self, input: RunPromptActivationsInput, info=None
    ) -> PromptActivationsResponse:
        """Run a prompt through the model with SAE hooks and return per-token features."""
        service = get_interpret_service()
        saes = [(s.layer, s.width) for s in input.saes] if input.saes else None
        try:
            async with service._lock:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        service.run_prompt_activations,
                        input.prompt,
                        input.layers,
                        input.width,
                        input.top_k,
                        input.model_id,
                        input.sae_id,
                        input.skip_chat_template,
                        input.filter_mode.value,
                        saes,
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

    # ------------------------------------------------------------------
    # Batch SAE document activations
    # ------------------------------------------------------------------

    @strawberry.mutation
    async def compute_document_activations(
        self, input: ComputeDocumentActivationsInput, info=None
    ) -> ComputeDocumentActivationsResult:
        """Run SAE inference on all documents in a collection and store activations.

        Parses layer/width/hook_type from the collection's SAE metadata.
        Supports resume: already-processed items are skipped.
        Emits progress via the embedding_progress subscription.
        """
        logger = logging.getLogger("orrery.mutations")
        db = get_duckdb_client()
        service = get_interpret_service()
        collection_name = input.collection_name

        # --- Resolve SAE config from collection metadata ---
        dataset_name = db.get_dataset_name_for_collection(collection_name)
        if not dataset_name:
            return ComputeDocumentActivationsResult(
                collection_name=collection_name,
                items_processed=0,
                total_items=0,
                duration_seconds=0.0,
                error=f"Collection '{collection_name}' not found.",
            )

        ds = db.get_dataset(dataset_name)
        raw_extra = (ds.get("extra_metadata") or {}) if ds else {}
        extra = json.loads(raw_extra) if isinstance(raw_extra, str) else raw_extra
        model_id = extra.get("sae_model_id")
        sae_id = extra.get("sae_id")
        if not model_id or not sae_id:
            return ComputeDocumentActivationsResult(
                collection_name=collection_name,
                items_processed=0,
                total_items=0,
                duration_seconds=0.0,
                error="Collection has no SAE metadata (sae_model_id / sae_id).",
            )

        # Parse sae_id → layer, hook_type, width
        # Format: "{layer}-gemmascope-{version}-{hookAbbrev}-{width}"
        parts = sae_id.split("-")
        layer = int(parts[0]) if parts else 9
        hook_abbrev = parts[3] if len(parts) > 3 else "res"
        width = parts[4] if len(parts) > 4 else "16k"
        hook_map = {"res": "resid_post", "mlp": "mlp_out", "att": "attn_out"}
        hook_type = hook_map.get(hook_abbrev, "resid_post")

        # Derive expected checkpoint from model_id (e.g. "gemma-3-4b-it")
        # and ensure the correct model is loaded before running inference.
        # model_id is the Neuronpedia ID (e.g. "gemma-3-1b"), but load_model
        # expects a HuggingFace checkpoint (e.g. "google/gemma-3-1b-pt").
        expected_checkpoint = f"google/{model_id}" if "/" not in model_id else model_id

        # --- Load all items ---
        all_items = db.get_filtered_items(dataset_name, filters=[], limit=100_000)
        if not all_items:
            return ComputeDocumentActivationsResult(
                collection_name=collection_name,
                items_processed=0,
                total_items=0,
                duration_seconds=0.0,
                error="No items found in dataset.",
            )

        # --- Resume: skip already-processed items ---
        existing_ids = db.get_document_activation_item_ids(collection_name)
        documents = [
            (item["id"], item.get("document") or "")
            for item in all_items
            if item["id"] not in existing_ids
        ]

        total_items = len(all_items)
        already_done = len(existing_ids)
        to_process = len(documents)

        if to_process == 0:
            return ComputeDocumentActivationsResult(
                collection_name=collection_name,
                items_processed=total_items,
                total_items=total_items,
                duration_seconds=0.0,
            )

        job_id = f"{collection_name}_sae_activations"

        def _run_batch():
            start = time.monotonic()

            def progress_cb(done: int, total: int):
                emit_progress_sync(
                    job_id=job_id,
                    status="running",
                    items_processed=already_done + done,
                    total_items=total_items,
                    current_batch=done,
                    total_batches=total,
                    message=f"SAE inference: {already_done + done}/{total_items}",
                )

            emit_progress_sync(
                job_id=job_id,
                status="running",
                items_processed=already_done,
                total_items=total_items,
                current_batch=0,
                total_batches=to_process,
                message=f"Starting SAE inference ({to_process} items)...",
            )

            FLUSH_EVERY = 50  # persist every N documents to avoid losing work
            pending_rows: list[dict[str, Any]] = []
            docs_since_flush = 0
            docs_processed = 0

            def on_result(item_id: str, activations: list) -> None:
                nonlocal docs_since_flush, docs_processed
                for f in activations:
                    pending_rows.append(
                        {
                            "collection_name": collection_name,
                            "item_id": item_id,
                            "feature_index": f.feature_index,
                            "activation": f.activation,
                        }
                    )
                docs_since_flush += 1
                docs_processed += 1
                if docs_since_flush >= FLUSH_EVERY and pending_rows:
                    db.insert_document_activations_bulk(pd.DataFrame(pending_rows))
                    pending_rows.clear()
                    docs_since_flush = 0

            service.run_batch_highlight(
                documents=documents,
                layer=layer,
                width=width,
                hook_type=hook_type,
                progress_callback=progress_cb,
                result_callback=on_result,
            )

            # Flush remaining
            if pending_rows:
                db.insert_document_activations_bulk(pd.DataFrame(pending_rows))

            elapsed = time.monotonic() - start

            emit_progress_sync(
                job_id=job_id,
                status="completed",
                items_processed=total_items,
                total_items=total_items,
                current_batch=to_process,
                total_batches=to_process,
                message="SAE document activations complete.",
            )

            return already_done + docs_processed, elapsed

        try:
            async with service._lock:
                # Ensure the correct model is loaded for this SAE
                status = service.get_status()
                if status.loaded and service._neuronpedia_model_id != expected_checkpoint:
                    await asyncio.to_thread(service.unload_model)
                    status = service.get_status()
                if not status.loaded:
                    await asyncio.to_thread(service.load_model, expected_checkpoint)
                items_processed, duration = await asyncio.to_thread(_run_batch)

            return ComputeDocumentActivationsResult(
                collection_name=collection_name,
                items_processed=items_processed,
                total_items=total_items,
                duration_seconds=round(duration, 2),
            )
        except Exception as e:
            logger.exception("compute_document_activations failed")
            emit_progress_sync(
                job_id=job_id,
                status="failed",
                items_processed=already_done,
                total_items=total_items,
                current_batch=0,
                total_batches=to_process,
                error=str(e),
            )
            return ComputeDocumentActivationsResult(
                collection_name=collection_name,
                items_processed=already_done,
                total_items=total_items,
                duration_seconds=0.0,
                error=str(e),
            )

    @strawberry.mutation
    def delete_document_activations(self, collection_name: str, info=None) -> int:
        """Delete all precomputed SAE document activations for a collection."""
        db = get_duckdb_client()
        return db.delete_document_activations(collection_name)

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    @strawberry.mutation
    def create_chat_session(self, input: CreateChatSessionInput, info=None) -> ChatSessionInfo:
        """Create a new chat session."""
        db = get_duckdb_client()
        config_str = json.dumps(input.config) if not isinstance(input.config, str) else input.config
        data = db.create_chat_session(input.id, input.title, config_str)
        return ChatSessionInfo(
            id=data["id"],
            title=data["title"],
            config=data["config"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    @strawberry.mutation
    def save_chat_message(self, input: SaveChatMessageInput, info=None) -> ChatSessionMessage:
        """Save a message to a chat session."""
        db = get_duckdb_client()
        parts_str = json.dumps(input.parts) if input.parts is not None else None
        steering_snapshot_str = (
            json.dumps(input.steering_snapshot) if input.steering_snapshot is not None else None
        )
        data = db.save_chat_message(
            input.id,
            input.session_id,
            input.role,
            input.content,
            parts_str,
            steering_snapshot_str,
        )
        return ChatSessionMessage(
            id=data["id"],
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            parts=data.get("parts"),
            steering_snapshot=data.get("steering_snapshot"),
            created_at=data["created_at"],
        )

    # ------------------------------------------------------------------
    # Prompt → SAE → document similarity search
    # ------------------------------------------------------------------

    @strawberry.mutation
    async def search_documents_by_prompt(
        self, input: SearchDocumentsByPromptInput, info=None
    ) -> PromptDocumentSearchResponse:
        """Run a prompt through model+SAE, then find documents with similar activations.

        1. Runs the prompt through the model with SAE hooks (same as runPromptHighlight).
        2. Takes the max-pooled activation vector.
        3. Computes sparse dot-product similarity against precomputed document activations.
        4. Returns ranked documents.
        """

        service = get_interpret_service()
        db = get_duckdb_client()
        collection_name = input.collection_name

        # Check document activations exist
        if not db.has_document_activations(collection_name):
            return PromptDocumentSearchResponse(
                results=[],
                prompt_feature_count=0,
                error="No document activations computed for this collection. "
                "Run computeDocumentActivations first.",
            )

        # Resolve SAE config from collection metadata
        dataset_name = db.get_dataset_name_for_collection(collection_name)
        if not dataset_name:
            return PromptDocumentSearchResponse(
                results=[],
                prompt_feature_count=0,
                error=f"Collection '{collection_name}' not found.",
            )

        ds = db.get_dataset(dataset_name)
        raw_extra = (ds.get("extra_metadata") or {}) if ds else {}
        extra = json.loads(raw_extra) if isinstance(raw_extra, str) else raw_extra
        sae_id = extra.get("sae_id")
        if not sae_id:
            return PromptDocumentSearchResponse(
                results=[],
                prompt_feature_count=0,
                error="Collection has no SAE metadata.",
            )

        # Parse sae_id → layer, hook_type, width
        parts = sae_id.split("-")
        layer = int(parts[0]) if parts else 9
        hook_abbrev = parts[3] if len(parts) > 3 else "res"
        width = parts[4] if len(parts) > 4 else "16k"
        hook_map = {"res": "resid_post", "mlp": "mlp_out", "att": "attn_out"}
        hook_type = hook_map.get(hook_abbrev, "resid_post")

        # Check model loaded
        if not service.get_status().loaded:
            return PromptDocumentSearchResponse(
                results=[],
                prompt_feature_count=0,
                error="Model not loaded. Call loadModel first.",
            )

        try:
            async with service._lock:
                activations = await asyncio.wait_for(
                    asyncio.to_thread(
                        service.run_prompt_highlight,
                        input.prompt,
                        layer,
                        width,
                        hook_type,
                    ),
                    timeout=120.0,
                )
        except (RuntimeError, ValueError) as e:
            return PromptDocumentSearchResponse(results=[], prompt_feature_count=0, error=str(e))
        except TimeoutError:
            return PromptDocumentSearchResponse(
                results=[],
                prompt_feature_count=0,
                error="Inference timed out after 120s.",
            )

        if not activations:
            return PromptDocumentSearchResponse(
                results=[],
                prompt_feature_count=0,
                error="No features activated for this prompt.",
            )

        # Search documents by sparse dot product
        acts_tuples = [(f.feature_index, f.activation) for f in activations]
        results = db.search_documents_by_activations(
            collection_name=collection_name,
            activations=acts_tuples,
            limit=input.limit,
            top_k=input.top_k_features,
        )

        return PromptDocumentSearchResponse(
            results=[
                PromptDocumentSearchResult(
                    item_id=r["item_id"],
                    document=r.get("document"),
                    metadata=r.get("metadata"),
                    score=r["score"],
                    shared_features=r["shared_features"],
                    row_index=r.get("row_index"),
                )
                for r in results
            ],
            prompt_feature_count=len(activations),
        )

    @strawberry.mutation
    def delete_chat_session(self, id: str, info=None) -> bool:
        """Delete a chat session and all its messages."""
        db = get_duckdb_client()
        return db.delete_chat_session(id)


def _steering_inputs_to_specs(inputs) -> list[SteeringSpec]:
    """Convert GraphQL SteeringInput list to service SteeringSpec list."""
    return [
        SteeringSpec(
            feature_index=s.feature_index,
            layer=s.layer,
            hook_type=s.hook_type.value,
            width=s.width,
            strength=s.strength,
            direction_name=s.direction_name,
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
            direction_name=s.direction_name,
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
