"""Embedding pipeline with template method pattern.

Provides a shared run() flow: embed -> compute projections -> extract topics -> complete job.
Subclasses only override _do_embed() to provide HuggingFace or local file embedding.
"""

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from ..embed_dataset import (
    EmbeddingConfig,
    EmbeddingResult,
    LocalFileEmbeddingConfig,
    compute_projections_for_collection,
    embed_huggingface_dataset as do_hf_embed,
    embed_local_file as do_local_embed,
)
from ..embedding_functions.config import ReEmbedConfig
from ..embedding_functions.embed_existing_dataset import embed_existing_dataset as do_reembed
from .cancel_registry import CANCELLED_MESSAGE, is_cancelled, report_cancelled
from .job_state import get_job_state_service
from .progress_emitter import emit_progress_sync
from .topic_extraction_service import (
    TopicExtractionConfig,
    extract_topics as do_extract_topics,
)

logger = logging.getLogger("orrery." + __name__)


@dataclass
class PipelineResult:
    """Result of the full embedding pipeline."""

    embedding_result: EmbeddingResult
    projections_computed: bool


class EmbeddingPipeline(ABC):
    """Abstract base class for embedding pipelines.

    Template method pattern: run() orchestrates the full flow,
    subclasses provide _do_embed() implementation.
    """

    def __init__(
        self,
        collection_name: str,
        compute_projections: bool = True,
        extract_topics: bool = False,
        topic_config: TopicExtractionConfig | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self.collection_name = collection_name
        self.compute_projections = compute_projections
        self.extract_topics = extract_topics
        self.topic_config = topic_config
        self.cancel_event = cancel_event

    @abstractmethod
    def _do_embed(self) -> EmbeddingResult:
        """Run the actual embedding. Called in a background thread."""
        ...

    async def run(self) -> PipelineResult:
        """Execute the full embedding pipeline.

        Flow: embed -> projections -> topics -> complete job -> emit completion.
        A cancel request is honoured between batches (inside _do_embed) and
        between steps; projections and topic extraction run to completion
        once started.
        """
        job_state = get_job_state_service()

        # Step 1: Embed
        result = await asyncio.to_thread(self._do_embed)

        if is_cancelled(self.cancel_event):
            return self._cancelled(result, projections_computed=False)

        # Step 2: Compute projections
        projections_computed = False
        if self.compute_projections and result.error is None and result.total_embedded > 0:
            emit_progress_sync(
                job_id=self.collection_name,
                status="running",
                items_processed=result.total_embedded,
                total_items=result.total_embedded,
                current_batch=0,
                total_batches=0,
                message="Computing projections (PCA/UMAP)...",
            )
            projections_computed = await asyncio.to_thread(
                compute_projections_for_collection,
                self.collection_name,
                job_id=self.collection_name,
            )

        if is_cancelled(self.cancel_event):
            return self._cancelled(result, projections_computed=projections_computed)

        # Step 3: Extract topics
        if self.extract_topics and projections_computed and result.error is None:
            await self._run_topic_extraction()

        # Step 4: Mark job complete
        job_state.complete_job(self.collection_name)

        # Step 5: Emit completion
        emit_progress_sync(
            job_id=self.collection_name,
            status="completed",
            items_processed=result.total_embedded,
            total_items=result.total_embedded,
            current_batch=0,
            total_batches=0,
            message="Complete!",
        )

        return PipelineResult(
            embedding_result=result,
            projections_computed=projections_computed,
        )

    def _cancelled(self, result: EmbeddingResult, projections_computed: bool) -> PipelineResult:
        """Stop the pipeline after a cancel request.

        The batch loops report a cancel they catch themselves (result.error is
        then already set); a cancel that lands after the last batch is
        reported here. The job stays resumable while projections are still
        missing (Resume recomputes them); once they exist the collection is
        usable and only topic extraction was skipped, so the job is closed.
        """
        if result.error is None:
            projections_missing = (
                self.compute_projections and result.total_embedded > 0 and not projections_computed
            )
            report_cancelled(
                self.collection_name,
                items_processed=result.total_embedded,
                total_items=result.total_embedded,
                resumable=projections_missing,
            )
            result = replace(result, error=CANCELLED_MESSAGE)
        return PipelineResult(embedding_result=result, projections_computed=projections_computed)

    async def _run_topic_extraction(self) -> bool:
        """Run topic extraction for the collection."""
        if self.topic_config is None:
            return False
        try:
            extraction_result = await asyncio.to_thread(do_extract_topics, self.topic_config)
            return extraction_result.error is None
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return False


class HuggingFaceEmbeddingPipeline(EmbeddingPipeline):
    """Pipeline for embedding HuggingFace datasets."""

    def __init__(self, config: EmbeddingConfig, **kwargs):
        super().__init__(collection_name=config.collection_name, **kwargs)
        self.config = config

    def _do_embed(self) -> EmbeddingResult:
        return do_hf_embed(self.config, cancel_event=self.cancel_event)


class LocalFileEmbeddingPipeline(EmbeddingPipeline):
    """Pipeline for embedding local files."""

    def __init__(self, config: LocalFileEmbeddingConfig, **kwargs):
        super().__init__(collection_name=config.collection_name, **kwargs)
        self.config = config

    def _do_embed(self) -> EmbeddingResult:
        return do_local_embed(self.config, cancel_event=self.cancel_event)


class ReEmbeddingPipeline(EmbeddingPipeline):
    """Pipeline for re-embedding an existing dataset with a new model."""

    def __init__(self, config: ReEmbedConfig, **kwargs):
        super().__init__(collection_name=config.collection_name, **kwargs)
        self.config = config

    def _do_embed(self) -> EmbeddingResult:
        return do_reembed(self.config, cancel_event=self.cancel_event)
