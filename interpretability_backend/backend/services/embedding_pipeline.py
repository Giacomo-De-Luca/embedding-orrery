"""Embedding pipeline with template method pattern.

Provides a shared run() flow: embed -> compute projections -> extract topics -> complete job.
Subclasses only override _do_embed() to provide HuggingFace or local file embedding.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..embed_dataset import (
    EmbeddingConfig,
    EmbeddingResult,
    LocalFileEmbeddingConfig,
    compute_projections_for_collection,
    embed_huggingface_dataset as do_hf_embed,
    embed_local_file as do_local_embed,
)
from .job_state import get_job_state_service
from .progress_emitter import emit_progress_sync
from .topic_extraction_service import (
    TopicExtractionConfig,
    extract_topics as do_extract_topics,
)

logger = logging.getLogger("star_map." + __name__)


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
    ):
        self.collection_name = collection_name
        self.compute_projections = compute_projections
        self.extract_topics = extract_topics
        self.topic_config = topic_config

    @abstractmethod
    def _do_embed(self) -> EmbeddingResult:
        """Run the actual embedding. Called in a background thread."""
        ...

    async def run(self) -> PipelineResult:
        """Execute the full embedding pipeline.

        Flow: embed -> projections -> topics -> complete job -> emit completion.
        """
        job_state = get_job_state_service()

        # Step 1: Embed
        result = await asyncio.to_thread(self._do_embed)

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
        return do_hf_embed(self.config)


class LocalFileEmbeddingPipeline(EmbeddingPipeline):
    """Pipeline for embedding local files."""

    def __init__(self, config: LocalFileEmbeddingConfig, **kwargs):
        super().__init__(collection_name=config.collection_name, **kwargs)
        self.config = config

    def _do_embed(self) -> EmbeddingResult:
        return do_local_embed(self.config)
