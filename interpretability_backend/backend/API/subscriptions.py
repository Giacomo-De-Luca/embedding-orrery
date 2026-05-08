"""
GraphQL subscriptions for real-time progress updates.

This module provides WebSocket-based subscriptions for monitoring
embedding job progress in real-time.

The actual event bus and emission logic is in services/progress_emitter.py
to avoid circular imports when embedding functions need to emit progress.
"""

import asyncio
from collections.abc import AsyncGenerator

import strawberry

from ..services.progress_emitter import (
    ProgressEvent,
    register_subscriber,
    unregister_subscriber,
)


@strawberry.type
class JobProgress:
    """Real-time progress update for an embedding job (GraphQL type)."""

    job_id: str  # collection_name
    status: str  # "running", "completed", "failed"
    items_processed: int
    total_items: int
    current_batch: int
    total_batches: int
    error: str | None = None
    message: str | None = None  # Status message (e.g., "Sorting batches", "Loading model")


@strawberry.type
class Subscription:
    """GraphQL subscription root for real-time updates."""

    @strawberry.subscription
    async def embedding_progress(self, job_id: str) -> AsyncGenerator[JobProgress, None]:
        """
        Subscribe to real-time progress updates for an embedding job.

        The subscription will emit JobProgress events as the embedding
        job processes batches. It will complete when the job finishes
        (status becomes "completed" or "failed").

        Args:
            job_id: The collection name / job identifier to monitor

        Yields:
            JobProgress events with current progress information
        """
        queue: asyncio.Queue[ProgressEvent] = asyncio.Queue(maxsize=100)

        # Register this subscriber with the shared event bus
        await register_subscriber(job_id, queue)

        try:
            while True:
                # Wait for next progress update
                event = await queue.get()

                # Convert ProgressEvent to GraphQL JobProgress type
                progress = JobProgress(
                    job_id=event.job_id,
                    status=event.status,
                    items_processed=event.items_processed,
                    total_items=event.total_items,
                    current_batch=event.current_batch,
                    total_batches=event.total_batches,
                    error=event.error,
                    message=event.message,
                )
                yield progress

                # Stop if job completed or failed
                if event.status in ("completed", "failed"):
                    break
        finally:
            # Always unregister when done
            await unregister_subscriber(job_id, queue)
