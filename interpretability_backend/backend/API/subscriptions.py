"""
GraphQL subscriptions for real-time progress and token streaming.

This module provides WebSocket-based subscriptions for monitoring
embedding job progress and streaming model generation in real-time.

Event buses live in services/progress_emitter.py and services/token_emitter.py
to avoid circular imports.
"""

import asyncio
import threading
import uuid
from collections.abc import AsyncGenerator

import strawberry

from ..services.interpret_service import SteeringSpec
from ..services.progress_emitter import (
    ProgressEvent,
    register_subscriber,
    unregister_subscriber,
)
from ..services.token_emitter import (
    TokenEvent,
    register_token_subscriber,
    unregister_token_subscriber,
)
from .interpret_instance import get_interpret_service
from .types import GenerateStreamInput, TokenChunk


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

    @strawberry.subscription
    async def generate_stream(
        self,
        input: GenerateStreamInput,
    ) -> AsyncGenerator[TokenChunk, None]:
        """Stream tokens from a multi-turn chat generation via WebSocket.

        The subscription starts generation, acquires the GPU lock for the
        duration, and yields TokenChunk events until the model produces
        an EOS/EOT token or reaches output_len.
        """
        service = get_interpret_service()
        stream_id = str(uuid.uuid4())
        queue: asyncio.Queue[TokenEvent] = asyncio.Queue(maxsize=500)
        await register_token_subscriber(stream_id, queue)

        if not input.turns:
            yield TokenChunk(
                stream_id=stream_id,
                token_index=0,
                token_id=0,
                text="",
                done=True,
                error="turns must not be empty",
            )
            return

        turns = [(t.role, t.content) for t in input.turns]
        cancel_event = threading.Event()
        task: asyncio.Task | None = None

        # Convert GraphQL SteeringInput list → service SteeringSpec list
        steering_specs: list[SteeringSpec] | None = None
        if input.steering:
            steering_specs = [
                SteeringSpec(
                    feature_index=s.feature_index,
                    layer=s.layer,
                    hook_type=s.hook_type.value,
                    width=s.width,
                    strength=s.strength,
                    direction_name=s.direction_name,
                )
                for s in input.steering
            ]

        try:
            async with service._lock:
                task = asyncio.ensure_future(
                    asyncio.to_thread(
                        service.generate_stream,
                        turns,
                        stream_id,
                        input.output_len,
                        input.temperature,
                        input.top_p,
                        input.top_k,
                        cancel_event=cancel_event,
                        steering_specs=steering_specs,
                        seed=input.seed,
                    )
                )
                while True:
                    # Timeout prevents holding the GPU lock forever
                    event = await asyncio.wait_for(queue.get(), timeout=300.0)
                    yield TokenChunk(
                        stream_id=event.stream_id,
                        token_index=event.token_index,
                        token_id=event.token_id,
                        text=event.text,
                        done=event.done,
                        error=event.error,
                    )
                    if event.done:
                        break
                await task
        except (asyncio.CancelledError, GeneratorExit):
            # Signal the model thread to stop after the current token
            cancel_event.set()
            # Wait for the GPU thread to finish before releasing the lock
            if task and not task.done():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            raise
        except TimeoutError:
            cancel_event.set()
            yield TokenChunk(
                stream_id=stream_id,
                token_index=0,
                token_id=0,
                text="",
                done=True,
                error="Generation timed out",
            )
        finally:
            # Wait for thread to complete if still running, then unregister
            if task and not task.done():
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            await unregister_token_subscriber(stream_id, queue)
