"""
Progress emission for embedding jobs.

This module provides the event bus and emission functions for real-time
progress updates. It's separate from API/subscriptions.py to avoid
circular imports when embedding functions need to emit progress.

The subscriptions module will import from here to register subscribers.
"""

import asyncio
from dataclasses import dataclass


@dataclass
class ProgressEvent:
    """Progress event data (plain dataclass, not Strawberry type)."""

    job_id: str  # collection_name
    status: str  # "running", "completed", "failed"
    items_processed: int
    total_items: int
    current_batch: int
    total_batches: int
    error: str | None = None
    message: str | None = None  # Status message (e.g., "Sorting batches", "Loading model")


# Global event bus for progress subscribers
# Maps collection_name -> list of subscriber queues
_progress_subscribers: dict[str, list] = {}
_lock = asyncio.Lock()


async def register_subscriber(job_id: str, queue: asyncio.Queue) -> None:
    """Register a subscriber for job progress updates."""
    async with _lock:
        if job_id not in _progress_subscribers:
            _progress_subscribers[job_id] = []
        _progress_subscribers[job_id].append(queue)


async def unregister_subscriber(job_id: str, queue: asyncio.Queue) -> None:
    """Unregister a subscriber for job progress updates."""
    async with _lock:
        if job_id in _progress_subscribers:
            try:
                _progress_subscribers[job_id].remove(queue)
                if not _progress_subscribers[job_id]:
                    del _progress_subscribers[job_id]
            except ValueError:
                pass  # Queue not found, already removed


def emit_progress(
    job_id: str,
    status: str,
    items_processed: int,
    total_items: int,
    current_batch: int,
    total_batches: int,
    error: str | None = None,
    message: str | None = None,
) -> None:
    """
    Emit a progress update to all subscribers for a job.

    This function is thread-safe and can be called from synchronous code.

    Args:
        job_id: The collection name / job identifier
        status: Job status ("running", "completed", "failed")
        items_processed: Number of items processed so far
        total_items: Total number of items to process
        current_batch: Current batch number
        total_batches: Total number of batches
        error: Optional error message if status is "failed"
        message: Optional status message (e.g., "Sorting batches", "Loading model")
    """
    if job_id not in _progress_subscribers:
        return  # No subscribers, skip

    event = ProgressEvent(
        job_id=job_id,
        status=status,
        items_processed=items_processed,
        total_items=total_items,
        current_batch=current_batch,
        total_batches=total_batches,
        error=error,
        message=message,
    )

    # Broadcast to all subscribers
    for queue in _progress_subscribers.get(job_id, []):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop update if queue is full


# Alias for backwards compatibility with embedding functions
emit_progress_sync = emit_progress
