"""Cancel event registry for long-running jobs.

Provides a global, thread-safe registry of threading.Event objects keyed by
job id (the progress job id: the collection name for embedding jobs,
``{collection}_llm_labeling`` for LLM labeling, ``{collection}_sae_activations``
for SAE document activations). A mutation registers an event for the job's
lifetime (``cancellable_job``), the ``cancelJob`` mutation looks it up and sets
it, and the job's work loop checks it between batches for cooperative
cancellation. Work done before the cancel is kept, and the job is left
resumable.
"""

import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from .job_state import get_job_state_service
from .progress_emitter import emit_progress_sync

# Error reported by a cancelled job (mutation results and the failed progress
# event). The frontend matches it (lib/utils/jobCancellation.ts) to tell a
# cancel apart from a failure.
CANCELLED_MESSAGE = "Cancelled by user"

_active_cancel_events: dict[str, threading.Event] = {}
# Ids no job may start under (their collection is being deleted).
_reserved_job_ids: set[str] = set()
_lock = threading.Lock()


class JobIdReservedError(RuntimeError):
    """A job cannot start: its id is reserved while its collection is deleted."""


class JobIdsActiveError(RuntimeError):
    """Ids could not be reserved: jobs are running under them."""

    def __init__(self, job_ids: list[str]) -> None:
        super().__init__(", ".join(job_ids))
        self.job_ids = job_ids


class JobCancelled(BaseException):  # noqa: N818 - mirrors asyncio.CancelledError
    """Raised inside a job's work to unwind after a cancel request.

    A BaseException, like ``asyncio.CancelledError``: the pipelines' broad
    ``except Exception`` fallbacks (degrade to projection coords, skip the
    centroid refresh, the SAE runner's catch-all) must not swallow a cancel
    and carry on. Job entry points catch it explicitly.
    """

    def __init__(self) -> None:
        super().__init__(CANCELLED_MESSAGE)


def register_cancel_event(job_id: str, event: threading.Event | None = None) -> threading.Event:
    """Store a cancel event for a job (a new one unless ``event`` is given). Returns it.

    Raises ``JobIdReservedError`` while the id is reserved (``reserved_job_ids``).
    """
    event = event or threading.Event()
    with _lock:
        if job_id in _reserved_job_ids:
            raise JobIdReservedError(
                f"Job '{job_id}' cannot start while its collection is being deleted"
            )
        _active_cancel_events[job_id] = event
    return event


@contextmanager
def reserved_job_ids(job_ids: Iterable[str]) -> Iterator[None]:
    """Keep jobs from starting under ``job_ids`` while the block runs.

    Raises ``JobIdsActiveError`` (naming them) when jobs already run under any
    of the ids, and ``JobIdReservedError`` when another block holds one.
    Checked and reserved under one lock, so no job slips in between.
    """
    job_ids = tuple(job_ids)
    with _lock:
        held = [job_id for job_id in job_ids if job_id in _reserved_job_ids]
        if held:
            raise JobIdReservedError(f"Already being deleted: {', '.join(held)}")
        active = [job_id for job_id in job_ids if job_id in _active_cancel_events]
        if active:
            raise JobIdsActiveError(active)
        _reserved_job_ids.update(job_ids)
    try:
        yield
    finally:
        with _lock:
            _reserved_job_ids.difference_update(job_ids)


def request_cancel(job_id: str) -> bool:
    """Signal cancellation for a running job. Returns True if a job was found."""
    with _lock:
        event = _active_cancel_events.get(job_id)
    if event is not None:
        event.set()
        return True
    return False


def is_active(job_id: str) -> bool:
    """Whether a job is registered under ``job_id`` (registered while it runs)."""
    with _lock:
        return job_id in _active_cancel_events


def unregister_cancel_event(job_id: str, event: threading.Event | None = None) -> None:
    """Remove the cancel event after a job finishes.

    With ``event``, only removes the registration if it is still that event,
    so a job finishing late cannot unregister a newer job under the same id.
    """
    with _lock:
        if event is None or _active_cancel_events.get(job_id) is event:
            _active_cancel_events.pop(job_id, None)


@contextmanager
def cancellable_job(job_id: str, event: threading.Event | None = None) -> Iterator[threading.Event]:
    """Register a cancel event for the duration of a job.

    Pass ``event`` to make a running job cancellable under a second id too
    (prepareSaeData's collection phase also answers to the collection's id,
    which is what the Jobs list cancels by).
    """
    event = register_cancel_event(job_id, event)
    try:
        yield event
    finally:
        unregister_cancel_event(job_id, event)


def is_cancelled(event: threading.Event | None) -> bool:
    """True when a cancel was requested (None means the job is not cancellable)."""
    return event is not None and event.is_set()


def raise_if_cancelled(event: threading.Event | None) -> None:
    """Raise JobCancelled when a cancel was requested."""
    if is_cancelled(event):
        raise JobCancelled()


def report_cancelled(
    job_id: str,
    *,
    items_processed: int,
    total_items: int,
    current_batch: int = 0,
    total_batches: int = 0,
    resumable: bool = True,
) -> None:
    """Record a cancelled job and emit a failed progress event.

    ``resumable`` leaves the job state interrupted so the Jobs list offers
    Resume; ``resumable=False`` closes it, for a cancel that only skipped
    optional work after the output was already usable.
    """
    job_states = get_job_state_service()
    if resumable:
        job_states.fail_job(job_id, CANCELLED_MESSAGE)
    else:
        job_states.complete_job(job_id)
    emit_cancelled(
        job_id,
        items_processed=items_processed,
        total_items=total_items,
        current_batch=current_batch,
        total_batches=total_batches,
    )


def emit_cancelled(
    job_id: str,
    *,
    items_processed: int = 0,
    total_items: int = 0,
    current_batch: int = 0,
    total_batches: int = 0,
) -> None:
    """Emit the failed progress event of a cancelled job, leaving job state alone
    (for jobs without a job-state record, or whose caller owns it)."""
    emit_progress_sync(
        job_id=job_id,
        status="failed",
        items_processed=items_processed,
        total_items=total_items,
        current_batch=current_batch,
        total_batches=total_batches,
        error=CANCELLED_MESSAGE,
        message=CANCELLED_MESSAGE,
    )
