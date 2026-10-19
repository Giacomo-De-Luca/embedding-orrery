"""Batch-loop bookkeeping shared by every embedder.

HuggingFace, local text, re-embed, image and vector embedding all run the same
loop around their own row → (ids, documents, vectors) step:

- ``open_embedding_collection`` opens the ChromaDB collection: on resume it
  reuses the existing one and returns the ids already embedded, otherwise it
  replaces any existing collection with a fresh one;
- ``EmbeddingBatchJob`` registers the job (or keeps the record of the job
  being resumed), yields batch bounds while checking the job's cancel event
  between batches, records progress after each batch (job state, WebSocket
  progress, optional callback) and builds the result: a cancelled run stops
  with the work done so far kept, the job left interrupted (resumable) and
  ``CANCELLED_MESSAGE`` as the error;
- ``report_embedding_failure`` turns an exception into the failed result,
  marking the job interrupted so it can be resumed.
"""

import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from tqdm import tqdm

from ..services.cancel_registry import CANCELLED_MESSAGE, is_cancelled, report_cancelled
from ..services.job_state import get_job_state_service
from ..services.progress_emitter import emit_progress_sync
from .config import EmbeddingModelConfig, EmbeddingResult


def open_embedding_collection(
    client,
    name: str,
    *,
    resume: bool,
    metadata: dict,
    embedding_function=None,
) -> tuple[Any, set[str]]:
    """Return ``(collection, ids already embedded)``.

    ``metadata`` (None values dropped, ChromaDB rejects them) is only used when
    a collection is created; a resumed collection keeps its own. Without an
    ``embedding_function`` the collection is created without one, as the
    image and vector paths always did.
    """
    try:
        existing = client.get_collection(name=name, embedding_function=embedding_function)
    except Exception:  # does not exist
        existing = None

    if existing is not None:
        if resume:
            ids = set(existing.get(include=[])["ids"])
            print(f"Resuming {name}: found {len(ids)} existing embeddings")
            return existing, ids
        client.delete_collection(name=name)
        print(f"Deleted existing collection: {name}")

    kwargs = {"embedding_function": embedding_function} if embedding_function is not None else {}
    collection = client.create_collection(
        name=name,
        metadata={k: v for k, v in metadata.items() if v is not None},
        **kwargs,
    )
    print(f"Created collection: {name}")
    return collection, set()


class EmbeddingBatchJob:
    """One embedding run: job state, per-batch progress, cancellation, result.

    Usage::

        job = EmbeddingBatchJob(...)
        job.start(already_embedded=len(existing_ids), resume=config.resume)
        for start, end in job.batches():
            ...embed rows[start:end], write them...
            job.record(added=len(ids), end=end)
        return job.result()
    """

    def __init__(
        self,
        *,
        collection_name: str,
        job_type: str,
        job_config: dict,
        n_rows: int,
        batch_size: int,
        embedding_dim: int,
        device: str,
        start_time: float,
        model_config: EmbeddingModelConfig | None = None,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        description: str = "Embedding batches",
    ):
        self.collection_name = collection_name
        self.job_type = job_type
        self.job_config = job_config
        self.n_rows = n_rows
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.start_time = start_time
        self.model_config = model_config
        self.cancel_event = cancel_event
        self.progress_callback = progress_callback
        self.description = description

        self.total_batches = (n_rows + batch_size - 1) // batch_size
        self.total_embedded = 0
        self.new_embedded = 0
        self.batches_completed = 0
        self.cancelled = False
        self._job_state = get_job_state_service()

    def start(self, already_embedded: int, resume: bool) -> None:
        """Register the job, or mark the one this run resumes running again
        (keeping its original record)."""
        self.total_embedded = already_embedded
        if resume and already_embedded > 0:
            self._job_state.resume_job(self.collection_name)
        else:
            self._job_state.start_job(
                collection_name=self.collection_name,
                job_type=self.job_type,
                total_expected=self.n_rows,
                total_batches=self.total_batches,
                config=self.job_config,
            )
        self._emit(status="running", current_batch=0, message="Starting embedding...")

    def batches(self) -> Iterator[tuple[int, int]]:
        """Yield ``(start, end)`` row bounds, stopping at a cancel request."""
        for start in tqdm(
            range(0, self.n_rows, self.batch_size), desc=self.description, unit="batch"
        ):
            if is_cancelled(self.cancel_event):
                self.cancelled = True
                report_cancelled(
                    self.collection_name,
                    items_processed=self.total_embedded,
                    total_items=self.n_rows,
                    current_batch=self.batches_completed,
                    total_batches=self.total_batches,
                )
                return
            yield start, min(start + self.batch_size, self.n_rows)

    def record(self, added: int, end: int) -> None:
        """Account for one finished batch that wrote ``added`` items."""
        self.total_embedded += added
        self.new_embedded += added
        self.batches_completed += 1
        self._job_state.update_progress(
            collection_name=self.collection_name,
            items_embedded=self.total_embedded,
            batches_completed=self.batches_completed,
        )
        self._emit(status="running", current_batch=self.batches_completed)
        if self.progress_callback:
            self.progress_callback(end, self.n_rows)

    def result(self) -> EmbeddingResult:
        duration = time.time() - self.start_time
        if not self.cancelled:
            # Embedding done; the pipeline's projections/topics follow on this job id.
            self._emit(status="progress", current_batch=self.total_batches)
            print(
                f"Embedded {self.new_embedded} new items ({self.total_embedded} total) "
                f"into {self.collection_name} in {duration:.2f}s"
            )
        return EmbeddingResult(
            collection_name=self.collection_name,
            total_embedded=self.total_embedded,
            embedding_dim=self.embedding_dim,
            device=self.device,
            duration_seconds=duration,
            error=CANCELLED_MESSAGE if self.cancelled else None,
            embedding_provider=self.model_config.provider.value if self.model_config else None,
            embedding_model=self.model_config.model_name if self.model_config else None,
        )

    def _emit(self, *, status: str, current_batch: int, message: str | None = None) -> None:
        emit_progress_sync(
            job_id=self.collection_name,
            status=status,
            items_processed=self.total_embedded,
            total_items=self.n_rows,
            current_batch=current_batch,
            total_batches=self.total_batches,
            message=message,
        )


def report_embedding_failure(
    collection_name: str,
    error: Exception,
    *,
    start_time: float,
    device: str,
    model_config: EmbeddingModelConfig | None = None,
    embedding_dim: int = 0,
) -> EmbeddingResult:
    """Failed result for an embedder exception; the job stays resumable."""
    get_job_state_service().fail_job(collection_name, str(error))
    emit_progress_sync(
        job_id=collection_name,
        status="failed",
        items_processed=0,
        total_items=0,
        current_batch=0,
        total_batches=0,
        error=str(error),
    )
    return EmbeddingResult(
        collection_name=collection_name,
        total_embedded=0,
        embedding_dim=embedding_dim,
        device=device,
        duration_seconds=time.time() - start_time,
        error=str(error),
        embedding_provider=model_config.provider.value if model_config else None,
        embedding_model=model_config.model_name if model_config else None,
    )
