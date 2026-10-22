"""Truncation warnings during an embedding run: shown live, kept in the result, never stopping it."""

import asyncio

from backend.API.subscriptions import JobProgress
from backend.embedding_functions import embedding_job
from backend.embedding_functions.embedding_job import EmbeddingBatchJob
from backend.services import progress_emitter
from backend.utils.truncation import TruncationTracker


class FakeJobState:
    def start_job(self, **kwargs):
        pass

    def resume_job(self, collection_name):
        pass

    def update_progress(self, **kwargs):
        pass


class TrackingEF:
    def __init__(self, max_length: int):
        self.truncation = TruncationTracker(max_length)


def _job(monkeypatch, source) -> tuple[EmbeddingBatchJob, list[dict]]:
    events: list[dict] = []
    monkeypatch.setattr(embedding_job, "get_job_state_service", lambda: FakeJobState())
    monkeypatch.setattr(embedding_job, "emit_progress_sync", lambda **kw: events.append(kw))
    job = EmbeddingBatchJob(
        collection_name="c",
        job_type="huggingface",
        job_config={},
        n_rows=4,
        batch_size=2,
        embedding_dim=2,
        device="cpu",
        start_time=0.0,
        truncation_source=source,
    )
    job.start(already_embedded=0, resume=False)
    return job, events


def _run(job, ef, lengths_per_batch):
    for (_, end), lengths in zip(job.batches(), lengths_per_batch, strict=True):
        if ef is not None:
            ef.truncation.observe(lengths)
        job.record(added=len(lengths), end=end)
    return job.result()


def test_truncation_warns_once_per_increase_and_the_run_completes(monkeypatch):
    ef = TrackingEF(max_length=3)
    job, events = _job(monkeypatch, ef)

    result = _run(job, ef, [[5, 1], [1, 1]])

    warnings = [e["warning"] for e in events if e.get("warning")]
    assert len(warnings) == 1 and "1 of 2 documents" in warnings[0]
    assert result.error is None and result.total_embedded == 4
    assert len(result.warnings) == 1 and "1 of 4 documents" in result.warnings[0]


def test_no_warning_without_truncation(monkeypatch):
    ef = TrackingEF(max_length=3)
    job, events = _job(monkeypatch, ef)

    result = _run(job, ef, [[1, 2], [3, 3]])

    assert not any(e.get("warning") for e in events)
    assert result.warnings == []


def test_sources_without_a_tracker_are_ignored(monkeypatch):
    job, events = _job(monkeypatch, lambda docs: [[0.0] for _ in docs])

    result = _run(job, None, [[1, 1], [1, 1]])

    assert result.warnings == [] and result.error is None


def test_progress_events_carry_the_warning_to_subscribers():
    async def scenario():
        queue: asyncio.Queue = asyncio.Queue()
        await progress_emitter.register_subscriber("job-w", queue)
        try:
            progress_emitter.emit_progress(
                job_id="job-w",
                status="running",
                items_processed=1,
                total_items=2,
                current_batch=1,
                total_batches=2,
                warning="3 of 10 documents exceeded the limit",
            )
            return queue.get_nowait()
        finally:
            await progress_emitter.unregister_subscriber("job-w", queue)

    event = asyncio.run(scenario())

    assert event.warning == "3 of 10 documents exceeded the limit"
    assert (
        JobProgress(
            job_id="j",
            status="running",
            items_processed=0,
            total_items=0,
            current_batch=0,
            total_batches=0,
        ).warning
        is None
    )
