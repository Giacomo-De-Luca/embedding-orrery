"""Shared embedding batch job (embedding_functions/embedding_job.py).

Covers the helpers every embedder (HF, local text, re-embed, images, vectors)
now runs through: opening/resuming the ChromaDB collection, job-state and
progress bookkeeping, cooperative cancellation between batches, and failure
reporting, plus regression tests for resume on the local-text and vector paths.
"""

import importlib
import json
import threading

import pytest

from backend.embedding_functions.config import (
    DataType,
    EmbeddingModelConfig,
    LocalFileEmbeddingConfig,
)
from backend.embedding_functions.embedding_job import (
    EmbeddingBatchJob,
    open_embedding_collection,
    report_embedding_failure,
)
from backend.services import cancel_registry, job_state
from backend.services.cancel_registry import CANCELLED_MESSAGE
from backend.services.job_state import JobState, JobStateService, JobStatus

# The package re-exports functions with the same names as these submodules.
local_module = importlib.import_module("backend.embedding_functions.embed_local_file")
vectors_module = importlib.import_module("backend.embedding_functions.embed_vectors")


@pytest.fixture(autouse=True)
def job_states(tmp_path, monkeypatch):
    service = JobStateService(state_file=tmp_path / "job_state.json")
    monkeypatch.setattr(job_state, "_job_state_service", service)
    return service


@pytest.fixture
def emitted(monkeypatch):
    events: list[dict] = []
    record = lambda **kw: events.append(kw)  # noqa: E731
    monkeypatch.setattr(cancel_registry, "emit_progress_sync", record)
    monkeypatch.setattr("backend.embedding_functions.embedding_job.emit_progress_sync", record)
    return events


class FakeCollection:
    def __init__(self, ids=()):
        self.ids = list(ids)
        self.added: list[str] = []
        self.on_add = None

    def get(self, include):
        return {"ids": list(self.ids)}

    def add(self, ids, embeddings):
        self.ids.extend(ids)
        self.added.extend(ids)
        if self.on_add:
            self.on_add()


class FakeClient:
    def __init__(self, existing: FakeCollection | None = None):
        self.existing = existing
        self.deleted: list[str] = []
        self.created: list[dict] = []

    def get_collection(self, name, embedding_function=None):
        if self.existing is None:
            raise ValueError(f"Collection {name} does not exist")
        return self.existing

    def delete_collection(self, name):
        self.deleted.append(name)
        self.existing = None

    def create_collection(self, name, metadata, **kwargs):
        self.created.append({"name": name, "metadata": metadata, **kwargs})
        self.existing = FakeCollection()
        return self.existing


# ---------------------------------------------------------------------------
# Collection opening
# ---------------------------------------------------------------------------


class TestOpenEmbeddingCollection:
    def test_creates_when_missing_and_drops_none_metadata(self):
        client = FakeClient()

        collection, existing_ids = open_embedding_collection(
            client, "c", resume=False, metadata={"a": 1, "b": None}
        )

        assert existing_ids == set()
        assert client.created == [{"name": "c", "metadata": {"a": 1}}]
        assert collection is client.existing

    def test_overwrite_deletes_existing(self):
        client = FakeClient(existing=FakeCollection(["x"]))

        _, existing_ids = open_embedding_collection(client, "c", resume=False, metadata={})

        assert client.deleted == ["c"] and existing_ids == set()
        assert len(client.created) == 1

    def test_resume_reuses_existing_ids(self):
        existing = FakeCollection(["x", "y"])
        client = FakeClient(existing=existing)

        collection, existing_ids = open_embedding_collection(
            client, "c", resume=True, metadata={}, embedding_function="ef"
        )

        assert collection is existing and existing_ids == {"x", "y"}
        assert client.created == [] and client.deleted == []

    def test_passes_embedding_function_only_when_given(self):
        client = FakeClient()
        open_embedding_collection(client, "c", resume=False, metadata={}, embedding_function="ef")
        assert client.created[0]["embedding_function"] == "ef"


# ---------------------------------------------------------------------------
# Batch job bookkeeping
# ---------------------------------------------------------------------------


def _job(**overrides) -> EmbeddingBatchJob:
    kwargs = dict(
        collection_name="c",
        job_type="local_file",
        job_config={"file_path": "f"},
        n_rows=5,
        batch_size=2,
        embedding_dim=3,
        device="cpu",
        start_time=0.0,
        model_config=EmbeddingModelConfig(),
    )
    kwargs.update(overrides)
    return EmbeddingBatchJob(**kwargs)


class TestEmbeddingBatchJob:
    def test_runs_all_batches_and_reports_progress(self, job_states, emitted):
        seen = []
        job = _job(progress_callback=lambda done, total: seen.append((done, total)))
        job.start(already_embedded=0, resume=False)

        for start, end in job.batches():
            job.record(added=end - start, end=end)
        result = job.result()

        assert result.error is None and result.total_embedded == 5
        assert seen == [(2, 5), (4, 5), (5, 5)]
        stored = job_states.get_job("c")
        assert stored.status is JobStatus.RUNNING and stored.items_embedded == 5
        assert stored.batches_completed == 3 and stored.config == {"file_path": "f"}
        assert emitted[0]["message"] == "Starting embedding..."

    def test_resume_keeps_the_original_job_record(self, job_states, emitted):
        job_states.start_job(
            "c", job_type="local_file", total_expected=5, total_batches=3, config={"orig": True}
        )
        job_states.fail_job("c", "interrupted earlier")
        job = _job(job_config={"new": True})

        job.start(already_embedded=2, resume=True)

        stored = job_states.get_job("c")
        assert stored.config == {"orig": True}
        # Regression: a resumed job stayed "interrupted" while running, so the
        # Jobs list offered Resume for a live job.
        assert stored.status is JobStatus.RUNNING
        assert job.total_embedded == 2

    def test_cancel_stops_between_batches(self, job_states, emitted):
        event = threading.Event()
        job = _job(cancel_event=event)
        job.start(already_embedded=0, resume=False)

        processed = []
        for start, end in job.batches():
            processed.append((start, end))
            job.record(added=end - start, end=end)
            event.set()  # user cancels during the first batch
        result = job.result()

        assert processed == [(0, 2)]
        assert result.error == CANCELLED_MESSAGE and result.total_embedded == 2
        assert job_states.get_job("c").status is JobStatus.INTERRUPTED
        assert emitted[-1]["error"] == CANCELLED_MESSAGE

    def test_failure_marks_job_interrupted(self, job_states, emitted):
        job_states.start_job(
            "c", job_type="local_file", total_expected=5, total_batches=3, config={}
        )

        result = report_embedding_failure(
            "c", RuntimeError("model exploded"), start_time=0.0, device="cpu"
        )

        assert result.error == "model exploded" and result.total_embedded == 0
        assert job_states.get_job("c").status is JobStatus.INTERRUPTED
        assert emitted[-1]["status"] == "failed"


# ---------------------------------------------------------------------------
# Embedders: resume + cancel regressions
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_duckdb_sync(monkeypatch):
    synced: list[list[str]] = []
    for module in (local_module, vectors_module):
        monkeypatch.setattr(module, "sync_dataset_and_collection", lambda *a, **k: "ds")
        monkeypatch.setattr(module, "sync_items", lambda ds, ids, docs, metas: synced.append(ids))
    return synced


class TestLocalTextResume:
    def test_resume_with_existing_collection_embeds_only_new_rows(
        self, job_states, emitted, fake_duckdb_sync, monkeypatch
    ):
        """Regression: resuming used an unbound collection_metadata and crashed."""
        monkeypatch.setattr(
            local_module,
            "create_embedding_function",
            lambda model_config, device: (lambda docs: [[0.0, 1.0, 0.0] for _ in docs], 3),
        )
        rows = [{"id": f"r{k}", "text": f"text number {k}"} for k in range(4)]
        existing = FakeCollection(["r0", "r1"])
        config = LocalFileEmbeddingConfig(
            file_path="f.csv",
            collection_name="c",
            columns=["text"],
            id_column="id",
            batch_size=2,
            resume=True,
        )

        result = local_module.embed_text_from_local(
            FakeClient(existing=existing), config, rows, 4, "cpu", 0.0
        )

        assert result.error is None
        assert sorted(existing.added) == ["r2", "r3"]
        assert result.total_embedded == 4


class TestVectorEmbedCancelAndResume:
    def _config(self, resume: bool) -> LocalFileEmbeddingConfig:
        return LocalFileEmbeddingConfig(
            file_path="f.parquet",
            collection_name="vecs",
            data_type=DataType.VECTOR,
            vector_column="vector",
            id_column="id",
            resume=resume,
        )

    def test_cancel_then_resume_finishes_the_rest(
        self, job_states, emitted, fake_duckdb_sync, monkeypatch
    ):
        monkeypatch.setattr(vectors_module, "EMBEDDING_BATCH_SIZE", 2)
        rows = [{"id": f"v{k}", "vector": [float(k), 1.0], "text": f"t{k}"} for k in range(5)]
        event = threading.Event()
        client = FakeClient()

        first = vectors_module.embed_vectors(
            client, self._config(resume=False), rows, 5, "cpu", 0.0, cancel_event=event
        )
        # nothing cancelled yet: everything embeds
        assert first.error is None and first.total_embedded == 5

        client = FakeClient()
        collection_ids: list[str] = []

        def cancel_after_first_batch():
            collection_ids.extend(client.existing.added)
            event.set()

        original_create = client.create_collection

        def create_and_hook(**kwargs):
            collection = original_create(**kwargs)
            collection.on_add = cancel_after_first_batch
            return collection

        client.create_collection = create_and_hook
        cancelled = vectors_module.embed_vectors(
            client, self._config(resume=False), rows, 5, "cpu", 0.0, cancel_event=event
        )
        assert cancelled.error == CANCELLED_MESSAGE and cancelled.total_embedded == 2
        assert job_states.get_job("vecs").status is JobStatus.INTERRUPTED

        resumed = vectors_module.embed_vectors(
            client, self._config(resume=True), rows, 5, "cpu", 0.0, cancel_event=threading.Event()
        )
        assert resumed.error is None and resumed.total_embedded == 5
        assert sorted(client.existing.ids) == [f"v{k}" for k in range(5)]


class TestCollectionExtraMetadata:
    """The SAE link (and other collection settings) live on the dataset row."""

    def test_reembedded_collection_updates_its_source_dataset(self, db):
        db._conn.execute("INSERT INTO datasets (name, item_count) VALUES ('ds', 1)")
        db._conn.execute(
            "INSERT INTO vector_collections (collection_name, dataset_name, backend, vector_type)"
            " VALUES ('ds_gemini', 'ds', 'chromadb', 'dense')"
        )
        db.update_dataset("ds", extra_metadata={"default_color_scheme": "x"})

        dataset_name = db.update_collection_extra_metadata(
            "ds_gemini", {"sae_model_id": "m", "sae_id": "s"}
        )

        assert dataset_name == "ds"
        extra = db.get_dataset("ds")["extra_metadata"]
        extra = json.loads(extra) if isinstance(extra, str) else extra
        assert extra == {"default_color_scheme": "x", "sae_model_id": "m", "sae_id": "s"}


class TestJobSource:
    def test_reembed_jobs_report_their_source_dataset(self):
        job = JobState.from_dict(
            {
                "collection_name": "ds_gemini",
                "status": "interrupted",
                "job_type": "re_embed",
                "items_embedded": 1,
                "total_expected": 2,
                "batches_completed": 1,
                "total_batches": 2,
                "config": {"source_dataset_name": "ds"},
                "started_at": "",
            }
        )

        assert job.source == "ds"
