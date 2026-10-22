"""Cooperative cancellation of long-running jobs (the progress modal's Cancel).

Covers the registry (`services/cancel_registry.py`), the `cancelJob`
mutation, and each cancellable job: the embedding pipeline (incl. re-embed),
LLM topic labeling and SAE document activations. A cancelled job stops at its
next checkpoint (batch / topic / document), keeps the work done so far, is
left resumable, and reports `CANCELLED_MESSAGE` as its error.
"""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from chromadb.errors import NotFoundError

from backend.API import mutations, schema
from backend.embedding_functions import embed_existing_dataset as reembed_module
from backend.embedding_functions.config import (
    EmbeddingModelConfig,
    EmbeddingProvider,
    EmbeddingResult,
    ReEmbedConfig,
)
from backend.services import (
    cancel_registry,
    embedding_pipeline,
    job_state,
    topic_extraction_service as topic_service,
)
from backend.services.cancel_registry import (
    CANCELLED_MESSAGE,
    cancellable_job,
    register_cancel_event,
    report_cancelled,
    request_cancel,
    unregister_cancel_event,
)
from backend.services.job_ids import JobIds
from backend.services.job_state import JobStateService, JobStatus
from backend.utils.embedding_model_catalog import TASK_PREFIX_FORMAT

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "embedding_visualization"


@pytest.fixture(autouse=True)
def job_states(tmp_path, monkeypatch):
    """Route every get_job_state_service() call to a temp state file, so no
    test touches the live resources/job_state.json."""
    service = JobStateService(state_file=tmp_path / "job_state.json")
    monkeypatch.setattr(job_state, "_job_state_service", service)
    return service


@pytest.fixture
def emitted(monkeypatch):
    """Progress events emitted by report_cancelled."""
    events: list[dict] = []
    monkeypatch.setattr(cancel_registry, "emit_progress_sync", lambda **kw: events.append(kw))
    return events


def _start(job_states: JobStateService, job_id: str) -> None:
    job_states.start_job(job_id, job_type="test", total_expected=10, total_batches=1, config={})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestCancelRegistry:
    def test_request_sets_registered_event(self):
        with cancellable_job("job_a") as event:
            assert not event.is_set()
            assert request_cancel("job_a") is True
            assert event.is_set()

    def test_request_unknown_job_returns_false(self):
        assert request_cancel("never_registered") is False

    def test_context_exit_unregisters(self):
        with cancellable_job("job_b"):
            pass
        assert request_cancel("job_b") is False

    def test_context_exit_unregisters_on_error(self):
        with pytest.raises(RuntimeError), cancellable_job("job_c"):
            raise RuntimeError("boom")
        assert request_cancel("job_c") is False

    def test_stale_exit_keeps_newer_registration(self):
        """A finished job must not unregister a newer job under the same id."""
        old = register_cancel_event("job_d")
        new = register_cancel_event("job_d")
        unregister_cancel_event("job_d", old)
        assert request_cancel("job_d") is True
        assert new.is_set() and not old.is_set()
        unregister_cancel_event("job_d", new)
        assert request_cancel("job_d") is False

    def test_report_cancelled_marks_job_resumable(self, job_states, emitted):
        _start(job_states, "job_e")
        report_cancelled("job_e", items_processed=4, total_items=10)

        job = job_states.get_job("job_e")
        assert job is not None and job.status is JobStatus.INTERRUPTED
        assert emitted == [
            {
                "job_id": "job_e",
                "status": "failed",
                "items_processed": 4,
                "total_items": 10,
                "current_batch": 0,
                "total_batches": 0,
                "error": CANCELLED_MESSAGE,
                "message": CANCELLED_MESSAGE,
            }
        ]

    def test_report_cancelled_not_resumable_closes_job(self, job_states, emitted):
        _start(job_states, "job_f")
        report_cancelled("job_f", items_processed=10, total_items=10, resumable=False)

        assert job_states.get_job("job_f") is None
        assert emitted[0]["error"] == CANCELLED_MESSAGE

    def test_frontend_mirrors_cancelled_message(self):
        """lib/utils/jobCancellation.ts matches this string to detect a cancel."""
        ts_source = (FRONTEND_DIR / "lib" / "utils" / "jobCancellation.ts").read_text()
        assert f"JOB_CANCELLED_MESSAGE = '{CANCELLED_MESSAGE}'" in ts_source


class TestCancelJobMutation:
    QUERY = "mutation Cancel($jobId: String!) { cancelJob(jobId: $jobId) }"

    def _cancel(self, job_id: str) -> bool:
        result = asyncio.run(schema.execute(self.QUERY, variable_values={"jobId": job_id}))
        assert result.errors is None, result.errors
        return result.data["cancelJob"]

    def test_signals_running_job(self):
        with cancellable_job("col_llm_labeling") as event:
            assert self._cancel("col_llm_labeling") is True
            assert event.is_set()

    def test_unknown_job_returns_false(self):
        assert self._cancel("col_llm_labeling") is False


# ---------------------------------------------------------------------------
# Embedding pipeline (shared by HF / local file / re-embed)
# ---------------------------------------------------------------------------


class _FakePipeline(embedding_pipeline.EmbeddingPipeline):
    """Embeds instantly; optionally 'clicks Cancel' while embedding."""

    def __init__(self, cancel_during_embed: bool, **kwargs):
        super().__init__(collection_name="col", **kwargs)
        self.cancel_during_embed = cancel_during_embed

    def _do_embed(self) -> EmbeddingResult:
        if self.cancel_during_embed:
            self.cancel_event.set()
        return EmbeddingResult(
            collection_name="col",
            total_embedded=5,
            embedding_dim=3,
            device="cpu",
            duration_seconds=0.1,
        )


class TestEmbeddingPipelineCancel:
    def test_cancel_during_embed_skips_projections(self, job_states, emitted, monkeypatch):
        _start(job_states, "col")

        def _no_projections(*args, **kwargs):
            raise AssertionError("projections must not run after a cancel")

        monkeypatch.setattr(
            embedding_pipeline, "compute_projections_for_collection", _no_projections
        )
        pipeline = _FakePipeline(cancel_during_embed=True, cancel_event=threading.Event())

        result = asyncio.run(pipeline.run())

        assert result.embedding_result.error == CANCELLED_MESSAGE
        assert result.projections_computed is False
        assert job_states.get_job("col").status is JobStatus.INTERRUPTED
        assert [e["status"] for e in emitted] == ["failed"]

    def test_cancel_during_projections_skips_topics(self, job_states, emitted, monkeypatch):
        _start(job_states, "col")
        event = threading.Event()

        def _projections(*args, **kwargs):
            event.set()  # user cancels while UMAP runs
            return True

        def _no_topics(*args, **kwargs):
            raise AssertionError("topic extraction must not run after a cancel")

        monkeypatch.setattr(embedding_pipeline, "compute_projections_for_collection", _projections)
        monkeypatch.setattr(embedding_pipeline, "do_extract_topics", _no_topics)
        pipeline = _FakePipeline(
            cancel_during_embed=False,
            cancel_event=event,
            extract_topics=True,
            topic_config=object(),
        )

        result = asyncio.run(pipeline.run())

        assert result.embedding_result.error == CANCELLED_MESSAGE
        assert result.projections_computed is True
        # The collection is usable (only topics were skipped): nothing to resume.
        assert job_states.get_job("col") is None
        assert [e["status"] for e in emitted] == ["failed"]

    def test_loop_reported_cancel_is_not_reported_twice(self, job_states, emitted, monkeypatch):
        """When the batch loop already reported the cancel, the pipeline just returns."""
        pipeline = _FakePipeline(cancel_during_embed=True, cancel_event=threading.Event())
        loop_result = EmbeddingResult(
            collection_name="col",
            total_embedded=2,
            embedding_dim=3,
            device="cpu",
            duration_seconds=0.1,
            error=CANCELLED_MESSAGE,
        )
        monkeypatch.setattr(pipeline, "_do_embed", lambda: loop_result)
        pipeline.cancel_event.set()

        result = asyncio.run(pipeline.run())

        assert result.embedding_result is loop_result
        assert emitted == []

    def test_reembed_pipeline_passes_cancel_event(self, monkeypatch):
        seen = {}

        def _fake_reembed(config, cancel_event=None):
            seen["event"] = cancel_event
            return EmbeddingResult(
                collection_name="col",
                total_embedded=0,
                embedding_dim=3,
                device="cpu",
                duration_seconds=0,
            )

        monkeypatch.setattr(embedding_pipeline, "do_reembed", _fake_reembed)
        event = threading.Event()
        pipeline = embedding_pipeline.ReEmbeddingPipeline(
            config=SimpleNamespace(collection_name="col"), cancel_event=event
        )
        pipeline._do_embed()
        assert seen["event"] is event


class _FakeReEmbedDB:
    def __init__(self, n_items: int):
        self.items = [{"id": f"i{k}", "document": f"document {k}"} for k in range(n_items)]

    def get_vector_collection(self, name):
        return None

    def get_dataset(self, name):
        return {"name": name}

    def get_filtered_items(self, name, filters, limit):
        return list(self.items)

    def register_vector_collection(self, *args, **kwargs):
        pass


class _FakeChromaCollection:
    def __init__(self, on_add):
        self.added: list[str] = []
        self._on_add = on_add

    def add(self, ids, embeddings):
        self.added.extend(ids)
        self._on_add()


class _FakeChromaClient:
    def __init__(self, collection):
        self._collection = collection

    def get_collection(self, **kwargs):
        raise NotFoundError("does not exist")  # what chromadb raises

    def create_collection(self, **kwargs):
        return self._collection


class TestReEmbedLoopCancel:
    def test_stops_between_batches(self, job_states, emitted, monkeypatch, tmp_path):
        event = threading.Event()
        collection = _FakeChromaCollection(on_add=event.set)  # cancel after batch 1

        monkeypatch.setattr(
            "backend.API.duckdb_instance.get_duckdb_client", lambda: _FakeReEmbedDB(n_items=6)
        )
        monkeypatch.setattr(reembed_module, "get_device", lambda: "cpu")
        monkeypatch.setattr(
            reembed_module,
            "create_embedding_function",
            lambda model_config, device: (lambda docs: [[0.0, 0.0, 0.0] for _ in docs], 3),
        )
        monkeypatch.setattr(reembed_module, "DB_PATH", tmp_path / "vector_db")
        monkeypatch.setattr(
            reembed_module.chromadb, "PersistentClient", lambda **kw: _FakeChromaClient(collection)
        )

        config = ReEmbedConfig(
            source_dataset_name="ds",
            collection_name="ds_new",
            embedding_model=EmbeddingModelConfig(),
            batch_size=2,
        )
        result = reembed_module.embed_existing_dataset(config, cancel_event=event)

        assert len(collection.added) == 2
        assert result.error == CANCELLED_MESSAGE
        assert result.total_embedded == 2
        assert job_states.get_job("ds_new").status is JobStatus.INTERRUPTED
        assert emitted[-1]["error"] == CANCELLED_MESSAGE


class TestReEmbedTaskPrefix:
    """The prefix row is the item's metadata plus ``__document__``, in both branches."""

    ITEMS = [
        {"id": "a", "document": "doc a", "metadata": {"title": "Alpha", "abstract": "abs a"}},
        # extract_metadata drops None values, so this item has no title key at all.
        {"id": "b", "document": "doc b", "metadata": {"abstract": "abs b"}},
    ]

    def _run(self, monkeypatch, tmp_path, prompt, columns=None):
        received: list[str] = []
        collection = _FakeChromaCollection(on_add=lambda: None)
        db = _FakeReEmbedDB(n_items=0)
        db.items = [dict(item) for item in self.ITEMS]
        monkeypatch.setattr("backend.API.duckdb_instance.get_duckdb_client", lambda: db)
        monkeypatch.setattr(reembed_module, "get_device", lambda: "cpu")

        def factory(model_config, device):
            def embed(docs):
                received.extend(docs)
                return [[0.0, 0.0, 0.0] for _ in docs]

            return embed, 3

        monkeypatch.setattr(reembed_module, "create_embedding_function", factory)
        monkeypatch.setattr(reembed_module, "DB_PATH", tmp_path / "vector_db")
        monkeypatch.setattr(
            reembed_module.chromadb, "PersistentClient", lambda **kw: _FakeChromaClient(collection)
        )
        config = ReEmbedConfig(
            source_dataset_name="ds",
            collection_name="ds_new",
            columns=columns,
            batch_size=10,
            embedding_model=EmbeddingModelConfig(
                provider=EmbeddingProvider.GEMINI,
                model_name="gemini-embedding-2",
                prompt=prompt,
                task="task: search result | query: ",
                prompt_format=TASK_PREFIX_FORMAT,
            ),
        )
        return reembed_module.embed_existing_dataset(config), received, collection

    def test_prefix_on_the_stored_documents(self, job_states, emitted, monkeypatch, tmp_path):
        result, received, _ = self._run(monkeypatch, tmp_path, "title: {title} | text: ")
        assert result.error is None
        assert sorted(received) == ["title: Alpha | text: doc a", "title: none | text: doc b"]

    def test_prefix_on_text_composed_from_metadata(
        self, job_states, emitted, monkeypatch, tmp_path
    ):
        result, received, _ = self._run(
            monkeypatch, tmp_path, "title: {title} | text: ", columns=["abstract"]
        )
        assert result.error is None
        assert sorted(received) == ["title: Alpha | text: abs a", "title: none | text: abs b"]

    def test_unknown_prefix_column_fails_before_any_vector(
        self, job_states, emitted, monkeypatch, tmp_path
    ):
        result, received, collection = self._run(
            monkeypatch, tmp_path, "title: {headline} | text: "
        )
        assert "headline" in result.error
        assert received == [] and collection.added == []


# ---------------------------------------------------------------------------
# LLM topic labeling
# ---------------------------------------------------------------------------


class _FakeTopicDB:
    """Three topics with keyword labels, one document each."""

    def __init__(self):
        self.saved: dict[int, str] = {}

    def get_active_topics(self, collection_name):
        return {
            "id": 1,
            "topic_hierarchy": None,
            "topics": [
                {
                    "topic_id": t,
                    "label": f"kw{t}_a_b",
                    "keywords": [{"word": f"kw{t}", "score": 1.0}],
                }
                for t in range(3)
            ],
        }

    def get_items_columns(self, collection_name, columns):
        return [(f"i{t}", f"document {t}") for t in range(3)]

    def get_topic_assignments_raw(self, extraction_id, columns):
        return [(f"i{t}", t, f"kw{t}_a_b", None, None) for t in range(3)]

    def update_topic_label(self, extraction_id, topic_id, label):
        self.saved[topic_id] = label


class TestLlmLabelingCancel:
    def test_stops_between_topics_and_keeps_saved_labels(self, job_states, monkeypatch):
        db = _FakeTopicDB()
        event = threading.Event()
        labeled: list[int] = []

        def _fake_label(topic_id, keywords, sample_documents, labeler):
            labeled.append(topic_id)
            event.set()  # user cancels while the first LLM call runs
            return f"Label {topic_id}"

        monkeypatch.setattr(topic_service, "_get_duckdb", lambda: db)
        monkeypatch.setattr(topic_service, "_create_labeler", lambda provider, model: object())
        monkeypatch.setattr(topic_service, "generate_llm_label_for_topic", _fake_label)

        result = topic_service.generate_llm_labels_for_collection(
            "col", label_scope="topics_only", cancel_event=event
        )

        assert labeled == [0]
        assert db.saved == {0: "Label 0"}
        assert result.error == CANCELLED_MESSAGE
        assert (result.topics_labeled, result.total_topics) == (1, 3)
        job = job_states.get_job(JobIds.llm_labeling("col"))
        assert job is not None and job.status is JobStatus.INTERRUPTED


# ---------------------------------------------------------------------------
# SAE document activations
# ---------------------------------------------------------------------------


class _FakeActivationsDB:
    def __init__(self, n_items: int):
        self.n_items = n_items
        self.inserted: list[dict] = []

    def get_dataset_name_for_collection(self, collection_name):
        return "ds"

    def get_dataset(self, dataset_name):
        return {
            "extra_metadata": {"sae_model_id": "gemma-3-4b-it", "sae_id": "9-gemmascope-2-res-16k"}
        }

    def get_filtered_items(self, dataset_name, filters, limit):
        return [{"id": f"d{k}", "document": f"doc {k}"} for k in range(self.n_items)]

    def get_document_activation_item_ids(self, collection_name):
        return set()

    def insert_document_activations_bulk(self, df):
        self.inserted.extend(df.to_dict("records"))


class _CancelWhileLoadingItemsDB(_FakeActivationsDB):
    """The user clicks Cancel while the resolver is still loading items."""

    def get_filtered_items(self, dataset_name, filters, limit):
        request_cancel("col_sae_activations")
        return super().get_filtered_items(dataset_name, filters, limit)


class _FakeInterpretService:
    """The user clicks Cancel after the first document."""

    def __init__(self, loaded: bool = True):
        self._lock = asyncio.Lock()
        self._neuronpedia_model_id = "gemma-3-4b-it"
        self.loaded = loaded
        self.batches_run = 0

    def get_status(self):
        return SimpleNamespace(loaded=self.loaded)

    def load_model(self, checkpoint):
        raise AssertionError("a cancelled job must not load the model")

    def run_batch_highlight(
        self, documents, layer, width, hook_type, progress_callback, result_callback, cancel_event
    ):
        self.batches_run += 1
        for done, (item_id, _text) in enumerate(documents, start=1):
            if cancel_event.is_set():
                break
            result_callback(item_id, [SimpleNamespace(feature_index=7, activation=1.5)])
            progress_callback(done, len(documents))
            request_cancel("col_sae_activations")


class TestDocumentActivationsCancel:
    QUERY = """
        mutation Compute($input: ComputeDocumentActivationsInput!) {
            computeDocumentActivations(input: $input) { itemsProcessed totalItems error }
        }
    """

    def _compute(self, monkeypatch, db, service) -> dict:
        monkeypatch.setattr(mutations, "get_duckdb_client", lambda: db)
        monkeypatch.setattr(mutations, "get_interpret_service", lambda: service)
        result = asyncio.run(
            schema.execute(self.QUERY, variable_values={"input": {"collectionName": "col"}})
        )
        assert result.errors is None, result.errors
        return result.data["computeDocumentActivations"]

    def test_stops_and_flushes_partial_results(self, monkeypatch, emitted):
        db = _FakeActivationsDB(n_items=4)

        data = self._compute(monkeypatch, db, _FakeInterpretService())

        assert data == {"itemsProcessed": 1, "totalItems": 4, "error": CANCELLED_MESSAGE}
        # The cancelled document batch is persisted (resume skips it).
        assert [row["item_id"] for row in db.inserted] == ["d0"]
        assert request_cancel("col_sae_activations") is False

    def test_cancel_before_inference_skips_model_load(self, monkeypatch, emitted):
        """A cancel while items load (or while waiting for the GPU) is honoured
        before the model load, not after it."""
        service = _FakeInterpretService(loaded=False)

        data = self._compute(monkeypatch, _CancelWhileLoadingItemsDB(n_items=4), service)

        assert data == {"itemsProcessed": 0, "totalItems": 4, "error": CANCELLED_MESSAGE}
        assert service.batches_run == 0
        assert emitted[-1]["error"] == CANCELLED_MESSAGE
