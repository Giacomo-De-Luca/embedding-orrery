"""Cancellation of the remaining long-running steps.

Projections, topic extraction (incl. its LLM labels), topic reduction, topic
evaluation and SAE preparation stop on a cancel, persist nothing past the
cancel point, and report CANCELLED_MESSAGE. Heavy kernels run through
CancellableTask; the isolated (child-process) path is exercised end to end for
projections and clustering so the kernels are known to import and run in a
spawned child. Also covers the embedding pipeline's topic step and SAE link,
and the mutations registering each job under its JobIds id.
"""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.API import mutations, schema
from backend.embedding_functions.config import EmbeddingResult
from backend.services import (
    cancel_registry,
    embedding_pipeline,
    job_state,
    sae_pipeline_service,
    topic_quality_service,
)
from backend.services import topic_extraction_service as topic_service
from backend.services.cancel_registry import CANCELLED_MESSAGE, JobCancelled, request_cancel
from backend.services.job_ids import JobIds
from backend.services.job_state import JobStateService
from backend.topic_extraction import llm_labeling
from backend.utils import cancellable_task, compute_projections
from backend.utils.cancellable_task import CancellableTaskSettings

INLINE = CancellableTaskSettings(isolate_min_rows=10**9)
ISOLATED = CancellableTaskSettings(isolate_min_rows=0, spill_min_bytes=0, poll_interval_ms=20)


@pytest.fixture(autouse=True)
def job_states(tmp_path, monkeypatch):
    service = JobStateService(state_file=tmp_path / "job_state.json")
    monkeypatch.setattr(job_state, "_job_state_service", service)
    return service


@pytest.fixture(autouse=True)
def emitted(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(cancel_registry, "emit_progress_sync", lambda **kw: events.append(kw))
    return events


def _use_settings(monkeypatch, settings: CancellableTaskSettings) -> None:
    monkeypatch.setattr(cancellable_task, "SETTINGS", settings)


def _blobs(n_per: int = 30, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(7)
    centers = rng.normal(scale=6.0, size=(3, dim))
    return np.vstack([c + rng.normal(scale=0.3, size=(n_per, dim)) for c in centers])


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


class FakeProjectionDB:
    def __init__(self, on_insert=None):
        self.inserted: dict[str, np.ndarray] = {}
        self.on_insert = on_insert

    def get_vector_collection(self, name):
        return {"dataset_name": name}

    def insert_projections_batch(self, collection, ids, projection_type, coords):
        self.inserted[projection_type] = np.asarray(coords)
        if self.on_insert:
            self.on_insert()

    def upsert_projection_metadata(self, collection, projection_type, variance=None):
        pass


def _fake_loader(vectors: np.ndarray):
    def iter_batches(collection_name, batch_size=5000):
        ids = [f"i{k}" for k in range(len(vectors))]
        yield ids[:40], vectors[:40]
        yield ids[40:], vectors[40:]

    return iter_batches


class TestProjections:
    def test_cancel_between_projections_keeps_finished_ones(self, monkeypatch):
        _use_settings(monkeypatch, INLINE)
        event = threading.Event()
        db = FakeProjectionDB(on_insert=event.set)  # cancel once PCA 2D is stored
        monkeypatch.setattr(compute_projections, "_get_duckdb", lambda: db)
        monkeypatch.setattr(compute_projections, "iter_embedding_batches", _fake_loader(_blobs()))

        with pytest.raises(JobCancelled):
            compute_projections.compute_projections_for_collection(
                "c", ["pca2d", "pca3d"], cancel_event=event
            )
        assert list(db.inserted) == ["pca_2d"]

    def test_cancel_while_loading_embeddings(self, monkeypatch):
        _use_settings(monkeypatch, INLINE)
        event = threading.Event()
        vectors = _blobs()

        def iter_batches(collection_name, batch_size=5000):
            yield [f"i{k}" for k in range(40)], vectors[:40]
            event.set()
            yield [f"i{k}" for k in range(40, 90)], vectors[40:]

        db = FakeProjectionDB()
        monkeypatch.setattr(compute_projections, "_get_duckdb", lambda: db)
        monkeypatch.setattr(compute_projections, "iter_embedding_batches", iter_batches)

        with pytest.raises(JobCancelled):
            compute_projections.compute_projections_for_collection("c", cancel_event=event)
        assert db.inserted == {}

    def test_isolated_child_computes_projections(self, monkeypatch):
        _use_settings(monkeypatch, ISOLATED)
        vectors = _blobs()
        db = FakeProjectionDB()
        monkeypatch.setattr(compute_projections, "_get_duckdb", lambda: db)
        monkeypatch.setattr(compute_projections, "iter_embedding_batches", _fake_loader(vectors))

        assert compute_projections.compute_projections_for_collection("c", ["pca2d", "pca3d"])
        assert db.inserted["pca_2d"].shape == (len(vectors), 2)
        assert db.inserted["pca_3d"].shape == (len(vectors), 3)


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------


class FakeTopicExtractionDB:
    def __init__(self, coords: np.ndarray, on_load=None):
        self.coords = coords
        self.on_load = on_load

    def get_projection_data(self, collection_name, projection_type):
        if self.on_load:
            self.on_load()
        n = len(self.coords)
        return {
            "ids": [f"i{k}" for k in range(n)],
            "documents": [f"cluster{k // 30} word{k % 5} text" for k in range(n)],
            "coordinates": self.coords.tolist(),
        }


@pytest.fixture
def synced(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        topic_service, "_sync_topics_to_duckdb", lambda **kw: calls.append(kw) or "ext-1"
    )
    monkeypatch.setattr(topic_service, "_refresh_topic_centroids", lambda *a, **k: None)
    return calls


def _extraction_config(**overrides):
    config = topic_service.TopicExtractionConfig(
        collection_name="c", min_topic_size=5, cluster_on="projection", projection_type="umap_2d"
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestTopicExtraction:
    def test_cancel_before_clustering_stores_nothing(self, monkeypatch, synced, emitted):
        _use_settings(monkeypatch, INLINE)
        event = threading.Event()
        db = FakeTopicExtractionDB(_blobs(dim=2), on_load=event.set)
        monkeypatch.setattr(topic_service, "_get_duckdb", lambda: db)

        result = topic_service.extract_topics(_extraction_config(), cancel_event=event)

        assert result.error == CANCELLED_MESSAGE
        assert synced == []
        assert emitted[-1]["error"] == CANCELLED_MESSAGE

    def test_isolated_clustering_runs_in_a_child(self, monkeypatch, synced):
        _use_settings(monkeypatch, ISOLATED)
        db = FakeTopicExtractionDB(_blobs(dim=2))
        monkeypatch.setattr(topic_service, "_get_duckdb", lambda: db)

        result = topic_service.extract_topics(_extraction_config(), cancel_event=threading.Event())

        assert result.error is None and result.num_topics >= 2
        assert len(synced) == 1

    def test_llm_labels_stop_between_topics(self, monkeypatch):
        event = threading.Event()
        calls: list[int] = []

        def _label(topic_id, keywords, sample_documents, labeler):
            calls.append(topic_id)
            event.set()
            return f"Label {topic_id}"

        monkeypatch.setattr(llm_labeling, "_create_labeler", lambda provider, model: object())
        monkeypatch.setattr(llm_labeling, "generate_llm_label_for_topic", _label)
        documents_df = pd.DataFrame({"Document_ID": [0, 1], "Document": ["a", "b"], "Topic": [0, 1]})

        with pytest.raises(JobCancelled):
            llm_labeling.generate_llm_labels(
                {0: [("a", 1.0)], 1: [("b", 1.0)]}, documents_df, cancel_event=event
            )
        assert calls == [0]


# ---------------------------------------------------------------------------
# Topic reduction
# ---------------------------------------------------------------------------


class FakeReductionDB:
    def __init__(self, on_items=None):
        self.on_items = on_items

    def get_active_topics(self, collection_name):
        return {
            "id": "ext-0",
            "topics": [
                {"topic_id": t, "label": f"t{t}", "keywords": [{"word": f"w{t}", "score": 1.0}]}
                for t in range(4)
            ],
        }

    def get_items_columns(self, collection_name, columns):
        if self.on_items:
            self.on_items()
        return [(f"i{k}", f"word{k % 4} doc") for k in range(20)]

    def get_topic_assignments_raw(self, extraction_id, columns):
        return [(f"i{k}", k % 4) for k in range(20)]


class TestTopicReduction:
    def test_cancel_before_reducing_stores_nothing(self, monkeypatch, synced, emitted):
        event = threading.Event()
        monkeypatch.setattr(topic_service, "_get_duckdb", lambda: FakeReductionDB(event.set))

        result = topic_service.reduce_existing_topics(
            "c",
            method="fixed_n",
            n_topics=2,
            use_ctfidf=True,
            regenerate_labels=False,
            llm_provider="gemini",
            llm_model="m",
            cancel_event=event,
        )

        assert result.error == CANCELLED_MESSAGE
        assert synced == []
        assert emitted[-1]["job_id"] == JobIds.topic_reduction("c")


# ---------------------------------------------------------------------------
# Topic evaluation
# ---------------------------------------------------------------------------


class FakeEvaluationDB:
    def __init__(self, on_projection=None):
        self.on_projection = on_projection
        self.persisted: list = []

    def get_active_topics(self, collection_name):
        return {"id": "ext-0", "config": {"projection_type": "umap_2d"}}

    def get_projection_data(self, collection_name, projection_type):
        if self.on_projection:
            self.on_projection()
        coords = _blobs(dim=2)
        return {
            "ids": [f"i{k}" for k in range(len(coords))],
            "documents": [f"cluster{k // 30} text" for k in range(len(coords))],
            "coordinates": coords.tolist(),
        }

    def get_topic_assignments_raw(self, extraction_id, columns):
        return [(f"i{k}", k // 30) for k in range(90)]

    def update_topic_quality_metrics(self, extraction_id, level, result):
        self.persisted.append(result)


class TestTopicEvaluation:
    def test_cancel_returns_cancelled_and_persists_nothing(self, monkeypatch, emitted):
        _use_settings(monkeypatch, INLINE)
        event = threading.Event()
        db = FakeEvaluationDB(on_projection=event.set)

        result = topic_quality_service.score_topic_quality(
            "c", metrics={"silhouette"}, duckdb=db, cancel_event=event
        )

        assert result["error"] == CANCELLED_MESSAGE
        assert db.persisted == []

    def test_completes_without_cancel(self, monkeypatch):
        _use_settings(monkeypatch, INLINE)
        db = FakeEvaluationDB()

        result = topic_quality_service.score_topic_quality(
            "c", metrics={"silhouette"}, duckdb=db, cancel_event=threading.Event()
        )

        assert "error" not in result and len(db.persisted) == 1


# ---------------------------------------------------------------------------
# SAE preparation
# ---------------------------------------------------------------------------


class _FakeRunner:
    """Stands in for SAEPipelineRunner: reports download progress batch by batch."""

    on_progress = None

    def __init__(self, config):
        self.config = config

    def run(self, progress_callback=None):
        for done in range(1, 4):
            progress_callback("download:download_features", done, 3)
            if _FakeRunner.on_progress:
                _FakeRunner.on_progress()
        return SimpleNamespace(error=None, features_parquet=None, activations_jsonl=None)


class TestSaePrepare:
    def test_cancel_during_download_skips_ingestion(self, monkeypatch, emitted):
        event = threading.Event()
        _FakeRunner.on_progress = event.set
        monkeypatch.setattr(sae_pipeline_service, "SAEPipelineRunner", _FakeRunner)
        monkeypatch.setattr(
            sae_pipeline_service,
            "_ingest_features",
            lambda *a, **k: pytest.fail("a cancelled preparation must not ingest"),
        )

        result = sae_pipeline_service.prepare_sae_data(
            layer=9, job_id="sae_job", cancel_event=event, isolate=False
        )

        assert result["status"] == "failed" and result["error"] == CANCELLED_MESSAGE
        assert emitted[-1]["error"] == CANCELLED_MESSAGE


# ---------------------------------------------------------------------------
# Embedding pipeline: topic step + SAE link
# ---------------------------------------------------------------------------


class _InstantPipeline(embedding_pipeline.EmbeddingPipeline):
    def _do_embed(self) -> EmbeddingResult:
        return EmbeddingResult(
            collection_name="col", total_embedded=5, embedding_dim=3, device="cpu", duration_seconds=0
        )


class TestPipelineSteps:
    def _pipeline(self, monkeypatch, topics_error=None, **kwargs):
        monkeypatch.setattr(
            embedding_pipeline, "compute_projections_for_collection", lambda *a, **k: True
        )
        seen = {}

        def _topics(config, cancel_event=None):
            seen["cancel_event"] = cancel_event
            return SimpleNamespace(error=topics_error)

        monkeypatch.setattr(embedding_pipeline, "do_extract_topics", _topics)
        linked: list = []
        monkeypatch.setattr(
            embedding_pipeline, "link_collection_to_sae", lambda name, link: linked.append(link)
        )
        pipeline = _InstantPipeline(
            collection_name="col",
            extract_topics=True,
            topic_config=object(),
            cancel_event=threading.Event(),
            **kwargs,
        )
        return pipeline, seen, linked

    def test_topic_step_receives_the_cancel_event(self, monkeypatch, job_states):
        pipeline, seen, _ = self._pipeline(monkeypatch)
        asyncio.run(pipeline.run())
        assert seen["cancel_event"] is pipeline.cancel_event

    def test_cancelled_topics_close_the_job(self, monkeypatch, job_states):
        job_states.start_job("col", job_type="t", total_expected=5, total_batches=1, config={})
        pipeline, _, linked = self._pipeline(
            monkeypatch,
            topics_error=CANCELLED_MESSAGE,
            sae_link={"sae_model_id": "m", "sae_id": "s"},
        )

        result = asyncio.run(pipeline.run())

        assert result.embedding_result.error == CANCELLED_MESSAGE
        assert job_states.get_job("col") is None
        assert linked == []

    def test_links_the_sae_after_a_successful_run(self, monkeypatch, job_states):
        link = {"sae_model_id": "m", "sae_id": "s"}
        pipeline, _, linked = self._pipeline(monkeypatch, sae_link=link)

        result = asyncio.run(pipeline.run())

        assert result.embedding_result.error is None
        assert linked == [link]


# ---------------------------------------------------------------------------
# Mutations register each job under its JobIds id
# ---------------------------------------------------------------------------


def _execute(query: str, variables: dict) -> dict:
    result = asyncio.run(schema.execute(query, variable_values=variables))
    assert result.errors is None, result.errors
    return result.data


def _cancel_from_inside(job_id: str) -> None:
    """Simulate the user's Cancel click while the job runs."""
    assert request_cancel(job_id), f"{job_id} was not registered as cancellable"


class TestMutationRegistration:
    def test_extract_topics(self, monkeypatch):
        seen = {}

        def _extract(config, cancel_event=None):
            _cancel_from_inside(JobIds.topic_extraction("c"))
            seen["set"] = cancel_event.is_set()
            return topic_service.TopicExtractionResult(
                collection_name="c", num_topics=0, num_noise_points=0, topics=[],
                duration_seconds=0, error=CANCELLED_MESSAGE,
            )

        monkeypatch.setattr(mutations, "do_extract_topics", _extract)
        data = _execute(
            "mutation($c: String!) { extractTopics(input: {collectionName: $c}) { error } }",
            {"c": "c"},
        )
        assert seen["set"] and data["extractTopics"]["error"] == CANCELLED_MESSAGE

    def test_reduce_topics(self, monkeypatch):
        seen = {}

        def _reduce(**kwargs):
            _cancel_from_inside(JobIds.topic_reduction("c"))
            seen["set"] = kwargs["cancel_event"].is_set()
            return topic_service.TopicExtractionResult(
                collection_name="c", num_topics=0, num_noise_points=0, topics=[],
                duration_seconds=0, error=CANCELLED_MESSAGE,
            )

        monkeypatch.setattr(topic_service, "reduce_existing_topics", _reduce)
        _execute(
            "mutation($c: String!) { reduceTopics(input: {collectionName: $c, method: \"auto\"})"
            " { error } }",
            {"c": "c"},
        )
        assert seen["set"]

    def test_evaluate_topics(self, monkeypatch):
        seen = {}

        def _score(collection_name, **kwargs):
            _cancel_from_inside(JobIds.topic_evaluation("c"))
            seen["set"] = kwargs["cancel_event"].is_set()
            return {"error": CANCELLED_MESSAGE}

        monkeypatch.setattr(topic_quality_service, "score_topic_quality", _score)
        _execute(
            "mutation($c: String!) { evaluateTopics(input: {collectionName: $c}) { error } }",
            {"c": "c"},
        )
        assert seen["set"]

    def test_prepare_sae_data(self, monkeypatch):
        seen = {}

        def _prepare(**kwargs):
            _cancel_from_inside(JobIds.sae_prepare("4b", "it", 9, "resid_post", "16k"))
            seen["set"] = kwargs["cancel_event"].is_set()
            return {
                "model_id": "m", "sae_id": "s", "duration_seconds": 0.0,
                "status": "failed", "error": CANCELLED_MESSAGE,
            }

        monkeypatch.setattr(sae_pipeline_service, "prepare_sae_data", _prepare)
        data = _execute(
            "mutation { prepareSaeData(input: {layer: 9, width: \"16k\", hookType: \"resid_post\","
            " modelSize: \"4b\", variant: \"it\"}) { error status } }",
            {},
        )
        assert seen["set"] and data["prepareSaeData"]["error"] == CANCELLED_MESSAGE
