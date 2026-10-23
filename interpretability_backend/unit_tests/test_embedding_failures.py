"""Results of embeddings that stop before embedding anything (``EmbeddingResult.failed``).

Every embedder returns the same shape when its input can't be embedded: nothing
embedded, the error, the model (when the run has one) and the time spent. Each
early return is pinned here through the embedder's public entry point.
"""

import importlib

import pytest

from backend.embedding_functions.config import (
    TEXT_EMBEDDING_DIMENSIONS,
    DataType,
    EmbeddingConfig,
    EmbeddingModelConfig,
    EmbeddingResult,
    LocalFileEmbeddingConfig,
    ReEmbedConfig,
)
from backend.embedding_functions.embedding_job import report_embedding_failure
from backend.services import job_state
from backend.services.job_state import JobStateService, JobStatus

local_module = importlib.import_module("backend.embedding_functions.embed_local_file")
vectors_module = importlib.import_module("backend.embedding_functions.embed_vectors")
hf_module = importlib.import_module("backend.embedding_functions.embed_huggingface")
reembed_module = importlib.import_module("backend.embedding_functions.embed_existing_dataset")

DEFAULT_MODEL = EmbeddingModelConfig()
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40


def _assert_failed(result, error, *, model=True, embedding_dim=0, collection="c"):
    assert result.collection_name == collection
    assert result.total_embedded == 0 and result.embedding_dim == embedding_dim
    assert result.device == "cpu" and result.duration_seconds >= 0
    assert result.error == error if isinstance(error, str) else error(result.error)
    expected = (DEFAULT_MODEL.provider.value, DEFAULT_MODEL.model_name) if model else (None, None)
    assert (result.embedding_provider, result.embedding_model) == expected
    assert result.warnings == []


class TestFailed:
    def test_records_the_model(self, monkeypatch):
        monkeypatch.setattr("backend.embedding_functions.config.time.time", lambda: 12.5)

        result = EmbeddingResult.failed(
            "c", "boom", device="cpu", start_time=10.0, model_config=DEFAULT_MODEL
        )

        assert result == EmbeddingResult(
            collection_name="c",
            total_embedded=0,
            embedding_dim=0,
            device="cpu",
            duration_seconds=2.5,
            error="boom",
            embedding_provider=DEFAULT_MODEL.provider.value,
            embedding_model=DEFAULT_MODEL.model_name,
        )

    def test_without_a_model(self):
        result = EmbeddingResult.failed("c", "boom", device="cpu", start_time=0.0)
        assert result.embedding_provider is None and result.embedding_model is None

    def test_keeps_a_known_dimension(self):
        result = EmbeddingResult.failed(
            "c", "boom", device="cpu", start_time=0.0, embedding_dim=384
        )
        assert result.embedding_dim == 384


class TestLocalFile:
    def test_no_rows(self, monkeypatch):
        monkeypatch.setattr(local_module, "get_device", lambda: "cpu")
        monkeypatch.setattr(local_module, "load_local_file_portion", lambda **kw: ([], 0))

        result = local_module.embed_local_file(
            LocalFileEmbeddingConfig(file_path="/data/x.csv", collection_name="c")
        )

        _assert_failed(
            result, "No rows loaded from file", model=False, embedding_dim=TEXT_EMBEDDING_DIMENSIONS
        )

    def test_no_text_columns(self):
        result = local_module.embed_text_from_local(
            None,
            LocalFileEmbeddingConfig(file_path="/data/x.csv", collection_name="c"),
            rows=[{"n": 1}],
            total=1,
            device="cpu",
            start_time=0.0,
        )
        _assert_failed(result, "No text columns found")

    def test_bytes_columns_are_never_picked_as_text(self):
        """The loader keeps bytes as bytes now; as a str(bytes) dump they were
        auto-picked as text and embedded as the document."""
        result = local_module.embed_text_from_local(
            None,
            LocalFileEmbeddingConfig(file_path="/data/x.parquet", collection_name="c"),
            rows=[{"img": PNG, "n": 1}],
            total=1,
            device="cpu",
            start_time=0.0,
        )
        _assert_failed(result, "No text columns found")


class TestReportEmbeddingFailure:
    def test_keeps_the_model_and_dimension(self, tmp_path, monkeypatch):
        states = JobStateService(state_file=tmp_path / "job_state.json")
        monkeypatch.setattr(job_state, "_job_state_service", states)
        monkeypatch.setattr(
            "backend.embedding_functions.embedding_job.emit_progress_sync", lambda **kw: None
        )
        states.start_job("c", job_type="local_file", total_expected=5, total_batches=3, config={})

        result = report_embedding_failure(
            "c",
            RuntimeError("boom"),
            start_time=0.0,
            device="cpu",
            model_config=DEFAULT_MODEL,
            embedding_dim=TEXT_EMBEDDING_DIMENSIONS,
        )

        _assert_failed(result, "boom", embedding_dim=TEXT_EMBEDDING_DIMENSIONS)
        assert states.get_job("c").status is JobStatus.INTERRUPTED


class TestVectors:
    def _run(self, rows):
        config = LocalFileEmbeddingConfig(
            file_path="/data/x.csv", collection_name="c", data_type=DataType.VECTOR
        )
        return vectors_module.embed_vectors(
            None, config, rows=rows, total=len(rows), device="cpu", start_time=0.0
        )

    def test_no_vector_column(self):
        _assert_failed(
            self._run([{"text": "a"}]), "No vector column found or specified", model=False
        )

    def test_unparsable_vector(self, monkeypatch):
        monkeypatch.setattr(vectors_module, "_is_vector_like", lambda value: True)
        _assert_failed(
            self._run([{"vec": "not a vector"}]),
            lambda error: error.startswith("Could not parse vector from column 'vec'"),
            model=False,
        )


class TestHuggingFace:
    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch):
        monkeypatch.setattr(hf_module, "get_device", lambda: "cpu")

    def _run(self, monkeypatch, rows):
        monkeypatch.setattr(hf_module, "load_dataset_portion", lambda **kw: (list(rows), len(rows)))
        return hf_module.embed_huggingface_dataset(
            EmbeddingConfig(dataset_id="org/x", collection_name="c")
        )

    def test_no_rows(self, monkeypatch):
        _assert_failed(self._run(monkeypatch, []), "No rows loaded from dataset")

    def test_no_text_columns(self, monkeypatch):
        _assert_failed(self._run(monkeypatch, [{"n": 1}]), "No text columns found in dataset")


class _FakeDB:
    def __init__(self, dataset, items):
        self.dataset, self.items = dataset, items

    def get_vector_collection(self, name):
        return None

    def get_dataset(self, name):
        return self.dataset

    def get_filtered_items(self, name, filters, limit):
        return list(self.items)


class TestReEmbed:
    def _run(self, monkeypatch, db):
        monkeypatch.setattr("backend.API.duckdb_instance.get_duckdb_client", lambda: db)
        monkeypatch.setattr(reembed_module, "get_device", lambda: "cpu")
        return reembed_module.embed_existing_dataset(
            ReEmbedConfig(source_dataset_name="ds", collection_name="c", batch_size=2)
        )

    def test_dataset_not_found(self, monkeypatch):
        _assert_failed(
            self._run(monkeypatch, _FakeDB(None, [])), "Dataset 'ds' not found in DuckDB"
        )

    def test_no_items(self, monkeypatch):
        _assert_failed(
            self._run(monkeypatch, _FakeDB({"name": "ds"}, [])), "Dataset 'ds' has no items"
        )
