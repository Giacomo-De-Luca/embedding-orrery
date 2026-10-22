"""The multimodal embedder (embedding_functions/embed_media.py).

One run per local-file or HuggingFace media job: detect the media columns,
check the token budget, load only the towers the media kinds need, decode
each row, embed text + media together, and store documents, media references
and vectors. Uses a fake ChromaDB client and a fake embedding function; the
images are real PNG bytes decoded with PIL.
"""

import importlib
import io
import json
import threading

import numpy as np
import pytest
from PIL import Image

from backend.embedding_functions.config import (
    DataType,
    EmbeddingModelConfig,
    LocalFileEmbeddingConfig,
)
from backend.services import cancel_registry, job_state
from backend.services.cancel_registry import CANCELLED_MESSAGE
from backend.services.job_state import JobStateService, JobStatus
from backend.utils.embedding_model_catalog import TASK_PREFIX_FORMAT
from backend.utils.media_kinds import MediaKind
from backend.utils.media_settings import MediaEmbeddingSettings, MediaOptions
from backend.utils.truncation import TruncationTracker

media_module = importlib.import_module("backend.embedding_functions.embed_media")
MediaSource = media_module.MediaSource
MultimodalEmbeddingRun = media_module.MultimodalEmbeddingRun

EG2 = "google/embeddinggemma-2"


def _png(size=(16, 12)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(10, 200, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def job_states(tmp_path, monkeypatch):
    service = JobStateService(state_file=tmp_path / "job_state.json")
    monkeypatch.setattr(job_state, "_job_state_service", service)
    return service


@pytest.fixture(autouse=True)
def emitted(monkeypatch):
    events: list[dict] = []
    record = lambda **kw: events.append(kw)  # noqa: E731
    monkeypatch.setattr(cancel_registry, "emit_progress_sync", record)
    monkeypatch.setattr("backend.embedding_functions.embedding_job.emit_progress_sync", record)
    monkeypatch.setattr(media_module, "emit_progress_sync", record)
    return events


@pytest.fixture(autouse=True)
def synced(monkeypatch):
    stored = {"datasets": [], "items": []}
    monkeypatch.setattr(
        media_module,
        "sync_dataset_and_collection",
        lambda name, metadata, **kw: stored["datasets"].append(metadata) or "ds",
    )
    monkeypatch.setattr(
        media_module,
        "sync_items",
        lambda ds, ids, docs, metas: stored["items"].extend(zip(ids, docs, metas, strict=True)),
    )
    return stored


class FakeCollection:
    def __init__(self, ids=()):
        self.ids = list(ids)
        self.added: list[str] = []
        self.on_add = None

    def get(self, include):
        return {"ids": list(self.ids)}

    def add(self, ids, embeddings):
        assert len(ids) == len(embeddings)
        self.ids.extend(ids)
        self.added.extend(ids)
        if self.on_add:
            self.on_add()


class FakeClient:
    def __init__(self, existing=None):
        self.existing = existing
        self.created: list[dict] = []

    def get_collection(self, name, embedding_function=None):
        from chromadb.errors import NotFoundError

        if self.existing is None:
            raise NotFoundError(name)
        return self.existing

    def create_collection(self, name, metadata, **kwargs):
        self.created.append({"name": name, "metadata": metadata})
        self.existing = FakeCollection()
        return self.existing


class FakeMediaEF:
    def __init__(self):
        self.batches: list[dict] = []
        self.required: list[tuple] = []
        self.text_caps: list[int] = []
        self.truncation = TruncationTracker(None)

    def cap_text_tokens(self, limit):
        self.text_caps.append(limit)
        self.truncation = TruncationTracker(limit)

    def embed_inputs(self, inputs, has_text, processing_kwargs=None, batch_size=None):
        self.batches.append(
            {
                "inputs": list(inputs),
                "has_text": list(has_text),
                "processing_kwargs": processing_kwargs,
                "batch_size": batch_size,
            }
        )
        return [np.ones(4, dtype=np.float32) for _ in inputs]

    def require_modalities(self, kinds, with_text):
        self.required.append((tuple(kinds), with_text))


class Factory:
    def __init__(self):
        self.ef = FakeMediaEF()
        self.calls: list[dict] = []

    def __call__(self, model_config, device, **kwargs):
        self.calls.append({"model": model_config.model_name, **kwargs})
        return self.ef, 4


def _config(**overrides) -> LocalFileEmbeddingConfig:
    values = dict(
        file_path="/data/pets.parquet",
        collection_name="pets",
        data_type=DataType.MULTIMODAL,
        media_columns=["img"],
        columns=["caption"],
        embedding_model=EmbeddingModelConfig(model_name=EG2),
        media_options=MediaOptions(),
        batch_size=2,
    )
    values.update(overrides)
    return LocalFileEmbeddingConfig(**values)


def _rows(n=3):
    return [{"caption": f"pet {i}", "img": _png(), "owner": f"o{i}"} for i in range(n)]


def _run(rows, config=None, client=None, factory=None, cancel_event=None, settings=None):
    factory = factory or Factory()
    run = MultimodalEmbeddingRun(
        client=client or FakeClient(),
        config=config or _config(),
        rows=rows,
        total=len(rows),
        device="cpu",
        start_time=0.0,
        source=MediaSource(
            job_type="local_file",
            collection_metadata={"description": "pets", "source_file": "/data/pets.parquet"},
            auto_id=lambda row, row_idx: f"pets_{row_idx}",
        ),
        cancel_event=cancel_event,
        settings=settings or MediaEmbeddingSettings(),
        embedding_function_factory=factory,
    )
    return run.run(), factory


class TestRun:
    def test_embeds_text_and_images_together(self, synced):
        result, factory = _run(_rows(3))

        assert result.error is None and result.total_embedded == 3
        inputs = [item for batch in factory.ef.batches for item in batch["inputs"]]
        assert [list(i) for i in inputs] == [["text", "image"]] * 3
        assert inputs[0]["text"] == "pet 0"
        assert isinstance(inputs[0]["image"], Image.Image)
        assert factory.ef.batches[0]["processing_kwargs"] == {"image": {"max_soft_tokens": 280}}
        assert factory.ef.batches[0]["batch_size"] == MediaEmbeddingSettings().encode_batch_size

    def test_text_is_capped_at_what_the_media_leave(self):
        _, factory = _run(_rows(1))
        settings = MediaEmbeddingSettings()
        assert factory.ef.text_caps == [
            settings.text_token_budget(settings.defaults, {MediaKind.IMAGE: 1})
        ]

    def test_truncated_text_is_reported(self):
        factory = Factory()
        original = factory.ef.embed_inputs

        def embed_and_truncate(inputs, has_text, **kwargs):
            factory.ef.truncation.observe([10**6] * len(inputs))
            return original(inputs, has_text, **kwargs)

        factory.ef.embed_inputs = embed_and_truncate
        result, _ = _run(_rows(2), factory=factory)
        assert any("truncated" in warning for warning in result.warnings)

    def test_loads_only_the_needed_towers(self):
        _, factory = _run(_rows(1))
        assert factory.calls[0]["modalities"] == ("text", "image")
        assert factory.ef.required == [(("image",), True)]

    def test_collection_metadata_describes_the_media(self, synced):
        client = FakeClient()
        _run(_rows(1), client=client)

        metadata = client.created[0]["metadata"]
        assert metadata["data_type"] == "multimodal"
        assert json.loads(metadata["media_columns"]) == {"img": "image"}
        assert json.loads(metadata["media_options"])["image_max_soft_tokens"] == 280
        assert json.loads(metadata["embedded_columns"]) == ["caption"]
        assert metadata["embedding_model"] == EG2
        assert synced["datasets"][0]["data_type"] == "multimodal"

    def test_items_store_references_never_bytes(self, synced):
        _run(_rows(2))

        ids = [item[0] for item in synced["items"]]
        documents = [item[1] for item in synced["items"]]
        metadatas = [item[2] for item in synced["items"]]
        assert ids == ["pets_0", "pets_1"] and documents == ["pet 0", "pet 1"]
        assert metadatas[0]["img"].startswith("[image ") and metadatas[0]["owner"] == "o0"
        assert all(not isinstance(v, bytes) for meta in metadatas for v in meta.values())

    def test_media_only_rows_are_labelled(self, synced):
        rows = [{"img": _png(), "path": "x"}]
        _run(rows, config=_config(columns=None))
        assert synced["items"][0][1] == "image"

    def test_task_prefix_goes_on_the_text(self):
        model = EmbeddingModelConfig(
            model_name=EG2, prompt="title: none | text: ", prompt_format=TASK_PREFIX_FORMAT
        )
        _, factory = _run(_rows(1), config=_config(embedding_model=model))
        assert factory.ef.batches[0]["inputs"][0]["text"] == "title: none | text: pet 0"

    def test_undecodable_rows_are_skipped_and_reported(self, synced):
        rows = _rows(3)
        rows[1]["img"] = b"\x89PNG\r\n\x1a\nbroken"

        result, _ = _run(rows)

        assert result.error is None and result.total_embedded == 2
        assert [item[0] for item in synced["items"]] == ["pets_0", "pets_2"]
        assert any("1 row" in warning for warning in result.warnings)

    def test_id_column(self, synced):
        rows = [
            {"caption": "a", "img": _png(), "pid": "x"},
            {"caption": "b", "img": _png(), "pid": "x"},
        ]
        _run(rows, config=_config(id_column="pid"))
        assert [item[0] for item in synced["items"]] == ["x", "x_1"]


class TestValidation:
    def test_budget_overflow_fails_before_the_model_loads(self):
        rows = [{"a": "a.mp4", "b": "b.mp4"}]
        factory = Factory()
        with pytest.raises(ValueError, match="tokens"):
            _run(rows, config=_config(media_columns=["a", "b"], columns=None), factory=factory)
        assert factory.calls == []

    def test_columns_without_media_fail(self):
        with pytest.raises(ValueError, match="'caption'"):
            _run(_rows(1), config=_config(media_columns=["caption"]))


class TestResumeAndCancel:
    def test_resume_skips_embedded_rows(self, synced):
        client = FakeClient(existing=FakeCollection(["pets_0"]))
        result, _ = _run(_rows(3), config=_config(resume=True), client=client)
        assert client.existing.added == ["pets_1", "pets_2"] and result.total_embedded == 3

    def test_cancel_stops_between_batches(self, job_states):
        event = threading.Event()
        client = FakeClient()
        original = client.create_collection

        def create_and_hook(**kwargs):
            collection = original(**kwargs)
            collection.on_add = event.set
            return collection

        client.create_collection = create_and_hook
        result, _ = _run(_rows(4), client=client, cancel_event=event)

        assert result.error == CANCELLED_MESSAGE and result.total_embedded == 2
        assert job_states.get_job("pets").status is JobStatus.INTERRUPTED
