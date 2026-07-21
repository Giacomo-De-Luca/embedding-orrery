"""Tests for configuration-driven seed snapshots."""

import json
from pathlib import Path
from types import SimpleNamespace

import chromadb
import duckdb
import pytest
from chromadb.config import Settings

from interpretability_backend.backend.clients.duckdb_client import DuckDBClient
from interpretability_backend.backend.utils.seed_snapshot import (
    ChromaSnapshotExporter,
    DuckDBSnapshotExporter,
    HuggingFaceSnapshotDownloader,
    HuggingFaceSnapshotPublisher,
    SeedSnapshotBuilder,
    SeedSnapshotConfig,
    SeedSnapshotConfigError,
    SnapshotIntegrity,
)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "snapshot.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _payload() -> dict:
    return {
        "schema_version": 1,
        "name": "demo",
        "output": "output/demo",
        "collections": [
            {"name": "documents"},
            {
                "name": "feature-vectors",
                "include": {
                    "vectors": True,
                    "projections": False,
                    "topics": False,
                    "probes": False,
                },
            },
        ],
        "sae_data": [
            {
                "model_id": "model-a",
                "sae_id": "sae-a",
                "features": True,
                "activation_examples": False,
                "document_activations": ["documents"],
                "explanation_vector_collection": "feature-vectors",
            }
        ],
        "publish": {
            "repo_id_env": "TEST_SEED_REPO",
            "private": True,
            "path": "snapshots/demo",
        },
    }


def _snapshot_metadata(config: SeedSnapshotConfig, counts: dict | None = None) -> dict:
    return {
        "snapshot_name": config.name,
        "config_sha256": config.config_sha256,
        "counts": counts or {},
    }


def _disable_all_sae_payloads(payload: dict) -> None:
    payload["sae_data"][0].update(
        {
            "features": False,
            "activation_examples": False,
            "document_activations": [],
            "explanation_vector_collection": None,
        }
    )


def _add_duplicate_document_activation_owner(payload: dict) -> None:
    payload["sae_data"].append(
        {
            "model_id": "model-b",
            "sae_id": "sae-b",
            "features": True,
            "activation_examples": False,
            "document_activations": ["documents"],
            "explanation_vector_collection": None,
        }
    )


def test_config_parses_defaults_and_resolves_paths(tmp_path: Path) -> None:
    config = SeedSnapshotConfig.from_file(
        _write_config(tmp_path, _payload()), project_root=tmp_path
    )

    assert config.output_dir == tmp_path / "output/demo"
    assert config.collection_names == ("documents", "feature-vectors")
    assert config.collections[0].include.vectors is True
    assert config.collections[0].include.projections is True
    assert config.collections[0].include.topics is True
    assert config.collections[0].include.probes is False
    assert config.sae_data[0].document_activations == ("documents",)
    assert config.publish is not None
    assert config.publish.lock_file == tmp_path / "snapshot.lock.json"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data["collections"].append({"name": "documents"}), "duplicate collection"),
        (
            lambda data: data["sae_data"][0].update(
                {"document_activations": ["missing-collection"]}
            ),
            "missing-collection",
        ),
        (
            lambda data: data["sae_data"][0].update(
                {"features": False, "activation_examples": True}
            ),
            "requires features",
        ),
        (lambda data: data.update({"unexpected": True}), "unknown keys"),
        (
            lambda data: data["publish"].update({"path": "../outside"}),
            "relative repository path",
        ),
        (_disable_all_sae_payloads, "selects no payloads"),
        (_add_duplicate_document_activation_owner, "assigned to multiple SAE"),
    ],
)
def test_config_rejects_invalid_manifests(tmp_path: Path, mutate, message: str) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(SeedSnapshotConfigError, match=message):
        SeedSnapshotConfig.from_file(_write_config(tmp_path, payload), project_root=tmp_path)


def _create_source_duckdb(path: Path) -> None:
    client = DuckDBClient(db_path=str(path))
    for dataset_name in ("dataset-a", "dataset-b"):
        client._exec(
            "INSERT INTO datasets (name, description, item_count) VALUES (?, ?, ?)",
            [dataset_name, dataset_name, 1],
        )
        client._ensure_items_table(dataset_name)
        client._exec(
            f"INSERT INTO {client._items_table(dataset_name)} VALUES (?, ?, ?, ?)",
            [f"item-{dataset_name}", dataset_name, "{}", 0],
        )
    client._exec(
        "UPDATE datasets SET extra_metadata = ? WHERE name = ?",
        [
            json.dumps({"sae_model_id": "model-a", "sae_id": "sae-a"}),
            "dataset-a",
        ],
    )

    client._exec(
        """
        INSERT INTO vector_collections (
            collection_name, dataset_name, backend, vector_type,
            item_count, has_projections, has_topics
        ) VALUES (?, ?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "documents",
            "dataset-a",
            "chromadb",
            "dense",
            1,
            True,
            True,
            "feature-vectors",
            "dataset-a",
            "chromadb",
            "dense",
            1,
            True,
            True,
            "excluded",
            "dataset-b",
            "chromadb",
            "dense",
            1,
            False,
            False,
        ],
    )
    client._exec(
        "INSERT INTO projections VALUES (?, ?, ?, ?), (?, ?, ?, ?)",
        [
            "documents",
            "item-dataset-a",
            "umap_3d",
            [1.0, 2.0, 3.0],
            "feature-vectors",
            "item-dataset-a",
            "umap_3d",
            [4.0, 5.0, 6.0],
        ],
    )
    client._exec(
        """
        INSERT INTO topic_extractions (
            id, collection_name, dataset_name, topic_count, is_active
        ) VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)
        """,
        [
            "topic-docs",
            "documents",
            "dataset-a",
            1,
            True,
            "topic-features",
            "feature-vectors",
            "dataset-a",
            1,
            True,
        ],
    )
    client._exec(
        "INSERT INTO probes (collection_name, target_field, kind) VALUES (?, ?, ?), (?, ?, ?)",
        ["documents", "year", "ridge", "feature-vectors", "density", "ridge"],
    )
    client._exec(
        """
        INSERT INTO probe_scores (collection_name, target_field, kind, item_id, score)
        VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)
        """,
        [
            "documents",
            "year",
            "ridge",
            "item-dataset-a",
            0.5,
            "feature-vectors",
            "density",
            "ridge",
            "item-dataset-a",
            0.7,
        ],
    )
    client._exec(
        """
        INSERT INTO sae_features (
            model_id, sae_id, feature_index, density, label, top_logits, bottom_logits
        ) VALUES (?, ?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "model-a",
            "sae-a",
            7,
            0.1,
            "selected",
            "[]",
            "[]",
            "model-b",
            "sae-b",
            8,
            0.2,
            "excluded",
            "[]",
            "[]",
        ],
    )
    client._exec(
        """
        INSERT INTO sae_activations (id, model_id, sae_id, feature_index, tokens, act_values)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ["activation-a", "model-a", "sae-a", 7, "[]", "[]"],
    )
    client._exec(
        "INSERT INTO sae_document_activations VALUES (?, ?, ?, ?)",
        ["documents", "item-dataset-a", 7, 3.5],
    )
    client.close()


def test_duckdb_export_selects_payloads_and_sae_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.duckdb"
    destination = tmp_path / "seed.duckdb"
    _create_source_duckdb(source)
    payload = _payload()
    payload["collections"][0]["include"] = {
        "vectors": True,
        "projections": True,
        "topics": True,
        "probes": True,
    }
    config = SeedSnapshotConfig.from_file(_write_config(tmp_path, payload), project_root=tmp_path)

    counts = DuckDBSnapshotExporter(source).export(config, destination)

    connection = duckdb.connect(str(destination), read_only=True)
    try:
        assert connection.execute("SELECT name FROM datasets").fetchall() == [("dataset-a",)]
        assert set(
            connection.execute("SELECT collection_name FROM vector_collections").fetchall()
        ) == {
            ("documents",),
            ("feature-vectors",),
        }
        flags = dict(
            connection.execute(
                "SELECT collection_name, has_projections FROM vector_collections"
            ).fetchall()
        )
        assert flags == {"documents": True, "feature-vectors": False}
        assert connection.execute("SELECT count(*) FROM projections").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM topic_extractions").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM probes").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM probe_scores").fetchone() == (1,)
        assert connection.execute("SELECT model_id, sae_id FROM sae_features").fetchall() == [
            ("model-a", "sae-a")
        ]
        assert connection.execute("SELECT count(*) FROM sae_activations").fetchone() == (0,)
        assert connection.execute(
            "SELECT collection_name, item_id FROM sae_document_activations"
        ).fetchall() == [("documents", "item-dataset-a")]
    finally:
        connection.close()

    assert counts["datasets"] == 1
    assert counts["sae_features"] == 1


def test_duckdb_export_optionally_includes_activation_examples(tmp_path: Path) -> None:
    source = tmp_path / "source.duckdb"
    destination = tmp_path / "seed.duckdb"
    _create_source_duckdb(source)
    payload = _payload()
    payload["sae_data"][0]["activation_examples"] = True
    config = SeedSnapshotConfig.from_file(_write_config(tmp_path, payload), project_root=tmp_path)

    DuckDBSnapshotExporter(source).export(config, destination)

    connection = duckdb.connect(str(destination), read_only=True)
    try:
        assert connection.execute("SELECT id FROM sae_activations").fetchall() == [
            ("activation-a",)
        ]
    finally:
        connection.close()


def _create_chroma(path: Path, *, empty_documents: bool = False) -> None:
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    documents = client.create_collection("documents", metadata={"kind": "documents"})
    if not empty_documents:
        documents.add(ids=["doc-1"], embeddings=[[1.0, 0.0]])
    features = client.create_collection(
        "feature-vectors",
        metadata={"sae_model_id": "model-a", "sae_id": "sae-a"},
    )
    features.add(ids=["7"], embeddings=[[0.0, 1.0]])


def test_chroma_export_preserves_metadata_and_uses_fallback(tmp_path: Path) -> None:
    source = tmp_path / "source-vectors"
    fallback = tmp_path / "fallback-vectors"
    destination = tmp_path / "seed-vectors"
    _create_chroma(source, empty_documents=True)
    _create_chroma(fallback)
    config = SeedSnapshotConfig.from_file(
        _write_config(tmp_path, _payload()), project_root=tmp_path
    )

    exporter = ChromaSnapshotExporter(source, fallback)
    exporter.validate(config)
    counts = exporter.export(config, destination)

    result = chromadb.PersistentClient(
        path=str(destination), settings=Settings(anonymized_telemetry=False)
    )
    assert result.get_collection("documents").count() == 1
    feature_collection = result.get_collection("feature-vectors")
    assert feature_collection.metadata["sae_model_id"] == "model-a"
    assert counts == {"documents": 1, "feature-vectors": 1}


def test_chroma_export_rejects_an_interrupted_partial_copy(tmp_path: Path, monkeypatch) -> None:
    payload = _payload()
    payload["collections"] = [{"name": "documents"}]
    payload["sae_data"] = []
    config = SeedSnapshotConfig.from_file(_write_config(tmp_path, payload), project_root=tmp_path)

    class SourceCollection:
        metadata = {"kind": "documents"}

        @staticmethod
        def count() -> int:
            return 2

        @staticmethod
        def get(**kwargs) -> dict:
            return {"ids": [], "embeddings": []}

    class DestinationCollection:
        @staticmethod
        def add(**kwargs) -> None:
            raise AssertionError("an empty batch must not be added")

        @staticmethod
        def count() -> int:
            return 0

    class SourceClient:
        @staticmethod
        def get_collection(name: str):
            return SourceCollection()

    class DestinationClient:
        @staticmethod
        def create_collection(**kwargs):
            return DestinationCollection()

    exporter = ChromaSnapshotExporter(tmp_path / "source")
    monkeypatch.setattr(
        exporter,
        "_client",
        lambda path: SourceClient() if path == exporter.source_path else DestinationClient(),
    )

    with pytest.raises(RuntimeError, match="incomplete Chroma export"):
        exporter.export(config, tmp_path / "destination")


def test_builder_writes_integrity_manifest_and_replaces_atomically(tmp_path: Path) -> None:
    source_duckdb = tmp_path / "source.duckdb"
    source_chroma = tmp_path / "source-vectors"
    _create_source_duckdb(source_duckdb)
    _create_chroma(source_chroma)
    config = SeedSnapshotConfig.from_file(
        _write_config(tmp_path, _payload()), project_root=tmp_path
    )
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "old.txt").write_text("old", encoding="utf-8")

    result = SeedSnapshotBuilder(
        source_duckdb_path=source_duckdb,
        source_chroma_path=source_chroma,
        committed_seed_dir=tmp_path / "no-fallback",
        project_root=tmp_path,
    ).build(config)

    assert not (config.output_dir / "old.txt").exists()
    assert (config.output_dir / "main.duckdb").is_file()
    assert result["manifest_sha256"] == SnapshotIntegrity.verify(config.output_dir)
    manifest = json.loads(
        (config.output_dir / "snapshot-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["counts"]["duckdb"]["vector_collections"] == 2
    assert manifest["counts"]["chroma"] == {"documents": 1, "feature-vectors": 1}


def test_failed_builder_validation_preserves_existing_snapshot(tmp_path: Path) -> None:
    source_duckdb = tmp_path / "source.duckdb"
    source_chroma = tmp_path / "source-vectors"
    _create_source_duckdb(source_duckdb)
    chroma_client = chromadb.PersistentClient(
        path=str(source_chroma), settings=Settings(anonymized_telemetry=False)
    )
    documents = chroma_client.create_collection("documents")
    documents.add(ids=["doc-1"], embeddings=[[1.0, 0.0]])
    config = SeedSnapshotConfig.from_file(
        _write_config(tmp_path, _payload()), project_root=tmp_path
    )
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "sentinel.txt").write_text("preserved", encoding="utf-8")

    with pytest.raises(ValueError, match="feature-vectors"):
        SeedSnapshotBuilder(
            source_duckdb_path=source_duckdb,
            source_chroma_path=source_chroma,
            committed_seed_dir=tmp_path / "no-fallback",
            project_root=tmp_path,
        ).build(config)

    assert (config.output_dir / "sentinel.txt").read_text(encoding="utf-8") == "preserved"


def test_missing_sae_reference_preserves_existing_snapshot(tmp_path: Path) -> None:
    source_duckdb = tmp_path / "source.duckdb"
    source_chroma = tmp_path / "source-vectors"
    _create_source_duckdb(source_duckdb)
    _create_chroma(source_chroma)
    payload = _payload()
    payload["sae_data"][0].update(
        {
            "model_id": "missing-model",
            "explanation_vector_collection": None,
        }
    )
    config = SeedSnapshotConfig.from_file(_write_config(tmp_path, payload), project_root=tmp_path)
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "sentinel.txt").write_text("preserved", encoding="utf-8")

    with pytest.raises(SeedSnapshotConfigError, match="SAE reference not found"):
        SeedSnapshotBuilder(
            source_duckdb_path=source_duckdb,
            source_chroma_path=source_chroma,
            committed_seed_dir=tmp_path / "no-fallback",
            project_root=tmp_path,
        ).build(config)

    assert (config.output_dir / "sentinel.txt").read_text(encoding="utf-8") == "preserved"


def test_document_activation_sae_mismatch_preserves_existing_snapshot(tmp_path: Path) -> None:
    source_duckdb = tmp_path / "source.duckdb"
    source_chroma = tmp_path / "source-vectors"
    _create_source_duckdb(source_duckdb)
    connection = duckdb.connect(str(source_duckdb))
    connection.execute(
        "UPDATE datasets SET extra_metadata = ? WHERE name = ?",
        [json.dumps({"sae_model_id": "model-b", "sae_id": "sae-b"}), "dataset-a"],
    )
    connection.close()
    _create_chroma(source_chroma)
    config = SeedSnapshotConfig.from_file(
        _write_config(tmp_path, _payload()), project_root=tmp_path
    )
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "sentinel.txt").write_text("preserved", encoding="utf-8")

    with pytest.raises(SeedSnapshotConfigError, match="document activation SAE metadata mismatch"):
        SeedSnapshotBuilder(
            source_duckdb_path=source_duckdb,
            source_chroma_path=source_chroma,
            committed_seed_dir=tmp_path / "no-fallback",
            project_root=tmp_path,
        ).build(config)

    assert (config.output_dir / "sentinel.txt").read_text(encoding="utf-8") == "preserved"


class _FakeHubApi:
    def __init__(self, *, private: bool = True, failed_uploads: int = 0) -> None:
        self.created = []
        self.uploaded = []
        self.private = private
        self.failed_uploads = failed_uploads

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def repo_info(self, **kwargs):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)
        if len(self.uploaded) <= self.failed_uploads:
            raise ConnectionError("transient upload failure")
        return SimpleNamespace(oid="hub-commit-123")


def test_publisher_creates_private_repo_and_writes_immutable_lock(tmp_path: Path) -> None:
    config = SeedSnapshotConfig.from_file(
        _write_config(tmp_path, _payload()), project_root=tmp_path
    )
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "payload.txt").write_text("seed", encoding="utf-8")
    SnapshotIntegrity.write_manifest(
        config.output_dir,
        _snapshot_metadata(config, {"documents": 1}),
    )
    fake_api = _FakeHubApi()

    lock = HuggingFaceSnapshotPublisher(
        api=fake_api,
        environ={"TEST_SEED_REPO": "owner/private-seed", "HF_TOKEN": "secret-token"},
    ).publish(config)

    assert fake_api.created[0]["private"] is True
    assert fake_api.uploaded[0]["path_in_repo"] == "snapshots/demo"
    assert fake_api.uploaded[0]["delete_patterns"] == "**"
    assert lock.revision == "hub-commit-123"
    persisted = json.loads(config.publish.lock_file.read_text(encoding="utf-8"))
    assert persisted["revision"] == "hub-commit-123"
    assert "secret-token" not in config.publish.lock_file.read_text(encoding="utf-8")


def test_publisher_retries_upload_and_rejects_public_repo(tmp_path: Path) -> None:
    config = SeedSnapshotConfig.from_file(
        _write_config(tmp_path, _payload()), project_root=tmp_path
    )
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "payload.txt").write_text("seed", encoding="utf-8")
    SnapshotIntegrity.write_manifest(config.output_dir, _snapshot_metadata(config))
    retrying_api = _FakeHubApi(failed_uploads=2)
    sleeps = []

    lock = HuggingFaceSnapshotPublisher(
        api=retrying_api,
        environ={"TEST_SEED_REPO": "owner/private-seed", "HF_TOKEN": "secret-token"},
        sleep_function=sleeps.append,
    ).publish(config)

    assert lock.revision == "hub-commit-123"
    assert len(retrying_api.uploaded) == 3
    assert sleeps == [1.0, 2.0]

    public_api = _FakeHubApi(private=False)
    with pytest.raises(SeedSnapshotConfigError, match="must be private"):
        HuggingFaceSnapshotPublisher(
            api=public_api,
            environ={"TEST_SEED_REPO": "owner/public-seed", "HF_TOKEN": "secret-token"},
            sleep_function=lambda _: None,
        ).publish(config)
    assert public_api.uploaded == []


def test_publisher_rejects_snapshot_built_from_stale_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _payload())
    original = SeedSnapshotConfig.from_file(config_path, project_root=tmp_path)
    original.output_dir.mkdir(parents=True)
    (original.output_dir / "payload.txt").write_text("seed", encoding="utf-8")
    SnapshotIntegrity.write_manifest(original.output_dir, _snapshot_metadata(original))
    changed_payload = _payload()
    changed_payload["collections"][0]["include"] = {"probes": True}
    config_path.write_text(json.dumps(changed_payload), encoding="utf-8")
    changed = SeedSnapshotConfig.from_file(config_path, project_root=tmp_path)
    fake_api = _FakeHubApi()

    with pytest.raises(ValueError, match="config_sha256 mismatch"):
        HuggingFaceSnapshotPublisher(
            api=fake_api,
            environ={"TEST_SEED_REPO": "owner/private-seed", "HF_TOKEN": "secret-token"},
            sleep_function=lambda _: None,
        ).publish(changed)

    assert fake_api.created == []


def test_downloader_verifies_lock_and_installs_snapshot_atomically(tmp_path: Path) -> None:
    config = SeedSnapshotConfig.from_file(
        _write_config(tmp_path, _payload()), project_root=tmp_path
    )
    remote_root = tmp_path / "remote"
    remote_snapshot = remote_root / "snapshots/demo"
    remote_snapshot.mkdir(parents=True)
    (remote_snapshot / "payload.txt").write_text("downloaded", encoding="utf-8")
    manifest_hash = SnapshotIntegrity.write_manifest(remote_snapshot, _snapshot_metadata(config))
    config.publish.lock_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_name": "demo",
                "repo_id": "owner/private-seed",
                "path": "snapshots/demo",
                "revision": "hub-commit-123",
                "manifest_sha256": manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(remote_root)

    config.output_dir.mkdir(parents=True)
    (config.output_dir / "old.txt").write_text("old", encoding="utf-8")

    HuggingFaceSnapshotDownloader(download_function=fake_download).download(
        config, token="read-token"
    )

    assert calls[0]["revision"] == "hub-commit-123"
    assert (config.output_dir / "payload.txt").read_text(encoding="utf-8") == "downloaded"
    assert not (config.output_dir / "old.txt").exists()


def test_downloader_uses_committed_seed_before_first_lock(tmp_path: Path) -> None:
    config = SeedSnapshotConfig.from_file(
        _write_config(tmp_path, _payload()), project_root=tmp_path
    )
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    (fallback / "main.duckdb").write_text("fallback-db", encoding="utf-8")

    result = HuggingFaceSnapshotDownloader(
        download_function=lambda **kwargs: pytest.fail("Hub download should not run")
    ).download(config, fallback_dir=fallback)

    assert result == "fallback"
    assert (config.output_dir / "main.duckdb").read_text(encoding="utf-8") == "fallback-db"


def test_integrity_failure_does_not_replace_existing_output(tmp_path: Path) -> None:
    config = SeedSnapshotConfig.from_file(
        _write_config(tmp_path, _payload()), project_root=tmp_path
    )
    remote_root = tmp_path / "remote"
    remote_snapshot = remote_root / "snapshots/demo"
    remote_snapshot.mkdir(parents=True)
    (remote_snapshot / "payload.txt").write_text("corrupt", encoding="utf-8")
    SnapshotIntegrity.write_manifest(remote_snapshot, _snapshot_metadata(config))
    config.publish.lock_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_name": "demo",
                "repo_id": "owner/private-seed",
                "path": "snapshots/demo",
                "revision": "hub-commit-123",
                "manifest_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "old.txt").write_text("preserved", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest checksum"):
        HuggingFaceSnapshotDownloader(download_function=lambda **kwargs: str(remote_root)).download(
            config, token="read-token"
        )

    assert (config.output_dir / "old.txt").read_text(encoding="utf-8") == "preserved"


def test_downloader_rejects_snapshot_built_from_stale_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _payload())
    original = SeedSnapshotConfig.from_file(config_path, project_root=tmp_path)
    remote_root = tmp_path / "remote"
    remote_snapshot = remote_root / "snapshots/demo"
    remote_snapshot.mkdir(parents=True)
    (remote_snapshot / "payload.txt").write_text("downloaded", encoding="utf-8")
    manifest_hash = SnapshotIntegrity.write_manifest(remote_snapshot, _snapshot_metadata(original))
    original.publish.lock_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_name": "demo",
                "repo_id": "owner/private-seed",
                "path": "snapshots/demo",
                "revision": "hub-commit-123",
                "manifest_sha256": manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    changed_payload = _payload()
    changed_payload["collections"][0]["include"] = {"probes": True}
    config_path.write_text(json.dumps(changed_payload), encoding="utf-8")
    changed = SeedSnapshotConfig.from_file(config_path, project_root=tmp_path)
    changed.output_dir.mkdir(parents=True)
    (changed.output_dir / "old.txt").write_text("preserved", encoding="utf-8")

    with pytest.raises(ValueError, match="config_sha256 mismatch"):
        HuggingFaceSnapshotDownloader(download_function=lambda **kwargs: str(remote_root)).download(
            changed, token="read-token"
        )

    assert (changed.output_dir / "old.txt").read_text(encoding="utf-8") == "preserved"


def test_integrity_rejects_untracked_snapshot_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "payload.txt").write_text("tracked", encoding="utf-8")
    SnapshotIntegrity.write_manifest(snapshot, {"counts": {}})
    (snapshot / "unexpected.txt").write_text("not checksummed", encoding="utf-8")

    with pytest.raises(ValueError, match="untracked files"):
        SnapshotIntegrity.verify(snapshot)


def test_integrity_rejects_nested_manifest_named_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "payload.txt").write_text("tracked", encoding="utf-8")
    SnapshotIntegrity.write_manifest(snapshot, {"counts": {}})
    nested = snapshot / "nested"
    nested.mkdir()
    (nested / "snapshot-manifest.json").write_text("untracked", encoding="utf-8")

    with pytest.raises(ValueError, match="untracked files"):
        SnapshotIntegrity.verify(snapshot)
