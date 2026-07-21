"""Configuration-driven seed snapshot generation and Hub distribution.

The classes in this module keep snapshot behavior reusable outside the command
line entry points. A snapshot is built in a staging directory, verified, and
only then swapped into place so a failed export never destroys the last good
copy.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
import duckdb
from chromadb.config import Settings
from huggingface_hub import HfApi, snapshot_download

from ..clients.duckdb_client import DuckDBClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "interpretability_backend/config/seed_snapshots/default.json"
MANIFEST_FILENAME = "snapshot-manifest.json"
LOCK_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1
CHROMA_BATCH_SIZE = 5000


class SeedSnapshotConfigError(ValueError):
    """Raised when a snapshot manifest is invalid."""


@dataclass(frozen=True)
class CollectionIncludeConfig:
    """Optional collection payloads beyond required dataset/item metadata."""

    vectors: bool = True
    projections: bool = True
    topics: bool = True
    probes: bool = False


@dataclass(frozen=True)
class SnapshotCollectionConfig:
    """One vector collection selected for a snapshot."""

    name: str
    include: CollectionIncludeConfig


@dataclass(frozen=True)
class SnapshotSAEConfig:
    """SAE rows selected independently from visualization collections."""

    model_id: str
    sae_id: str
    features: bool
    activation_examples: bool
    document_activations: tuple[str, ...]
    explanation_vector_collection: str | None


@dataclass(frozen=True)
class SnapshotPublishConfig:
    """Private Hugging Face Dataset destination for a snapshot."""

    repo_id_env: str
    private: bool
    path: str
    lock_file: Path


@dataclass(frozen=True)
class SnapshotLock:
    """Immutable reference to one published snapshot revision."""

    snapshot_name: str
    repo_id: str
    path: str
    revision: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LOCK_SCHEMA_VERSION,
            "snapshot_name": self.snapshot_name,
            "repo_id": self.repo_id,
            "path": self.path,
            "revision": self.revision,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_file(cls, path: Path) -> "SnapshotLock":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise SeedSnapshotConfigError(f"snapshot lock not found: {path}") from error
        except json.JSONDecodeError as error:
            raise SeedSnapshotConfigError(f"invalid snapshot lock JSON: {error}") from error

        expected = {
            "schema_version",
            "snapshot_name",
            "repo_id",
            "path",
            "revision",
            "manifest_sha256",
        }
        extra = set(payload) - expected
        missing = expected - set(payload)
        if extra or missing:
            raise SeedSnapshotConfigError(
                f"invalid snapshot lock keys; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if payload["schema_version"] != LOCK_SCHEMA_VERSION:
            raise SeedSnapshotConfigError(
                f"unsupported snapshot lock schema_version: {payload['schema_version']}"
            )
        return cls(
            snapshot_name=cls._nonempty(payload["snapshot_name"], "snapshot_name"),
            repo_id=cls._nonempty(payload["repo_id"], "repo_id"),
            path=SeedSnapshotConfig._repository_path(payload["path"], "path"),
            revision=cls._nonempty(payload["revision"], "revision"),
            manifest_sha256=cls._checksum(payload["manifest_sha256"]),
        )

    @staticmethod
    def _nonempty(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SeedSnapshotConfigError(f"snapshot lock {field} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _checksum(value: Any) -> str:
        checksum = SnapshotLock._nonempty(value, "manifest_sha256")
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise SeedSnapshotConfigError("snapshot lock manifest_sha256 must be lowercase SHA-256")
        return checksum


@dataclass(frozen=True)
class SeedSnapshotConfig:
    """Validated snapshot manifest."""

    schema_version: int
    name: str
    output_dir: Path
    collections: tuple[SnapshotCollectionConfig, ...]
    sae_data: tuple[SnapshotSAEConfig, ...]
    publish: SnapshotPublishConfig | None
    config_path: Path
    config_sha256: str

    @property
    def collection_names(self) -> tuple[str, ...]:
        return tuple(collection.name for collection in self.collections)

    @classmethod
    def from_file(
        cls,
        path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        project_root: Path = PROJECT_ROOT,
    ) -> "SeedSnapshotConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            raw_bytes = config_path.read_bytes()
            payload = json.loads(raw_bytes)
        except FileNotFoundError as error:
            raise SeedSnapshotConfigError(f"snapshot config not found: {config_path}") from error
        except json.JSONDecodeError as error:
            raise SeedSnapshotConfigError(f"invalid snapshot config JSON: {error}") from error

        if not isinstance(payload, dict):
            raise SeedSnapshotConfigError("snapshot config root must be an object")
        cls._reject_unknown(
            payload,
            {"schema_version", "name", "output", "collections", "sae_data", "publish"},
            "snapshot config",
        )
        schema_version = payload.get("schema_version")
        if schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise SeedSnapshotConfigError(
                f"unsupported snapshot schema_version: {schema_version!r}"
            )

        name = cls._nonempty_string(payload.get("name"), "name")
        output_value = cls._nonempty_string(payload.get("output"), "output")
        output_dir = Path(output_value).expanduser()
        if not output_dir.is_absolute():
            output_dir = (project_root / output_dir).resolve()

        collections = cls._parse_collections(payload.get("collections"))
        sae_data = cls._parse_sae_data(payload.get("sae_data", []), collections)
        publish = cls._parse_publish(payload.get("publish"), config_path, project_root)
        return cls(
            schema_version=schema_version,
            name=name,
            output_dir=output_dir,
            collections=collections,
            sae_data=sae_data,
            publish=publish,
            config_path=config_path,
            config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )

    @classmethod
    def _parse_collections(cls, value: Any) -> tuple[SnapshotCollectionConfig, ...]:
        if not isinstance(value, list) or not value:
            raise SeedSnapshotConfigError("collections must be a non-empty list")
        result: list[SnapshotCollectionConfig] = []
        seen: set[str] = set()
        include_keys = {"vectors", "projections", "topics", "probes"}
        for index, entry in enumerate(value):
            if not isinstance(entry, dict):
                raise SeedSnapshotConfigError(f"collections[{index}] must be an object")
            cls._reject_unknown(entry, {"name", "include"}, f"collections[{index}]")
            name = cls._nonempty_string(entry.get("name"), f"collections[{index}].name")
            if name in seen:
                raise SeedSnapshotConfigError(f"duplicate collection: {name}")
            seen.add(name)
            include_raw = entry.get("include", {})
            if not isinstance(include_raw, dict):
                raise SeedSnapshotConfigError(f"collections[{index}].include must be an object")
            cls._reject_unknown(include_raw, include_keys, f"collections[{index}].include")
            for key, raw_value in include_raw.items():
                if not isinstance(raw_value, bool):
                    raise SeedSnapshotConfigError(
                        f"collections[{index}].include.{key} must be boolean"
                    )
            result.append(
                SnapshotCollectionConfig(
                    name=name,
                    include=CollectionIncludeConfig(
                        vectors=include_raw.get("vectors", True),
                        projections=include_raw.get("projections", True),
                        topics=include_raw.get("topics", True),
                        probes=include_raw.get("probes", False),
                    ),
                )
            )
        return tuple(result)

    @classmethod
    def _parse_sae_data(
        cls,
        value: Any,
        collections: tuple[SnapshotCollectionConfig, ...],
    ) -> tuple[SnapshotSAEConfig, ...]:
        if not isinstance(value, list):
            raise SeedSnapshotConfigError("sae_data must be a list")
        collection_map = {collection.name: collection for collection in collections}
        result: list[SnapshotSAEConfig] = []
        seen: set[tuple[str, str]] = set()
        document_activation_owners: dict[str, tuple[str, str]] = {}
        allowed = {
            "model_id",
            "sae_id",
            "features",
            "activation_examples",
            "document_activations",
            "explanation_vector_collection",
        }
        for index, entry in enumerate(value):
            if not isinstance(entry, dict):
                raise SeedSnapshotConfigError(f"sae_data[{index}] must be an object")
            cls._reject_unknown(entry, allowed, f"sae_data[{index}]")
            model_id = cls._nonempty_string(entry.get("model_id"), f"sae_data[{index}].model_id")
            sae_id = cls._nonempty_string(entry.get("sae_id"), f"sae_data[{index}].sae_id")
            pair = (model_id, sae_id)
            if pair in seen:
                raise SeedSnapshotConfigError(f"duplicate SAE selection: {model_id}/{sae_id}")
            seen.add(pair)

            features = entry.get("features", True)
            activation_examples = entry.get("activation_examples", False)
            if not isinstance(features, bool) or not isinstance(activation_examples, bool):
                raise SeedSnapshotConfigError(
                    f"sae_data[{index}] features and activation_examples must be boolean"
                )
            if activation_examples and not features:
                raise SeedSnapshotConfigError(
                    f"sae_data[{index}] activation_examples requires features"
                )

            documents_raw = entry.get("document_activations", [])
            if not isinstance(documents_raw, list) or not all(
                isinstance(name, str) and name.strip() for name in documents_raw
            ):
                raise SeedSnapshotConfigError(
                    f"sae_data[{index}].document_activations must be a list of names"
                )
            document_activations = tuple(name.strip() for name in documents_raw)
            if len(set(document_activations)) != len(document_activations):
                raise SeedSnapshotConfigError(
                    f"sae_data[{index}].document_activations contains duplicates"
                )
            for collection_name in document_activations:
                if collection_name not in collection_map:
                    raise SeedSnapshotConfigError(
                        f"document activation collection is not exported: {collection_name}"
                    )
                owner = document_activation_owners.get(collection_name)
                if owner is not None:
                    raise SeedSnapshotConfigError(
                        "document activation collection is assigned to multiple SAE selections: "
                        f"{collection_name} ({owner[0]}/{owner[1]} and {model_id}/{sae_id})"
                    )
                document_activation_owners[collection_name] = pair

            explanation = entry.get("explanation_vector_collection")
            if explanation is not None:
                explanation = cls._nonempty_string(
                    explanation, f"sae_data[{index}].explanation_vector_collection"
                )
                if explanation not in collection_map:
                    raise SeedSnapshotConfigError(
                        f"explanation vector collection is not exported: {explanation}"
                    )
                if not collection_map[explanation].include.vectors:
                    raise SeedSnapshotConfigError(
                        f"explanation vector collection must include vectors: {explanation}"
                    )
            if not features and not document_activations and explanation is None:
                raise SeedSnapshotConfigError(
                    f"sae_data[{index}] selects no payloads for {model_id}/{sae_id}"
                )

            result.append(
                SnapshotSAEConfig(
                    model_id=model_id,
                    sae_id=sae_id,
                    features=features,
                    activation_examples=activation_examples,
                    document_activations=document_activations,
                    explanation_vector_collection=explanation,
                )
            )
        return tuple(result)

    @classmethod
    def _parse_publish(
        cls,
        value: Any,
        config_path: Path,
        project_root: Path,
    ) -> SnapshotPublishConfig | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise SeedSnapshotConfigError("publish must be an object")
        cls._reject_unknown(value, {"repo_id_env", "private", "path", "lock_file"}, "publish")
        repo_id_env = cls._nonempty_string(value.get("repo_id_env"), "publish.repo_id_env")
        private = value.get("private", True)
        if not isinstance(private, bool):
            raise SeedSnapshotConfigError("publish.private must be boolean")
        repo_path = cls._repository_path(value.get("path"), "publish.path")
        lock_raw = value.get("lock_file")
        if lock_raw is None:
            lock_file = config_path.with_suffix(".lock.json")
        else:
            lock_file = Path(cls._nonempty_string(lock_raw, "publish.lock_file")).expanduser()
            if not lock_file.is_absolute():
                lock_file = (project_root / lock_file).resolve()
        return SnapshotPublishConfig(
            repo_id_env=repo_id_env,
            private=private,
            path=repo_path,
            lock_file=lock_file,
        )

    @staticmethod
    def _reject_unknown(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
        unknown = set(value) - allowed
        if unknown:
            raise SeedSnapshotConfigError(f"{context} has unknown keys: {sorted(unknown)}")

    @staticmethod
    def _nonempty_string(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SeedSnapshotConfigError(f"{field} must be a non-empty string")
        return value.strip()

    @classmethod
    def _repository_path(cls, value: Any, field: str) -> str:
        normalized = cls._nonempty_string(value, field).strip("/")
        parts = Path(normalized).parts
        if not parts or any(part in {".", ".."} for part in parts):
            raise SeedSnapshotConfigError(f"{field} must be a safe relative repository path")
        return Path(*parts).as_posix()


class SnapshotIntegrity:
    """Create and verify per-file snapshot checksums."""

    @classmethod
    def write_manifest(cls, snapshot_dir: Path, metadata: Mapping[str, Any]) -> str:
        files = cls._file_inventory(snapshot_dir)
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            **dict(metadata),
            "files": files,
        }
        manifest_path = snapshot_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return cls.sha256_file(manifest_path)

    @classmethod
    def verify(
        cls,
        snapshot_dir: Path,
        expected_manifest_sha256: str | None = None,
        *,
        expected_snapshot_name: str | None = None,
        expected_config_sha256: str | None = None,
    ) -> str:
        manifest_path = snapshot_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ValueError(f"snapshot manifest not found: {manifest_path}")
        manifest_sha256 = cls.sha256_file(manifest_path)
        if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "snapshot manifest checksum mismatch: "
                f"expected {expected_manifest_sha256}, got {manifest_sha256}"
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid snapshot manifest JSON: {error}") from error
        if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported snapshot manifest schema_version: {payload.get('schema_version')}"
            )
        for field, expected_value in (
            ("snapshot_name", expected_snapshot_name),
            ("config_sha256", expected_config_sha256),
        ):
            if expected_value is not None and payload.get(field) != expected_value:
                raise ValueError(
                    f"snapshot manifest {field} mismatch: "
                    f"expected {expected_value}, got {payload.get(field)}"
                )
        files = payload.get("files")
        if not isinstance(files, dict):
            raise ValueError("snapshot manifest files must be an object")
        actual_files = {
            path.relative_to(snapshot_dir).as_posix()
            for path in snapshot_dir.rglob("*")
            if path.is_file() and path != manifest_path
        }
        declared_files = set(files)
        unexpected = actual_files - declared_files
        missing = declared_files - actual_files
        if unexpected:
            raise ValueError(f"snapshot contains untracked files: {sorted(unexpected)}")
        if missing:
            raise ValueError(f"snapshot files missing: {sorted(missing)}")
        for relative, expected in files.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or any(
                part in {".", ".."} for part in relative_path.parts
            ):
                raise ValueError(f"invalid snapshot file path: {relative}")
            if not isinstance(expected, dict):
                raise ValueError(f"invalid file record for {relative}")
            file_path = snapshot_dir / relative
            if not file_path.is_file():
                raise ValueError(f"snapshot file missing: {relative}")
            actual_sha256 = cls.sha256_file(file_path)
            if actual_sha256 != expected.get("sha256"):
                raise ValueError(f"snapshot file checksum mismatch: {relative}")
            if file_path.stat().st_size != expected.get("size"):
                raise ValueError(f"snapshot file size mismatch: {relative}")
        return manifest_sha256

    @staticmethod
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _file_inventory(cls, snapshot_dir: Path) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for file_path in sorted(path for path in snapshot_dir.rglob("*") if path.is_file()):
            if file_path == snapshot_dir / MANIFEST_FILENAME:
                continue
            relative = file_path.relative_to(snapshot_dir).as_posix()
            result[relative] = {
                "sha256": cls.sha256_file(file_path),
                "size": file_path.stat().st_size,
            }
        return result


class SnapshotDirectoryInstaller:
    """Atomically install a staged snapshot directory."""

    @staticmethod
    def install(staged_dir: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        had_destination = destination.exists()
        backup: Path | None = None
        if had_destination:
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.backup_",
                    dir=destination.parent,
                )
            )
            backup.rmdir()
            destination.rename(backup)
        try:
            staged_dir.rename(destination)
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                backup.rename(destination)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


class DuckDBSnapshotExporter:
    """Export selected relational data from the live DuckDB store."""

    def __init__(self, source_path: str | Path):
        self.source_path = Path(source_path).expanduser().resolve()

    def resolve_datasets(self, config: SeedSnapshotConfig) -> dict[str, str]:
        if not self.source_path.is_file():
            raise FileNotFoundError(f"source DuckDB not found: {self.source_path}")
        connection = duckdb.connect(str(self.source_path), read_only=True)
        try:
            placeholders = ", ".join("?" for _ in config.collection_names)
            rows = connection.execute(
                "SELECT collection_name, dataset_name FROM vector_collections "
                f"WHERE collection_name IN ({placeholders})",
                list(config.collection_names),
            ).fetchall()
        finally:
            connection.close()
        resolved = dict(rows)
        missing = [name for name in config.collection_names if name not in resolved]
        if missing:
            raise SeedSnapshotConfigError(f"collections not found in DuckDB: {missing}")
        return {name: resolved[name] for name in config.collection_names}

    def validate(self, config: SeedSnapshotConfig) -> dict[str, str]:
        """Validate every relational reference before creating staged output."""
        resolved = self.resolve_datasets(config)
        if not config.sae_data:
            return resolved
        connection = duckdb.connect(str(self.source_path), read_only=True)
        try:
            for sae in config.sae_data:
                feature_count = connection.execute(
                    "SELECT count(*) FROM sae_features WHERE model_id = ? AND sae_id = ?",
                    [sae.model_id, sae.sae_id],
                ).fetchone()[0]
                if feature_count == 0:
                    raise SeedSnapshotConfigError(
                        f"SAE reference not found: {sae.model_id}/{sae.sae_id}"
                    )
                if sae.activation_examples:
                    activation_count = connection.execute(
                        "SELECT count(*) FROM sae_activations WHERE model_id = ? AND sae_id = ?",
                        [sae.model_id, sae.sae_id],
                    ).fetchone()[0]
                    if activation_count == 0:
                        raise SeedSnapshotConfigError(
                            f"SAE activation examples not found: {sae.model_id}/{sae.sae_id}"
                        )
                for collection_name in sae.document_activations:
                    dataset_name = resolved[collection_name]
                    identity = connection.execute(
                        "SELECT json_extract_string(extra_metadata, '$.sae_model_id'), "
                        "json_extract_string(extra_metadata, '$.sae_id') "
                        "FROM datasets WHERE name = ?",
                        [dataset_name],
                    ).fetchone()
                    actual_identity = tuple(identity) if identity is not None else (None, None)
                    expected_identity = (sae.model_id, sae.sae_id)
                    if actual_identity != expected_identity:
                        raise SeedSnapshotConfigError(
                            "document activation SAE metadata mismatch for "
                            f"{collection_name}: expected {expected_identity}, "
                            f"got {actual_identity}"
                        )
                    document_count = connection.execute(
                        "SELECT count(*) FROM sae_document_activations WHERE collection_name = ?",
                        [collection_name],
                    ).fetchone()[0]
                    if document_count == 0:
                        raise SeedSnapshotConfigError(
                            f"SAE document activations not found for collection: {collection_name}"
                        )
        finally:
            connection.close()
        return resolved

    def export(self, config: SeedSnapshotConfig, destination_path: Path) -> dict[str, int]:
        collection_datasets = self.resolve_datasets(config)
        datasets = tuple(dict.fromkeys(collection_datasets.values()))
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        seed_client = DuckDBClient(db_path=str(destination_path))
        item_tables: dict[str, str] = {}
        for dataset in datasets:
            seed_client._ensure_items_table(dataset)
            item_tables[dataset] = seed_client._items_table(dataset)
        seed_client.close()

        counts: dict[str, int] = {}
        connection = duckdb.connect(str(destination_path))
        attached = False
        try:
            escaped_source = str(self.source_path).replace("'", "''")
            connection.execute(f"ATTACH '{escaped_source}' AS prod (READ_ONLY)")
            attached = True
            counts["datasets"] = self._copy_filtered(connection, "datasets", "name", datasets)
            for dataset, table in item_tables.items():
                counts[f"items:{dataset}"] = self._insert_count(
                    connection, f"INSERT INTO {table} BY NAME SELECT * FROM prod.{table}"
                )
            counts["vector_collections"] = self._copy_filtered(
                connection,
                "vector_collections",
                "collection_name",
                config.collection_names,
            )

            projection_names = self._selected(config, "projections")
            counts["projections"] = self._copy_filtered(
                connection, "projections", "collection_name", projection_names
            )
            counts["projection_metadata"] = self._copy_filtered(
                connection, "projection_metadata", "collection_name", projection_names
            )

            topic_names = self._selected(config, "topics")
            counts["topic_extractions"] = self._copy_filtered(
                connection, "topic_extractions", "collection_name", topic_names
            )
            counts["topic_info"] = self._copy_topic_children(connection, "topic_info", topic_names)
            counts["topic_assignments"] = self._copy_topic_children(
                connection, "topic_assignments", topic_names
            )

            probe_names = self._selected(config, "probes")
            counts["probes"] = self._copy_filtered(
                connection, "probes", "collection_name", probe_names
            )
            counts["probe_scores"] = self._copy_filtered(
                connection, "probe_scores", "collection_name", probe_names
            )

            counts["sae_features"] = self._copy_sae_rows(
                connection,
                "sae_features",
                tuple(sae for sae in config.sae_data if sae.features),
            )
            counts["sae_activations"] = self._copy_sae_rows(
                connection,
                "sae_activations",
                tuple(sae for sae in config.sae_data if sae.activation_examples),
            )
            document_activation_names = tuple(
                dict.fromkeys(name for sae in config.sae_data for name in sae.document_activations)
            )
            counts["sae_document_activations"] = self._copy_filtered(
                connection,
                "sae_document_activations",
                "collection_name",
                document_activation_names,
            )

            self._normalize_flags(connection, config)
            connection.execute("DETACH prod")
            attached = False
            connection.execute("CHECKPOINT")
        finally:
            if attached:
                try:
                    connection.execute("DETACH prod")
                except duckdb.Error:
                    pass
            connection.close()
        return counts

    @staticmethod
    def _selected(config: SeedSnapshotConfig, payload: str) -> tuple[str, ...]:
        return tuple(
            collection.name
            for collection in config.collections
            if getattr(collection.include, payload)
        )

    @staticmethod
    def _insert_count(
        connection: duckdb.DuckDBPyConnection,
        sql: str,
        parameters: list[Any] | None = None,
    ) -> int:
        row = connection.execute(sql, parameters or []).fetchone()
        return int(row[0]) if row else 0

    @classmethod
    def _copy_filtered(
        cls,
        connection: duckdb.DuckDBPyConnection,
        table: str,
        column: str,
        values: tuple[str, ...],
    ) -> int:
        if not values:
            return 0
        placeholders = ", ".join("?" for _ in values)
        return cls._insert_count(
            connection,
            f"INSERT INTO {table} BY NAME SELECT * FROM prod.{table} "
            f"WHERE {column} IN ({placeholders})",
            list(values),
        )

    @classmethod
    def _copy_topic_children(
        cls,
        connection: duckdb.DuckDBPyConnection,
        table: str,
        collection_names: tuple[str, ...],
    ) -> int:
        if not collection_names:
            return 0
        placeholders = ", ".join("?" for _ in collection_names)
        return cls._insert_count(
            connection,
            f"INSERT INTO {table} BY NAME SELECT child.* FROM prod.{table} child "
            "JOIN prod.topic_extractions extraction ON extraction.id = child.extraction_id "
            f"WHERE extraction.collection_name IN ({placeholders})",
            list(collection_names),
        )

    @classmethod
    def _copy_sae_rows(
        cls,
        connection: duckdb.DuckDBPyConnection,
        table: str,
        selections: tuple[SnapshotSAEConfig, ...],
    ) -> int:
        copied = 0
        for selection in selections:
            copied += cls._insert_count(
                connection,
                f"INSERT INTO {table} BY NAME SELECT * FROM prod.{table} "
                "WHERE model_id = ? AND sae_id = ?",
                [selection.model_id, selection.sae_id],
            )
        return copied

    @staticmethod
    def _normalize_flags(
        connection: duckdb.DuckDBPyConnection,
        config: SeedSnapshotConfig,
    ) -> None:
        disabled_projections = tuple(
            collection.name
            for collection in config.collections
            if not collection.include.projections
        )
        disabled_topics = tuple(
            collection.name for collection in config.collections if not collection.include.topics
        )
        for column, values in (
            ("has_projections", disabled_projections),
            ("has_topics", disabled_topics),
        ):
            if not values:
                continue
            placeholders = ", ".join("?" for _ in values)
            connection.execute(
                f"UPDATE vector_collections SET {column} = FALSE "
                f"WHERE collection_name IN ({placeholders})",
                list(values),
            )


class ChromaSnapshotExporter:
    """Rebuild selected Chroma collections into a clean snapshot store."""

    def __init__(
        self,
        source_path: str | Path,
        fallback_path: str | Path | None = None,
    ):
        self.source_path = Path(source_path).expanduser().resolve()
        self.fallback_path = (
            Path(fallback_path).expanduser().resolve() if fallback_path is not None else None
        )

    def validate(self, config: SeedSnapshotConfig) -> None:
        source = self._client(self.source_path)
        fallback = self._optional_client(self.fallback_path)
        selections = self._vector_selections(config)
        for selection in selections:
            collection, _ = self._source_collection(selection.name, source, fallback)
            if collection.count() == 0:
                raise ValueError(f"collection has no vectors: {selection.name}")
        for sae in config.sae_data:
            if sae.explanation_vector_collection is None:
                continue
            collection, _ = self._source_collection(
                sae.explanation_vector_collection, source, fallback
            )
            metadata = dict(collection.metadata or {})
            actual = (metadata.get("sae_model_id"), metadata.get("sae_id"))
            expected = (sae.model_id, sae.sae_id)
            if actual != expected:
                raise SeedSnapshotConfigError(
                    "explanation vector collection SAE metadata mismatch for "
                    f"{sae.explanation_vector_collection}: expected {expected}, got {actual}"
                )

    def export(self, config: SeedSnapshotConfig, destination_path: Path) -> dict[str, int]:
        source = self._client(self.source_path)
        fallback = self._optional_client(self.fallback_path)
        destination = self._client(destination_path)
        counts: dict[str, int] = {}
        for selection in self._vector_selections(config):
            source_collection, _ = self._source_collection(selection.name, source, fallback)
            destination_collection = destination.create_collection(
                name=selection.name,
                metadata=dict(source_collection.metadata or {}),
            )
            total = source_collection.count()
            copied = 0
            while copied < total:
                data = source_collection.get(
                    include=["embeddings"],
                    limit=min(CHROMA_BATCH_SIZE, total - copied),
                    offset=copied,
                )
                ids = data["ids"]
                if not ids:
                    raise RuntimeError(
                        f"incomplete Chroma export for {selection.name}: "
                        f"copied {copied} of {total} vectors"
                    )
                embeddings = [list(vector) for vector in data["embeddings"]]
                destination_collection.add(ids=ids, embeddings=embeddings)
                copied += len(ids)
            source_count = source_collection.count()
            destination_count = destination_collection.count()
            if copied != total or source_count != total or destination_count != total:
                raise RuntimeError(
                    f"incomplete Chroma export for {selection.name}: "
                    f"source={source_count}, expected={total}, copied={copied}, "
                    f"destination={destination_count}"
                )
            counts[selection.name] = copied
        return counts

    @staticmethod
    def _vector_selections(
        config: SeedSnapshotConfig,
    ) -> tuple[SnapshotCollectionConfig, ...]:
        return tuple(collection for collection in config.collections if collection.include.vectors)

    @staticmethod
    def _client(path: Path):
        return chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )

    @classmethod
    def _optional_client(cls, path: Path | None):
        if path is None or not path.exists():
            return None
        return cls._client(path)

    @staticmethod
    def _source_collection(name: str, source, fallback):
        try:
            collection = source.get_collection(name)
        except Exception as error:
            collection = None
            source_error = error
        else:
            source_error = None
        if collection is not None and collection.count() > 0:
            return collection, "source"
        if fallback is not None:
            try:
                fallback_collection = fallback.get_collection(name)
            except Exception:
                fallback_collection = None
            if fallback_collection is not None and fallback_collection.count() > 0:
                return fallback_collection, "fallback"
        if collection is None:
            raise ValueError(f"Chroma collection not found: {name}") from source_error
        return collection, "source"


class SeedSnapshotBuilder:
    """Coordinate validated DuckDB and Chroma exports."""

    def __init__(
        self,
        source_duckdb_path: str | Path,
        source_chroma_path: str | Path,
        committed_seed_dir: str | Path,
        *,
        project_root: Path = PROJECT_ROOT,
    ):
        self.source_duckdb_path = Path(source_duckdb_path).expanduser().resolve()
        self.source_chroma_path = Path(source_chroma_path).expanduser().resolve()
        self.committed_seed_dir = Path(committed_seed_dir).expanduser().resolve()
        self.project_root = project_root.resolve()

    def build(self, config: SeedSnapshotConfig) -> dict[str, Any]:
        duckdb_exporter = DuckDBSnapshotExporter(self.source_duckdb_path)
        duckdb_exporter.validate(config)

        fallback_temp: Path | None = None
        fallback_vector_path: Path | None = None
        committed_vectors = self.committed_seed_dir / "vector_db"
        if committed_vectors.exists():
            fallback_temp = Path(tempfile.mkdtemp(prefix="orrery_seed_fallback_"))
            fallback_vector_path = fallback_temp / "vector_db"
            shutil.copytree(committed_vectors, fallback_vector_path)

        config.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staged_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{config.output_dir.name}.staging_",
                dir=config.output_dir.parent,
            )
        )
        try:
            chroma_exporter = ChromaSnapshotExporter(
                self.source_chroma_path,
                fallback_vector_path,
            )
            chroma_exporter.validate(config)
            duckdb_counts = duckdb_exporter.export(config, staged_dir / "main.duckdb")
            chroma_counts = chroma_exporter.export(config, staged_dir / "vector_db")
            metadata = {
                "snapshot_name": config.name,
                "config_sha256": config.config_sha256,
                "source_git_commit": self._git_commit(),
                "collections": list(config.collection_names),
                "counts": {"duckdb": duckdb_counts, "chroma": chroma_counts},
            }
            manifest_sha256 = SnapshotIntegrity.write_manifest(staged_dir, metadata)
            SnapshotIntegrity.verify(
                staged_dir,
                manifest_sha256,
                expected_snapshot_name=config.name,
                expected_config_sha256=config.config_sha256,
            )
            SnapshotDirectoryInstaller.install(staged_dir, config.output_dir)
            return {
                "output": str(config.output_dir),
                "manifest_sha256": manifest_sha256,
                **metadata,
            }
        except Exception:
            if staged_dir.exists():
                shutil.rmtree(staged_dir, ignore_errors=True)
            raise
        finally:
            if fallback_temp is not None:
                shutil.rmtree(fallback_temp, ignore_errors=True)

    def _git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None


class HuggingFaceSnapshotPublisher:
    """Publish a verified snapshot and persist its immutable Hub revision."""

    def __init__(
        self,
        *,
        api: Any | None = None,
        environ: Mapping[str, str] | None = None,
        sleep_function: Callable[[float], None] = time.sleep,
    ):
        self.environ = environ if environ is not None else os.environ
        self.api = api
        self.sleep_function = sleep_function

    def publish(self, config: SeedSnapshotConfig) -> SnapshotLock:
        if config.publish is None:
            raise SeedSnapshotConfigError("snapshot config has no publish section")
        repo_id = self.environ.get(config.publish.repo_id_env, "").strip()
        if not repo_id:
            raise SeedSnapshotConfigError(
                f"missing Hugging Face repository environment variable: "
                f"{config.publish.repo_id_env}"
            )
        token = self.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise SeedSnapshotConfigError("HF_TOKEN is required to publish a snapshot")
        manifest_sha256 = SnapshotIntegrity.verify(
            config.output_dir,
            expected_snapshot_name=config.name,
            expected_config_sha256=config.config_sha256,
        )
        api = self.api if self.api is not None else HfApi(token=token)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=config.publish.private,
            exist_ok=True,
        )
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        actual_private = getattr(repo_info, "private", None)
        if actual_private is None or bool(actual_private) != config.publish.private:
            expected_visibility = "private" if config.publish.private else "public"
            raise SeedSnapshotConfigError(
                f"Dataset repository {repo_id} must be {expected_visibility}"
            )
        commit = self._upload_with_retry(
            api,
            folder_path=config.output_dir,
            repo_id=repo_id,
            path_in_repo=config.publish.path,
            commit_message=f"Publish Orrery seed snapshot {config.name}",
        )
        revision = getattr(commit, "oid", None)
        if not isinstance(revision, str) or not revision:
            raise RuntimeError("Hugging Face upload did not return an immutable commit oid")
        lock = SnapshotLock(
            snapshot_name=config.name,
            repo_id=repo_id,
            path=config.publish.path,
            revision=revision,
            manifest_sha256=manifest_sha256,
        )
        self._write_lock(config.publish.lock_file, lock)
        return lock

    def _upload_with_retry(
        self,
        api: Any,
        *,
        folder_path: Path,
        repo_id: str,
        path_in_repo: str,
        commit_message: str,
    ) -> Any:
        for attempt in range(1, 4):
            try:
                return api.upload_folder(
                    folder_path=str(folder_path),
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_in_repo=path_in_repo,
                    # Patterns are relative to path_in_repo when both are set.
                    delete_patterns="**",
                    commit_message=commit_message,
                )
            except Exception:
                if attempt == 3:
                    raise
                self.sleep_function(float(2 ** (attempt - 1)))
        raise AssertionError("unreachable")

    @staticmethod
    def _write_lock(path: Path, lock: SnapshotLock) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(path.suffix + ".tmp")
        staged.write_text(
            json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged.replace(path)


class HuggingFaceSnapshotDownloader:
    """Install a locked private Dataset snapshot after checksum verification."""

    def __init__(
        self,
        *,
        download_function: Callable[..., str] = snapshot_download,
    ):
        self.download_function = download_function

    def download(
        self,
        config: SeedSnapshotConfig,
        *,
        token: str | None = None,
        token_file: Path | None = None,
        fallback_dir: Path | None = None,
    ) -> str:
        if config.publish is None:
            raise SeedSnapshotConfigError("snapshot config has no publish section")
        if not config.publish.lock_file.exists():
            if fallback_dir is None:
                raise SeedSnapshotConfigError(
                    f"snapshot lock not found: {config.publish.lock_file}"
                )
            return self._install_fallback(config, fallback_dir)
        lock = SnapshotLock.from_file(config.publish.lock_file)
        if lock.snapshot_name != config.name:
            raise SeedSnapshotConfigError(
                f"snapshot lock name {lock.snapshot_name!r} does not match {config.name!r}"
            )
        resolved_token = self._resolve_token(token, token_file)
        cache_root = Path(
            self.download_function(
                repo_id=lock.repo_id,
                repo_type="dataset",
                revision=lock.revision,
                allow_patterns=f"{lock.path}/**",
                token=resolved_token,
            )
        )
        source = cache_root / lock.path
        SnapshotIntegrity.verify(
            source,
            lock.manifest_sha256,
            expected_snapshot_name=config.name,
            expected_config_sha256=config.config_sha256,
        )
        staged = self._stage_copy(source, config.output_dir)
        try:
            SnapshotDirectoryInstaller.install(staged, config.output_dir)
        except Exception:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return lock.manifest_sha256

    def _install_fallback(self, config: SeedSnapshotConfig, fallback_dir: Path) -> str:
        fallback = fallback_dir.expanduser().resolve()
        if not (fallback / "main.duckdb").is_file():
            raise SeedSnapshotConfigError(f"fallback seed is invalid: {fallback}")
        staged = self._stage_copy(fallback, config.output_dir)
        try:
            SnapshotDirectoryInstaller.install(staged, config.output_dir)
        except Exception:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        manifest = config.output_dir / MANIFEST_FILENAME
        return SnapshotIntegrity.sha256_file(manifest) if manifest.exists() else "fallback"

    @staticmethod
    def _resolve_token(token: str | None, token_file: Path | None) -> str:
        if token is not None and token.strip():
            return token.strip()
        if token_file is not None and token_file.is_file():
            value = token_file.read_text(encoding="utf-8").strip()
            if value:
                return value
        raise SeedSnapshotConfigError(
            "a read-only Hugging Face token is required for the locked private seed"
        )

    @staticmethod
    def _stage_copy(source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.download_",
                dir=destination.parent,
            )
        )
        shutil.rmtree(staged)
        shutil.copytree(source, staged)
        return staged
