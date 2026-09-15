"""Read-only, audited legacy document activations from the live DuckDB store."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import duckdb
import numpy as np
import pyarrow as pa

from ...clients.duckdb_client import DuckDBClient
from .contract import GraphBundleWriter, SaeIdentity
from .counting import EventActivity


class DuckDBDocumentSource:
    """Fingerprint an explicit corpus and every stored activation, without inference.

    The old sparse table has no processing ledger. Therefore every source document
    must have at least one stored row. Thresholding may subsequently empty an event;
    such events still belong to the explicit source-document universe.
    """

    def __init__(self, path: Path, config: dict[str, Any], identity: SaeIdentity):
        self.path, self.config, self.identity = path, config, identity
        self.connection: duckdb.DuckDBPyConnection | None = None
        self._temporary: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> "DuckDBDocumentSource":
        self._temporary = tempfile.TemporaryDirectory(prefix="sae-graph-duckdb-")
        try:
            self.connection = duckdb.connect(str(self.path), read_only=True)
            self.connection.execute("SET memory_limit='512MB'")
            self.connection.execute("SET threads=2")
            self.connection.execute("SET temp_directory=?", [self._temporary.name])
            self.connection.execute("BEGIN TRANSACTION")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_args) -> None:
        if self.connection is not None:
            self.connection.close()
        if self._temporary is not None:
            self._temporary.cleanup()

    def _dataset(self, collection: str) -> tuple[str, str]:
        row = self.connection.execute(
            "SELECT d.name,d.extra_metadata FROM vector_collections v "
            "JOIN datasets d ON d.name=v.dataset_name WHERE v.collection_name=?", [collection]
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown collection: {collection}")
        metadata = json.loads(row[1] or "{}")
        if (metadata.get("sae_model_id"), metadata.get("sae_id")) != (
                self.identity.modelId, self.identity.saeId):
            raise ValueError(f"SAE metadata mismatch: {collection}")
        table = '"items_' + DuckDBClient._sanitize_table_name(row[0]) + '"'
        return row[0], table

    def _validate_linked_collections(self) -> None:
        for collection in self.config.get("linkedCollectionNames", []):
            _, table = self._dataset(collection)
            fields = {row[0] for row in self.connection.execute(f"DESCRIBE {table}").fetchall()}
            if "metadata" not in fields:
                raise ValueError(f"not a feature collection: {collection}")
            count, invalid, unique_count = self.connection.execute(f"""
                SELECT count(*), count(*) FILTER (
                    WHERE try_cast(json_extract(metadata, '$.index') AS DOUBLE) IS NULL
                       OR try_cast(json_extract(metadata, '$.density') AS DOUBLE) IS NULL
                       OR try_cast(json_extract(metadata, '$.index') AS DOUBLE) < 0
                       OR try_cast(json_extract(metadata, '$.index') AS DOUBLE) >= ?
                       OR try_cast(json_extract(metadata, '$.index') AS DOUBLE) !=
                          floor(try_cast(json_extract(metadata, '$.index') AS DOUBLE))),
                    count(DISTINCT try_cast(json_extract(metadata, '$.index') AS DOUBLE))
                FROM {table}
            """, [self.identity.width]).fetchone()
            if not count or invalid or unique_count != count:
                raise ValueError(f"invalid feature collection: {collection}")

    def read(self, progress: Callable[[str], None] | None = None
             ) -> tuple[EventActivity, list[str], dict[str, Any], str]:
        if self.config.get("allowLegacyProvenance") is not True:
            raise ValueError("legacy source requires explicit provenance acknowledgment")
        collection = self.config["collectionName"]
        threshold = self.config.get("threshold", 0.)
        if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        dataset, table = self._dataset(collection)
        self._validate_linked_collections()
        documents = self.connection.execute(f"SELECT id,document FROM {table} ORDER BY id").fetchall()
        document_ids = [row[0] for row in documents]
        if not documents or len(set(document_ids)) != len(documents):
            raise ValueError("empty or duplicate source documents")
        if len(documents) != self.config["expectedDocumentCount"]:
            raise ValueError("source document coverage differs from configured expectation")
        corpus_hash = hashlib.sha256()
        for row in documents:
            corpus_hash.update(GraphBundleWriter.json_bytes(row))
        self.connection.register("graph_events", pa.table({
            "item_id": document_ids, "event_index": np.arange(len(documents), dtype=np.uint32)}))
        activity = EventActivity(self.identity.width, len(documents))
        raw_counts = np.zeros(len(documents), dtype=np.int64)
        source_rows = 0
        source_hash = hashlib.sha256()
        last_pair = -1
        activation_min, activation_max = float("inf"), float("-inf")
        # Numeric event indices eliminate repeated document strings in Arrow batches.
        # The sort can spill to the temporary directory; there is no pairwise SQL join.
        reader = self.connection.execute("""
            SELECT e.event_index, a.feature_index, a.activation
            FROM sae_document_activations a
            LEFT JOIN graph_events e ON a.item_id=e.item_id
            WHERE a.collection_name=?
            ORDER BY e.event_index, a.feature_index
        """, [collection]).to_arrow_reader(batch_size=131072)
        for batch in reader:
            if any(column.null_count for column in batch.columns):
                raise ValueError("null activation or activation outside source event universe")
            events = batch.column(0).to_numpy().astype(np.int64)
            features = batch.column(1).to_numpy().astype(np.int64)
            values = batch.column(2).to_numpy()
            if (np.any(features < 0) or np.any(features >= self.identity.width)
                    or not np.isfinite(values).all() or np.any(values <= 0)):
                raise ValueError("invalid source feature/activation")
            pairs = events * self.identity.width + features
            if pairs.size and (pairs[0] <= last_pair or np.any(np.diff(pairs) <= 0)):
                raise ValueError("duplicate or unordered source feature/event rows")
            if pairs.size:
                last_pair = int(pairs[-1])
                activation_min = min(activation_min, float(values.min()))
                activation_max = max(activation_max, float(values.max()))
            # Fixed little-endian interleaved records: hash independent of batch size.
            records = np.empty(len(events), dtype=[("event", "<u4"), ("feature", "<u4"),
                                                   ("activation", "<f4")])
            records["event"], records["feature"], records["activation"] = events, features, values
            source_hash.update(records.tobytes())
            raw_counts += np.bincount(events, minlength=len(documents))
            keep = values > threshold
            activity.add(events[keep], features[keep])
            source_rows += len(events)
            if progress and source_rows % (131072 * 40) == 0:
                progress(f"Read {source_rows:,} activation rows")
        reader.close()
        if source_rows != self.config["expectedSparseRowCount"]:
            raise ValueError("source rows differ from expectation; possible truncated seed")
        if np.any(raw_counts == 0):
            raise ValueError("incomplete processing coverage: sparse absence cannot prove a zero event")
        labels = [f"Feature {i}" for i in range(self.identity.width)]
        features = self.connection.execute(
            "SELECT feature_index,label FROM sae_features WHERE model_id=? AND sae_id=? "
            "ORDER BY feature_index", [self.identity.modelId, self.identity.saeId]).fetchall()
        seen = set()
        for feature, label in features:
            if feature in seen or not 0 <= feature < self.identity.width:
                raise ValueError("invalid feature dictionary")
            seen.add(feature)
            if label and label.strip():
                labels[feature] = label.strip()
        combined = {"dataset": dataset, "collection": collection,
                    "corpusSha256": corpus_hash.hexdigest(),
                    "activationRowsSha256": source_hash.hexdigest()}
        revision = "sha256:" + hashlib.sha256(GraphBundleWriter.json_bytes(combined)).hexdigest()
        # Count events made empty by the configured activity threshold without unpacking.
        any_feature = np.bitwise_or.reduce(activity.bits, axis=0)
        active_events = int(np.bitwise_count(any_feature).sum(dtype=np.uint64))
        coverage = {**combined, "sourceDocumentCount": len(documents),
                    "processedEventCount": len(documents), "sparseRowCount": source_rows,
                    "sourceEventsWithoutStoredRows": 0,
                    "zeroActivationEventCountAfterThreshold": len(documents) - active_events,
                    "storedFeaturesPerDocument": GraphBundleWriter.distribution(raw_counts),
                    "storedActivationMinimum": activation_min, "storedActivationMaximum": activation_max,
                    "dictionaryMetadataRows": len(features),
                    "fallbackLabelCount": sum(label == f"Feature {i}" for i, label in enumerate(labels)),
                    "identityEvidence": "dataset SAE metadata; explicit legacy-source acknowledgment",
                    "historicalInferenceRevision": None,
                    "coveragePolicy": "every corpus document has stored rows; absent rows fail closed"}
        return activity, labels, coverage, revision
