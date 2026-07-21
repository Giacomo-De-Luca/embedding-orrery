"""
DuckDB client — central orchestrator for document storage, metadata,
projections, topic assignments, and vector collection references.

Uses table-per-dataset for items: each dataset gets its own
"items_{dataset_id}" table. This ensures FTS indexes, scans,
and queries are scoped to a single dataset with zero cross-dataset overhead.

Global tables (small, registry-style): datasets, vector_collections,
projections, projection_metadata, topic_extractions, topic_info, topic_assignments.
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from ..utils.resource_paths import DUCKDB_PATH

logger = logging.getLogger("orrery." + __name__)

# Fields to strip from item metadata (stored in their own tables)
_PROJECTION_KEYS = frozenset({"pca_2d", "pca_3d", "umap_2d", "umap_3d"})
_TOPIC_KEYS = frozenset(
    {
        "topic_id",
        "topic_label",
        "subtopic_id",
        "subtopic_label",
        "ctfidf_label",
        "ctfidf_subtopic_map",
    }
)

_SNIPPET_RADIUS = 100


def _sanitize_for_json(d: dict, json_columns: tuple[str, ...] = ()) -> dict:
    """Convert non-JSON-serializable values (datetime, etc.) to strings.

    DuckDB returns ``JSON``-typed columns as Python strings; pass their names
    in ``json_columns`` to parse them back into dicts/lists for API responses.
    """
    import datetime

    out = {}
    for k, v in d.items():
        if isinstance(v, (datetime.datetime, datetime.date)):
            out[k] = v.isoformat()
        elif k in json_columns and isinstance(v, str):
            out[k] = json.loads(v)
        else:
            out[k] = v
    return out


class DuckDBClient:
    """Central data orchestrator using DuckDB as the document/metadata store.

    Items are stored in per-dataset tables: "items_{dataset_id}".
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(DUCKDB_PATH.resolve())
        elif db_path == ":memory:":
            pass
        else:
            db_path = str(Path(db_path).resolve())

        self.db_path = db_path
        self._conn = duckdb.connect(db_path)
        # Lock for DataFrame-based inserts that must use self._conn.execute()
        # directly (DuckDB's replacement scan only finds Python variables in
        # the immediate caller's frame — cursor-based _exec adds an extra frame).
        # Non-DataFrame writes go through _exec (cursor) and rely on DuckDB's
        # internal write serialization.
        self._write_lock = threading.Lock()
        self._fts_dirty: set[str] = set()  # dataset_ids needing FTS rebuild
        self._ensure_schema()

    def _exec(self, sql, params=None):
        """Execute SQL on a fresh cursor (thread-safe).

        DuckDB cursors from the same connection are safe for concurrent use
        from different threads.  Each call creates a short-lived cursor so
        background threads (embedding, topic extraction, SAE inference) and
        the main event-loop thread never contend on the same query state.

        **Not for DataFrame references**: DuckDB's replacement scan resolves
        Python variable names from the immediate caller's frame.  Because
        ``_exec`` adds an intermediary frame, ``SELECT * FROM df`` won't
        find a caller's local ``df``.  Use ``self._write_lock`` +
        ``self._conn.execute()`` directly for those (see insert methods).

        Returns the cursor (supports ``.fetchall()``, ``.fetchone()``,
        ``.description``).
        """
        cur = self._conn.cursor()
        if params is not None:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        return cur

    def _execmany(self, sql, values):
        """executemany on a fresh cursor (thread-safe). Returns the cursor."""
        cur = self._conn.cursor()
        cur.executemany(sql, values)
        return cur

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _sanitize_table_name(name: str) -> str:
        """Sanitize a name for use as a SQL table identifier."""
        import re

        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    def _items_table_name(self, dataset_name: str) -> str:
        """Unquoted table name for a dataset's items (for FTS PRAGMA)."""
        return f"items_{self._sanitize_table_name(dataset_name)}"

    def _items_table(self, dataset_name: str) -> str:
        """Quoted table name for a dataset's items (for SQL statements)."""
        return f'"{self._items_table_name(dataset_name)}"'

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self):
        """Create global tables if they don't exist. Idempotent."""
        self._exec("""
            CREATE TABLE IF NOT EXISTS datasets (
                name                VARCHAR PRIMARY KEY,
                description         VARCHAR,
                source_type         VARCHAR,
                source_dataset      VARCHAR,
                source_config       VARCHAR,
                source_split        VARCHAR,
                source_file         VARCHAR,
                embedded_columns    JSON,
                data_type           VARCHAR,
                total_in_source     INTEGER,
                item_count          INTEGER DEFAULT 0,
                created_at          TIMESTAMP DEFAULT current_timestamp,
                extra_metadata      JSON DEFAULT '{}'
            )
        """)

        self._exec("""
            CREATE TABLE IF NOT EXISTS vector_collections (
                collection_name     VARCHAR PRIMARY KEY,
                dataset_name        VARCHAR NOT NULL REFERENCES datasets(name),
                backend             VARCHAR NOT NULL,
                vector_type         VARCHAR NOT NULL,
                embedding_provider  VARCHAR,
                embedding_model     VARCHAR,
                embedding_dim       INTEGER,
                embedding_task      VARCHAR,
                embedding_task_type VARCHAR,
                embedding_prompt    VARCHAR,
                item_count          INTEGER DEFAULT 0,
                has_projections     BOOLEAN DEFAULT FALSE,
                has_topics          BOOLEAN DEFAULT FALSE,
                created_at          TIMESTAMP DEFAULT current_timestamp
            )
        """)

        self._exec("""
            CREATE TABLE IF NOT EXISTS projections (
                collection_name     VARCHAR NOT NULL REFERENCES vector_collections(collection_name),
                item_id             VARCHAR NOT NULL,
                projection_type     VARCHAR NOT NULL,
                coordinates         FLOAT[] NOT NULL,
                PRIMARY KEY (collection_name, item_id, projection_type)
            )
        """)

        self._exec("""
            CREATE TABLE IF NOT EXISTS projection_metadata (
                collection_name     VARCHAR NOT NULL REFERENCES vector_collections(collection_name),
                projection_type     VARCHAR NOT NULL,
                variance            FLOAT[],
                computed_at         TIMESTAMP,
                PRIMARY KEY (collection_name, projection_type)
            )
        """)

        self._exec("""
            CREATE TABLE IF NOT EXISTS topic_extractions (
                id                          VARCHAR PRIMARY KEY,
                collection_name             VARCHAR NOT NULL REFERENCES vector_collections(collection_name),
                dataset_name                VARCHAR NOT NULL REFERENCES datasets(name),
                config                      JSON,
                extracted_at                TIMESTAMP DEFAULT current_timestamp,
                topic_count                 INTEGER,
                reduction_applied           BOOLEAN DEFAULT FALSE,
                reduction_method            VARCHAR,
                reduction_target            INTEGER,
                num_topics_before_reduction INTEGER,
                topic_hierarchy             JSON,
                quality_metrics             JSON,
                is_active                   BOOLEAN DEFAULT TRUE
            )
        """)

        # Migration for DBs created before quality metrics existed. Guarded by an
        # explicit column check and followed by a CHECKPOINT: replaying an
        # ALTER TABLE on this FK-referencing table from the WAL crashes DuckDB
        # 1.4/1.5 ("GetDefaultDatabase with no default database set"), so the
        # ALTER must never be a repeated no-op nor linger in the WAL.
        has_quality_metrics = self._exec(
            "SELECT count(*) FROM pragma_table_info('topic_extractions') "
            "WHERE name = 'quality_metrics'"
        ).fetchone()[0]
        if not has_quality_metrics:
            self._exec("ALTER TABLE topic_extractions ADD COLUMN quality_metrics JSON")
            self._exec("CHECKPOINT")

        self._exec("""
            CREATE TABLE IF NOT EXISTS topic_info (
                extraction_id VARCHAR NOT NULL REFERENCES topic_extractions(id),
                topic_id      INTEGER NOT NULL,
                label         VARCHAR,
                ctfidf_label  VARCHAR,
                count         INTEGER DEFAULT 0,
                keywords      JSON,
                subtopics     JSON,
                PRIMARY KEY (extraction_id, topic_id)
            )
        """)

        self._exec("""
            CREATE TABLE IF NOT EXISTS topic_assignments (
                extraction_id  VARCHAR NOT NULL REFERENCES topic_extractions(id),
                item_id        VARCHAR NOT NULL,
                topic_id       INTEGER NOT NULL,
                topic_label    VARCHAR,
                subtopic_id    INTEGER,
                subtopic_label VARCHAR,
                PRIMARY KEY (extraction_id, item_id)
            )
        """)

        # -- SAE (Sparse Autoencoder) tables --

        self._exec("""
            CREATE TABLE IF NOT EXISTS sae_features (
                model_id        VARCHAR NOT NULL,
                sae_id          VARCHAR NOT NULL,
                feature_index   INTEGER NOT NULL,
                density         FLOAT,
                label           VARCHAR,
                top_logits      JSON,
                bottom_logits   JSON,
                created_at      TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (model_id, sae_id, feature_index)
            )
        """)

        self._exec("""
            CREATE TABLE IF NOT EXISTS sae_activations (
                id                   VARCHAR PRIMARY KEY,
                model_id             VARCHAR NOT NULL,
                sae_id               VARCHAR NOT NULL,
                feature_index        INTEGER NOT NULL,
                tokens               JSON NOT NULL,
                act_values           JSON NOT NULL,
                max_value            FLOAT,
                max_value_token_idx  INTEGER,
                min_value            FLOAT,
                qualifying_token_idx INTEGER
            )
        """)

        self._exec("""
            CREATE INDEX IF NOT EXISTS idx_sae_act_feature
            ON sae_activations (model_id, sae_id, feature_index)
        """)

        self._exec("""
            CREATE INDEX IF NOT EXISTS idx_sae_features_model
            ON sae_features (model_id)
        """)

        # -- SAE document activation table (per-document max-pooled activations) --
        self._exec("""
            CREATE TABLE IF NOT EXISTS sae_document_activations (
                collection_name  VARCHAR NOT NULL,
                item_id          VARCHAR NOT NULL,
                feature_index    INTEGER NOT NULL,
                activation       FLOAT   NOT NULL,
                PRIMARY KEY (collection_name, item_id, feature_index)
            )
        """)

        self._exec("""
            CREATE INDEX IF NOT EXISTS idx_sae_doc_act_feature
            ON sae_document_activations (collection_name, feature_index)
        """)

        # -- Probe tables (embedding-space probes on numeric metadata fields) --
        # No FK to vector_collections: delete_dataset cleans up manually in
        # dependency order (mirrors sae_document_activations).
        self._exec("""
            CREATE TABLE IF NOT EXISTS probes (
                collection_name VARCHAR NOT NULL,
                target_field    VARCHAR NOT NULL,
                kind            VARCHAR NOT NULL,
                config          JSON,
                metrics         JSON,
                direction       FLOAT[],
                scaler_mean     FLOAT[],
                scaler_scale    FLOAT[],
                intercept       FLOAT,
                artifact_path   VARCHAR,
                n_train         INTEGER,
                n_val           INTEGER,
                created_at      TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (collection_name, target_field, kind)
            )
        """)

        self._exec("""
            CREATE TABLE IF NOT EXISTS probe_scores (
                collection_name VARCHAR NOT NULL,
                target_field    VARCHAR NOT NULL,
                kind            VARCHAR NOT NULL,
                item_id         VARCHAR NOT NULL,
                score           FLOAT NOT NULL,
                residual        FLOAT,
                PRIMARY KEY (collection_name, target_field, kind, item_id)
            )
        """)

        # -- Chat history tables --

        self._exec("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id          VARCHAR PRIMARY KEY,
                title       VARCHAR NOT NULL,
                config      JSON NOT NULL,
                created_at  TIMESTAMP DEFAULT current_timestamp,
                updated_at  TIMESTAMP DEFAULT current_timestamp
            )
        """)

        self._exec("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id                VARCHAR PRIMARY KEY,
                session_id        VARCHAR NOT NULL REFERENCES chat_sessions(id),
                role              VARCHAR NOT NULL,
                content           VARCHAR NOT NULL,
                parts             JSON,
                steering_snapshot JSON,
                created_at        TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # Idempotent migration for DBs created before steering_snapshot existed.
        self._exec("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS steering_snapshot JSON")

        self._exec("""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session
            ON chat_messages (session_id, created_at)
        """)

        # FTS extension
        self._exec("INSTALL fts")
        self._exec("LOAD fts")

        logger.info("DuckDB schema ensured at %s", self.db_path)

    def _ensure_items_table(self, dataset_name: str):
        """Create the per-dataset items table if it doesn't exist."""
        table = self._items_table(dataset_name)
        self._exec(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id          VARCHAR PRIMARY KEY,
                document    VARCHAR,
                metadata    JSON,
                row_index   INTEGER
            )
        """)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        name: str,
        *,
        description: str = None,
        source_type: str = None,
        source_dataset: str = None,
        source_config: str = None,
        source_split: str = None,
        source_file: str = None,
        embedded_columns: list = None,
        data_type: str = None,
        total_in_source: int = None,
        extra_metadata: dict = None,
    ) -> str:
        """Create a new dataset + its items table. Returns the dataset name."""
        self._exec(
            """
            INSERT INTO datasets (name, description, source_type, source_dataset,
                source_config, source_split, source_file, embedded_columns,
                data_type, total_in_source, extra_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                name,
                description,
                source_type,
                source_dataset,
                source_config,
                source_split,
                source_file,
                json.dumps(embedded_columns) if embedded_columns else None,
                data_type,
                total_in_source,
                json.dumps(extra_metadata or {}),
            ],
        )
        self._ensure_items_table(name)
        logger.info("Created dataset %r", name)
        return name

    def list_datasets(self) -> list[dict[str, Any]]:
        """List all collections (one entry per vector_collection). Returns [{name, metadata, count}].

        Each vector_collection is returned as a separate entry, enriched with its
        parent dataset's item count and metadata. Datasets without any vector_collections
        are also included (for re-embedding as a data source).
        """
        ds_cur = self._exec("SELECT * FROM datasets ORDER BY created_at DESC")
        rows = ds_cur.fetchall()
        columns = [desc[0] for desc in ds_cur.description]
        datasets_by_name: dict[str, dict[str, Any]] = {}
        for row in rows:
            d = _sanitize_for_json(dict(zip(columns, row, strict=True)))
            datasets_by_name[d["name"]] = d

        # Get all vector_collections
        vc_cur = self._exec("SELECT * FROM vector_collections ORDER BY created_at DESC")
        vc_rows = vc_cur.fetchall()
        vc_columns = [desc[0] for desc in vc_cur.description]
        all_vcs = [dict(zip(vc_columns, vr, strict=True)) for vr in vc_rows]

        # Build lookup of active topic extractions keyed by collection_name
        te_rows = self._exec(
            "SELECT collection_name, topic_count, topic_hierarchy FROM topic_extractions WHERE is_active = TRUE"
        ).fetchall()
        te_by_collection: dict[str, dict[str, Any]] = {}
        for tr in te_rows:
            te_by_collection[tr[0]] = {"topic_count": tr[1], "topic_hierarchy": tr[2]}

        # Track which datasets have at least one vector_collection
        datasets_with_vc: set[str] = set()
        result = []

        # One entry per vector_collection
        for vc in all_vcs:
            ds_name = vc["dataset_name"]
            datasets_with_vc.add(ds_name)
            ds = datasets_by_name.get(ds_name)
            if not ds:
                continue

            d = dict(ds)  # Copy dataset fields
            count = d.pop("item_count", 0)

            # Override with vector_collection-specific fields
            d["embedding_provider"] = vc.get("embedding_provider")
            d["embedding_model"] = vc.get("embedding_model")
            d["embedding_dim"] = vc.get("embedding_dim")
            d["has_projections"] = vc.get("has_projections", False)
            d["has_topics"] = vc.get("has_topics", False)
            d["dataset_name"] = ds_name  # Track parent dataset

            # Enrich with topic extraction data
            te = te_by_collection.get(vc["collection_name"])
            if te:
                d["topic_count"] = te["topic_count"]
                if te["topic_hierarchy"]:
                    d["topic_hierarchy"] = te["topic_hierarchy"]

            # Parse extra_metadata JSON into top-level keys
            extra = d.pop("extra_metadata", None)
            if extra:
                if isinstance(extra, str):
                    try:
                        extra = json.loads(extra)
                    except (json.JSONDecodeError, TypeError):
                        extra = None
                if isinstance(extra, dict):
                    d.update(extra)

            # Use collection_name as the entry name (not dataset name)
            d["name"] = vc["collection_name"]
            result.append({"name": vc["collection_name"], "metadata": d, "count": count})

        # Include datasets without any vector_collections (bare datasets)
        for ds_name, ds in datasets_by_name.items():
            if ds_name not in datasets_with_vc:
                d = dict(ds)
                count = d.pop("item_count", 0)
                extra = d.pop("extra_metadata", None)
                if extra:
                    if isinstance(extra, str):
                        try:
                            extra = json.loads(extra)
                        except (json.JSONDecodeError, TypeError):
                            extra = None
                    if isinstance(extra, dict):
                        d.update(extra)
                result.append({"name": d["name"], "metadata": d, "count": count})

        return result

    def get_dataset(self, name: str) -> dict[str, Any] | None:
        """Get dataset by name."""
        cur = self._exec("SELECT * FROM datasets WHERE name = ?", [name])
        rows = cur.fetchall()
        if not rows:
            return None
        columns = [desc[0] for desc in cur.description]
        d = dict(zip(columns, rows[0], strict=True))
        d["count"] = d.get("item_count", 0)
        return d

    def update_dataset(self, name: str, **kwargs) -> None:
        """Update dataset fields by name.

        For ``extra_metadata``, new keys are **merged** into the existing JSON
        rather than replacing it.  Keys explicitly set to ``None`` are removed.
        """
        if not kwargs:
            return

        if "extra_metadata" in kwargs and isinstance(kwargs["extra_metadata"], dict):
            existing = self._exec(
                "SELECT extra_metadata FROM datasets WHERE name = ?", [name]
            ).fetchone()
            old = {}
            if existing and existing[0]:
                old = json.loads(existing[0]) if isinstance(existing[0], str) else existing[0]
            merged = {**old, **kwargs["extra_metadata"]}
            merged = {k: v for k, v in merged.items() if v is not None}
            kwargs["extra_metadata"] = merged

        set_parts = []
        values = []
        for key, val in kwargs.items():
            set_parts.append(f"{key} = ?")
            if isinstance(val, (dict, list)):
                values.append(json.dumps(val))
            else:
                values.append(val)
        values.append(name)
        self._exec(
            f"UPDATE datasets SET {', '.join(set_parts)} WHERE name = ?",
            values,
        )

    def delete_dataset(self, name: str) -> bool:
        """Delete dataset, its items table, and all related data."""
        ds = self.get_dataset(name)
        if not ds:
            return False

        # Delete in dependency order
        self._exec(
            """
            DELETE FROM topic_assignments WHERE extraction_id IN (
                SELECT id FROM topic_extractions WHERE dataset_name = ?
            )
        """,
            [name],
        )
        self._exec(
            """
            DELETE FROM topic_info WHERE extraction_id IN (
                SELECT id FROM topic_extractions WHERE dataset_name = ?
            )
        """,
            [name],
        )
        self._exec("DELETE FROM topic_extractions WHERE dataset_name = ?", [name])

        self._exec(
            """
            DELETE FROM projection_metadata WHERE collection_name IN (
                SELECT collection_name FROM vector_collections WHERE dataset_name = ?
            )
        """,
            [name],
        )
        self._exec(
            """
            DELETE FROM projections WHERE collection_name IN (
                SELECT collection_name FROM vector_collections WHERE dataset_name = ?
            )
        """,
            [name],
        )

        self._exec(
            """
            DELETE FROM sae_document_activations WHERE collection_name IN (
                SELECT collection_name FROM vector_collections WHERE dataset_name = ?
            )
        """,
            [name],
        )

        self._exec(
            """
            DELETE FROM probe_scores WHERE collection_name IN (
                SELECT collection_name FROM vector_collections WHERE dataset_name = ?
            )
        """,
            [name],
        )
        self._exec(
            """
            DELETE FROM probes WHERE collection_name IN (
                SELECT collection_name FROM vector_collections WHERE dataset_name = ?
            )
        """,
            [name],
        )

        self._exec("DELETE FROM vector_collections WHERE dataset_name = ?", [name])

        # Drop per-dataset items table
        self._exec(f"DROP TABLE IF EXISTS {self._items_table(name)}")

        self._exec("DELETE FROM datasets WHERE name = ?", [name])

        self._fts_dirty.discard(name)
        logger.info("Deleted dataset %r", name)
        return True

    # ------------------------------------------------------------------
    # Items (per-dataset tables)
    # ------------------------------------------------------------------

    def insert_items_batch(
        self,
        dataset_name: str,
        ids: list[str],
        documents: list[str | None],
        metadatas: list[dict | None],
    ) -> int:
        """Bulk insert items into the dataset's table. Returns number inserted."""
        if not ids:
            return 0

        self._ensure_items_table(dataset_name)

        col_metas = []
        provided_row_indices = []
        for meta in metadatas:
            clean_meta = None
            row_index = None
            if meta:
                clean = {
                    k: v
                    for k, v in meta.items()
                    if k not in _PROJECTION_KEYS and k not in _TOPIC_KEYS
                }
                row_index = meta.get("row_index")
                clean.pop("row_index", None)
                clean_meta = json.dumps(clean) if clean else None
            col_metas.append(clean_meta)
            provided_row_indices.append(row_index)

        table = self._items_table(dataset_name)
        with self._write_lock:
            current_max = self._conn.execute(
                f"SELECT COALESCE(MAX(row_index), -1) FROM {table}"
            ).fetchone()[0]
            legacy_ids = [
                row[0]
                for row in self._conn.execute(
                    f"SELECT id FROM {table} WHERE row_index IS NULL ORDER BY rowid"
                ).fetchall()
            ]
            if legacy_ids:
                legacy_updates = [
                    (current_max + offset, item_id)
                    for offset, item_id in enumerate(legacy_ids, start=1)
                ]
                self._conn.executemany(
                    f"UPDATE {table} SET row_index = ? WHERE id = ?",
                    legacy_updates,
                )
                current_max += len(legacy_ids)

            explicit_max = max(
                (int(value) for value in provided_row_indices if value is not None),
                default=-1,
            )
            next_row_index = max(current_max, explicit_max) + 1
            col_row_indices = []
            for value in provided_row_indices:
                if value is None:
                    col_row_indices.append(next_row_index)
                    next_row_index += 1
                else:
                    col_row_indices.append(value)

            df = pd.DataFrame(  # noqa: F841  # DuckDB reads this DataFrame by name in the SQL below
                {
                    "id": ids,
                    "document": documents,
                    "metadata": col_metas,
                    "row_index": col_row_indices,
                }
            )
            self._conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM df")

        # Update cached item count
        count = self._exec(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        self._exec("UPDATE datasets SET item_count = ? WHERE name = ?", [count, dataset_name])

        self._fts_dirty.add(dataset_name)
        return len(ids)

    def get_item_ids(self, dataset_name: str) -> set[str]:
        """Get all item IDs for a dataset."""
        table = self._items_table(dataset_name)
        try:
            rows = self._exec(f"SELECT id FROM {table}").fetchall()
        except duckdb.CatalogException:
            return set()
        return {r[0] for r in rows}

    def get_items_by_ids(self, dataset_name: str, ids: list[str]) -> list[dict[str, Any]]:
        """Get items by IDs. Used to enrich search results."""
        if not ids:
            return []
        table = self._items_table(dataset_name)
        placeholders = ", ".join(["?"] * len(ids))
        rows = self._exec(
            f"SELECT id, document, metadata, row_index FROM {table} WHERE id IN ({placeholders})",
            list(ids),
        ).fetchall()
        result = []
        for r in rows:
            item = {"id": r[0], "document": r[1], "row_index": r[3]}
            if r[2]:
                item["metadata"] = json.loads(r[2]) if isinstance(r[2], str) else r[2]
            else:
                item["metadata"] = {}
            result.append(item)
        return result

    def _build_metadata_where(self, filters: list[dict]) -> tuple:
        """Build SQL WHERE clause from metadata filter dicts.

        Returns (where_sql, params) where where_sql is "cond1 AND cond2 ..."
        or "TRUE" if no filters.
        """
        where_parts: list[str] = []
        params: list = []

        for f in filters:
            field = f["field"]
            op = f["operator"]
            value = f["value"]
            json_path = f"$.{field}"

            extract = "json_extract_string(metadata, ?)"
            params.append(json_path)

            if op == "$eq":
                where_parts.append(f"{extract} = ?")
                params.append(str(value))
            elif op == "$ne":
                where_parts.append(f"{extract} != ?")
                params.append(str(value))
            elif op == "$gt":
                where_parts.append(f"CAST({extract} AS DOUBLE) > ?")
                params.append(float(value))
            elif op == "$gte":
                where_parts.append(f"CAST({extract} AS DOUBLE) >= ?")
                params.append(float(value))
            elif op == "$lt":
                where_parts.append(f"CAST({extract} AS DOUBLE) < ?")
                params.append(float(value))
            elif op == "$lte":
                where_parts.append(f"CAST({extract} AS DOUBLE) <= ?")
                params.append(float(value))
            elif op == "$in":
                if isinstance(value, list):
                    ph = ", ".join(["?"] * len(value))
                    where_parts.append(f"{extract} IN ({ph})")
                    params.extend([str(v) for v in value])
            elif op == "$nin":
                if isinstance(value, list):
                    ph = ", ".join(["?"] * len(value))
                    where_parts.append(f"{extract} NOT IN ({ph})")
                    params.extend([str(v) for v in value])

        where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
        return where_sql, params

    def get_filtered_items(
        self, dataset_name: str, filters: list[dict], limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get items with JSON metadata filtering.

        filters: list of {field, operator, value} dicts.
                 operator: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
        """
        table = self._items_table(dataset_name)
        where_sql, params = self._build_metadata_where(filters)
        params.extend([limit, offset])

        rows = self._exec(
            f"SELECT id, document, metadata, row_index FROM {table} WHERE {where_sql} ORDER BY row_index LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        result = []
        for r in rows:
            item = {"id": r[0], "document": r[1], "row_index": r[3]}
            if r[2]:
                item["metadata"] = json.loads(r[2]) if isinstance(r[2], str) else r[2]
            else:
                item["metadata"] = {}
            result.append(item)
        return result

    # ------------------------------------------------------------------
    # Vector Collections
    # ------------------------------------------------------------------

    def register_vector_collection(
        self,
        dataset_name: str,
        backend: str,
        collection_name: str,
        vector_type: str,
        *,
        embedding_provider: str = None,
        embedding_model: str = None,
        embedding_dim: int = None,
        embedding_task: str = None,
        embedding_task_type: str = None,
        embedding_prompt: str = None,
        item_count: int = 0,
    ) -> str:
        """Register a vector collection. Returns the collection_name (PK)."""
        self._exec(
            """
            INSERT INTO vector_collections (collection_name, dataset_name, backend,
                vector_type, embedding_provider, embedding_model, embedding_dim,
                embedding_task, embedding_task_type, embedding_prompt, item_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                collection_name,
                dataset_name,
                backend,
                vector_type,
                embedding_provider,
                embedding_model,
                embedding_dim,
                embedding_task,
                embedding_task_type,
                embedding_prompt,
                item_count,
            ],
        )
        return collection_name

    def get_vector_collections(self, dataset_name: str) -> list[dict[str, Any]]:
        """Get all vector collections for a dataset."""
        cur = self._exec(
            "SELECT * FROM vector_collections WHERE dataset_name = ? ORDER BY created_at",
            [dataset_name],
        )
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, r, strict=True)) for r in rows]

    def get_vector_collection(self, collection_name: str) -> dict[str, Any] | None:
        """Get a vector collection by its collection_name (PK)."""
        cur = self._exec(
            "SELECT * FROM vector_collections WHERE collection_name = ?",
            [collection_name],
        )
        rows = cur.fetchall()
        if not rows:
            return None
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, rows[0], strict=True))

    def get_dataset_name_for_collection(self, collection_name: str) -> str | None:
        """Return the dataset_name for a vector collection, or None."""
        row = self._exec(
            "SELECT dataset_name FROM vector_collections WHERE collection_name = ?",
            [collection_name],
        ).fetchone()
        return row[0] if row else None

    def set_collection_has_topics(self, collection_name: str) -> None:
        """Mark a vector collection as having topic data."""
        self._exec(
            "UPDATE vector_collections SET has_topics = TRUE WHERE collection_name = ?",
            [collection_name],
        )

    def update_topic_extraction(self, extraction_id: str, **kwargs) -> None:
        """Update fields on a topic_extractions row."""
        if not kwargs:
            return
        set_parts = []
        values = []
        for key, val in kwargs.items():
            set_parts.append(f"{key} = ?")
            if isinstance(val, (dict, list)):
                values.append(json.dumps(val))
            else:
                values.append(val)
        values.append(extraction_id)
        self._exec(
            f"UPDATE topic_extractions SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )

    def get_topic_assignments_raw(
        self, extraction_id: str, columns: list[str] | None = None
    ) -> list[tuple]:
        """Return topic assignment rows for an extraction."""
        col_str = ", ".join(columns) if columns else "*"
        return self._exec(
            f"SELECT {col_str} FROM topic_assignments WHERE extraction_id = ?",
            [extraction_id],
        ).fetchall()

    def get_items_columns(
        self,
        dataset_name: str,
        columns: tuple[str, ...] = ("id", "document"),
        *,
        where_ids: list[str] | None = None,
    ) -> list[tuple]:
        """Return rows from the items table with the specified columns.

        If *where_ids* is provided, only those items are returned (unordered).
        Otherwise all items are returned ordered by row_index.
        """
        col_str = ", ".join(columns)
        table = self._items_table(dataset_name)
        if where_ids:
            placeholders = ", ".join(["?"] * len(where_ids))
            return self._exec(
                f"SELECT {col_str} FROM {table} WHERE id IN ({placeholders})",
                list(where_ids),
            ).fetchall()
        return self._exec(f"SELECT {col_str} FROM {table} ORDER BY row_index").fetchall()

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def insert_projections_batch(
        self,
        collection_name: str,
        item_ids: list[str],
        projection_type: str,
        coordinates: list[list[float]],
    ) -> int:
        """Bulk insert projection coordinates. Returns count inserted."""
        if not item_ids:
            return 0

        df = pd.DataFrame(  # noqa: F841  # DuckDB reads this DataFrame by name in the SQL below
            {
                "collection_name": collection_name,
                "item_id": item_ids,
                "projection_type": projection_type,
                "coordinates": coordinates,
            }
        )

        with self._write_lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO projections
                SELECT collection_name, item_id, projection_type, coordinates
                FROM df
            """)
        return len(item_ids)

    def upsert_projection_metadata(
        self,
        collection_name: str,
        projection_type: str,
        *,
        variance: list[float] = None,
        computed_at: str = None,
    ) -> None:
        """Insert or update projection metadata (variance, timestamp)."""
        if computed_at is None:
            computed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._exec(
            """
            INSERT OR REPLACE INTO projection_metadata
                (collection_name, projection_type, variance, computed_at)
            VALUES (?, ?, ?, ?)
        """,
            [collection_name, projection_type, variance, computed_at],
        )
        self._exec(
            "UPDATE vector_collections SET has_projections = TRUE WHERE collection_name = ?",
            [collection_name],
        )

    def _load_projection_rows(
        self,
        collection_name: str,
        projection_type: str,
        *,
        include_items: bool,
    ) -> tuple[list[tuple], dict[str, Any], str] | None:
        """Load ordered projection rows, optionally including shared item columns."""
        vc = self.get_vector_collection(collection_name)
        if not vc:
            return None
        dataset_name = vc["dataset_name"]
        table = self._items_table(dataset_name)
        selected_columns = (
            "i.id, i.document, i.metadata, p.coordinates"
            if include_items
            else "i.id, p.coordinates"
        )

        rows = self._exec(
            f"""
            SELECT {selected_columns}
            FROM {table} i
            INNER JOIN projections p ON p.item_id = i.id
            WHERE p.collection_name = ?
              AND p.projection_type = ?
            ORDER BY i.row_index NULLS LAST, i.rowid
        """,
            [collection_name, projection_type],
        ).fetchall()

        if not rows:
            return None

        return rows, vc, dataset_name

    @staticmethod
    def item_id_signature(item_ids: list[str]) -> str:
        """Return a stable digest of ordered item membership."""
        digest = hashlib.sha256()
        for item_id in item_ids:
            encoded = str(item_id).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def get_projection_coordinates(
        self, collection_name: str, projection_type: str
    ) -> dict[str, Any] | None:
        """Load one ordered projection plus a compact item-membership digest."""
        loaded = self._load_projection_rows(
            collection_name,
            projection_type,
            include_items=False,
        )
        if loaded is None:
            return None

        rows, _, _ = loaded
        dimensions = 3 if projection_type.endswith("_3d") else 2
        ids = [row[0] for row in rows]
        return {
            "coordinates": [
                list(row[1]) if row[1] else [0.0] * dimensions for row in rows
            ],
            "item_signature": self.item_id_signature(ids),
        }

    def get_projection_data(
        self, collection_name: str, projection_type: str
    ) -> dict[str, Any] | None:
        """Load items + one projection type for visualization."""
        loaded = self._load_projection_rows(
            collection_name,
            projection_type,
            include_items=True,
        )
        if loaded is None:
            return None

        rows, vc, dataset_name = loaded
        ids = []
        documents = []
        item_metadata = []
        coordinates = []
        available_fields = set()

        for r in rows:
            ids.append(r[0])
            documents.append(r[1])
            meta = json.loads(r[2]) if isinstance(r[2], str) and r[2] else {}
            item_metadata.append(meta)
            available_fields.update(meta.keys())
            coordinates.append(list(r[3]) if r[3] else [0.0, 0.0])

        ds = self.get_dataset(dataset_name)

        pm_rows = self._exec(
            """
            SELECT projection_type, variance
            FROM projection_metadata WHERE collection_name = ?
        """,
            [collection_name],
        ).fetchall()
        variance_map = {r[0]: r[1] for r in pm_rows}

        # Parse extra_metadata (stored as JSON string)
        raw_extra = ds.get("extra_metadata") if ds else None
        extra = json.loads(raw_extra) if isinstance(raw_extra, str) else (raw_extra or {})

        metadata = {
            "total_items": len(ids),
            "embedding_dim": vc.get("embedding_dim"),
            "embedding_provider": vc.get("embedding_provider"),
            "embedding_model": vc.get("embedding_model"),
            "embedding_prompt": vc.get("embedding_prompt"),
            "has_projections": vc.get("has_projections", False),
            "pca_2d_variance": variance_map.get("pca_2d"),
            "pca_3d_variance": variance_map.get("pca_3d"),
            "timestamp": str(ds.get("created_at", "")) if ds else "",
            "source_dataset": ds.get("source_dataset") if ds else None,
            "source_split": ds.get("source_split") if ds else None,
            "source_file": ds.get("source_file") if ds else None,
            "embedded_columns": ds.get("embedded_columns") if ds else None,
            "sae_model_id": extra.get("sae_model_id"),
            "sae_id": extra.get("sae_id"),
        }

        return {
            "ids": ids,
            "item_signature": self.item_id_signature(ids),
            "documents": documents,
            "item_metadata": item_metadata,
            "available_fields": sorted(available_fields),
            "coordinates": coordinates,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Text Search (per-dataset, no cross-dataset pollution)
    # ------------------------------------------------------------------

    def _rebuild_fts_for_dataset(self, dataset_name: str):
        """Rebuild the FTS index for one dataset's items table."""
        table = self._items_table(dataset_name)
        table_unquoted = self._items_table_name(dataset_name)

        count = self._exec(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count == 0:
            return

        try:
            self._exec(f"PRAGMA drop_fts_index('{table_unquoted}')")
        except duckdb.CatalogException:
            pass

        self._exec(f"""
            PRAGMA create_fts_index('{table_unquoted}', 'id', 'document',
                stemmer='porter', stopwords='english', lower=1, overwrite=1)
        """)
        self._fts_dirty.discard(dataset_name)
        logger.info("Rebuilt FTS index for %s (%d rows)", dataset_name, count)

    @staticmethod
    def _text_match_condition(column_sql: str, mode: str, case_sensitive: bool) -> str:
        """SQL condition matching `column_sql` against one `?` query placeholder."""
        if mode == "exact":
            if case_sensitive:
                return f"{column_sql} = ?"
            return f"LOWER({column_sql}) = LOWER(?)"
        if case_sensitive:
            return f"{column_sql} LIKE '%' || ? || '%'"
        return f"{column_sql} ILIKE '%' || ? || '%'"

    _METADATA_MATCH_COLUMN = "CAST(json_extract_string(metadata, ?) AS VARCHAR)"

    def text_search(
        self,
        dataset_name: str,
        query: str,
        fields: list[str] | None = None,
        mode: str = "contains",
        case_sensitive: bool = False,
        filters: list[dict] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Search documents and/or metadata fields within one dataset.

        When filters are provided, results are restricted to items matching
        the metadata filters (same operators as get_filtered_items).

        `total_matches` is always the full deduplicated match count; when
        `limit` is set, `matches` is truncated to at most that many rows so
        broad queries don't ship the whole table over the wire.
        """
        ds = self.get_dataset(dataset_name)
        if not ds:
            return {"matches": [], "total_matches": 0}
        table = self._items_table(dataset_name)

        if fields is None:
            fields = ["__document__"]
        meta_fields = [f for f in fields if f != "__document__"]

        filter_sql: str | None = None
        filter_params: list = []
        if filters:
            filter_sql, filter_params = self._build_metadata_where(filters)

        # True match count across all selected fields (rows deduplicated by OR).
        conditions: list[str] = []
        count_params: list = []
        if "__document__" in fields:
            conditions.append(self._text_match_condition("document", mode, case_sensitive))
            count_params.append(query)
        for field in meta_fields:
            conditions.append(
                self._text_match_condition(self._METADATA_MATCH_COLUMN, mode, case_sensitive)
            )
            count_params.extend([f"$.{field}", query])
        if not conditions:
            return {"matches": [], "total_matches": 0}
        count_where = "(" + " OR ".join(conditions) + ")"
        if filter_sql:
            count_where += f" AND ({filter_sql})"
            count_params.extend(filter_params)
        total_matches = self._exec(
            f"SELECT COUNT(*) FROM {table} WHERE {count_where}", count_params
        ).fetchone()[0]

        matches = []
        seen_ids = set()

        if "__document__" in fields:
            doc_matches = self._search_documents(
                table, query, mode, case_sensitive, filter_sql, filter_params, limit
            )
            for m in doc_matches:
                if m["id"] not in seen_ids:
                    matches.append(m)
                    seen_ids.add(m["id"])

        if meta_fields and (limit is None or len(matches) < limit):
            meta_matches = self._search_metadata_fields(
                table, query, meta_fields, mode, case_sensitive, filter_sql, filter_params, limit
            )
            for m in meta_matches:
                if m["id"] not in seen_ids:
                    matches.append(m)
                    seen_ids.add(m["id"])

        if limit is not None:
            matches = matches[:limit]

        return {"matches": matches, "total_matches": total_matches}

    def _search_documents(
        self,
        table: str,
        query: str,
        mode: str,
        case_sensitive: bool,
        filter_sql: str | None = None,
        filter_params: list | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Search the document column of a per-dataset items table."""
        condition = self._text_match_condition("document", mode, case_sensitive)
        sql = f"SELECT id, document FROM {table} WHERE {condition}"
        params: list = [query]
        if filter_sql:
            sql += f" AND ({filter_sql})"
            params.extend(filter_params or [])
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self._exec(sql, params).fetchall()

        matches = []
        query_lower = query.lower()
        for r in rows:
            doc = r[1] or ""
            pos = doc.lower().find(query_lower)
            snippet = None
            if pos >= 0:
                start = max(0, pos - _SNIPPET_RADIUS)
                end = min(len(doc), pos + len(query) + _SNIPPET_RADIUS)
                snippet = (
                    ("..." if start > 0 else "")
                    + doc[start:end]
                    + ("..." if end < len(doc) else "")
                )
            matches.append({"id": r[0], "matched_field": "__document__", "snippet": snippet})
        return matches

    def _search_metadata_fields(
        self,
        table: str,
        query: str,
        fields: list[str],
        mode: str,
        case_sensitive: bool,
        filter_sql: str | None = None,
        filter_params: list | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Search metadata JSON fields in a per-dataset items table."""
        condition = self._text_match_condition(self._METADATA_MATCH_COLUMN, mode, case_sensitive)
        matches = []
        for field in fields:
            sql = f"SELECT id FROM {table} WHERE {condition}"
            params: list = [f"$.{field}", query]
            if filter_sql:
                sql += f" AND ({filter_sql})"
                params.extend(filter_params or [])
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            rows = self._exec(sql, params).fetchall()

            for r in rows:
                matches.append({"id": r[0], "matched_field": field, "snippet": None})
        return matches

    def text_search_bm25(
        self, dataset_name: str, query: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Word-level BM25 search using per-dataset FTS index."""
        ds = self.get_dataset(dataset_name)
        if not ds:
            return []
        table = self._items_table(dataset_name)
        table_unquoted = self._items_table_name(dataset_name)
        fts_schema = f"fts_main_{table_unquoted}"

        # Rebuild FTS if dirty
        if dataset_name in self._fts_dirty:
            self._rebuild_fts_for_dataset(dataset_name)

        rows = self._exec(
            f"""
            SELECT id, document, metadata,
                   {fts_schema}.match_bm25(id, ?) AS score
            FROM {table}
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        """,
            [query, limit],
        ).fetchall()

        results = []
        for r in rows:
            results.append(
                {
                    "id": r[0],
                    "document": r[1],
                    "metadata": json.loads(r[2]) if isinstance(r[2], str) and r[2] else {},
                    "score": r[3],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    def create_topic_extraction(
        self, collection_name: str, dataset_name: str, config: dict = None
    ) -> str:
        """Create a new topic extraction. Deactivates previous."""
        self._exec(
            """
            UPDATE topic_extractions SET is_active = FALSE
            WHERE collection_name = ? AND is_active = TRUE
        """,
            [collection_name],
        )

        extraction_id = str(uuid.uuid4())
        self._exec(
            """
            INSERT INTO topic_extractions (id, collection_name, dataset_name, config, is_active)
            VALUES (?, ?, ?, ?, TRUE)
        """,
            [extraction_id, collection_name, dataset_name, json.dumps(config) if config else None],
        )
        return extraction_id

    def insert_topic_info_batch(self, extraction_id: str, topics: list[dict]) -> int:
        """Insert topic info records."""
        values = []
        for t in topics:
            values.append(
                (
                    extraction_id,
                    t["topic_id"],
                    t.get("label"),
                    t.get("ctfidf_label"),
                    t.get("count", 0),
                    json.dumps(t.get("keywords")) if t.get("keywords") else None,
                    json.dumps(t.get("subtopics")) if t.get("subtopics") else None,
                )
            )
        self._execmany(
            """
            INSERT INTO topic_info (extraction_id, topic_id, label, ctfidf_label,
                count, keywords, subtopics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            values,
        )
        return len(values)

    def insert_topic_assignments_batch(self, extraction_id: str, assignments: list[dict]) -> int:
        """Insert topic assignments via DataFrame."""
        if not assignments:
            return 0

        df = pd.DataFrame(  # noqa: F841  # DuckDB reads this DataFrame by name in the SQL below
            {
                "extraction_id": extraction_id,
                "item_id": [a["item_id"] for a in assignments],
                "topic_id": [a["topic_id"] for a in assignments],
                "topic_label": [a.get("topic_label") for a in assignments],
                "subtopic_id": [a.get("subtopic_id") for a in assignments],
                "subtopic_label": [a.get("subtopic_label") for a in assignments],
            }
        )

        with self._write_lock:
            self._conn.execute("""
                INSERT INTO topic_assignments
                SELECT extraction_id, item_id, topic_id,
                       topic_label, subtopic_id, subtopic_label
                FROM df
            """)
        return len(assignments)

    def get_active_topics(self, collection_name: str) -> dict[str, Any] | None:
        """Get active topic extraction with all topic info."""
        ext_cur = self._exec(
            """
            SELECT * FROM topic_extractions
            WHERE collection_name = ? AND is_active = TRUE LIMIT 1
        """,
            [collection_name],
        )
        rows = ext_cur.fetchall()
        if not rows:
            return None

        columns = [desc[0] for desc in ext_cur.description]
        extraction = dict(zip(columns, rows[0], strict=True))
        if isinstance(extraction.get("quality_metrics"), str):
            extraction["quality_metrics"] = json.loads(extraction["quality_metrics"])

        ti_cur = self._exec(
            """
            SELECT * FROM topic_info WHERE extraction_id = ? ORDER BY topic_id
        """,
            [extraction["id"]],
        )
        ti_rows = ti_cur.fetchall()
        ti_columns = [desc[0] for desc in ti_cur.description]
        topics = []
        for r in ti_rows:
            t = dict(zip(ti_columns, r, strict=True))
            if isinstance(t.get("keywords"), str):
                t["keywords"] = json.loads(t["keywords"])
            if isinstance(t.get("subtopics"), str):
                t["subtopics"] = json.loads(t["subtopics"])
            topics.append(t)

        extraction["topics"] = topics
        return extraction

    def update_topic_quality_metrics(self, extraction_id: str, level: str, metrics: dict) -> None:
        """Store quality metrics on an extraction, keyed by level.

        Read-modify-write so re-scoring one level ("topic"/"subtopic") preserves
        the other level's stored metrics.
        """
        cur = self._exec(
            "SELECT quality_metrics FROM topic_extractions WHERE id = ?", [extraction_id]
        )
        rows = cur.fetchall()
        if not rows:
            raise ValueError(f"No topic extraction with id {extraction_id!r}")

        existing = rows[0][0]
        if isinstance(existing, str):
            existing = json.loads(existing)
        merged = dict(existing or {})
        merged[level] = metrics

        self._exec(
            "UPDATE topic_extractions SET quality_metrics = ? WHERE id = ?",
            [json.dumps(merged), extraction_id],
        )

    def get_items_for_topic(self, extraction_id: str, topic_id: int) -> list[str]:
        """Get item IDs assigned to a specific topic."""
        rows = self._exec(
            "SELECT item_id FROM topic_assignments WHERE extraction_id = ? AND topic_id = ?",
            [extraction_id, topic_id],
        ).fetchall()
        return [r[0] for r in rows]

    def update_topic_label(self, extraction_id: str, topic_id: int, new_label: str) -> None:
        """Update a topic label in both topic_info and topic_assignments."""
        self._exec(
            "UPDATE topic_info SET label = ? WHERE extraction_id = ? AND topic_id = ?",
            [new_label, extraction_id, topic_id],
        )
        self._exec(
            "UPDATE topic_assignments SET topic_label = ? WHERE extraction_id = ? AND topic_id = ?",
            [new_label, extraction_id, topic_id],
        )

    def update_subtopic_label(self, extraction_id: str, subtopic_id: int, new_label: str) -> None:
        """Update a subtopic label in topic_assignments and topic_info.subtopics JSON."""
        # Get old label before update
        old_row = self._exec(
            "SELECT DISTINCT subtopic_label FROM topic_assignments WHERE extraction_id = ? AND subtopic_id = ?",
            [extraction_id, subtopic_id],
        ).fetchone()
        old_label = old_row[0] if old_row else None

        # Update topic_assignments
        self._exec(
            "UPDATE topic_assignments SET subtopic_label = ? WHERE extraction_id = ? AND subtopic_id = ?",
            [new_label, extraction_id, subtopic_id],
        )

        # Also update topic_info.subtopics JSON array
        if old_label:
            topic_row = self._exec(
                "SELECT DISTINCT topic_id FROM topic_assignments WHERE extraction_id = ? AND subtopic_id = ?",
                [extraction_id, subtopic_id],
            ).fetchone()
            if topic_row:
                topic_id = topic_row[0]
                info_row = self._exec(
                    "SELECT subtopics FROM topic_info WHERE extraction_id = ? AND topic_id = ?",
                    [extraction_id, topic_id],
                ).fetchone()
                if info_row and info_row[0]:
                    subtopics = (
                        json.loads(info_row[0]) if isinstance(info_row[0], str) else info_row[0]
                    )
                    subtopics = [new_label if s == old_label else s for s in subtopics]
                    self._exec(
                        "UPDATE topic_info SET subtopics = ? WHERE extraction_id = ? AND topic_id = ?",
                        [json.dumps(subtopics), extraction_id, topic_id],
                    )

    # ------------------------------------------------------------------
    # Field Analysis
    # ------------------------------------------------------------------

    def compute_field_analysis(self, dataset_name: str) -> dict[str, Any]:
        """Compute metadata field statistics via SQL aggregation."""
        ds = self.get_dataset(dataset_name)
        if not ds:
            return {}
        table = self._items_table(dataset_name)

        sample_rows = self._exec(
            f"SELECT metadata FROM {table} WHERE metadata IS NOT NULL LIMIT 100"
        ).fetchall()

        if not sample_rows:
            return {}

        all_keys = set()
        for r in sample_rows:
            meta = json.loads(r[0]) if isinstance(r[0], str) else r[0]
            if meta:
                all_keys.update(meta.keys())

        analysis = {}
        for key in sorted(all_keys):
            row = self._exec(f"""
                SELECT
                    COUNT(*) AS total,
                    COUNT(DISTINCT json_extract_string(metadata, '$.{key}')) AS distinct_count
                FROM {table}
                WHERE json_extract_string(metadata, '$.{key}') IS NOT NULL
            """).fetchone()
            analysis[key] = {"total": row[0], "distinct_count": row[1]}

        return analysis

    # ------------------------------------------------------------------
    # SAE Features
    # ------------------------------------------------------------------

    def insert_sae_features_batch(self, model_id: str, sae_id: str, df: pd.DataFrame) -> int:
        """Bulk-insert SAE feature rows.

        Expected DataFrame columns: feature_index, density, label,
        top_logits (JSON str), bottom_logits (JSON str).
        """
        insert_df = pd.DataFrame(  # noqa: F841  # DuckDB reads this DataFrame by name in the SQL below
            {
                "model_id": model_id,
                "sae_id": sae_id,
                "feature_index": df["feature_index"],
                "density": df["density"],
                "label": df["label"],
                "top_logits": df["top_logits"],
                "bottom_logits": df["bottom_logits"],
                "created_at": pd.Timestamp.now(),
            }
        )
        with self._write_lock:
            self._conn.execute("INSERT OR REPLACE INTO sae_features SELECT * FROM insert_df")
        count = self._exec(
            "SELECT COUNT(*) FROM sae_features WHERE model_id = ? AND sae_id = ?",
            [model_id, sae_id],
        ).fetchone()[0]
        logger.info("Inserted SAE features for %s/%s — %d total", model_id, sae_id, count)
        return int(count)

    def insert_sae_activations_batch(self, df: pd.DataFrame) -> int:
        """Bulk-insert SAE activation rows.

        Expected DataFrame columns: id, model_id, sae_id, feature_index,
        tokens (JSON str), act_values (JSON str), max_value,
        max_value_token_idx, min_value, qualifying_token_idx.
        """
        with self._write_lock:
            self._conn.execute("INSERT OR REPLACE INTO sae_activations SELECT * FROM df")
        return len(df)

    def get_sae_feature(
        self, model_id: str, sae_id: str, feature_index: int
    ) -> dict[str, Any] | None:
        """Return a single SAE feature or None."""
        row = self._exec(
            "SELECT feature_index, density, label, top_logits, bottom_logits "
            "FROM sae_features "
            "WHERE model_id = ? AND sae_id = ? AND feature_index = ?",
            [model_id, sae_id, feature_index],
        ).fetchone()
        if not row:
            return None
        return {
            "model_id": model_id,
            "sae_id": sae_id,
            "feature_index": row[0],
            "density": row[1],
            "label": row[2],
            "top_logits": json.loads(row[3]) if isinstance(row[3], str) else row[3],
            "bottom_logits": json.loads(row[4]) if isinstance(row[4], str) else row[4],
        }

    def get_sae_feature_labels_batch(
        self, model_id: str, sae_id: str, feature_indices: list[int]
    ) -> dict[int, tuple[str, float | None]]:
        """Return labels and densities for a batch of feature indices.

        Returns a dict mapping feature_index -> (label, density).
        """
        if not feature_indices:
            return {}
        placeholders = ",".join("?" * len(feature_indices))
        rows = self._exec(
            f"SELECT feature_index, label, density FROM sae_features "
            f"WHERE model_id = ? AND sae_id = ? AND feature_index IN ({placeholders})",
            [model_id, sae_id, *feature_indices],
        ).fetchall()
        return {r[0]: (r[1] or "", r[2]) for r in rows}

    def get_sae_activations(
        self, model_id: str, sae_id: str, feature_index: int, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return top activations for a feature, ordered by max_value desc."""
        rows = self._exec(
            "SELECT id, tokens, act_values, max_value, max_value_token_idx "
            "FROM sae_activations "
            "WHERE model_id = ? AND sae_id = ? AND feature_index = ? "
            "ORDER BY max_value DESC LIMIT ?",
            [model_id, sae_id, feature_index, limit],
        ).fetchall()
        return [
            {
                "id": r[0],
                "tokens": json.loads(r[1]) if isinstance(r[1], str) else r[1],
                "values": json.loads(r[2]) if isinstance(r[2], str) else r[2],
                "max_value": r[3],
                "max_value_token_index": r[4],
            }
            for r in rows
        ]

    def update_sae_feature_label(
        self, model_id: str, sae_id: str, feature_index: int, label: str
    ) -> bool:
        """Update the label of a single SAE feature. Returns True if a row was updated."""
        with self._write_lock:
            result = self._conn.execute(
                "UPDATE sae_features SET label = ? "
                "WHERE model_id = ? AND sae_id = ? AND feature_index = ? "
                "RETURNING feature_index",
                [label, model_id, sae_id, feature_index],
            )
            return result.fetchone() is not None

    def search_sae_features(
        self,
        model_id: str | None = None,
        sae_id: str | None = None,
        sae_ids: list[str] | None = None,
        query: str | None = None,
        min_density: float | None = None,
        max_density: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search features by label (ILIKE) and/or density range.

        All filter parameters are optional, enabling cross-SAE search:
        - ``model_id`` — filter to a single model (None = all models)
        - ``sae_id`` — filter to a single SAE (None = all SAEs)
        - ``sae_ids`` — filter to a set of SAEs (takes precedence over ``sae_id``)
        """
        conditions: list[str] = []
        params: list = []

        if model_id is not None:
            conditions.append("model_id = ?")
            params.append(model_id)
        if sae_ids is not None and len(sae_ids) > 0:
            placeholders = ", ".join(["?"] * len(sae_ids))
            conditions.append(f"sae_id IN ({placeholders})")
            params.extend(sae_ids)
        elif sae_id is not None:
            conditions.append("sae_id = ?")
            params.append(sae_id)

        if query:
            conditions.append("label ILIKE ?")
            params.append(f"%{query}%")
        if min_density is not None:
            conditions.append("density >= ?")
            params.append(min_density)
        if max_density is not None:
            conditions.append("density <= ?")
            params.append(max_density)

        where = " AND ".join(conditions) if conditions else "TRUE"
        params.extend([limit, offset])

        rows = self._exec(
            f"SELECT model_id, sae_id, feature_index, density, label, top_logits, bottom_logits "
            f"FROM sae_features WHERE {where} "
            f"ORDER BY density DESC "
            f"LIMIT ? OFFSET ?",
            params,
        ).fetchall()

        results = []
        for r in rows:
            feat = {
                "model_id": r[0],
                "sae_id": r[1],
                "feature_index": r[2],
                "density": r[3],
                "label": r[4],
                "top_logits": json.loads(r[5]) if isinstance(r[5], str) else r[5],
                "bottom_logits": json.loads(r[6]) if isinstance(r[6], str) else r[6],
            }
            results.append(feat)
        return results

    def list_sae_models(self) -> list[dict[str, Any]]:
        """List distinct (model_id, sae_id) pairs with counts."""
        rows = self._exec("""
            SELECT f.model_id, f.sae_id, COUNT(*) AS feature_count,
                   COALESCE(a.act_count, 0) AS activation_count
            FROM sae_features f
            LEFT JOIN (
                SELECT model_id, sae_id, COUNT(*) AS act_count
                FROM sae_activations GROUP BY model_id, sae_id
            ) a ON f.model_id = a.model_id AND f.sae_id = a.sae_id
            GROUP BY f.model_id, f.sae_id, a.act_count
        """).fetchall()
        return [
            {
                "model_id": r[0],
                "sae_id": r[1],
                "feature_count": r[2],
                "activation_count": r[3],
            }
            for r in rows
        ]

    def get_sae_feature_densities(self, model_id: str, sae_id: str) -> list[float]:
        """Return all non-null density values for a model/sae pair."""
        rows = self._exec(
            "SELECT density FROM sae_features "
            "WHERE model_id = ? AND sae_id = ? AND density IS NOT NULL "
            "ORDER BY feature_index",
            [model_id, sae_id],
        ).fetchall()
        return [r[0] for r in rows]

    def get_sae_activations_by_quantile(
        self,
        model_id: str,
        sae_id: str,
        feature_index: int,
        n_quantiles: int = 5,
        per_quantile_limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return activations grouped into quantile bins by max_value.

        Uses NTILE window function. Quantile 1 = highest activations.
        Returns list of dicts with keys: quantile, bin_min, bin_max, activations.
        """
        rows = self._exec(
            """
            WITH ranked AS (
                SELECT id, tokens, act_values, max_value, max_value_token_idx,
                       NTILE(?) OVER (ORDER BY max_value DESC) AS quantile
                FROM sae_activations
                WHERE model_id = ? AND sae_id = ? AND feature_index = ?
            ),
            numbered AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY quantile ORDER BY max_value DESC
                ) AS rn
                FROM ranked
            )
            SELECT quantile, id, tokens, act_values, max_value, max_value_token_idx
            FROM numbered
            WHERE rn <= ?
            ORDER BY quantile, max_value DESC
        """,
            [n_quantiles, model_id, sae_id, feature_index, per_quantile_limit],
        ).fetchall()

        # Also get bin boundaries per quantile
        bounds = self._exec(
            """
            WITH ranked AS (
                SELECT max_value,
                       NTILE(?) OVER (ORDER BY max_value DESC) AS quantile
                FROM sae_activations
                WHERE model_id = ? AND sae_id = ? AND feature_index = ?
            )
            SELECT quantile, MIN(max_value), MAX(max_value)
            FROM ranked GROUP BY quantile ORDER BY quantile
        """,
            [n_quantiles, model_id, sae_id, feature_index],
        ).fetchall()

        bounds_map = {b[0]: (b[1], b[2]) for b in bounds}

        # Group rows by quantile
        groups: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            q = r[0]
            act = {
                "id": r[1],
                "tokens": json.loads(r[2]) if isinstance(r[2], str) else r[2],
                "values": json.loads(r[3]) if isinstance(r[3], str) else r[3],
                "max_value": r[4],
                "max_value_token_index": r[5],
            }
            groups.setdefault(q, []).append(act)

        return [
            {
                "quantile": q,
                "bin_min": bounds_map.get(q, (0, 0))[0],
                "bin_max": bounds_map.get(q, (0, 0))[1],
                "activations": acts,
            }
            for q, acts in sorted(groups.items())
        ]

    def delete_sae_data(self, model_id: str, sae_id: str) -> bool:
        """Delete all SAE features and activations for a model/sae pair."""
        self._exec(
            "DELETE FROM sae_activations WHERE model_id = ? AND sae_id = ?",
            [model_id, sae_id],
        )
        self._exec(
            "DELETE FROM sae_features WHERE model_id = ? AND sae_id = ?",
            [model_id, sae_id],
        )
        logger.info("Deleted SAE data for %s/%s", model_id, sae_id)
        return True

    # ------------------------------------------------------------------
    # SAE Document Activations (per-document max-pooled sparse vectors)
    # ------------------------------------------------------------------

    def insert_document_activations_batch(
        self,
        collection_name: str,
        item_id: str,
        activations: list[tuple[int, float]],
    ) -> int:
        """Insert one document's nonzero SAE activations.

        Parameters
        ----------
        collection_name:
            The vector collection the document belongs to.
        item_id:
            The document's ID in the items table.
        activations:
            List of ``(feature_index, activation)`` pairs (nonzero only).
        """
        if not activations:
            return 0
        df = pd.DataFrame(  # noqa: F841  # DuckDB reads this DataFrame by name in the SQL below
            {
                "collection_name": collection_name,
                "item_id": item_id,
                "feature_index": [a[0] for a in activations],
                "activation": [a[1] for a in activations],
            }
        )
        with self._write_lock:
            self._conn.execute("INSERT OR REPLACE INTO sae_document_activations SELECT * FROM df")
        return len(activations)

    def insert_document_activations_bulk(
        self,
        df: pd.DataFrame,
    ) -> int:
        """Bulk-insert document activations from a DataFrame.

        Expected columns: collection_name, item_id, feature_index, activation.
        """
        if df.empty:
            return 0
        with self._write_lock:
            self._conn.execute("INSERT OR REPLACE INTO sae_document_activations SELECT * FROM df")
        return len(df)

    def get_document_activation_item_ids(self, collection_name: str) -> set[str]:
        """Return item IDs that already have activations stored (for resume)."""
        rows = self._exec(
            "SELECT DISTINCT item_id FROM sae_document_activations WHERE collection_name = ?",
            [collection_name],
        ).fetchall()
        return {r[0] for r in rows}

    def has_document_activations(self, collection_name: str) -> bool:
        """Check if any document activations exist for a collection."""
        row = self._exec(
            "SELECT 1 FROM sae_document_activations WHERE collection_name = ? LIMIT 1",
            [collection_name],
        ).fetchone()
        return row is not None

    def delete_document_activations(self, collection_name: str) -> int:
        """Delete all document activations for a collection. Returns rows deleted."""
        count = self._exec(
            "SELECT COUNT(*) FROM sae_document_activations WHERE collection_name = ?",
            [collection_name],
        ).fetchone()[0]
        self._exec(
            "DELETE FROM sae_document_activations WHERE collection_name = ?",
            [collection_name],
        )
        logger.info("Deleted %d document activation rows for %s", count, collection_name)
        return int(count)

    # ------------------------------------------------------------------
    # Probes (embedding-space probes on numeric metadata fields)
    # ------------------------------------------------------------------

    def upsert_probe(
        self,
        collection_name: str,
        target_field: str,
        kind: str,
        *,
        config: dict | None = None,
        metrics: dict | None = None,
        direction: list[float] | None = None,
        scaler_mean: list[float] | None = None,
        scaler_scale: list[float] | None = None,
        intercept: float | None = None,
        artifact_path: str | None = None,
        n_train: int = 0,
        n_val: int = 0,
    ) -> None:
        """Insert or replace a probe row (one probe per collection+field+kind)."""
        self._exec(
            """
            INSERT OR REPLACE INTO probes (
                collection_name, target_field, kind, config, metrics,
                direction, scaler_mean, scaler_scale, intercept,
                artifact_path, n_train, n_val
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                collection_name,
                target_field,
                kind,
                json.dumps(config) if config is not None else None,
                json.dumps(metrics) if metrics is not None else None,
                direction,
                scaler_mean,
                scaler_scale,
                intercept,
                artifact_path,
                n_train,
                n_val,
            ],
        )

    def list_probes(self, collection_name: str) -> list[dict]:
        """Return all probes for a collection, JSON fields parsed, oldest first."""
        rows = self._exec(
            """
            SELECT collection_name, target_field, kind, config, metrics,
                   direction, scaler_mean, scaler_scale, intercept,
                   artifact_path, n_train, n_val, created_at
            FROM probes WHERE collection_name = ?
            ORDER BY created_at, target_field, kind
            """,
            [collection_name],
        ).fetchall()

        def _parse_json(value):
            return json.loads(value) if isinstance(value, str) else value

        return [
            {
                "collection_name": r[0],
                "target_field": r[1],
                "kind": r[2],
                "config": _parse_json(r[3]),
                "metrics": _parse_json(r[4]),
                "direction": list(r[5]) if r[5] is not None else None,
                "scaler_mean": list(r[6]) if r[6] is not None else None,
                "scaler_scale": list(r[7]) if r[7] is not None else None,
                "intercept": r[8],
                "artifact_path": r[9],
                "n_train": r[10],
                "n_val": r[11],
                "created_at": str(r[12]) if r[12] is not None else "",
            }
            for r in rows
        ]

    def insert_probe_scores_bulk(
        self,
        collection_name: str,
        target_field: str,
        kind: str,
        scores_df: pd.DataFrame,
    ) -> int:
        """Replace all scores for a probe key with the given rows.

        Expected columns: item_id, score, residual (residual may hold None).
        Existing rows for the key are deleted first so a retrain with fewer
        valid items never leaves strays.
        """
        self._exec(
            "DELETE FROM probe_scores WHERE collection_name = ? AND target_field = ? AND kind = ?",
            [collection_name, target_field, kind],
        )
        if scores_df.empty:
            return 0
        n = len(scores_df)
        residual = (
            scores_df["residual"] if "residual" in scores_df.columns else pd.Series([None] * n)
        )
        # Nullable Float64 keeps None as SQL NULL (plain float64 would coerce
        # None to NaN, which DuckDB stores as NaN, not NULL).
        insert_df = pd.DataFrame(  # noqa: F841  # DuckDB reads this DataFrame by name in the SQL below
            {
                "collection_name": [collection_name] * n,
                "target_field": [target_field] * n,
                "kind": [kind] * n,
                "item_id": scores_df["item_id"].astype(str),
                "score": scores_df["score"].astype("float64"),
                "residual": pd.array(
                    [None if pd.isna(v) else float(v) for v in residual],
                    dtype="Float64",
                ),
            }
        )
        with self._write_lock:
            self._conn.execute("INSERT INTO probe_scores SELECT * FROM insert_df")
        return n

    def get_probe_scores(self, collection_name: str, target_field: str, kind: str) -> dict | None:
        """Return {"item_ids", "scores", "residuals"} for a probe, or None if absent."""
        probe = self._exec(
            "SELECT 1 FROM probes WHERE collection_name = ? AND target_field = ? AND kind = ?",
            [collection_name, target_field, kind],
        ).fetchone()
        if probe is None:
            return None
        rows = self._exec(
            """
            SELECT item_id, score, residual FROM probe_scores
            WHERE collection_name = ? AND target_field = ? AND kind = ?
            ORDER BY item_id
            """,
            [collection_name, target_field, kind],
        ).fetchall()

        def _clean(value):
            if value is None or value != value:  # NULL or NaN
                return None
            return float(value)

        return {
            "item_ids": [r[0] for r in rows],
            "scores": [float(r[1]) for r in rows],
            "residuals": [_clean(r[2]) for r in rows],
        }

    def delete_probe(self, collection_name: str, target_field: str, kind: str) -> bool:
        """Delete a probe and its scores. Returns True if a probe row existed."""
        existed = self._exec(
            "SELECT 1 FROM probes WHERE collection_name = ? AND target_field = ? AND kind = ?",
            [collection_name, target_field, kind],
        ).fetchone()
        self._exec(
            "DELETE FROM probe_scores WHERE collection_name = ? AND target_field = ? AND kind = ?",
            [collection_name, target_field, kind],
        )
        self._exec(
            "DELETE FROM probes WHERE collection_name = ? AND target_field = ? AND kind = ?",
            [collection_name, target_field, kind],
        )
        if existed:
            logger.info("Deleted probe %s/%s/%s", collection_name, target_field, kind)
        return existed is not None

    def get_numeric_metadata_field(
        self, dataset_name: str, field: str
    ) -> list[tuple[str, float | None]]:
        """Read one metadata field as numbers for all items, ordered by row_index.

        Uses a quoted JSONPath so dotted field names (e.g. "Conc.M") are treated
        as literal keys, not nested paths. Values that are missing or fail
        TRY_CAST come back as None.
        """
        if not self.get_dataset(dataset_name):
            return []
        escaped = field.replace("\\", "\\\\").replace('"', '\\"')
        json_path = f'$."{escaped}"'
        table = self._items_table(dataset_name)
        rows = self._exec(
            f"""
            SELECT id, TRY_CAST(json_extract_string(metadata, ?) AS DOUBLE)
            FROM {table}
            ORDER BY row_index
            """,
            [json_path],
        ).fetchall()

        def _clean(value):
            if value is None or value != value:  # NULL or NaN
                return None
            return float(value)

        return [(r[0], _clean(r[1])) for r in rows]

    def get_text_metadata_field(
        self, dataset_name: str, field: str
    ) -> list[tuple[str, str | None]]:
        """Read one metadata field as raw strings for all items, ordered by row_index.

        Same quoted-JSONPath handling as ``get_numeric_metadata_field``; used
        as the fallback for binary categorical probe targets (e.g.
        "safe"/"unsafe"). Missing keys and JSON nulls come back as None;
        scalar non-strings come back in their JSON string form.
        """
        if not self.get_dataset(dataset_name):
            return []
        escaped = field.replace("\\", "\\\\").replace('"', '\\"')
        json_path = f'$."{escaped}"'
        table = self._items_table(dataset_name)
        rows = self._exec(
            f"""
            SELECT id, json_extract_string(metadata, ?)
            FROM {table}
            ORDER BY row_index
            """,
            [json_path],
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def search_documents_by_feature_labels(
        self,
        collection_name: str,
        query: str,
        model_id: str,
        sae_id: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Two-hop search: label text → matching features → ranked documents.

        1. Finds SAE features whose label matches *query* (ILIKE).
        2. Looks up which documents activate those features.
        3. Ranks documents by MAX(activation) across matching features.

        Returns dict with keys: results (list[dict]), matched_feature_count (int),
        matched_features (list[dict]).
        """
        # Hop 1 — find matching feature indices
        matched_features = self.search_sae_features(
            query=query,
            model_id=model_id,
            sae_id=sae_id,
            limit=500,
        )
        if not matched_features:
            return {
                "results": [],
                "matched_feature_count": 0,
                "matched_features": matched_features,
            }

        feature_indices = [f["feature_index"] for f in matched_features]
        placeholders = ", ".join(["?"] * len(feature_indices))

        # Hop 2 — rank documents by MAX activation
        rows = self._exec(
            f"SELECT item_id, MAX(activation) AS score, COUNT(*) AS matching_features "
            f"FROM sae_document_activations "
            f"WHERE collection_name = ? AND feature_index IN ({placeholders}) "
            f"GROUP BY item_id "
            f"ORDER BY score DESC "
            f"LIMIT ?",
            [collection_name, *feature_indices, limit],
        ).fetchall()

        if not rows:
            return {
                "results": [],
                "matched_feature_count": len(matched_features),
                "matched_features": matched_features,
            }

        # Enrich with documents + metadata
        item_ids = [r[0] for r in rows]
        scores = {r[0]: (r[1], r[2]) for r in rows}

        # Resolve dataset name from collection
        vc_row = self._exec(
            "SELECT dataset_name FROM vector_collections WHERE collection_name = ?",
            [collection_name],
        ).fetchone()

        enriched: dict[str, dict[str, Any]] = {}
        if vc_row:
            items = self.get_items_by_ids(vc_row[0], item_ids)
            for item in items:
                enriched[item["id"]] = item

        results = []
        for item_id in item_ids:
            score, matching_count = scores[item_id]
            item = enriched.get(item_id, {})
            results.append(
                {
                    "item_id": item_id,
                    "document": item.get("document"),
                    "metadata": item.get("metadata"),
                    "score": score,
                    "matching_features": matching_count,
                    "row_index": item.get("row_index"),
                }
            )

        return {
            "results": results,
            "matched_feature_count": len(matched_features),
            "matched_features": matched_features,
        }

    # Ranking modes for search_documents_by_feature_indices. `norm` is the
    # activation scaled by that feature's max over the whole collection, so
    # scaled modes treat selected features as equally important regardless
    # of their absolute magnitudes (which differ ~100x).
    _FEATURE_RANKING_MODES = {
        "max": ("MAX(activation)", "score DESC, item_id"),
        "sum": ("SUM(activation)", "score DESC, item_id"),
        "scaled_sum": ("SUM(norm)", "score DESC, item_id"),
        "matching_features": (
            "SUM(norm)",
            "matching_features DESC, score DESC, item_id",
        ),
    }

    def search_documents_by_feature_indices(
        self,
        collection_name: str,
        feature_indices: list[int],
        limit: int = 50,
        ranking: str = "scaled_sum",
    ) -> dict[str, Any]:
        """Rank documents over explicit feature indices.

        This is hop-2 only — the caller has already selected which features
        to search for (e.g. from a multi-select combobox in the frontend).

        Ranking modes (see ``_FEATURE_RANKING_MODES``): ``max`` (strongest
        single activation), ``sum`` (raw sum), ``scaled_sum`` (default; sum
        of activations each normalized by that feature's collection-wide
        max), ``matching_features`` (number of selected features present,
        tie-broken by scaled sum).

        Returns ``{"results": [...], "total_matches": int}`` where each
        result dict has keys: item_id, document, metadata, score,
        matching_features, row_index. ``total_matches`` is the number of
        matching documents before ``limit`` is applied.
        """
        if not feature_indices:
            return {"results": [], "total_matches": 0}

        if ranking not in self._FEATURE_RANKING_MODES:
            raise ValueError(
                f"Unknown ranking mode {ranking!r}; "
                f"expected one of {sorted(self._FEATURE_RANKING_MODES)}"
            )
        score_expr, order_clause = self._FEATURE_RANKING_MODES[ranking]

        placeholders = ", ".join(["?"] * len(feature_indices))
        # COUNT(*) OVER () is evaluated after GROUP BY but before LIMIT,
        # so every returned row carries the total number of matching docs.
        rows = self._exec(
            f"WITH scoped AS ("
            f"    SELECT item_id, activation, "
            f"           activation / MAX(activation) OVER (PARTITION BY feature_index) AS norm "
            f"    FROM sae_document_activations "
            f"    WHERE collection_name = ? AND feature_index IN ({placeholders}) "
            f") "
            f"SELECT item_id, {score_expr} AS score, COUNT(*) AS matching_features, "
            f"       COUNT(*) OVER () AS total_matches "
            f"FROM scoped "
            f"GROUP BY item_id "
            f"ORDER BY {order_clause} "
            f"LIMIT ?",
            [collection_name, *feature_indices, limit],
        ).fetchall()

        if not rows:
            return {"results": [], "total_matches": 0}

        total_matches = rows[0][3]
        item_ids = [r[0] for r in rows]
        scores = {r[0]: (r[1], r[2]) for r in rows}

        vc_row = self._exec(
            "SELECT dataset_name FROM vector_collections WHERE collection_name = ?",
            [collection_name],
        ).fetchone()

        enriched: dict[str, dict[str, Any]] = {}
        if vc_row:
            items = self.get_items_by_ids(vc_row[0], item_ids)
            for item in items:
                enriched[item["id"]] = item

        return {
            "results": [
                {
                    "item_id": item_id,
                    "document": enriched.get(item_id, {}).get("document"),
                    "metadata": enriched.get(item_id, {}).get("metadata"),
                    "score": scores[item_id][0],
                    "matching_features": scores[item_id][1],
                    "row_index": enriched.get(item_id, {}).get("row_index"),
                }
                for item_id in item_ids
            ],
            "total_matches": total_matches,
        }

    def search_documents_by_activations(
        self,
        collection_name: str,
        activations: list[tuple[int, float]],
        limit: int = 50,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find documents whose SAE activation pattern is most similar to a query vector.

        Computes sparse dot product between *activations* (e.g. from a prompt)
        and each document's stored activations.  Only shared nonzero features
        contribute to the score.

        Parameters
        ----------
        collection_name:
            Vector collection with precomputed document activations.
        activations:
            Query vector as ``(feature_index, activation)`` pairs (nonzero only).
        limit:
            Maximum number of documents to return.
        top_k:
            If set, use only the top-K strongest features from *activations*.

        Returns
        -------
        List of dicts with keys: item_id, document, metadata, score, shared_features, row_index.
        """
        if not activations:
            return []

        # Optionally trim to top_k features
        if top_k is not None and len(activations) > top_k:
            activations = sorted(activations, key=lambda x: x[1], reverse=True)[:top_k]

        query_weights = {a[0]: a[1] for a in activations}

        # Sparse dot product via CTE: SUM(doc_activation * query_weight) for shared features
        values_rows = ", ".join([f"({idx}, {weight})" for idx, weight in query_weights.items()])
        rows = self._exec(
            f"""
            WITH query_vec(feature_index, weight) AS (
                VALUES {values_rows}
            )
            SELECT d.item_id,
                   SUM(d.activation * q.weight) AS score,
                   COUNT(*) AS shared_features
            FROM sae_document_activations d
            JOIN query_vec q ON d.feature_index = q.feature_index
            WHERE d.collection_name = ?
            GROUP BY d.item_id
            ORDER BY score DESC
            LIMIT ?
            """,
            [collection_name, limit],
        ).fetchall()

        if not rows:
            return []

        # Enrich with documents + metadata
        item_ids = [r[0] for r in rows]
        scores = {r[0]: (r[1], r[2]) for r in rows}

        vc_row = self._exec(
            "SELECT dataset_name FROM vector_collections WHERE collection_name = ?",
            [collection_name],
        ).fetchone()

        enriched: dict[str, dict[str, Any]] = {}
        if vc_row:
            items = self.get_items_by_ids(vc_row[0], item_ids)
            for item in items:
                enriched[item["id"]] = item

        return [
            {
                "item_id": item_id,
                "document": enriched.get(item_id, {}).get("document"),
                "metadata": enriched.get(item_id, {}).get("metadata"),
                "score": scores[item_id][0],
                "shared_features": scores[item_id][1],
                "row_index": enriched.get(item_id, {}).get("row_index"),
            }
            for item_id in item_ids
        ]

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def create_chat_session(self, session_id: str, title: str, config_json: str) -> dict:
        """Create a new chat session. Returns the created session dict."""
        self._exec(
            """
            INSERT INTO chat_sessions (id, title, config)
            VALUES (?, ?, ?::JSON)
            """,
            [session_id, title, config_json],
        )
        cur = self._exec("SELECT * FROM chat_sessions WHERE id = ?", [session_id])
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
        return _sanitize_for_json(dict(zip(cols, row, strict=True)), json_columns=("config",))

    def list_chat_sessions(self, limit: int = 50) -> list[dict]:
        """List chat sessions ordered by most recently updated."""
        cur = self._exec(
            """
            SELECT id, title, config, created_at, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            [limit],
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [_sanitize_for_json(dict(zip(cols, r, strict=True)), json_columns=("config",)) for r in rows]

    def get_chat_session_with_messages(self, session_id: str) -> dict | None:
        """Get a session with all its messages."""
        cur = self._exec("SELECT * FROM chat_sessions WHERE id = ?", [session_id])
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        session = _sanitize_for_json(dict(zip(cols, row, strict=True)), json_columns=("config",))

        msg_cur = self._exec(
            """
            SELECT id, session_id, role, content, parts, steering_snapshot, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            [session_id],
        )
        msg_rows = msg_cur.fetchall()
        msg_cols = [d[0] for d in msg_cur.description]
        session["messages"] = [
            _sanitize_for_json(
                dict(zip(msg_cols, m, strict=True)),
                json_columns=("parts", "steering_snapshot"),
            )
            for m in msg_rows
        ]
        return session

    def save_chat_message(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        parts_json: str | None = None,
        steering_snapshot_json: str | None = None,
    ) -> dict:
        """Save a single chat message and bump session updated_at."""
        self._exec(
            """
            INSERT INTO chat_messages (id, session_id, role, content, parts, steering_snapshot)
            VALUES (?, ?, ?, ?, ?::JSON, ?::JSON)
            """,
            [message_id, session_id, role, content, parts_json, steering_snapshot_json],
        )
        self._exec(
            "UPDATE chat_sessions SET updated_at = current_timestamp WHERE id = ?",
            [session_id],
        )
        cur = self._exec("SELECT * FROM chat_messages WHERE id = ?", [message_id])
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
        return _sanitize_for_json(
            dict(zip(cols, row, strict=True)),
            json_columns=("parts", "steering_snapshot"),
        )

    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages."""
        self._exec("DELETE FROM chat_messages WHERE session_id = ?", [session_id])
        self._exec("DELETE FROM chat_sessions WHERE id = ?", [session_id])
        return True

    def update_chat_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a chat session."""
        self._exec(
            "UPDATE chat_sessions SET title = ? WHERE id = ?",
            [title, session_id],
        )
