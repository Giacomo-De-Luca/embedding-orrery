"""
DuckDB client — central orchestrator for document storage, metadata,
projections, topic assignments, and vector collection references.

Uses table-per-dataset for items: each dataset gets its own
"items_{dataset_id}" table. This ensures FTS indexes, scans,
and queries are scoped to a single dataset with zero cross-dataset overhead.

Global tables (small, registry-style): datasets, vector_collections,
projections, projection_metadata, topic_extractions, topic_info, topic_assignments.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from ..embedding_functions.config import DUCKDB_PATH

logger = logging.getLogger("star_map." + __name__)

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


def _sanitize_for_json(d: dict) -> dict:
    """Convert non-JSON-serializable values (datetime, etc.) to strings."""
    import datetime

    out = {}
    for k, v in d.items():
        if isinstance(v, (datetime.datetime, datetime.date)):
            out[k] = v.isoformat()
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
        self._fts_dirty: set[str] = set()  # dataset_ids needing FTS rebuild
        self._ensure_schema()

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
        self._conn.execute("""
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

        self._conn.execute("""
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

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS projections (
                collection_name     VARCHAR NOT NULL REFERENCES vector_collections(collection_name),
                item_id             VARCHAR NOT NULL,
                projection_type     VARCHAR NOT NULL,
                coordinates         FLOAT[] NOT NULL,
                PRIMARY KEY (collection_name, item_id, projection_type)
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS projection_metadata (
                collection_name     VARCHAR NOT NULL REFERENCES vector_collections(collection_name),
                projection_type     VARCHAR NOT NULL,
                variance            FLOAT[],
                computed_at         TIMESTAMP,
                PRIMARY KEY (collection_name, projection_type)
            )
        """)

        self._conn.execute("""
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
                is_active                   BOOLEAN DEFAULT TRUE
            )
        """)

        self._conn.execute("""
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

        self._conn.execute("""
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

        self._conn.execute("""
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

        self._conn.execute("""
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

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sae_act_feature
            ON sae_activations (model_id, sae_id, feature_index)
        """)

        # FTS extension
        self._conn.execute("INSTALL fts")
        self._conn.execute("LOAD fts")

        logger.info("DuckDB schema ensured at %s", self.db_path)

    def _ensure_items_table(self, dataset_name: str):
        """Create the per-dataset items table if it doesn't exist."""
        table = self._items_table(dataset_name)
        self._conn.execute(f"""
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
        self._conn.execute(
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
        """List all datasets with cached item counts. Returns [{name, metadata, count}].

        Enriches each dataset with vector_collection fields (embedding_provider,
        embedding_model, embedding_dim, has_projections, has_topics) from the
        first matching vector_collection row.
        """
        rows = self._conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        columns = [desc[0] for desc in self._conn.description]

        # Build lookup of vector_collections keyed by dataset_name
        vc_rows = self._conn.execute(
            "SELECT * FROM vector_collections ORDER BY created_at"
        ).fetchall()
        vc_columns = [desc[0] for desc in self._conn.description]
        vc_by_dataset: dict[str, dict[str, Any]] = {}
        for vr in vc_rows:
            vc = dict(zip(vc_columns, vr))
            ds_name = vc["dataset_name"]
            if ds_name not in vc_by_dataset:
                vc_by_dataset[ds_name] = vc

        # Build lookup of active topic extractions keyed by collection_name
        te_rows = self._conn.execute(
            "SELECT collection_name, topic_count, topic_hierarchy FROM topic_extractions WHERE is_active = TRUE"
        ).fetchall()
        te_by_collection: dict[str, dict[str, Any]] = {}
        for tr in te_rows:
            te_by_collection[tr[0]] = {"topic_count": tr[1], "topic_hierarchy": tr[2]}

        result = []
        for row in rows:
            d = _sanitize_for_json(dict(zip(columns, row)))
            count = d.pop("item_count", 0)
            # Enrich with vector collection data
            vc = vc_by_dataset.get(d["name"])
            if vc:
                d["embedding_provider"] = vc.get("embedding_provider")
                d["embedding_model"] = vc.get("embedding_model")
                d["embedding_dim"] = vc.get("embedding_dim")
                d["has_projections"] = vc.get("has_projections", False)
                d["has_topics"] = vc.get("has_topics", False)
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
            result.append({"name": d["name"], "metadata": d, "count": count})
        return result

    def get_dataset(self, name: str) -> dict[str, Any] | None:
        """Get dataset by name."""
        rows = self._conn.execute("SELECT * FROM datasets WHERE name = ?", [name]).fetchall()
        if not rows:
            return None
        columns = [desc[0] for desc in self._conn.description]
        d = dict(zip(columns, rows[0]))
        d["count"] = d.get("item_count", 0)
        return d

    def update_dataset(self, name: str, **kwargs) -> None:
        """Update dataset fields by name."""
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
        values.append(name)
        self._conn.execute(
            f"UPDATE datasets SET {', '.join(set_parts)} WHERE name = ?",
            values,
        )

    def delete_dataset(self, name: str) -> bool:
        """Delete dataset, its items table, and all related data."""
        ds = self.get_dataset(name)
        if not ds:
            return False

        # Delete in dependency order
        self._conn.execute(
            """
            DELETE FROM topic_assignments WHERE extraction_id IN (
                SELECT id FROM topic_extractions WHERE dataset_name = ?
            )
        """,
            [name],
        )
        self._conn.execute(
            """
            DELETE FROM topic_info WHERE extraction_id IN (
                SELECT id FROM topic_extractions WHERE dataset_name = ?
            )
        """,
            [name],
        )
        self._conn.execute("DELETE FROM topic_extractions WHERE dataset_name = ?", [name])

        self._conn.execute(
            """
            DELETE FROM projection_metadata WHERE collection_name IN (
                SELECT collection_name FROM vector_collections WHERE dataset_name = ?
            )
        """,
            [name],
        )
        self._conn.execute(
            """
            DELETE FROM projections WHERE collection_name IN (
                SELECT collection_name FROM vector_collections WHERE dataset_name = ?
            )
        """,
            [name],
        )

        self._conn.execute("DELETE FROM vector_collections WHERE dataset_name = ?", [name])

        # Drop per-dataset items table
        self._conn.execute(f"DROP TABLE IF EXISTS {self._items_table(name)}")

        self._conn.execute("DELETE FROM datasets WHERE name = ?", [name])

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
        col_row_indices = []
        for meta in metadatas:
            row_index = None
            clean_meta = None
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
            col_row_indices.append(row_index)

        df = pd.DataFrame(
            {
                "id": ids,
                "document": documents,
                "metadata": col_metas,
                "row_index": col_row_indices,
            }
        )

        table = self._items_table(dataset_name)
        self._conn.execute(f"INSERT INTO {table} SELECT * FROM df")

        # Update cached item count
        count = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        self._conn.execute(
            "UPDATE datasets SET item_count = ? WHERE name = ?", [count, dataset_name]
        )

        self._fts_dirty.add(dataset_name)
        return len(ids)

    def get_item_ids(self, dataset_name: str) -> set[str]:
        """Get all item IDs for a dataset."""
        table = self._items_table(dataset_name)
        try:
            rows = self._conn.execute(f"SELECT id FROM {table}").fetchall()
        except duckdb.CatalogException:
            return set()
        return {r[0] for r in rows}

    def get_items_by_ids(self, dataset_name: str, ids: list[str]) -> list[dict[str, Any]]:
        """Get items by IDs. Used to enrich search results."""
        if not ids:
            return []
        table = self._items_table(dataset_name)
        placeholders = ", ".join(["?"] * len(ids))
        rows = self._conn.execute(
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

        rows = self._conn.execute(
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
        self._conn.execute(
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
        rows = self._conn.execute(
            "SELECT * FROM vector_collections WHERE dataset_name = ? ORDER BY created_at",
            [dataset_name],
        ).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, r)) for r in rows]

    def get_vector_collection(self, collection_name: str) -> dict[str, Any] | None:
        """Get a vector collection by its collection_name (PK)."""
        rows = self._conn.execute(
            "SELECT * FROM vector_collections WHERE collection_name = ?",
            [collection_name],
        ).fetchall()
        if not rows:
            return None
        columns = [desc[0] for desc in self._conn.description]
        return dict(zip(columns, rows[0]))

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

        df = pd.DataFrame(
            {
                "collection_name": collection_name,
                "item_id": item_ids,
                "projection_type": projection_type,
                "coordinates": coordinates,
            }
        )

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
        self._conn.execute(
            """
            INSERT OR REPLACE INTO projection_metadata
                (collection_name, projection_type, variance, computed_at)
            VALUES (?, ?, ?, ?)
        """,
            [collection_name, projection_type, variance, computed_at],
        )
        self._conn.execute(
            "UPDATE vector_collections SET has_projections = TRUE WHERE collection_name = ?",
            [collection_name],
        )

    def get_projection_data(
        self, collection_name: str, projection_type: str
    ) -> dict[str, Any] | None:
        """Load items + one projection type for visualization."""
        vc = self.get_vector_collection(collection_name)
        if not vc:
            return None
        dataset_name = vc["dataset_name"]
        table = self._items_table(dataset_name)

        rows = self._conn.execute(
            f"""
            SELECT i.id, i.document, i.metadata, p.coordinates
            FROM {table} i
            INNER JOIN projections p ON p.item_id = i.id
            WHERE p.collection_name = ?
              AND p.projection_type = ?
            ORDER BY i.row_index
        """,
            [collection_name, projection_type],
        ).fetchall()

        if not rows:
            return None

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

        pm_rows = self._conn.execute(
            """
            SELECT projection_type, variance
            FROM projection_metadata WHERE collection_name = ?
        """,
            [collection_name],
        ).fetchall()
        variance_map = {r[0]: r[1] for r in pm_rows}

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
        }

        return {
            "ids": ids,
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

        count = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count == 0:
            return

        try:
            self._conn.execute(f"PRAGMA drop_fts_index('{table_unquoted}')")
        except duckdb.CatalogException:
            pass

        self._conn.execute(f"""
            PRAGMA create_fts_index('{table_unquoted}', 'id', 'document',
                stemmer='porter', stopwords='english', lower=1, overwrite=1)
        """)
        self._fts_dirty.discard(dataset_name)
        logger.info("Rebuilt FTS index for %s (%d rows)", dataset_name, count)

    def text_search(
        self,
        dataset_name: str,
        query: str,
        fields: list[str] | None = None,
        mode: str = "contains",
        case_sensitive: bool = False,
        filters: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Search documents and/or metadata fields within one dataset.

        When filters are provided, results are restricted to items matching
        the metadata filters (same operators as get_filtered_items).
        """
        ds = self.get_dataset(dataset_name)
        if not ds:
            return {"matches": [], "total_matches": 0}
        table = self._items_table(dataset_name)

        # Pre-filter: compute allowed IDs from metadata filters
        allowed_ids: set[str] | None = None
        if filters:
            where_sql, where_params = self._build_metadata_where(filters)
            rows = self._conn.execute(
                f"SELECT id FROM {table} WHERE {where_sql}",
                where_params,
            ).fetchall()
            allowed_ids = {r[0] for r in rows}
            if not allowed_ids:
                return {"matches": [], "total_matches": 0}

        if fields is None:
            fields = ["__document__"]

        matches = []
        seen_ids = set()

        if "__document__" in fields:
            doc_matches = self._search_documents(table, query, mode, case_sensitive)
            if allowed_ids is not None:
                doc_matches = [m for m in doc_matches if m["id"] in allowed_ids]
            for m in doc_matches:
                if m["id"] not in seen_ids:
                    matches.append(m)
                    seen_ids.add(m["id"])

        meta_fields = [f for f in fields if f != "__document__"]
        if meta_fields:
            meta_matches = self._search_metadata_fields(
                table, query, meta_fields, mode, case_sensitive
            )
            if allowed_ids is not None:
                meta_matches = [m for m in meta_matches if m["id"] in allowed_ids]
            for m in meta_matches:
                if m["id"] not in seen_ids:
                    matches.append(m)
                    seen_ids.add(m["id"])

        return {"matches": matches, "total_matches": len(matches)}

    def _search_documents(
        self, table: str, query: str, mode: str, case_sensitive: bool
    ) -> list[dict]:
        """Search the document column of a per-dataset items table."""
        if mode == "exact":
            if case_sensitive:
                rows = self._conn.execute(
                    f"SELECT id, document FROM {table} WHERE document = ?", [query]
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT id, document FROM {table} WHERE LOWER(document) = LOWER(?)", [query]
                ).fetchall()
        else:
            if case_sensitive:
                rows = self._conn.execute(
                    f"SELECT id, document FROM {table} WHERE document LIKE '%' || ? || '%'", [query]
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT id, document FROM {table} WHERE document ILIKE '%' || ? || '%'",
                    [query],
                ).fetchall()

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
        self, table: str, query: str, fields: list[str], mode: str, case_sensitive: bool
    ) -> list[dict]:
        """Search metadata JSON fields in a per-dataset items table."""
        matches = []
        for field in fields:
            json_path = f"$.{field}"
            if mode == "exact":
                if case_sensitive:
                    rows = self._conn.execute(
                        f"SELECT id FROM {table} WHERE json_extract_string(metadata, ?) = ?",
                        [json_path, query],
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT id FROM {table} WHERE LOWER(CAST(json_extract_string(metadata, ?) AS VARCHAR)) = LOWER(?)",
                        [json_path, query],
                    ).fetchall()
            else:
                if case_sensitive:
                    rows = self._conn.execute(
                        f"SELECT id FROM {table} WHERE CAST(json_extract_string(metadata, ?) AS VARCHAR) LIKE '%' || ? || '%'",
                        [json_path, query],
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT id FROM {table} WHERE CAST(json_extract_string(metadata, ?) AS VARCHAR) ILIKE '%' || ? || '%'",
                        [json_path, query],
                    ).fetchall()

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

        rows = self._conn.execute(
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
        self._conn.execute(
            """
            UPDATE topic_extractions SET is_active = FALSE
            WHERE collection_name = ? AND is_active = TRUE
        """,
            [collection_name],
        )

        extraction_id = str(uuid.uuid4())
        self._conn.execute(
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
        self._conn.executemany(
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

        df = pd.DataFrame(
            {
                "extraction_id": extraction_id,
                "item_id": [a["item_id"] for a in assignments],
                "topic_id": [a["topic_id"] for a in assignments],
                "topic_label": [a.get("topic_label") for a in assignments],
                "subtopic_id": [a.get("subtopic_id") for a in assignments],
                "subtopic_label": [a.get("subtopic_label") for a in assignments],
            }
        )

        self._conn.execute("""
            INSERT INTO topic_assignments
            SELECT extraction_id, item_id, topic_id,
                   topic_label, subtopic_id, subtopic_label
            FROM df
        """)
        return len(assignments)

    def get_active_topics(self, collection_name: str) -> dict[str, Any] | None:
        """Get active topic extraction with all topic info."""
        rows = self._conn.execute(
            """
            SELECT * FROM topic_extractions
            WHERE collection_name = ? AND is_active = TRUE LIMIT 1
        """,
            [collection_name],
        ).fetchall()
        if not rows:
            return None

        columns = [desc[0] for desc in self._conn.description]
        extraction = dict(zip(columns, rows[0]))

        ti_rows = self._conn.execute(
            """
            SELECT * FROM topic_info WHERE extraction_id = ? ORDER BY topic_id
        """,
            [extraction["id"]],
        ).fetchall()
        ti_columns = [desc[0] for desc in self._conn.description]
        topics = []
        for r in ti_rows:
            t = dict(zip(ti_columns, r))
            if isinstance(t.get("keywords"), str):
                t["keywords"] = json.loads(t["keywords"])
            if isinstance(t.get("subtopics"), str):
                t["subtopics"] = json.loads(t["subtopics"])
            topics.append(t)

        extraction["topics"] = topics
        return extraction

    def get_items_for_topic(self, extraction_id: str, topic_id: int) -> list[str]:
        """Get item IDs assigned to a specific topic."""
        rows = self._conn.execute(
            "SELECT item_id FROM topic_assignments WHERE extraction_id = ? AND topic_id = ?",
            [extraction_id, topic_id],
        ).fetchall()
        return [r[0] for r in rows]

    def update_topic_label(self, extraction_id: str, topic_id: int, new_label: str) -> None:
        """Update a topic label in both topic_info and topic_assignments."""
        self._conn.execute(
            "UPDATE topic_info SET label = ? WHERE extraction_id = ? AND topic_id = ?",
            [new_label, extraction_id, topic_id],
        )
        self._conn.execute(
            "UPDATE topic_assignments SET topic_label = ? WHERE extraction_id = ? AND topic_id = ?",
            [new_label, extraction_id, topic_id],
        )

    def update_subtopic_label(self, extraction_id: str, subtopic_id: int, new_label: str) -> None:
        """Update a subtopic label in topic_assignments and topic_info.subtopics JSON."""
        # Get old label before update
        old_row = self._conn.execute(
            "SELECT DISTINCT subtopic_label FROM topic_assignments WHERE extraction_id = ? AND subtopic_id = ?",
            [extraction_id, subtopic_id],
        ).fetchone()
        old_label = old_row[0] if old_row else None

        # Update topic_assignments
        self._conn.execute(
            "UPDATE topic_assignments SET subtopic_label = ? WHERE extraction_id = ? AND subtopic_id = ?",
            [new_label, extraction_id, subtopic_id],
        )

        # Also update topic_info.subtopics JSON array
        if old_label:
            topic_row = self._conn.execute(
                "SELECT DISTINCT topic_id FROM topic_assignments WHERE extraction_id = ? AND subtopic_id = ?",
                [extraction_id, subtopic_id],
            ).fetchone()
            if topic_row:
                topic_id = topic_row[0]
                info_row = self._conn.execute(
                    "SELECT subtopics FROM topic_info WHERE extraction_id = ? AND topic_id = ?",
                    [extraction_id, topic_id],
                ).fetchone()
                if info_row and info_row[0]:
                    subtopics = (
                        json.loads(info_row[0]) if isinstance(info_row[0], str) else info_row[0]
                    )
                    subtopics = [new_label if s == old_label else s for s in subtopics]
                    self._conn.execute(
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

        sample_rows = self._conn.execute(
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
            row = self._conn.execute(f"""
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
        insert_df = pd.DataFrame(
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
        self._conn.execute("INSERT OR REPLACE INTO sae_features SELECT * FROM insert_df")
        count = self._conn.execute(
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
        self._conn.execute("INSERT OR REPLACE INTO sae_activations SELECT * FROM df")
        return len(df)

    def get_sae_feature(
        self, model_id: str, sae_id: str, feature_index: int
    ) -> dict[str, Any] | None:
        """Return a single SAE feature or None."""
        row = self._conn.execute(
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

    def get_sae_activations(
        self, model_id: str, sae_id: str, feature_index: int, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return top activations for a feature, ordered by max_value desc."""
        rows = self._conn.execute(
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

    def search_sae_features(
        self,
        model_id: str,
        sae_id: str,
        query: str | None = None,
        min_density: float | None = None,
        max_density: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search features by label (ILIKE) and/or density range."""
        conditions = ["model_id = ?", "sae_id = ?"]
        params: list = [model_id, sae_id]

        if query:
            conditions.append("label ILIKE ?")
            params.append(f"%{query}%")
        if min_density is not None:
            conditions.append("density >= ?")
            params.append(min_density)
        if max_density is not None:
            conditions.append("density <= ?")
            params.append(max_density)

        where = " AND ".join(conditions)
        params.extend([limit, offset])

        rows = self._conn.execute(
            f"SELECT feature_index, density, label, top_logits, bottom_logits "
            f"FROM sae_features WHERE {where} "
            f"ORDER BY density DESC "
            f"LIMIT ? OFFSET ?",
            params,
        ).fetchall()

        results = []
        for r in rows:
            feat = {
                "model_id": model_id,
                "sae_id": sae_id,
                "feature_index": r[0],
                "density": r[1],
                "label": r[2],
                "top_logits": json.loads(r[3]) if isinstance(r[3], str) else r[3],
                "bottom_logits": json.loads(r[4]) if isinstance(r[4], str) else r[4],
            }
            results.append(feat)
        return results

    def list_sae_models(self) -> list[dict[str, Any]]:
        """List distinct (model_id, sae_id) pairs with counts."""
        rows = self._conn.execute("""
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
        rows = self._conn.execute(
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
        rows = self._conn.execute(
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
        bounds = self._conn.execute(
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
        self._conn.execute(
            "DELETE FROM sae_activations WHERE model_id = ? AND sae_id = ?",
            [model_id, sae_id],
        )
        self._conn.execute(
            "DELETE FROM sae_features WHERE model_id = ? AND sae_id = ?",
            [model_id, sae_id],
        )
        logger.info("Deleted SAE data for %s/%s", model_id, sae_id)
        return True
