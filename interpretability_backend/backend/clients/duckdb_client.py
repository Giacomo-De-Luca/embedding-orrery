"""
DuckDB client — central orchestrator for document storage, metadata,
projections, topic assignments, and vector collection references.

Replaces ChromaDB as the primary data access layer. ChromaDB is retained
only for dense vector storage and similarity search.
"""

import duckdb
import json
import uuid
import logging
import time
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple

from ..embedding_functions.config import DUCKDB_PATH

logger = logging.getLogger("star_map." + __name__)

# Projection fields to strip from item metadata (stored separately in projections table)
_PROJECTION_KEYS = frozenset({"pca_2d", "pca_3d", "umap_2d", "umap_3d"})
# Topic fields to strip from item metadata (stored separately in topic_assignments table)
_TOPIC_KEYS = frozenset({"topic_id", "topic_label", "subtopic_id", "subtopic_label",
                          "ctfidf_label", "ctfidf_subtopic_map"})

# Snippet radius for text search (chars before/after match)
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
    """Central data orchestrator using DuckDB as the document/metadata store."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(DUCKDB_PATH.resolve())
        elif db_path == ":memory:":
            pass  # in-memory for tests
        else:
            db_path = str(Path(db_path).resolve())

        self.db_path = db_path
        self._conn = duckdb.connect(db_path)
        self._fts_dirty = True  # rebuild FTS on first search
        self._ensure_schema()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self):
        """Create all tables if they don't exist. Idempotent."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id                  VARCHAR PRIMARY KEY,
                name                VARCHAR UNIQUE NOT NULL,
                description         VARCHAR,
                source_type         VARCHAR,
                source_dataset      VARCHAR,
                source_config       VARCHAR,
                source_split        VARCHAR,
                source_file         VARCHAR,
                embedded_columns    JSON,
                data_type           VARCHAR,
                total_in_source     INTEGER,
                created_at          TIMESTAMP DEFAULT current_timestamp,
                extra_metadata      JSON DEFAULT '{}'
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id          VARCHAR NOT NULL,
                dataset_id  VARCHAR NOT NULL REFERENCES datasets(id),
                document    VARCHAR,
                metadata    JSON,
                row_index   INTEGER,
                PRIMARY KEY (dataset_id, id)
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_collections (
                id                  VARCHAR PRIMARY KEY,
                dataset_id          VARCHAR NOT NULL REFERENCES datasets(id),
                backend             VARCHAR NOT NULL,
                collection_name     VARCHAR NOT NULL,
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
                vector_collection_id VARCHAR NOT NULL REFERENCES vector_collections(id),
                item_id              VARCHAR NOT NULL,
                dataset_id           VARCHAR NOT NULL,
                projection_type      VARCHAR NOT NULL,
                coordinates          FLOAT[] NOT NULL,
                PRIMARY KEY (vector_collection_id, item_id, projection_type),
                FOREIGN KEY (dataset_id, item_id) REFERENCES items(dataset_id, id)
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS projection_metadata (
                vector_collection_id VARCHAR NOT NULL REFERENCES vector_collections(id),
                projection_type      VARCHAR NOT NULL,
                variance             FLOAT[],
                computed_at          TIMESTAMP,
                PRIMARY KEY (vector_collection_id, projection_type)
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS topic_extractions (
                id                          VARCHAR PRIMARY KEY,
                vector_collection_id        VARCHAR NOT NULL REFERENCES vector_collections(id),
                dataset_id                  VARCHAR NOT NULL REFERENCES datasets(id),
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
                dataset_id     VARCHAR NOT NULL,
                item_id        VARCHAR NOT NULL,
                topic_id       INTEGER NOT NULL,
                topic_label    VARCHAR,
                subtopic_id    INTEGER,
                subtopic_label VARCHAR,
                PRIMARY KEY (extraction_id, dataset_id, item_id),
                FOREIGN KEY (dataset_id, item_id) REFERENCES items(dataset_id, id)
            )
        """)

        # Install FTS extension (idempotent)
        self._conn.execute("INSTALL fts")
        self._conn.execute("LOAD fts")

        logger.info("DuckDB schema ensured at %s", self.db_path)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def create_dataset(self, name: str, *, description: str = None,
                       source_type: str = None, source_dataset: str = None,
                       source_config: str = None, source_split: str = None,
                       source_file: str = None, embedded_columns: list = None,
                       data_type: str = None, total_in_source: int = None,
                       extra_metadata: dict = None) -> str:
        """Create a new dataset. Returns the dataset UUID."""
        dataset_id = str(uuid.uuid4())
        self._conn.execute("""
            INSERT INTO datasets (id, name, description, source_type, source_dataset,
                source_config, source_split, source_file, embedded_columns,
                data_type, total_in_source, extra_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            dataset_id, name, description, source_type, source_dataset,
            source_config, source_split, source_file,
            json.dumps(embedded_columns) if embedded_columns else None,
            data_type, total_in_source,
            json.dumps(extra_metadata or {}),
        ])
        logger.info("Created dataset %r (id=%s)", name, dataset_id)
        return dataset_id

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets with item counts. Returns [{name, metadata, count}]."""
        rows = self._conn.execute("""
            SELECT d.*, COUNT(i.id) AS item_count
            FROM datasets d
            LEFT JOIN items i ON i.dataset_id = d.id
            GROUP BY ALL
            ORDER BY d.created_at DESC
        """).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        result = []
        for row in rows:
            d = _sanitize_for_json(dict(zip(columns, row)))
            count = d.pop("item_count", 0)
            name = d["name"]
            result.append({"name": name, "metadata": d, "count": count})
        return result

    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get dataset by name, including item count."""
        rows = self._conn.execute("""
            SELECT d.*, COUNT(i.id) AS item_count
            FROM datasets d
            LEFT JOIN items i ON i.dataset_id = d.id
            WHERE d.name = ?
            GROUP BY ALL
        """, [name]).fetchall()
        if not rows:
            return None
        columns = [desc[0] for desc in self._conn.description]
        d = dict(zip(columns, rows[0]))
        d["count"] = d.pop("item_count", 0)
        return d

    def get_dataset_by_id(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset by ID."""
        rows = self._conn.execute(
            "SELECT * FROM datasets WHERE id = ?", [dataset_id]
        ).fetchall()
        if not rows:
            return None
        columns = [desc[0] for desc in self._conn.description]
        return dict(zip(columns, rows[0]))

    def update_dataset(self, name: str, **kwargs) -> None:
        """Update dataset fields by name. Only provided kwargs are updated."""
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
        """Delete dataset and all related data. Returns True if found."""
        ds = self.get_dataset(name)
        if not ds:
            return False
        dataset_id = ds["id"]

        # Delete in dependency order
        # topic_assignments → topic_info → topic_extractions
        self._conn.execute("""
            DELETE FROM topic_assignments WHERE extraction_id IN (
                SELECT id FROM topic_extractions WHERE dataset_id = ?
            )
        """, [dataset_id])
        self._conn.execute("""
            DELETE FROM topic_info WHERE extraction_id IN (
                SELECT id FROM topic_extractions WHERE dataset_id = ?
            )
        """, [dataset_id])
        self._conn.execute("DELETE FROM topic_extractions WHERE dataset_id = ?", [dataset_id])

        # projection_metadata → projections
        self._conn.execute("""
            DELETE FROM projection_metadata WHERE vector_collection_id IN (
                SELECT id FROM vector_collections WHERE dataset_id = ?
            )
        """, [dataset_id])
        self._conn.execute("""
            DELETE FROM projections WHERE vector_collection_id IN (
                SELECT id FROM vector_collections WHERE dataset_id = ?
            )
        """, [dataset_id])

        # vector_collections → items → dataset
        self._conn.execute("DELETE FROM vector_collections WHERE dataset_id = ?", [dataset_id])
        self._conn.execute("DELETE FROM items WHERE dataset_id = ?", [dataset_id])
        self._conn.execute("DELETE FROM datasets WHERE id = ?", [dataset_id])

        self._fts_dirty = True
        logger.info("Deleted dataset %r", name)
        return True

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def insert_items_batch(self, dataset_id: str, ids: List[str],
                           documents: List[Optional[str]],
                           metadatas: List[Optional[Dict]]) -> int:
        """Bulk insert items via DataFrame for columnar speed. Returns number inserted."""
        if not ids:
            return 0

        # Pre-process metadata: strip projection/topic keys, extract row_index
        col_metas = []
        col_row_indices = []
        for meta in metadatas:
            row_index = None
            clean_meta = None
            if meta:
                clean = {k: v for k, v in meta.items()
                         if k not in _PROJECTION_KEYS and k not in _TOPIC_KEYS}
                row_index = meta.get("row_index")
                clean.pop("row_index", None)
                clean_meta = json.dumps(clean) if clean else None
            col_metas.append(clean_meta)
            col_row_indices.append(row_index)

        df = pd.DataFrame({
            "id": ids,
            "dataset_id": dataset_id,
            "document": documents,
            "metadata": col_metas,
            "row_index": col_row_indices,
        })

        self._conn.execute("INSERT INTO items SELECT * FROM df")
        self._fts_dirty = True
        return len(ids)

    def get_item_ids(self, dataset_name: str) -> Set[str]:
        """Get all item IDs for a dataset. Used for resume checks."""
        ds = self.get_dataset(dataset_name)
        if not ds:
            return set()
        rows = self._conn.execute(
            "SELECT id FROM items WHERE dataset_id = ?", [ds["id"]]
        ).fetchall()
        return {r[0] for r in rows}

    def get_items_by_ids(self, dataset_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        """Get items by IDs. Used to enrich search results from vector DBs."""
        ds = self.get_dataset(dataset_name)
        if not ds or not ids:
            return []
        # Use VALUES list for the IN clause
        placeholders = ", ".join(["?"] * len(ids))
        rows = self._conn.execute(
            f"SELECT id, document, metadata, row_index FROM items WHERE dataset_id = ? AND id IN ({placeholders})",
            [ds["id"]] + list(ids),
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

    def get_filtered_items(self, dataset_name: str, filters: List[Dict],
                           limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get items with JSON metadata filtering.

        Args:
            dataset_name: dataset name
            filters: list of {field, operator, value} dicts.
                     operator: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
            limit: max items
            offset: skip N items

        Returns: list of {id, document, metadata, row_index}
        """
        ds = self.get_dataset(dataset_name)
        if not ds:
            return []

        where_parts = []
        params = [ds["id"]]

        for f in filters:
            field = f["field"]
            op = f["operator"]
            value = f["value"]
            json_path = f"$.{field}"

            # DuckDB json_extract returns typed values; cast to VARCHAR for string ops
            extract = f"json_extract_string(metadata, ?)"
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
                    placeholders = ", ".join(["?"] * len(value))
                    where_parts.append(f"{extract} IN ({placeholders})")
                    params.extend([str(v) for v in value])
            elif op == "$nin":
                if isinstance(value, list):
                    placeholders = ", ".join(["?"] * len(value))
                    where_parts.append(f"{extract} NOT IN ({placeholders})")
                    params.extend([str(v) for v in value])

        where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
        sql = f"""
            SELECT id, document, metadata, row_index FROM items
            WHERE dataset_id = ? AND {where_sql}
            ORDER BY row_index
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = self._conn.execute(sql, params).fetchall()
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

    def register_vector_collection(self, dataset_id: str, backend: str,
                                   collection_name: str, vector_type: str, *,
                                   embedding_provider: str = None,
                                   embedding_model: str = None,
                                   embedding_dim: int = None,
                                   embedding_task: str = None,
                                   embedding_task_type: str = None,
                                   embedding_prompt: str = None,
                                   item_count: int = 0) -> str:
        """Register a vector collection. Returns UUID."""
        vc_id = str(uuid.uuid4())
        self._conn.execute("""
            INSERT INTO vector_collections (id, dataset_id, backend, collection_name,
                vector_type, embedding_provider, embedding_model, embedding_dim,
                embedding_task, embedding_task_type, embedding_prompt, item_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            vc_id, dataset_id, backend, collection_name, vector_type,
            embedding_provider, embedding_model, embedding_dim,
            embedding_task, embedding_task_type, embedding_prompt, item_count,
        ])
        return vc_id

    def get_vector_collections(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get all vector collections for a dataset."""
        ds = self.get_dataset(dataset_name)
        if not ds:
            return []
        rows = self._conn.execute(
            "SELECT * FROM vector_collections WHERE dataset_id = ? ORDER BY created_at",
            [ds["id"]],
        ).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, r)) for r in rows]

    def get_vector_collection_by_name(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get a vector collection by its name in the vector DB."""
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

    def insert_projections_batch(self, vector_collection_id: str, dataset_id: str,
                                 item_ids: List[str], projection_type: str,
                                 coordinates: List[List[float]]) -> int:
        """Bulk insert projection coordinates via DataFrame. Returns count inserted."""
        if not item_ids:
            return 0

        df = pd.DataFrame({
            "vector_collection_id": vector_collection_id,
            "item_id": item_ids,
            "dataset_id": dataset_id,
            "projection_type": projection_type,
            "coordinates": coordinates,
        })

        self._conn.execute("""
            INSERT OR REPLACE INTO projections
            SELECT vector_collection_id, item_id, dataset_id, projection_type, coordinates
            FROM df
        """)
        return len(item_ids)

    def upsert_projection_metadata(self, vector_collection_id: str,
                                   projection_type: str, *,
                                   variance: List[float] = None,
                                   computed_at: str = None) -> None:
        """Insert or update projection metadata (variance, timestamp)."""
        if computed_at is None:
            computed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._conn.execute("""
            INSERT OR REPLACE INTO projection_metadata
                (vector_collection_id, projection_type, variance, computed_at)
            VALUES (?, ?, ?, ?)
        """, [vector_collection_id, projection_type, variance, computed_at])

        # Mark vector collection as having projections
        self._conn.execute(
            "UPDATE vector_collections SET has_projections = TRUE WHERE id = ?",
            [vector_collection_id],
        )

    def get_projection_data(self, vector_collection_name: str,
                            projection_type: str) -> Optional[Dict[str, Any]]:
        """Load items + one projection type for visualization.

        Returns dict with: ids, documents, item_metadata, available_fields,
        coordinates, metadata (collection-level info).
        """
        vc = self.get_vector_collection_by_name(vector_collection_name)
        if not vc:
            return None
        dataset_id = vc["dataset_id"]
        vc_id = vc["id"]

        rows = self._conn.execute("""
            SELECT i.id, i.document, i.metadata, p.coordinates
            FROM items i
            INNER JOIN projections p
                ON p.dataset_id = i.dataset_id AND p.item_id = i.id
            WHERE i.dataset_id = ?
              AND p.vector_collection_id = ?
              AND p.projection_type = ?
            ORDER BY i.row_index
        """, [dataset_id, vc_id, projection_type]).fetchall()

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

        # Load dataset-level metadata
        ds = self.get_dataset_by_id(dataset_id)

        # Load projection variance
        pm_rows = self._conn.execute("""
            SELECT projection_type, variance, computed_at
            FROM projection_metadata
            WHERE vector_collection_id = ?
        """, [vc_id]).fetchall()
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
    # Text Search
    # ------------------------------------------------------------------

    def _rebuild_fts_if_needed(self):
        """Rebuild the FTS index on items.document if it's stale."""
        if not self._fts_dirty:
            return

        # Check if items table has rows
        count = self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        if count == 0:
            self._fts_dirty = False
            return

        try:
            self._conn.execute("PRAGMA drop_fts_index('items')")
        except duckdb.CatalogException:
            pass  # index doesn't exist yet

        self._conn.execute("""
            PRAGMA create_fts_index('items', 'id', 'document',
                stemmer='porter', stopwords='english', lower=1, overwrite=1)
        """)
        self._fts_dirty = False
        logger.info("Rebuilt FTS index on items.document (%d rows)", count)

    def text_search(self, dataset_name: str, query: str,
                    fields: Optional[List[str]] = None,
                    mode: str = "contains",
                    case_sensitive: bool = False) -> Dict[str, Any]:
        """Search documents and/or metadata fields.

        Args:
            dataset_name: dataset to search
            query: search string
            fields: list of fields to search. Use "__document__" for document text.
                    None searches documents only.
            mode: "contains" (substring) or "exact" (full value)
            case_sensitive: whether the search is case-sensitive

        Returns: {matches: [{id, matched_field, snippet}], total_matches: int}
        """
        ds = self.get_dataset(dataset_name)
        if not ds:
            return {"matches": [], "total_matches": 0}
        dataset_id = ds["id"]

        if fields is None:
            fields = ["__document__"]

        matches = []
        seen_ids = set()

        # Document search
        if "__document__" in fields:
            doc_matches = self._search_documents(dataset_id, query, mode, case_sensitive)
            for m in doc_matches:
                if m["id"] not in seen_ids:
                    matches.append(m)
                    seen_ids.add(m["id"])

        # Metadata field search
        meta_fields = [f for f in fields if f != "__document__"]
        if meta_fields:
            meta_matches = self._search_metadata_fields(
                dataset_id, query, meta_fields, mode, case_sensitive
            )
            for m in meta_matches:
                if m["id"] not in seen_ids:
                    matches.append(m)
                    seen_ids.add(m["id"])

        return {"matches": matches, "total_matches": len(matches)}

    def _search_documents(self, dataset_id: str, query: str,
                          mode: str, case_sensitive: bool) -> List[Dict]:
        """Search the document column."""
        if mode == "exact":
            if case_sensitive:
                rows = self._conn.execute(
                    "SELECT id, document FROM items WHERE dataset_id = ? AND document = ?",
                    [dataset_id, query],
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, document FROM items WHERE dataset_id = ? AND LOWER(document) = LOWER(?)",
                    [dataset_id, query],
                ).fetchall()
        else:
            # Contains mode — use ILIKE for substring (vectorized columnar scan)
            if case_sensitive:
                rows = self._conn.execute(
                    "SELECT id, document FROM items WHERE dataset_id = ? AND document LIKE '%' || ? || '%'",
                    [dataset_id, query],
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, document FROM items WHERE dataset_id = ? AND document ILIKE '%' || ? || '%'",
                    [dataset_id, query],
                ).fetchall()

        matches = []
        query_lower = query.lower()
        for r in rows:
            doc = r[1] or ""
            # Build snippet
            pos = doc.lower().find(query_lower)
            snippet = None
            if pos >= 0:
                start = max(0, pos - _SNIPPET_RADIUS)
                end = min(len(doc), pos + len(query) + _SNIPPET_RADIUS)
                snippet = ("..." if start > 0 else "") + doc[start:end] + ("..." if end < len(doc) else "")
            matches.append({"id": r[0], "matched_field": "__document__", "snippet": snippet})
        return matches

    def _search_metadata_fields(self, dataset_id: str, query: str,
                                fields: List[str], mode: str,
                                case_sensitive: bool) -> List[Dict]:
        """Search specific metadata JSON fields."""
        matches = []
        for field in fields:
            if mode == "exact":
                if case_sensitive:
                    rows = self._conn.execute("""
                        SELECT id FROM items
                        WHERE dataset_id = ? AND json_extract_string(metadata, ?) = ?
                    """, [dataset_id, f"$.{field}", query]).fetchall()
                else:
                    rows = self._conn.execute("""
                        SELECT id FROM items
                        WHERE dataset_id = ?
                          AND LOWER(CAST(json_extract_string(metadata, ?) AS VARCHAR)) = LOWER(?)
                    """, [dataset_id, f"$.{field}", query]).fetchall()
            else:
                if case_sensitive:
                    rows = self._conn.execute("""
                        SELECT id FROM items
                        WHERE dataset_id = ?
                          AND CAST(json_extract_string(metadata, ?) AS VARCHAR) LIKE '%' || ? || '%'
                    """, [dataset_id, f"$.{field}", query]).fetchall()
                else:
                    rows = self._conn.execute("""
                        SELECT id FROM items
                        WHERE dataset_id = ?
                          AND CAST(json_extract_string(metadata, ?) AS VARCHAR) ILIKE '%' || ? || '%'
                    """, [dataset_id, f"$.{field}", query]).fetchall()

            for r in rows:
                matches.append({"id": r[0], "matched_field": field, "snippet": None})
        return matches

    def text_search_bm25(self, dataset_name: str, query: str,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Word-level BM25 search using FTS index. Returns scored results."""
        ds = self.get_dataset(dataset_name)
        if not ds:
            return []

        self._rebuild_fts_if_needed()

        rows = self._conn.execute("""
            SELECT i.id, i.document, i.metadata,
                   fts_main_items.match_bm25(i.id, ?) AS score
            FROM items i
            WHERE i.dataset_id = ? AND score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        """, [query, ds["id"], limit]).fetchall()

        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "document": r[1],
                "metadata": json.loads(r[2]) if isinstance(r[2], str) and r[2] else {},
                "score": r[3],
            })
        return results

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    def create_topic_extraction(self, vector_collection_id: str, dataset_id: str,
                                config: Dict = None) -> str:
        """Create a new topic extraction record. Returns UUID.
        Marks any previous extraction on this vector_collection as inactive."""
        # Deactivate previous
        self._conn.execute("""
            UPDATE topic_extractions SET is_active = FALSE
            WHERE vector_collection_id = ? AND is_active = TRUE
        """, [vector_collection_id])

        extraction_id = str(uuid.uuid4())
        self._conn.execute("""
            INSERT INTO topic_extractions (id, vector_collection_id, dataset_id, config, is_active)
            VALUES (?, ?, ?, ?, TRUE)
        """, [extraction_id, vector_collection_id, dataset_id,
              json.dumps(config) if config else None])
        return extraction_id

    def insert_topic_info_batch(self, extraction_id: str,
                                topics: List[Dict]) -> int:
        """Insert topic info records. Each dict: {topic_id, label, count, keywords, ...}."""
        values = []
        for t in topics:
            values.append((
                extraction_id,
                t["topic_id"],
                t.get("label"),
                t.get("ctfidf_label"),
                t.get("count", 0),
                json.dumps(t.get("keywords")) if t.get("keywords") else None,
                json.dumps(t.get("subtopics")) if t.get("subtopics") else None,
            ))
        self._conn.executemany("""
            INSERT INTO topic_info (extraction_id, topic_id, label, ctfidf_label,
                count, keywords, subtopics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, values)
        return len(values)

    def insert_topic_assignments_batch(self, extraction_id: str, dataset_id: str,
                                       assignments: List[Dict]) -> int:
        """Insert topic assignments via DataFrame. Each dict: {item_id, topic_id, ...}."""
        if not assignments:
            return 0

        df = pd.DataFrame({
            "extraction_id": extraction_id,
            "dataset_id": dataset_id,
            "item_id": [a["item_id"] for a in assignments],
            "topic_id": [a["topic_id"] for a in assignments],
            "topic_label": [a.get("topic_label") for a in assignments],
            "subtopic_id": [a.get("subtopic_id") for a in assignments],
            "subtopic_label": [a.get("subtopic_label") for a in assignments],
        })

        self._conn.execute("""
            INSERT INTO topic_assignments
            SELECT extraction_id, dataset_id, item_id, topic_id,
                   topic_label, subtopic_id, subtopic_label
            FROM df
        """)
        return len(assignments)

    def get_active_topics(self, vector_collection_name: str) -> Optional[Dict[str, Any]]:
        """Get active topic extraction with all topic info for a vector collection."""
        vc = self.get_vector_collection_by_name(vector_collection_name)
        if not vc:
            return None

        # Get active extraction
        rows = self._conn.execute("""
            SELECT * FROM topic_extractions
            WHERE vector_collection_id = ? AND is_active = TRUE
            LIMIT 1
        """, [vc["id"]]).fetchall()
        if not rows:
            return None

        columns = [desc[0] for desc in self._conn.description]
        extraction = dict(zip(columns, rows[0]))

        # Get topic info
        ti_rows = self._conn.execute("""
            SELECT * FROM topic_info
            WHERE extraction_id = ?
            ORDER BY topic_id
        """, [extraction["id"]]).fetchall()
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

    def get_items_for_topic(self, extraction_id: str, topic_id: int) -> List[str]:
        """Get item IDs assigned to a specific topic."""
        rows = self._conn.execute(
            "SELECT item_id FROM topic_assignments WHERE extraction_id = ? AND topic_id = ?",
            [extraction_id, topic_id],
        ).fetchall()
        return [r[0] for r in rows]

    def update_topic_label(self, extraction_id: str, topic_id: int,
                           new_label: str) -> None:
        """Update a topic label in both topic_info and topic_assignments."""
        self._conn.execute(
            "UPDATE topic_info SET label = ? WHERE extraction_id = ? AND topic_id = ?",
            [new_label, extraction_id, topic_id],
        )
        self._conn.execute(
            "UPDATE topic_assignments SET topic_label = ? WHERE extraction_id = ? AND topic_id = ?",
            [new_label, extraction_id, topic_id],
        )

    def update_subtopic_label(self, extraction_id: str, subtopic_id: int,
                              new_label: str) -> None:
        """Update a subtopic label in topic_assignments."""
        self._conn.execute(
            "UPDATE topic_assignments SET subtopic_label = ? WHERE extraction_id = ? AND subtopic_id = ?",
            [new_label, extraction_id, subtopic_id],
        )

    # ------------------------------------------------------------------
    # Field Analysis (computed via SQL)
    # ------------------------------------------------------------------

    def compute_field_analysis(self, dataset_name: str) -> Dict[str, Any]:
        """Compute metadata field statistics via SQL aggregation."""
        ds = self.get_dataset(dataset_name)
        if not ds:
            return {}

        # Get a sample of metadata to discover fields
        sample_rows = self._conn.execute("""
            SELECT metadata FROM items
            WHERE dataset_id = ? AND metadata IS NOT NULL
            LIMIT 100
        """, [ds["id"]]).fetchall()

        if not sample_rows:
            return {}

        # Discover all unique keys across samples
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
                FROM items
                WHERE dataset_id = ? AND json_extract_string(metadata, '$.{key}') IS NOT NULL
            """, [ds["id"]]).fetchone()
            analysis[key] = {
                "total": row[0],
                "distinct_count": row[1],
            }

        return analysis
