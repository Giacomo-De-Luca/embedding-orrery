"""
Migrate all existing ChromaDB collections into DuckDB.

Reads from ChromaDB persistent storage and populates:
  - datasets (from collection-level metadata)
  - items (documents + per-item metadata, stripping projection/topic keys)
  - vector_collections (one per ChromaDB collection)
  - projections (parsed from JSON strings in per-item metadata)
  - topic_extractions + topic_info + topic_assignments (from topic metadata)

Safe to re-run: skips collections already present in DuckDB.

Usage:
    uv run python -m interpretability_backend.scripts.migrate_chromadb_to_duckdb
    uv run python -m interpretability_backend.scripts.migrate_chromadb_to_duckdb --collection emotion
    uv run python -m interpretability_backend.scripts.migrate_chromadb_to_duckdb --force  # re-migrate existing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.clients.duckdb_client import DuckDBClient
from backend.utils.resource_paths import CHROMA_DB_PATH as DB_PATH

# Projection keys stored as JSON strings in ChromaDB per-item metadata
PROJECTION_KEYS = {"pca_2d", "pca_3d", "umap_2d", "umap_3d"}

# Topic keys stored in ChromaDB per-item metadata
TOPIC_KEYS = {
    "topic_id",
    "topic_label",
    "subtopic_id",
    "subtopic_label",
    "ctfidf_label",
    "ctfidf_subtopic_map",
}


def get_chromadb_client():
    return chromadb.PersistentClient(
        path=str(DB_PATH.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )


def migrate_collection(db: DuckDBClient, chroma_client, collection_name: str) -> dict:
    """Migrate a single ChromaDB collection to DuckDB.

    Returns a stats dict with counts of migrated entities.
    """
    stats = {
        "collection": collection_name,
        "items": 0,
        "projections": {},
        "topics": 0,
        "topic_assignments": 0,
        "error": None,
    }

    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=None)
    except Exception as e:
        stats["error"] = f"Collection not found: {e}"
        return stats

    count = collection.count()
    meta = collection.metadata or {}

    # ---- 1. Create dataset ----
    source_type = "huggingface" if meta.get("source_dataset") else "local_file"
    if meta.get("data_type") == "image":
        source_type = "image"
    elif meta.get("data_type") == "vector":
        source_type = "vector"

    # Parse embedded_columns (stored as JSON string in ChromaDB)
    embedded_columns = meta.get("embedded_columns")
    if isinstance(embedded_columns, str):
        try:
            embedded_columns = json.loads(embedded_columns)
        except (json.JSONDecodeError, TypeError):
            pass

    db.create_dataset(
        collection_name,
        description=meta.get("description"),
        source_type=source_type,
        source_dataset=meta.get("source_dataset"),
        source_config=meta.get("source_config"),
        source_split=meta.get("source_split"),
        source_file=meta.get("source_file"),
        embedded_columns=embedded_columns if isinstance(embedded_columns, list) else None,
        data_type=meta.get("data_type"),
        total_in_source=meta.get("total_in_split") or meta.get("total_in_file"),
    )

    # ---- 2. Register vector collection ----
    embedding_dim = meta.get("embedding_dim")
    if isinstance(embedding_dim, str):
        try:
            embedding_dim = int(embedding_dim)
        except (ValueError, TypeError):
            embedding_dim = None

    db.register_vector_collection(
        collection_name,
        "chromadb",
        collection_name,
        "dense",
        embedding_provider=meta.get("embedding_provider"),
        embedding_model=meta.get("embedding_model"),
        embedding_dim=embedding_dim,
        embedding_task=meta.get("embedding_task"),
        embedding_task_type=meta.get("embedding_task_type"),
        embedding_prompt=meta.get("embedding_prompt"),
        item_count=count,
    )

    # ---- 3. Load all items in batches ----
    # Accumulate everything in memory (user confirmed collections fit in memory)
    all_ids = []
    all_docs = []
    all_metas = []
    proj_data = {}  # ptype -> (item_ids[], coordinates[])
    topic_item_data = []  # [{item_id, topic_id, topic_label, subtopic_id, subtopic_label}]

    batch_size = 5000
    for offset in range(0, count, batch_size):
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas", "documents"],
        )
        batch_ids = results["ids"]
        batch_docs = results["documents"] or [""] * len(batch_ids)
        batch_metas = results["metadatas"] or [{}] * len(batch_ids)

        all_ids.extend(batch_ids)
        all_docs.extend(batch_docs)

        for item_id, item_meta in zip(batch_ids, batch_metas, strict=True):
            if item_meta is None:
                item_meta = {}
            # Extract projections (JSON strings -> float lists)
            for ptype in PROJECTION_KEYS:
                raw = item_meta.get(ptype)
                if raw is not None:
                    try:
                        coords = json.loads(raw) if isinstance(raw, str) else raw
                        if isinstance(coords, list) and len(coords) >= 2:
                            if ptype not in proj_data:
                                proj_data[ptype] = ([], [])
                            proj_data[ptype][0].append(item_id)
                            proj_data[ptype][1].append(coords)
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Extract topic assignments
            topic_id_raw = item_meta.get("topic_id")
            if topic_id_raw is not None:
                try:
                    topic_item_data.append(
                        {
                            "item_id": item_id,
                            "topic_id": int(topic_id_raw),
                            "topic_label": item_meta.get("topic_label"),
                            "subtopic_id": int(item_meta["subtopic_id"])
                            if item_meta.get("subtopic_id") is not None
                            else None,
                            "subtopic_label": item_meta.get("subtopic_label"),
                        }
                    )
                except (ValueError, TypeError):
                    pass

            all_metas.append(item_meta)

    # ---- 4. Bulk insert items ----
    if all_ids:
        db.insert_items_batch(collection_name, all_ids, all_docs, all_metas)
        stats["items"] = len(all_ids)

    # ---- 5. Bulk insert projections per type ----
    for ptype, (item_ids, coords) in proj_data.items():
        db.insert_projections_batch(collection_name, item_ids, ptype, coords)
        stats["projections"][ptype] = len(item_ids)

    # ---- 6. Projection metadata (variance) ----
    pca_2d_var = meta.get("pca_2d_variance")
    pca_3d_var = meta.get("pca_3d_variance")
    if pca_2d_var:
        try:
            variance = json.loads(pca_2d_var) if isinstance(pca_2d_var, str) else pca_2d_var
            db.upsert_projection_metadata(collection_name, "pca_2d", variance=variance)
        except (json.JSONDecodeError, TypeError):
            pass
    if pca_3d_var:
        try:
            variance = json.loads(pca_3d_var) if isinstance(pca_3d_var, str) else pca_3d_var
            db.upsert_projection_metadata(collection_name, "pca_3d", variance=variance)
        except (json.JSONDecodeError, TypeError):
            pass

    if proj_data:
        db._conn.execute(
            "UPDATE vector_collections SET has_projections = TRUE WHERE collection_name = ?",
            [collection_name],
        )

    # ---- 7. Topics ----
    if meta.get("has_topics") and topic_item_data:
        # Parse topic_summary from collection metadata
        topic_summary_raw = meta.get("topic_summary", "[]")
        try:
            topic_summary = (
                json.loads(topic_summary_raw)
                if isinstance(topic_summary_raw, str)
                else topic_summary_raw
            )
        except (json.JSONDecodeError, TypeError):
            topic_summary = []

        # Parse topic config
        topic_config_raw = meta.get("topic_config")
        topic_config = None
        if topic_config_raw:
            try:
                topic_config = (
                    json.loads(topic_config_raw)
                    if isinstance(topic_config_raw, str)
                    else topic_config_raw
                )
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse topic hierarchy
        topic_hierarchy_raw = meta.get("topic_hierarchy")
        topic_hierarchy = None
        if topic_hierarchy_raw:
            try:
                topic_hierarchy = (
                    json.loads(topic_hierarchy_raw)
                    if isinstance(topic_hierarchy_raw, str)
                    else topic_hierarchy_raw
                )
            except (json.JSONDecodeError, TypeError):
                pass

        # Create topic extraction record
        ext_id = db.create_topic_extraction(collection_name, collection_name, config=topic_config)

        # Update extraction with reduction info if present
        if meta.get("reduction_applied"):
            db._conn.execute(
                """
                UPDATE topic_extractions SET
                    reduction_applied = TRUE,
                    reduction_method = ?,
                    reduction_target = ?,
                    num_topics_before_reduction = ?,
                    topic_hierarchy = ?,
                    topic_count = ?
                WHERE id = ?
            """,
                [
                    meta.get("reduction_method"),
                    meta.get("reduction_target"),
                    meta.get("num_topics_before_reduction"),
                    json.dumps(topic_hierarchy) if topic_hierarchy else None,
                    meta.get("topic_count"),
                    ext_id,
                ],
            )
        else:
            db._conn.execute(
                "UPDATE topic_extractions SET topic_count = ? WHERE id = ?",
                [meta.get("topic_count"), ext_id],
            )

        # Insert topic info from summary
        topic_info_records = []
        for entry in topic_summary:
            topic_info_records.append(
                {
                    "topic_id": entry["topic_id"],
                    "label": entry.get("label"),
                    "ctfidf_label": entry.get("ctfidf_label"),
                    "count": entry.get("count", 0),
                    "keywords": entry.get("keywords"),
                    "subtopics": entry.get("subtopics"),
                }
            )
        if topic_info_records:
            db.insert_topic_info_batch(ext_id, topic_info_records)
            stats["topics"] = len(topic_info_records)

        # Insert topic assignments
        db.insert_topic_assignments_batch(ext_id, topic_item_data)
        stats["topic_assignments"] = len(topic_item_data)

        db._conn.execute(
            "UPDATE vector_collections SET has_topics = TRUE WHERE collection_name = ?",
            [collection_name],
        )

    return stats


def verify_migration(db: DuckDBClient, chroma_client, collection_name: str) -> dict:
    """Verify DuckDB data matches ChromaDB for a collection."""
    collection = chroma_client.get_collection(name=collection_name, embedding_function=None)
    chroma_count = collection.count()

    ds = db.get_dataset(collection_name)
    duckdb_count = ds["count"] if ds else 0

    chroma_ids = set(collection.get(include=[])["ids"])
    duckdb_ids = db.get_item_ids(collection_name)

    missing_in_duckdb = chroma_ids - duckdb_ids
    extra_in_duckdb = duckdb_ids - chroma_ids

    return {
        "collection": collection_name,
        "chroma_count": chroma_count,
        "duckdb_count": duckdb_count,
        "match": chroma_count == duckdb_count and not missing_in_duckdb,
        "missing_in_duckdb": len(missing_in_duckdb),
        "extra_in_duckdb": len(extra_in_duckdb),
    }


def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB collections to DuckDB")
    parser.add_argument("--collection", type=str, help="Migrate a single collection by name")
    parser.add_argument(
        "--force", action="store_true", help="Re-migrate collections already in DuckDB"
    )
    parser.add_argument("--verify", action="store_true", help="Verify migration without migrating")
    parser.add_argument(
        "--db-path", type=str, default=None, help="DuckDB path (default: resources/main.duckdb)"
    )
    args = parser.parse_args()

    chroma_client = get_chromadb_client()
    db = DuckDBClient(db_path=args.db_path)

    # Determine which collections to migrate
    if args.collection:
        collection_names = [args.collection]
    else:
        collection_names = sorted([c.name for c in chroma_client.list_collections()])

    if args.verify:
        print(f"\nVerifying {len(collection_names)} collections...\n")
        all_ok = True
        for name in collection_names:
            v = verify_migration(db, chroma_client, name)
            status = "OK" if v["match"] else "MISMATCH"
            if not v["match"]:
                all_ok = False
            print(
                f"  [{status:8}] {name:<40} chroma={v['chroma_count']:>8,}  duckdb={v['duckdb_count']:>8,}  missing={v['missing_in_duckdb']}"
            )
        print(f"\n{'All collections verified.' if all_ok else 'Some collections have mismatches!'}")
        return

    # Migration
    print(f"\nMigrating {len(collection_names)} collections from ChromaDB to DuckDB...")
    print(f"  ChromaDB: {DB_PATH.resolve()}")
    print(f"  DuckDB:   {db.db_path}\n")

    total_items = 0
    total_time = 0
    results = []

    for i, name in enumerate(collection_names, 1):
        # Skip if already in DuckDB (unless --force)
        existing = db.get_dataset(name)
        if existing and not args.force:
            print(
                f"  [{i}/{len(collection_names)}] {name:<40} SKIP (already in DuckDB, {existing['count']:,} items)"
            )
            continue
        elif existing and args.force:
            print(
                f"  [{i}/{len(collection_names)}] {name:<40} deleting existing...",
                end=" ",
                flush=True,
            )
            db.delete_dataset(name)

        t0 = time.perf_counter()
        stats = migrate_collection(db, chroma_client, name)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        if stats["error"]:
            print(f"  [{i}/{len(collection_names)}] {name:<40} ERROR: {stats['error']}")
        else:
            total_items += stats["items"]
            proj_summary = ", ".join(f"{k}={v}" for k, v in stats["projections"].items()) or "none"
            topic_summary = (
                f"{stats['topics']} topics, {stats['topic_assignments']} assignments"
                if stats["topics"]
                else "none"
            )
            print(
                f"  [{i}/{len(collection_names)}] {name:<40} {stats['items']:>8,} items  projections=[{proj_summary}]  topics=[{topic_summary}]  {elapsed:.1f}s"
            )

        results.append(stats)

    # Verify
    print(f"\n{'=' * 70}")
    print(f"  Migration complete: {total_items:,} items in {total_time:.1f}s")
    print(f"{'=' * 70}")

    print("\nVerifying...\n")
    all_ok = True
    for name in collection_names:
        v = verify_migration(db, chroma_client, name)
        status = "OK" if v["match"] else "MISMATCH"
        if not v["match"]:
            all_ok = False
            print(
                f"  [{status:8}] {name:<40} chroma={v['chroma_count']:>8,}  duckdb={v['duckdb_count']:>8,}  missing={v['missing_in_duckdb']}"
            )
        else:
            print(f"  [{status:8}] {name:<40} {v['duckdb_count']:>8,} items")

    if all_ok:
        print(f"\nAll {len(collection_names)} collections verified successfully.")
    else:
        print("\nSome collections have mismatches! Run with --verify for details.")


if __name__ == "__main__":
    main()
