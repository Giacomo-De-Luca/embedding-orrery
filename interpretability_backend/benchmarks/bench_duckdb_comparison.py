"""
DuckDB comparison benchmarks — mirrors the ChromaDB baseline to compare
projection load, text search, and memory usage.

Populates DuckDB from existing ChromaDB collections, then benchmarks
the same operations.

Run: uv run python -m interpretability_backend.benchmarks.bench_duckdb_comparison
"""

import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import chromadb
from chromadb.config import Settings

# Add backend to path so we can import the client
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.clients.duckdb_client import DuckDBClient

CHROMA_DB_PATH = Path(__file__).parent.parent / "resources" / "vector_db"

BENCH_COLLECTIONS = {
    "small": "emotion",
    "medium": "ag_news",
    "large": "lacan_sentences_gemini_document",
}

WARMUP_RUNS = 1
TIMED_RUNS = 3


def populate_duckdb_from_chromadb(db: DuckDBClient, chroma_client, collection_name: str):
    """Migrate one ChromaDB collection into DuckDB for benchmarking."""
    collection = chroma_client.get_collection(name=collection_name, embedding_function=None)
    count = collection.count()

    # Create dataset
    metadata = collection.metadata or {}
    ds_id = db.create_dataset(
        collection_name,
        description=metadata.get("description"),
        source_type="huggingface" if metadata.get("source_dataset") else "local_file",
        source_dataset=metadata.get("source_dataset"),
    )

    # Register vector collection
    vc_id = db.register_vector_collection(
        ds_id,
        "chromadb",
        collection_name,
        "dense",
        embedding_provider=metadata.get("embedding_provider"),
        embedding_model=metadata.get("embedding_model"),
        embedding_dim=metadata.get("embedding_dim"),
    )

    # Accumulate all data first, then bulk insert
    all_ids = []
    all_docs = []
    all_metas = []
    proj_data = {}  # ptype -> (item_ids, coords)

    batch_size = 5000
    for offset in range(0, count, batch_size):
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas", "documents"],
        )
        ids = results["ids"]
        documents = results["documents"] or [""] * len(ids)
        metadatas = results["metadatas"] or [{}] * len(ids)

        all_ids.extend(ids)
        all_docs.extend(documents)

        for item_id, meta in zip(ids, metadatas):
            for ptype in ("pca_2d", "pca_3d", "umap_2d", "umap_3d"):
                if ptype in meta:
                    try:
                        coords = json.loads(meta[ptype])
                        if ptype not in proj_data:
                            proj_data[ptype] = ([], [])
                        proj_data[ptype][0].append(item_id)
                        proj_data[ptype][1].append(coords)
                    except (json.JSONDecodeError, TypeError):
                        pass
            all_metas.append(meta)

    # Bulk insert items
    db.insert_items_batch(ds_id, all_ids, all_docs, all_metas)

    # Bulk insert projections per type
    for ptype, (item_ids, coords) in proj_data.items():
        db.insert_projections_batch(vc_id, ds_id, item_ids, ptype, coords)

    return ds_id, vc_id


def bench_projection_load_duckdb(
    db: DuckDBClient, collection_name: str, item_count: int, n_runs: int = TIMED_RUNS
):
    """Benchmark: load projection data from DuckDB."""
    # Warmup
    for _ in range(WARMUP_RUNS):
        db.get_projection_data(collection_name, "umap_2d")

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        data = db.get_projection_data(collection_name, "umap_2d")
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "operation": "projection_load",
        "backend": "duckdb",
        "collection": collection_name,
        "item_count": item_count,
        "mean_s": round(statistics.mean(times), 4),
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
        "runs": n_runs,
    }


def bench_text_search_document_duckdb(
    db: DuckDBClient,
    collection_name: str,
    item_count: int,
    query: str = "the",
    n_runs: int = TIMED_RUNS,
):
    """Benchmark: text search on document field via DuckDB ILIKE."""
    # Warmup
    for _ in range(WARMUP_RUNS):
        db.text_search(collection_name, query)

    times = []
    match_counts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        results = db.text_search(collection_name, query)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        match_counts.append(results["total_matches"])

    return {
        "operation": "text_search_document",
        "backend": "duckdb",
        "collection": collection_name,
        "item_count": item_count,
        "query": query,
        "matches": match_counts[0],
        "mean_s": round(statistics.mean(times), 4),
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "runs": n_runs,
    }


def bench_text_search_metadata_duckdb(
    db: DuckDBClient,
    collection_name: str,
    item_count: int,
    field: str,
    query: str,
    n_runs: int = TIMED_RUNS,
):
    """Benchmark: metadata field search via DuckDB json_extract + ILIKE."""
    # Warmup
    for _ in range(WARMUP_RUNS):
        db.text_search(collection_name, query, fields=[field])

    times = []
    match_counts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        results = db.text_search(collection_name, query, fields=[field])
        t1 = time.perf_counter()
        times.append(t1 - t0)
        match_counts.append(results["total_matches"])

    return {
        "operation": "text_search_metadata",
        "backend": "duckdb",
        "collection": collection_name,
        "item_count": item_count,
        "field": field,
        "query": query,
        "matches": match_counts[0],
        "mean_s": round(statistics.mean(times), 4),
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "runs": n_runs,
    }


def bench_bm25_search_duckdb(
    db: DuckDBClient,
    collection_name: str,
    item_count: int,
    query: str = "the",
    n_runs: int = TIMED_RUNS,
):
    """Benchmark: BM25 full-text search via DuckDB FTS extension."""
    # Warmup (also triggers FTS rebuild)
    for _ in range(WARMUP_RUNS):
        db.text_search_bm25(collection_name, query)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        results = db.text_search_bm25(collection_name, query)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "operation": "text_search_bm25",
        "backend": "duckdb",
        "collection": collection_name,
        "item_count": item_count,
        "query": query,
        "matches": len(results),
        "mean_s": round(statistics.mean(times), 4),
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "runs": n_runs,
    }


def bench_memory_projection_load_duckdb(db: DuckDBClient, collection_name: str, item_count: int):
    """Benchmark: peak memory for loading projection data from DuckDB."""
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    data = db.get_projection_data(collection_name, "umap_2d")

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_delta_mb = sum(s.size_diff for s in stats) / (1024 * 1024)
    total_mb = sum(s.size for s in snapshot_after.statistics("lineno")) / (1024 * 1024)

    return {
        "operation": "memory_projection_load",
        "backend": "duckdb",
        "collection": collection_name,
        "item_count": item_count,
        "total_allocated_mb": round(total_mb, 2),
        "delta_mb": round(total_delta_mb, 2),
    }


def run_benchmarks():
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )

    # Use a temporary DuckDB file for benchmarks
    bench_db_path = str(Path(__file__).parent / "_bench_temp.duckdb")
    db = DuckDBClient(db_path=bench_db_path)

    results = []

    # Load baseline results for comparison
    baseline_path = Path(__file__).parent / "results_chromadb_baseline.json"
    baseline = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            for r in json.load(f):
                key = (r["collection"], r["operation"])
                baseline[key] = r

    for size_label, coll_name in BENCH_COLLECTIONS.items():
        try:
            collection = chroma_client.get_collection(name=coll_name, embedding_function=None)
        except Exception:
            print(f"  SKIP {coll_name} (not found)")
            continue

        count = collection.count()
        print(f"\n{'=' * 70}")
        print(f"  {size_label.upper()}: {coll_name} ({count} items)")
        print(f"{'=' * 70}")

        # Populate DuckDB
        print("  Populating DuckDB from ChromaDB...")
        t0 = time.perf_counter()
        ds_id, vc_id = populate_duckdb_from_chromadb(db, chroma_client, coll_name)
        t_populate = time.perf_counter() - t0
        print(f"    Populated in {t_populate:.2f}s")

        # Projection load
        print("  Benchmarking projection load...")
        r = bench_projection_load_duckdb(db, coll_name, count)
        results.append(r)
        chroma_baseline = baseline.get((coll_name, "projection_load"), {})
        chroma_time = chroma_baseline.get("mean_s", 0)
        speedup = chroma_time / r["mean_s"] if r["mean_s"] > 0 else 0
        print(
            f"    DuckDB: {r['mean_s']}s  |  ChromaDB: {chroma_time}s  |  {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}"
        )

        # Document text search
        print("  Benchmarking document text search (ILIKE)...")
        r = bench_text_search_document_duckdb(db, coll_name, count, query="the")
        results.append(r)
        chroma_baseline = baseline.get((coll_name, "text_search_document"), {})
        chroma_time = chroma_baseline.get("mean_s", 0)
        speedup = chroma_time / r["mean_s"] if r["mean_s"] > 0 else 0
        print(
            f"    DuckDB: {r['mean_s']}s ({r['matches']} matches)  |  ChromaDB: {chroma_time}s  |  {speedup:.1f}x"
        )

        # BM25 search
        print("  Benchmarking BM25 full-text search...")
        r = bench_bm25_search_duckdb(db, coll_name, count, query="the")
        results.append(r)
        print(f"    BM25: {r['mean_s']}s ({r['matches']} results)")

        # Metadata field search
        meta_baseline = baseline.get((coll_name, "text_search_metadata"))
        if meta_baseline:
            field = meta_baseline["field"]
            query = meta_baseline["query"]
            print(f"  Benchmarking metadata search (field={field}, query={query!r})...")
            r = bench_text_search_metadata_duckdb(db, coll_name, count, field, query)
            results.append(r)
            chroma_time = meta_baseline.get("mean_s", 0)
            speedup = chroma_time / r["mean_s"] if r["mean_s"] > 0 else 0
            print(
                f"    DuckDB: {r['mean_s']}s ({r['matches']} matches)  |  ChromaDB: {chroma_time}s  |  {speedup:.1f}x"
            )

        # Memory
        print("  Benchmarking memory usage...")
        r = bench_memory_projection_load_duckdb(db, coll_name, count)
        results.append(r)
        chroma_baseline = baseline.get((coll_name, "memory_projection_load"), {})
        chroma_mb = chroma_baseline.get("total_allocated_mb", 0)
        savings = (1 - r["total_allocated_mb"] / chroma_mb) * 100 if chroma_mb > 0 else 0
        print(
            f"    DuckDB: {r['total_allocated_mb']}MB  |  ChromaDB: {chroma_mb}MB  |  {savings:.0f}% {'savings' if savings > 0 else 'more'}"
        )

    # Summary
    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Collection':<32} {'Operation':<25} {'DuckDB':>10} {'ChromaDB':>10} {'Speedup':>10}")
    print(f"  {'-' * 32} {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in results:
        if "mean_s" in r:
            key = (r["collection"], r["operation"])
            chroma_r = baseline.get(key, {})
            chroma_time = chroma_r.get("mean_s", "N/A")
            if isinstance(chroma_time, (int, float)) and r["mean_s"] > 0:
                speedup = f"{chroma_time / r['mean_s']:.1f}x"
            else:
                speedup = "N/A"
                chroma_time = "N/A"
            print(
                f"  {r['collection'][:32]:<32} {r['operation']:<25} {r['mean_s']:>9.4f}s {str(chroma_time):>9}s {speedup:>10}"
            )
        else:
            key = (r["collection"], r["operation"])
            chroma_r = baseline.get(key, {})
            chroma_mb = chroma_r.get("total_allocated_mb", "N/A")
            print(
                f"  {r['collection'][:32]:<32} {r['operation']:<25} {r['total_allocated_mb']:>8.1f}MB {str(chroma_mb):>8}MB"
            )

    # Save results
    out_path = Path(__file__).parent / "results_duckdb_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Cleanup temp DB
    db.close()
    import os

    try:
        os.remove(bench_db_path)
        os.remove(bench_db_path + ".wal")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    run_benchmarks()
