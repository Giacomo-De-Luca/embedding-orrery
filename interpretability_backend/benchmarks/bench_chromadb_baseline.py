"""
Baseline benchmarks for current ChromaDB-only architecture.

Measures:
  - Projection data load time (the main visualization query)
  - Text search time (document + metadata field)
  - Memory usage for full collection load

Run: uv run python -m interpretability_backend.benchmarks.bench_chromadb_baseline
"""

import json
import statistics
import time
import tracemalloc
from pathlib import Path

import chromadb
from chromadb.config import Settings

DB_PATH = Path(__file__).parent.parent / "resources" / "vector_db"

# Collections to benchmark (small, medium, large)
BENCH_COLLECTIONS = {
    "small": "emotion",  # 1k items
    "medium": "ag_news",  # 10k items
    "large": "lacan_sentences_gemini_document",  # 153k items
}

WARMUP_RUNS = 1
TIMED_RUNS = 3


def get_client():
    return chromadb.PersistentClient(
        path=str(DB_PATH.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )


def bench_projection_load(client, collection_name: str, n_runs: int = TIMED_RUNS):
    """Benchmark: load all items with metadata + parse projections (the main viz query)."""
    collection = client.get_collection(name=collection_name, embedding_function=None)
    count = collection.count()

    # Warmup
    for _ in range(WARMUP_RUNS):
        collection.get(include=["metadatas", "documents"])

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        results = collection.get(include=["metadatas", "documents"])
        # Simulate the JSON parsing that get_projection_data() does
        ids = results["ids"]
        documents = results["documents"] or []
        metadatas = results["metadatas"] or []
        projections = {"pca_2d": [], "umap_2d": []}
        for meta in metadatas:
            for key in ("pca_2d", "umap_2d"):
                try:
                    projections[key].append(json.loads(meta.get(key, "[0,0]")))
                except (json.JSONDecodeError, TypeError):
                    projections[key].append([0.0, 0.0])
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "operation": "projection_load",
        "collection": collection_name,
        "item_count": count,
        "mean_s": round(statistics.mean(times), 4),
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
        "runs": n_runs,
    }


def bench_text_search_document(
    client, collection_name: str, query: str = "the", n_runs: int = TIMED_RUNS
):
    """Benchmark: text search on document field via where_document."""
    collection = client.get_collection(name=collection_name, embedding_function=None)
    count = collection.count()

    # Warmup
    for _ in range(WARMUP_RUNS):
        collection.get(where_document={"$contains": query}, include=[])

    times = []
    match_counts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        results = collection.get(where_document={"$contains": query}, include=[])
        t1 = time.perf_counter()
        times.append(t1 - t0)
        match_counts.append(len(results["ids"]))

    return {
        "operation": "text_search_document",
        "collection": collection_name,
        "item_count": count,
        "query": query,
        "matches": match_counts[0],
        "mean_s": round(statistics.mean(times), 4),
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "runs": n_runs,
    }


def bench_text_search_metadata(
    client, collection_name: str, field: str, query: str, n_runs: int = TIMED_RUNS
):
    """Benchmark: text search on a metadata field (Python-side filtering)."""
    collection = client.get_collection(name=collection_name, embedding_function=None)
    count = collection.count()

    # Warmup
    for _ in range(WARMUP_RUNS):
        results = collection.get(include=["metadatas"])
        for meta in results["metadatas"] or []:
            val = str(meta.get(field, ""))
            if query.lower() in val.lower():
                pass

    times = []
    match_counts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        results = collection.get(include=["metadatas"])
        matches = 0
        for meta in results["metadatas"] or []:
            val = str(meta.get(field, ""))
            if query.lower() in val.lower():
                matches += 1
        t1 = time.perf_counter()
        times.append(t1 - t0)
        match_counts.append(matches)

    return {
        "operation": "text_search_metadata",
        "collection": collection_name,
        "item_count": count,
        "field": field,
        "query": query,
        "matches": match_counts[0],
        "mean_s": round(statistics.mean(times), 4),
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "runs": n_runs,
    }


def bench_memory_projection_load(client, collection_name: str):
    """Benchmark: peak memory for loading projection data."""
    collection = client.get_collection(name=collection_name, embedding_function=None)
    count = collection.count()

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    results = collection.get(include=["metadatas", "documents"])
    ids = results["ids"]
    documents = results["documents"] or []
    metadatas = results["metadatas"] or []
    projections = {"pca_2d": [], "umap_2d": []}
    for meta in metadatas:
        for key in ("pca_2d", "umap_2d"):
            try:
                projections[key].append(json.loads(meta.get(key, "[0,0]")))
            except (json.JSONDecodeError, TypeError):
                projections[key].append([0.0, 0.0])

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Compute memory delta
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_delta_mb = sum(s.size_diff for s in stats) / (1024 * 1024)
    current_mb, peak_mb = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)

    # Get total size of snapshot_after
    total_mb = sum(s.size for s in snapshot_after.statistics("lineno")) / (1024 * 1024)

    return {
        "operation": "memory_projection_load",
        "collection": collection_name,
        "item_count": count,
        "total_allocated_mb": round(total_mb, 2),
        "delta_mb": round(total_delta_mb, 2),
    }


def find_metadata_field(client, collection_name: str) -> str | None:
    """Find a searchable metadata field in the collection."""
    collection = client.get_collection(name=collection_name, embedding_function=None)
    sample = collection.get(limit=5, include=["metadatas"])
    if not sample["metadatas"]:
        return None
    meta = sample["metadatas"][0]
    # Pick first string field that isn't a projection or topic
    skip = {
        "pca_2d",
        "pca_3d",
        "umap_2d",
        "umap_3d",
        "topic_id",
        "topic_label",
        "subtopic_id",
        "subtopic_label",
        "row_index",
        "mapped_colour",
        "mapped_colour_scale",
    }
    for key, val in meta.items():
        if key not in skip and isinstance(val, str) and len(val) > 3:
            return key
    return None


def run_benchmarks():
    client = get_client()
    results = []

    for size_label, coll_name in BENCH_COLLECTIONS.items():
        try:
            collection = client.get_collection(name=coll_name, embedding_function=None)
        except Exception:
            print(f"  SKIP {coll_name} (not found)")
            continue

        count = collection.count()
        print(f"\n{'=' * 60}")
        print(f"  {size_label.upper()}: {coll_name} ({count} items)")
        print(f"{'=' * 60}")

        # Projection load
        print("  Benchmarking projection load...")
        r = bench_projection_load(client, coll_name)
        results.append(r)
        print(f"    mean={r['mean_s']}s  median={r['median_s']}s")

        # Document text search
        print("  Benchmarking document text search...")
        r = bench_text_search_document(client, coll_name, query="the")
        results.append(r)
        print(f"    mean={r['mean_s']}s  matches={r['matches']}")

        # Metadata field search
        meta_field = find_metadata_field(client, coll_name)
        if meta_field:
            sample = collection.get(limit=1, include=["metadatas"])
            sample_val = str(sample["metadatas"][0].get(meta_field, ""))[:5]
            if len(sample_val) >= 3:
                print(
                    f"  Benchmarking metadata search (field={meta_field}, query={sample_val!r})..."
                )
                r = bench_text_search_metadata(client, coll_name, meta_field, sample_val)
                results.append(r)
                print(f"    mean={r['mean_s']}s  matches={r['matches']}")

        # Memory
        print("  Benchmarking memory usage...")
        r = bench_memory_projection_load(client, coll_name)
        results.append(r)
        print(f"    allocated={r['total_allocated_mb']}MB  delta={r['delta_mb']}MB")

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        if "mean_s" in r:
            print(
                f"  [{r['collection'][:30]:30}] {r['operation']:30} {r['mean_s']:>8.4f}s  ({r['item_count']} items)"
            )
        else:
            print(
                f"  [{r['collection'][:30]:30}] {r['operation']:30} {r['total_allocated_mb']:>8.2f}MB ({r['item_count']} items)"
            )

    # Save results as JSON for later comparison
    out_path = Path(__file__).parent / "results_chromadb_baseline.json"
    import json as json_mod

    with open(out_path, "w") as f:
        json_mod.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_benchmarks()
