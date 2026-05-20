"""
Build a small, shippable seed snapshot from the production data stores.

Exports a fixed set of demo collections from the (large) production
``main.duckdb`` + ``vector_db`` into a tiny ``resources/seed/`` directory that
is committed to git. On a fresh clone the backend copies this seed into place
on first startup (see ``backend.utils.seed_bootstrap``), so the dashboard
renders a populated default with no setup, network, or model download.

What gets exported (see SEED_COLLECTIONS):
  - emotion              (1000 rows, all-MiniLM-L6-v2, Gemini topic labels)
  - xkcd_hilbert_gemini  (954 rows, gemini-embedding, rainbow mapped_colour)

DuckDB side: the seed DB is created with the canonical schema (via
``DuckDBClient``), then production rows for the target datasets/collections are
copied in FK-dependency order via ``ATTACH ... (READ_ONLY)`` + ``INSERT SELECT``.

ChromaDB side: vectors are read from each source collection and re-added to a
fresh ``PersistentClient`` (rebuilds a clean HNSW index with the installed
Chroma version → version-robust). The source collection ``metadata`` is
preserved verbatim so live semantic search can reconstruct the embedding
function (see ``ChromaDBClient.get_collection``).

IMPORTANT: run with the backend STOPPED — DuckDB is single-writer and the
production DB is locked while the server is running.

Usage:
    uv run python -m interpretability_backend.scripts.build_seed_snapshot
"""

import shutil
import sys

import chromadb
import duckdb
from chromadb.config import Settings

from interpretability_backend.backend.clients.chromadb_client import ChromaDBClient
from interpretability_backend.backend.clients.duckdb_client import DuckDBClient
from interpretability_backend.backend.embedding_functions.config import DB_PATH, DUCKDB_PATH

# Collections to ship, and the datasets that back them.
# (A dataset's items table is shared by all its vector_collections.)
SEED_COLLECTIONS = ["emotion", "xkcd_hilbert_gemini"]
SEED_DATASETS = ["emotion", "xkcd_hilbert"]

# Seed output paths (committed to git; un-ignored in .gitignore).
SEED_DIR = DUCKDB_PATH.parent / "seed"
SEED_DUCKDB_PATH = SEED_DIR / "main.duckdb"
SEED_VECTOR_DB = SEED_DIR / "vector_db"

# Tables copied wholesale-filtered. Per-dataset items tables are handled
# separately. Order matters: parents before children (FK constraints).
_DATASET_FILTER = "name IN ({})".format(", ".join(f"'{d}'" for d in SEED_DATASETS))
_COLLECTION_FILTER = "collection_name IN ({})".format(", ".join(f"'{c}'" for c in SEED_COLLECTIONS))


def _quoted_items_table(dataset_name: str) -> str:
    """Mirror DuckDBClient items-table naming for a dataset (quoted identifier)."""
    return f'"items_{DuckDBClient._sanitize_table_name(dataset_name)}"'


def export_duckdb() -> None:
    """Create the seed DuckDB with schema, then copy filtered production rows."""
    print(f"[duckdb] creating seed schema at {SEED_DUCKDB_PATH}")
    seed_client = DuckDBClient(db_path=str(SEED_DUCKDB_PATH))
    for dataset in SEED_DATASETS:
        seed_client._ensure_items_table(dataset)
    seed_client.close()

    print(f"[duckdb] attaching production DB {DUCKDB_PATH} (read-only)")
    con = duckdb.connect(str(SEED_DUCKDB_PATH))
    try:
        # ATTACH does not support bound parameters; the path is from config.
        con.execute(f"ATTACH '{DUCKDB_PATH.resolve()}' AS prod (READ_ONLY)")

        # Parents first, then children (FK order).
        statements = [
            (
                "datasets",
                f"INSERT INTO datasets SELECT * FROM prod.datasets WHERE {_DATASET_FILTER}",
            ),
        ]
        for dataset in SEED_DATASETS:
            tbl = _quoted_items_table(dataset)
            statements.append((f"items({dataset})", f"INSERT INTO {tbl} SELECT * FROM prod.{tbl}"))
        statements += [
            (
                "vector_collections",
                f"INSERT INTO vector_collections SELECT * FROM prod.vector_collections WHERE {_COLLECTION_FILTER}",
            ),
            (
                "projections",
                f"INSERT INTO projections SELECT * FROM prod.projections WHERE {_COLLECTION_FILTER}",
            ),
            (
                "projection_metadata",
                f"INSERT INTO projection_metadata SELECT * FROM prod.projection_metadata WHERE {_COLLECTION_FILTER}",
            ),
            (
                "topic_extractions",
                f"INSERT INTO topic_extractions SELECT * FROM prod.topic_extractions WHERE {_COLLECTION_FILTER}",
            ),
            (
                "topic_info",
                f"INSERT INTO topic_info SELECT * FROM prod.topic_info WHERE extraction_id IN (SELECT id FROM prod.topic_extractions WHERE {_COLLECTION_FILTER})",
            ),
            (
                "topic_assignments",
                f"INSERT INTO topic_assignments SELECT * FROM prod.topic_assignments WHERE extraction_id IN (SELECT id FROM prod.topic_extractions WHERE {_COLLECTION_FILTER})",
            ),
        ]

        for label, sql in statements:
            # DuckDB INSERT returns a one-row result with the inserted count.
            row = con.execute(sql).fetchone()
            count = row[0] if row else 0
            print(f"[duckdb]   {label:<22} {count:>7} rows")

        con.execute("DETACH prod")
    finally:
        con.close()


def export_chromadb() -> None:
    """Read vectors from each source collection and re-add to a fresh store."""
    print(f"[chroma] reading source vectors from {DB_PATH}")
    src = ChromaDBClient(db_path=str(DB_PATH))
    dest = chromadb.PersistentClient(
        path=str(SEED_VECTOR_DB.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )

    for name in SEED_COLLECTIONS:
        src_col = src.get_collection(name)
        total = src_col.count()
        data = src_col.get(include=["embeddings"], limit=total)
        ids = data["ids"]
        embeddings = [list(vec) for vec in data["embeddings"]]

        # Preserve source metadata verbatim so the EF config (provider, model,
        # dim, task) survives for live semantic search.
        dest_col = dest.create_collection(name=name, metadata=dict(src_col.metadata or {}))
        dest_col.add(ids=ids, embeddings=embeddings)
        print(
            f"[chroma]   {name:<22} {len(ids):>7} vectors (dim={len(embeddings[0]) if embeddings else 0})"
        )


def main() -> int:
    if not DUCKDB_PATH.exists():
        print(f"ERROR: production DuckDB not found at {DUCKDB_PATH}", file=sys.stderr)
        return 1

    if SEED_DIR.exists():
        print(f"[clean] removing existing seed dir {SEED_DIR}")
        shutil.rmtree(SEED_DIR)
    SEED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        export_duckdb()
    except duckdb.IOException as e:
        print(
            f"ERROR: could not open production DuckDB (is the backend running?): {e}",
            file=sys.stderr,
        )
        return 1

    export_chromadb()

    size = sum(f.stat().st_size for f in SEED_DIR.rglob("*") if f.is_file())
    print(f"\nSeed snapshot built at {SEED_DIR} ({size / 1e6:.1f} MB)")
    print("Collections:", ", ".join(SEED_COLLECTIONS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
