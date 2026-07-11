"""
Build a small, shippable seed snapshot from the production data stores.

Exports a set of demo collections from the (large) production ``main.duckdb``
+ ``vector_db`` into a self-contained seed directory. On a fresh install the
backend copies the seed into place on first startup (see
``backend.utils.seed_bootstrap``), so the dashboard renders a populated
default with no setup, network, or model download.

Two seeds are built from the same script:
  - default (committed to git, ``resources/seed/``):
      emotion              (1000 rows, all-MiniLM-L6-v2, Gemini topic labels)
      xkcd_hilbert_gemini  (954 rows, gemini-embedding, rainbow mapped_colour)
  - demo seed for the HuggingFace Space (``resources/seed_demo/``, gitignored —
    ~313 MB, ships only in the Space repo/image):
      the two above plus acl_abstracts_emnlp_findings
      (13,980 EMNLP abstracts, gemini-embedding-2, LLM topic labels)

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
    # default committed seed (emotion + xkcd → resources/seed/)
    uv run python -m interpretability_backend.scripts.build_seed_snapshot

    # HF Space demo seed (adds the EMNLP collection, separate output dir)
    uv run python -m interpretability_backend.scripts.build_seed_snapshot \
        --collections emotion xkcd_hilbert_gemini acl_abstracts_emnlp_findings \
        --datasets emotion xkcd_hilbert acl_abstracts_emnlp_findings \
        --output interpretability_backend/resources/seed_demo
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import chromadb
import duckdb
from chromadb.config import Settings

from interpretability_backend.backend.clients.chromadb_client import ChromaDBClient
from interpretability_backend.backend.clients.duckdb_client import DuckDBClient
from interpretability_backend.backend.utils.resource_paths import (
    CHROMA_DB_PATH as DB_PATH,
    DUCKDB_PATH,
)

# Default collections to ship, and the datasets that back them.
# (A dataset's items table is shared by all its vector_collections.)
SEED_COLLECTIONS = ["emotion", "xkcd_hilbert_gemini"]
SEED_DATASETS = ["emotion", "xkcd_hilbert"]

# Default seed output path (committed to git; un-ignored in .gitignore).
SEED_DIR = DUCKDB_PATH.parent / "seed"


def _sql_name_filter(column: str, names: list[str]) -> str:
    """Build a ``column IN (...)`` filter for trusted, CLI-provided names."""
    return "{} IN ({})".format(column, ", ".join("'{}'".format(n.replace("'", "''")) for n in names))


def export_duckdb(datasets: list[str], collections: list[str], seed_duckdb_path: Path) -> None:
    """Create the seed DuckDB with schema, then copy filtered production rows."""
    dataset_filter = _sql_name_filter("name", datasets)
    collection_filter = _sql_name_filter("collection_name", collections)

    print(f"[duckdb] creating seed schema at {seed_duckdb_path}")
    seed_client = DuckDBClient(db_path=str(seed_duckdb_path))
    # Reuse the canonical items-table naming from DuckDBClient (single source
    # of truth) while the client is alive.
    items_tables = {}
    for dataset in datasets:
        seed_client._ensure_items_table(dataset)
        items_tables[dataset] = seed_client._items_table(dataset)
    seed_client.close()

    print(f"[duckdb] attaching production DB {DUCKDB_PATH} (read-only)")
    con = duckdb.connect(str(seed_duckdb_path))
    try:
        # ATTACH does not support bound parameters; the path is from config.
        con.execute(f"ATTACH '{DUCKDB_PATH.resolve()}' AS prod (READ_ONLY)")

        # Parents first, then children (FK order). BY NAME maps columns by
        # name rather than position — the production tables gain columns over
        # time via guarded ALTERs (e.g. topic_extractions.quality_metrics), so
        # their column order can differ from a freshly created schema.
        statements = [
            (
                "datasets",
                f"INSERT INTO datasets BY NAME SELECT * FROM prod.datasets WHERE {dataset_filter}",
            ),
        ]
        for dataset in datasets:
            tbl = items_tables[dataset]
            statements.append(
                (f"items({dataset})", f"INSERT INTO {tbl} BY NAME SELECT * FROM prod.{tbl}")
            )
        statements += [
            (
                "vector_collections",
                f"INSERT INTO vector_collections BY NAME SELECT * FROM prod.vector_collections WHERE {collection_filter}",
            ),
            (
                "projections",
                f"INSERT INTO projections BY NAME SELECT * FROM prod.projections WHERE {collection_filter}",
            ),
            (
                "projection_metadata",
                f"INSERT INTO projection_metadata BY NAME SELECT * FROM prod.projection_metadata WHERE {collection_filter}",
            ),
            (
                "topic_extractions",
                f"INSERT INTO topic_extractions BY NAME SELECT * FROM prod.topic_extractions WHERE {collection_filter}",
            ),
            (
                "topic_info",
                f"INSERT INTO topic_info BY NAME SELECT * FROM prod.topic_info WHERE extraction_id IN (SELECT id FROM prod.topic_extractions WHERE {collection_filter})",
            ),
            (
                "topic_assignments",
                f"INSERT INTO topic_assignments BY NAME SELECT * FROM prod.topic_assignments WHERE extraction_id IN (SELECT id FROM prod.topic_extractions WHERE {collection_filter})",
            ),
        ]

        for label, sql in statements:
            # DuckDB INSERT returns a one-row result with the inserted count.
            row = con.execute(sql).fetchone()
            count = row[0] if row else 0
            print(f"[duckdb]   {label:<22} {count:>7} rows")

        con.execute("DETACH prod")
        # Fold the WAL into the main file so the seed ships as a single,
        # self-contained .duckdb with no sidecar.
        con.execute("CHECKPOINT")
    finally:
        con.close()


def export_chromadb(
    collections: list[str], seed_vector_db: Path, fallback_vector_db: Path | None
) -> None:
    """Read vectors from each source collection and re-add to a fresh store.

    A collection can be empty in the production store while its vectors
    survive in the committed seed (observed live for ``emotion``); in that
    case ``fallback_vector_db`` — a TEMP COPY of the committed seed store,
    snapshotted by main() before the output dir is wiped — is used as the
    vector source. A copy for two reasons: the output may BE the committed
    seed being rebuilt, and ``PersistentClient`` has no read-only mode, so
    opening the committed store directly would dirty a tracked binary.
    """
    print(f"[chroma] reading source vectors from {DB_PATH}")
    src = ChromaDBClient(db_path=str(DB_PATH))
    fallback = None
    if fallback_vector_db is not None and fallback_vector_db.exists():
        fallback = chromadb.PersistentClient(
            path=str(fallback_vector_db.resolve()),
            settings=Settings(anonymized_telemetry=False),
        )
    dest = chromadb.PersistentClient(
        path=str(seed_vector_db.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )

    for name in collections:
        src_col = src.get_collection(name)
        source_label = "prod"
        if src_col.count() == 0 and fallback is not None:
            try:
                seed_col = fallback.get_collection(name)
            except Exception:
                seed_col = None
            if seed_col is not None and seed_col.count() > 0:
                src_col = seed_col
                source_label = "committed seed (prod collection is empty)"
        total = src_col.count()
        if total == 0:
            raise RuntimeError(
                f"collection '{name}' has no vectors in the production store "
                f"(and no committed-seed fallback was available)"
            )

        # Preserve source metadata verbatim so the EF config (provider, model,
        # dim, task) survives for live semantic search. Copy in 5k batches —
        # Chroma caps a single add() at its max batch size (~5.4k), and this
        # also bounds memory for large collections.
        dest_col = dest.create_collection(name=name, metadata=dict(src_col.metadata or {}))
        batch_size = 5000
        copied = 0
        dim = 0
        while copied < total:
            data = src_col.get(
                include=["embeddings"], limit=min(batch_size, total - copied), offset=copied
            )
            ids = data["ids"]
            if not ids:
                break
            embeddings = [list(vec) for vec in data["embeddings"]]
            dim = len(embeddings[0])
            dest_col.add(ids=ids, embeddings=embeddings)
            copied += len(ids)
        print(f"[chroma]   {name:<22} {copied:>7} vectors (dim={dim}, source: {source_label})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--collections",
        nargs="+",
        default=SEED_COLLECTIONS,
        help="vector collections to export (default: the committed demo pair)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=SEED_DATASETS,
        help="datasets backing those collections (vector_collections.dataset_name)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SEED_DIR,
        help=f"seed output directory (default: {SEED_DIR})",
    )
    args = parser.parse_args(argv)

    seed_dir: Path = args.output
    seed_duckdb_path = seed_dir / "main.duckdb"
    seed_vector_db = seed_dir / "vector_db"

    if not DUCKDB_PATH.exists():
        print(f"ERROR: production DuckDB not found at {DUCKDB_PATH}", file=sys.stderr)
        return 1

    # Snapshot the committed seed's vector store to a temp dir BEFORE any
    # destructive step: it serves as the fallback vector source for
    # collections that are empty in production, it may itself be the output
    # being rebuilt, and Chroma has no read-only open mode (a direct open
    # would dirty the tracked sqlite file).
    committed_vector_db = SEED_DIR / "vector_db"
    fallback_tmp: str | None = None
    fallback_vector_db: Path | None = None
    if committed_vector_db.exists():
        fallback_tmp = tempfile.mkdtemp(prefix="orrery_seed_fallback_")
        fallback_vector_db = Path(fallback_tmp) / "vector_db"
        shutil.copytree(committed_vector_db, fallback_vector_db)

    if seed_dir.exists():
        print(f"[clean] removing existing seed dir {seed_dir}")
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Both stores are locked while the server runs. On any failure, remove the
    # partially-built seed dir so we never leave a half-built snapshot behind.
    try:
        export_duckdb(args.datasets, args.collections, seed_duckdb_path)
        export_chromadb(args.collections, seed_vector_db, fallback_vector_db)
    except duckdb.IOException as e:
        shutil.rmtree(seed_dir, ignore_errors=True)
        print(
            f"ERROR: could not open production DuckDB (is the backend running?): {e}",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        shutil.rmtree(seed_dir, ignore_errors=True)
        print(f"ERROR: seed build failed (is the backend running?): {e}", file=sys.stderr)
        return 1
    finally:
        if fallback_tmp is not None:
            shutil.rmtree(fallback_tmp, ignore_errors=True)

    size = sum(f.stat().st_size for f in seed_dir.rglob("*") if f.is_file())
    print(f"\nSeed snapshot built at {seed_dir} ({size / 1e6:.1f} MB)")
    print("Collections:", ", ".join(args.collections))
    return 0


if __name__ == "__main__":
    sys.exit(main())
