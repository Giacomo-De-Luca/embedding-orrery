"""First-run seed bootstrap.

On a fresh clone the live data stores (``resources/main.duckdb`` +
``resources/vector_db/``) do not exist — they are gitignored. This module
copies the committed seed snapshot (``resources/seed/``, built by
``scripts.build_seed_snapshot``) into place so the dashboard renders a
populated default on first launch.

The copy only happens when the live DuckDB is **absent**. It never overwrites
an existing database, so a developer's real (large) data store is always safe.
"""

import logging
import shutil
from pathlib import Path

from ..embedding_functions.config import DB_PATH, DUCKDB_PATH

logger = logging.getLogger("star_map." + __name__)

# Committed seed snapshot paths (mirror scripts/build_seed_snapshot.py).
SEED_DIR = DUCKDB_PATH.parent / "seed"
SEED_DUCKDB_PATH = SEED_DIR / "main.duckdb"
SEED_VECTOR_DB = SEED_DIR / "vector_db"


def ensure_seed_loaded() -> bool:
    """Populate the live data stores from the seed snapshot if needed.

    Returns True if the seed was copied into place, False otherwise (live DB
    already present, or no seed shipped).
    """
    live_duckdb = Path(DUCKDB_PATH)
    if live_duckdb.exists():
        # A real or already-seeded database exists — never clobber it.
        return False

    if not SEED_DUCKDB_PATH.exists():
        logger.info("No seed snapshot at %s; starting with empty stores.", SEED_DUCKDB_PATH)
        return False

    logger.info("No live database found; seeding from %s", SEED_DIR)

    live_duckdb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SEED_DUCKDB_PATH, live_duckdb)

    if SEED_VECTOR_DB.exists():
        shutil.copytree(SEED_VECTOR_DB, Path(DB_PATH), dirs_exist_ok=True)

    logger.info("Seed snapshot loaded into %s and %s", live_duckdb, DB_PATH)
    return True
