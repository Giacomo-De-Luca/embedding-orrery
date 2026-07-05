"""Runtime resource path configuration.

Local development keeps using ``interpretability_backend/resources``. Docker
can set ``ORRERY_RESOURCE_DIR`` to move mutable data into a named volume while
leaving committed seed assets in the image.
"""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESOURCE_DIR = PACKAGE_ROOT / "resources"


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


RESOURCE_DIR = _env_path("ORRERY_RESOURCE_DIR", DEFAULT_RESOURCE_DIR)
SEED_DIR = _env_path("ORRERY_SEED_DIR", DEFAULT_RESOURCE_DIR / "seed")
DIRECTIONS_DIR = _env_path("ORRERY_DIRECTIONS_DIR", DEFAULT_RESOURCE_DIR / "directions")

DUCKDB_PATH = RESOURCE_DIR / "main.duckdb"
CHROMA_DB_PATH = RESOURCE_DIR / "vector_db"
UPLOADS_DIR = RESOURCE_DIR / "uploads"
JOB_STATE_PATH = RESOURCE_DIR / "job_state.json"
SAE_LABELS_DIR = RESOURCE_DIR / "sae_labels"
SAE_VECTORS_DIR = RESOURCE_DIR / "sae_vectors"
PROBING_RESULTS_DIR = RESOURCE_DIR / "probing_results"
