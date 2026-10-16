"""Compute topic centroids for existing topic extractions.

Centroids back semantic topic search; extractions created before they existed
have none. Config: ``backfill_topic_centroids_config.toml`` next to this file
(override with ``ORRERY_TOPIC_CENTROIDS_CONFIG``).

Run with the backend stopped because the live DuckDB store is single-writer:
    uv run python -m interpretability_backend.scripts.backfill_topic_centroids
"""

import sys
from pathlib import Path

import duckdb

from interpretability_backend.backend.clients.duckdb_client import DuckDBClient
from interpretability_backend.backend.services.topic_centroid_service import (
    TOPIC_LEVEL,
    TopicCentroidService,
)
from interpretability_backend.evaluation.utils.runner_common import (
    load_config,
    resolve_config_path,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("backfill_topic_centroids_config.toml")


def _targets(db: DuckDBClient, collections) -> list[tuple[str, str | None]]:
    """``(collection, active extraction id or None)`` for the configured collections."""
    if collections == "all":
        return db.get_active_extractions()
    return [(name, db.get_active_extraction_id(name)) for name in collections]


def main() -> int:
    config_path = resolve_config_path("ORRERY_TOPIC_CENTROIDS_CONFIG", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)
    skip_existing = config.get("skip_existing", True)

    try:
        db = DuckDBClient()
    except duckdb.IOException as error:
        print(f"ERROR: could not open DuckDB (is the backend running?): {error}", file=sys.stderr)
        return 1

    service = TopicCentroidService(duckdb=db)
    failures = 0
    try:
        for collection_name, extraction_id in _targets(db, config.get("collections", "all")):
            if extraction_id is None:
                print(f"skip  {collection_name}: no active topic extraction")
                continue
            if skip_existing and db.get_topic_centroids(extraction_id, TOPIC_LEVEL) is not None:
                print(f"skip  {collection_name}: centroids already stored")
                continue
            try:
                rows = service.refresh(collection_name, extraction_id)
                print(f"done  {collection_name}: {rows} centroid rows")
            except Exception as error:
                failures += 1
                print(f"FAIL  {collection_name}: {error}", file=sys.stderr)
    finally:
        db.close()

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
