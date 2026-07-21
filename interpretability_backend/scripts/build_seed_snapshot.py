"""Build a seed snapshot from a versioned JSON manifest.

Run with the backend stopped because the live DuckDB store is single-writer.

Usage:
    uv run python -m interpretability_backend.scripts.build_seed_snapshot
    uv run python -m interpretability_backend.scripts.build_seed_snapshot \
        --config interpretability_backend/config/seed_snapshots/demo.json
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb

from interpretability_backend.backend.utils.resource_paths import (
    CHROMA_DB_PATH,
    DUCKDB_PATH,
)
from interpretability_backend.backend.utils.seed_snapshot import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    SeedSnapshotBuilder,
    SeedSnapshotConfig,
)

COMMITTED_SEED_DIR = PROJECT_ROOT / "interpretability_backend/resources/seed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"snapshot JSON manifest (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args(argv)

    try:
        config = SeedSnapshotConfig.from_file(args.config)
        result = SeedSnapshotBuilder(
            source_duckdb_path=DUCKDB_PATH,
            source_chroma_path=CHROMA_DB_PATH,
            committed_seed_dir=COMMITTED_SEED_DIR,
        ).build(config)
    except duckdb.IOException as error:
        print(
            f"ERROR: could not read the production DuckDB (is the backend running?): {error}",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(f"ERROR: seed build failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
