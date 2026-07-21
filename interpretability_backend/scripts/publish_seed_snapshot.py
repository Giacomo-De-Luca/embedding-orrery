"""Publish an existing verified seed snapshot to its configured HF Dataset."""

import argparse
import json
import sys
from pathlib import Path

from interpretability_backend.backend.utils.seed_snapshot import (
    DEFAULT_CONFIG_PATH,
    HuggingFaceSnapshotPublisher,
    SeedSnapshotConfig,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"snapshot JSON manifest (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args(argv)
    try:
        config = SeedSnapshotConfig.from_file(args.config)
        lock = HuggingFaceSnapshotPublisher().publish(config)
    except Exception as error:
        print(f"ERROR: snapshot publication failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(lock.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
