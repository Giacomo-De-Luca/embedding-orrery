"""Download and verify a locked private HF Dataset seed snapshot."""

import argparse
import sys
from pathlib import Path

from interpretability_backend.backend.utils.seed_snapshot import (
    HuggingFaceSnapshotDownloader,
    SeedSnapshotConfig,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="snapshot JSON manifest")
    parser.add_argument(
        "--token-file",
        type=Path,
        help="BuildKit secret file containing a read-only Hugging Face token",
    )
    parser.add_argument(
        "--fallback",
        type=Path,
        help="local seed used only while no published lock exists",
    )
    args = parser.parse_args(argv)
    try:
        config = SeedSnapshotConfig.from_file(args.config)
        result = HuggingFaceSnapshotDownloader().download(
            config,
            token_file=args.token_file,
            fallback_dir=args.fallback,
        )
    except Exception as error:
        print(f"ERROR: snapshot download failed: {error}", file=sys.stderr)
        return 1
    print(f"Seed snapshot installed at {config.output_dir} ({result})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
