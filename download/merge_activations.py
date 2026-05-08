#!/usr/bin/env python3
"""Merge gzipped activation batches into a single JSONL per SAE source.

Reads batch-*.jsonl.gz files from each source subdirectory under
activations/ and concatenates them into one sorted JSONL file per source.

Usage:
    uv run python -m interpret.download.merge_activations
    uv run python -m interpret.download.merge_activations --sources 22-gemmascope-2-res-16k
    uv run python -m interpret.download.merge_activations --input-dir /path/to/labels/dir
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MODEL_ID = "gemma-3-4b-it"
DEFAULT_LABELS_DIR = (
    Path.home() / "Colour_vectors/resources/sae_labels" / f"neuronpedia_{MODEL_ID}"
)


def merge_source(source_dir: Path, output_file: Path) -> int:
    """Decompress and merge all batch files for one source into a single JSONL."""
    batch_files = sorted(source_dir.glob("batch-*.jsonl.gz"))
    if not batch_files:
        print(f"  No batch files in {source_dir.name}")
        return 0

    records: list[tuple[int, str]] = []
    pbar = (
        tqdm(batch_files, desc=f"  {source_dir.name}", unit="batch", ncols=80, leave=False)
        if tqdm else None
    )

    for batch_file in (pbar or batch_files):
        with gzip.open(batch_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx = int(json.loads(line)["index"])
                records.append((idx, line))

    if pbar:
        pbar.close()

    # Sort by feature index for consistent ordering
    records.sort(key=lambda r: r[0])

    with open(output_file, "w") as out:
        for _, line in records:
            out.write(line + "\n")

    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Merge gzipped activation batches into single JSONL per source",
    )
    parser.add_argument(
        "--input-dir", type=str, default=str(DEFAULT_LABELS_DIR),
        help=f"Labels directory containing activations/ subdir (default: {DEFAULT_LABELS_DIR})",
    )
    parser.add_argument(
        "--sources", nargs="+", default=None,
        help="Source IDs to merge (default: all in activations/)",
    )
    args = parser.parse_args()

    labels_dir = Path(args.input_dir)
    act_dir = labels_dir / "activations"
    if not act_dir.is_dir():
        print(f"No activations/ directory found in {labels_dir}")
        sys.exit(1)

    # Find source directories
    if args.sources:
        source_dirs = [act_dir / s for s in args.sources]
        missing = [d for d in source_dirs if not d.is_dir()]
        if missing:
            print(f"Source directories not found: {[d.name for d in missing]}")
            sys.exit(1)
    else:
        source_dirs = sorted(d for d in act_dir.iterdir() if d.is_dir())

    print(f"Merging activations for {len(source_dirs)} sources\n")

    for source_dir in source_dirs:
        output_file = labels_dir / f"{MODEL_ID}_{source_dir.name}_activations.jsonl"
        print(f"[{source_dir.name}]")
        count = merge_source(source_dir, output_file)
        size_mb = output_file.stat().st_size / 1024**2
        print(f"  {count} records -> {output_file.name} ({size_mb:.0f} MB)\n")

    print("Done.")


if __name__ == "__main__":
    main()
