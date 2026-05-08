#!/usr/bin/env python3
"""Download Neuronpedia SAE feature data from the public S3 dataset bucket.

Downloads three data types per source:

- **features/**:     density, top/bottom logits, histograms  (~17 MB/source gz)
- **explanations/**:  autointerpreter label, 256-dim embedding (~20 MB/source gz)
- **activations/**:   token-level activation examples          (~336 MB/source gz)

Features and explanations are merged into a single JSONL compatible with
``FeatureLabelStore``. Activations are stored as raw gzipped batches in a
subdirectory (too large to merge, and better accessed per-batch).

Bucket index:
    https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html

Usage:
    uv run python -m interpret.download.download_neuronpedia_s3
    uv run python -m interpret.download.download_neuronpedia_s3 --sources 22-gemmascope-2-res-16k
    uv run python -m interpret.download.download_neuronpedia_s3 --list
    uv run python -m interpret.download.download_neuronpedia_s3 --skip-activations
"""

import argparse
import gzip
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree

try:
    import requests
except ImportError:
    print("Missing dependency. Install with: uv add requests")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── Configuration ────────────────────────────────────────────────────────────

# Default model ID — used by the CLI entry point. Exported functions accept
# model_id as an explicit parameter so callers are not tied to this default.
DEFAULT_MODEL_ID = "gemma-3-4b-it"
S3_BUCKET = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"

DEFAULT_OUTPUT_DIR = (
    Path.home() / "Colour_vectors/resources/sae_labels" / f"neuronpedia_{DEFAULT_MODEL_ID}"
)

# ── S3 helpers ───────────────────────────────────────────────────────────────

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _s3_prefix(model_id: str) -> str:
    """Build the S3 key prefix for a given model."""
    return f"v1/{model_id}"


def list_sources(
    session: requests.Session,
    model_id: str = DEFAULT_MODEL_ID,
) -> list[str]:
    """List available source IDs under the model prefix via S3 ListObjectsV2."""
    prefix = _s3_prefix(model_id)
    resp = session.get(
        S3_BUCKET,
        params={"prefix": f"{prefix}/", "delimiter": "/", "list-type": "2"},
        timeout=30,
    )
    resp.raise_for_status()
    root = ElementTree.fromstring(resp.content)
    sources = []
    for cp in root.findall(".//s3:CommonPrefixes/s3:Prefix", S3_NS):
        parts = cp.text.rstrip("/").split("/")
        sources.append(parts[-1])
    return sorted(sources)


def list_batch_keys(
    session: requests.Session,
    source: str,
    data_dir: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> list[str]:
    """List batch-*.jsonl.gz keys for a source/data_dir combination."""
    prefix = f"{_s3_prefix(model_id)}/{source}/{data_dir}/"
    resp = session.get(
        S3_BUCKET,
        params={"prefix": prefix, "list-type": "2"},
        timeout=30,
    )
    resp.raise_for_status()
    root = ElementTree.fromstring(resp.content)
    keys = []
    for contents in root.findall(".//s3:Contents", S3_NS):
        key = contents.find("s3:Key", S3_NS).text
        if key.endswith(".jsonl.gz"):
            keys.append(key)
    return sorted(keys)


def download_batch(session: requests.Session, key: str, retries: int = 3) -> bytes:
    """Download a single gzipped batch file with retries."""
    url = f"{S3_BUCKET}/{key}"
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 2**attempt
                print(f"    Retry {attempt + 1}/{retries} for {key}: {e}")
                time.sleep(wait)
            else:
                raise


def parse_batch(raw_gz: bytes) -> list[dict]:
    """Decompress and parse a gzipped JSONL batch."""
    text = gzip.decompress(raw_gz).decode("utf-8")
    return [json.loads(line) for line in text.strip().split("\n") if line]


# ── Merge features + explanations ────────────────────────────────────────────


def download_and_index(
    session: requests.Session,
    source: str,
    data_dir: str,
    desc: str,
    model_id: str = DEFAULT_MODEL_ID,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[int, dict]:
    """Download all batches for a data_dir, returning {index: first_record}."""
    keys = list_batch_keys(session, source, data_dir, model_id=model_id)
    if not keys:
        return {}

    by_index: dict[int, dict] = {}
    pbar = tqdm(keys, desc=desc, unit="batch", ncols=80, leave=False) if tqdm else None

    for i, key in enumerate(pbar or keys):
        raw = download_batch(session, key)
        for record in parse_batch(raw):
            idx = int(record["index"])
            if idx not in by_index:
                by_index[idx] = record
        if progress_callback:
            progress_callback(i + 1, len(keys))

    if pbar:
        pbar.close()
    return by_index


def merge_feature_record(
    feature: dict | None,
    explanation: dict | None,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict:
    """Merge a features/ record and an explanations/ record into store format."""
    feat = feature or {}
    expl = explanation or {}

    idx = int(feat.get("index", expl.get("index", -1)))

    # Build explanation entry preserving embedding
    explanations = []
    if expl:
        entry = {"text": expl.get("description", "")}
        if "embedding" in expl:
            entry["embedding"] = expl["embedding"]
        if "typeName" in expl:
            entry["method"] = expl["typeName"]
        if "explanationModelName" in expl:
            entry["explainer_model"] = expl["explanationModelName"]
        explanations.append(entry)

    # Top/bottom logits from features/
    top_logits = None
    bottom_logits = None
    if feat.get("pos_str"):
        top_logits = list(zip(feat["pos_str"], feat.get("pos_values", [])))
    if feat.get("neg_str"):
        bottom_logits = list(zip(feat["neg_str"], feat.get("neg_values", [])))

    return {
        "model_id": feat.get("modelId", expl.get("modelId", model_id)),
        "source": feat.get("layer", expl.get("layer", "")),
        "index": idx,
        "density": feat.get("frac_nonzero", 0.0),
        "explanations": explanations,
        "top_logits": top_logits,
        "bottom_logits": bottom_logits,
    }


# ── Download activations as raw batches ──────────────────────────────────────


def download_activations_raw(
    session: requests.Session,
    source: str,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Download activations/ batches as-is into a subdirectory."""
    act_dir = output_dir / "activations" / source
    act_dir.mkdir(parents=True, exist_ok=True)

    keys = list_batch_keys(session, source, "activations", model_id=model_id)
    if not keys:
        print("    No activation batches found")
        return 0

    # Skip already-downloaded batches
    existing = {f.name for f in act_dir.glob("*.jsonl.gz")}
    to_download = [k for k in keys if k.rsplit("/", 1)[-1] not in existing]

    if not to_download:
        print(f"    All {len(keys)} activation batches already downloaded")
        return len(keys)

    if existing:
        print(f"    {len(existing)} batches already downloaded, {len(to_download)} remaining")

    pbar = (
        tqdm(to_download, desc="    activations", unit="batch", ncols=80, leave=False)
        if tqdm
        else None
    )

    for i, key in enumerate(pbar or to_download):
        raw = download_batch(session, key)
        filename = key.rsplit("/", 1)[-1]
        (act_dir / filename).write_bytes(raw)
        if progress_callback:
            progress_callback(i + 1, len(to_download))

    if pbar:
        pbar.close()
    return len(keys)


# ── Source-level orchestration ───────────────────────────────────────────────


def download_source(
    session: requests.Session,
    source: str,
    output_dir: Path,
    skip_activations: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> int:
    """Download all data for one source.

    Args:
        session: HTTP session for S3 requests.
        source: Neuronpedia source ID (e.g. ``"9-gemmascope-2-res-65k"``).
        output_dir: Directory to write JSONL + activation batches into.
        skip_activations: If True, skip the (large) activation download.
        model_id: Neuronpedia model ID (e.g. ``"gemma-3-4b-it"``).
        progress_callback: ``(stage, done, total)`` progress reporter.
            Stages: ``"download_features"``, ``"download_explanations"``,
            ``"download_activations"``.

    Returns:
        Number of newly merged feature records.
    """
    output_file = output_dir / f"{model_id}_{source}_features.jsonl"

    # Resume: collect already-merged indices
    existing_indices: set[int] = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    existing_indices.add(json.loads(line)["index"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if existing_indices:
            print(f"  {len(existing_indices)} features already merged")

    # Helper to create per-stage progress sub-callbacks
    def _sub_progress(stage: str) -> Callable[[int, int], None] | None:
        if progress_callback is None:
            return None
        return lambda done, total: progress_callback(stage, done, total)

    # Download features/ and explanations/
    print("  Downloading features/...")
    features_by_idx = download_and_index(
        session,
        source,
        "features",
        "    features",
        model_id=model_id,
        progress_callback=_sub_progress("download_features"),
    )

    print("  Downloading explanations/...")
    explanations_by_idx = download_and_index(
        session,
        source,
        "explanations",
        "    explanations",
        model_id=model_id,
        progress_callback=_sub_progress("download_explanations"),
    )

    # Merge and write JSONL
    all_indices = sorted(set(features_by_idx) | set(explanations_by_idx))
    new_count = 0

    with open(output_file, "a") as out:
        for idx in all_indices:
            if idx in existing_indices:
                continue
            merged = merge_feature_record(
                feature=features_by_idx.get(idx),
                explanation=explanations_by_idx.get(idx),
                model_id=model_id,
            )
            out.write(json.dumps(merged) + "\n")
            new_count += 1

    print(f"  Merged {new_count} new features -> {output_file.name}")

    # Download activations as raw batches
    if not skip_activations:
        print("  Downloading activations/...")
        n_batches = download_activations_raw(
            session,
            source,
            output_dir,
            model_id=model_id,
            progress_callback=_sub_progress("download_activations"),
        )
        print(f"  {n_batches} activation batches stored")
    else:
        print("  Skipping activations/")

    return new_count


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download Neuronpedia SAE features from S3 (bulk, fast)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Source IDs to download (e.g. 22-gemmascope-2-res-16k). Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_sources",
        help="List available sources and exit",
    )
    parser.add_argument(
        "--skip-activations",
        action="store_true",
        help="Skip downloading activations/ (much faster, smaller files)",
    )
    args = parser.parse_args()

    model_id = DEFAULT_MODEL_ID

    session = requests.Session()
    session.headers["Accept-Encoding"] = "gzip"

    print("=" * 60)
    print(f"Neuronpedia S3 Bulk Downloader — {model_id}")
    print("=" * 60)

    # List mode
    if args.list_sources:
        sources = list_sources(session, model_id=model_id)
        print(f"\nAvailable sources ({len(sources)}):")
        for s in sources:
            print(f"  {s}")
        return

    # Determine sources
    if args.sources:
        sources = args.sources
    else:
        sources = list_sources(session, model_id=model_id)
        print(f"Found {len(sources)} sources")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")

    total = 0
    for i, source in enumerate(sources):
        print(f"\n[{i + 1}/{len(sources)}] {source}")
        count = download_source(
            session,
            source,
            output_dir,
            args.skip_activations,
            model_id=model_id,
        )
        total += count

    # Summary
    print("\n" + "=" * 60)
    print(f"Done — {total} new features merged across {len(sources)} sources")
    for f in sorted(output_dir.glob("*.jsonl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")
    act_dir = output_dir / "activations"
    if act_dir.exists():
        total_act = sum(f.stat().st_size for f in act_dir.rglob("*.jsonl.gz"))
        print(f"  activations/ ({total_act / 1024**2:.0f} MB total)")
    print("=" * 60)


if __name__ == "__main__":
    main()
