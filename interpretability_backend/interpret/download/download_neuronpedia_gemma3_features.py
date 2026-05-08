#!/usr/bin/env python3
"""
Download Gemma 3 4B autointerpreter features from Neuronpedia.

This script downloads SAE (Sparse Autoencoder) features with their
autointerpreter explanations from Neuronpedia's API for the Gemma 3 4B model.

Usage:
    # 1. Install dependencies:
    pip install requests tqdm

    # 2. (Optional) Set your API key for higher rate limits:
    export NEURONPEDIA_API_KEY="your_key_here"
    # Get a free key at: https://www.neuronpedia.org/account

    # 3. Run the script:
    python download_neuronpedia_gemma3_4b.py

    # Or customize:
    python download_neuronpedia_gemma3_4b.py --layers 0 5 10 --width 16k --hook res --max-features 1000

API Reference:
    - Docs: https://docs.neuronpedia.org/api
    - Interactive: https://neuronpedia.org/api-doc
    - Gemma Scope 2: https://www.neuronpedia.org/gemma-scope-2
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("Missing dependency. Install with: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    tqdm = None

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_ID = "gemma-3-4b-it"
BASE_URL = "https://neuronpedia.org/api"

# Gemma Scope 2 SAE naming convention:
#   {layer}-gemmascope-2-{hook}-{width}
# Hooks: res (residual), att (attention out), mlp (MLP out)
# Widths: 16k, 65k (most common)
# Gemma 3 4B has 34 layers (0-33)

HOOKS = ["res", "att", "mlp"]
WIDTHS = ["16k"]
NUM_LAYERS = 34  # Gemma 3 4B has 34 transformer layers
DEFAULT_OUTPUT_DIR = Path.home() / "Colour_vectors/resources/sae_labels" / f"neuronpedia_{MODEL_ID}" 
DEFAULT_MAX_FEATURES = 16384  # 16k SAE width
DEFAULT_BATCH_SIZE = 100
DEFAULT_WORKERS = 4
DEFAULT_RATE_LIMIT_DELAY = 0.1  # seconds between requests





# ── API Client ───────────────────────────────────────────────────────────────

class NeuronpediaClient:
    """Simple client for the Neuronpedia REST API."""

    def __init__(self, api_key=None, rate_limit_delay=DEFAULT_RATE_LIMIT_DELAY):
        self.session = requests.Session()
        self.session.headers["Accept-Encoding"] = "gzip"
        if api_key:
            self.session.headers["X-Api-Key"] = api_key
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0

    def _throttle(self):
        """Simple rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def get_feature(self, model_id, source, index, retries=3):
        """
        Fetch a single feature from the API.
        Returns the full JSON response including explanations and activations.
        """
        url = f"{BASE_URL}/feature/{model_id}/{source}/{index}"
        for attempt in range(retries):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return None  # Feature doesn't exist
                elif resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    print(f"\n  Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"\n  Request error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"\n  Failed after {retries} attempts: {e}")
                    return None
        return None

    def list_model_sources(self, model_id):
        """
        Try to list available sources for a model.
        Falls back to common Gemma Scope 2 naming patterns if the endpoint isn't available.
        """
        # Try the undocumented model page API first
        url = f"{BASE_URL}/model/{model_id}"
        try:
            self._throttle()
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if "sources" in data:
                    return data["sources"]
        except Exception:
            pass

        return None

    def probe_source(self, model_id, source):
        """Check if a source exists by trying to fetch feature 0."""
        result = self.get_feature(model_id, source, 0)
        return result is not None


def build_source_id(layer, hook, width):
    """Build the Neuronpedia source ID for a Gemma Scope 2 SAE."""
    return f"{layer}-gemmascope-2-{hook}-{width}"


# ── Download Logic ───────────────────────────────────────────────────────────

def extract_autointerp_data(raw_feature):
    """Extract the useful autointerpreter fields from a raw API response."""
    if raw_feature is None:
        return None

    explanations = []
    for exp in raw_feature.get("explanations", []):
        explanations.append({
            "text": exp.get("description", ""),
            "method": exp.get("typeName"),
            "explainer_model": exp.get("explanationModelName"),
            "score": exp.get("score"),
        })

    # Extract top activating tokens (compact form)
    top_activations = []
    for act in raw_feature.get("activations", []):
        tokens = act.get("tokens", [])
        values = act.get("values", [])
        top_activations.append({
            "tokens": tokens,
            "values": values,
            "bin_min": act.get("binMin"),
            "bin_max": act.get("binMax"),
        })

    return {
        "model_id": raw_feature.get("modelId"),
        "source": raw_feature.get("layer"),
        "index": raw_feature.get("index"),
        "density": raw_feature.get("frac_nonzero"),
        "explanations": explanations,
        "activations": top_activations,
        "top_logits": raw_feature.get("topLogits"),
        "bottom_logits": raw_feature.get("bottomLogits"),
    }


def download_source_features(client, model_id, source, max_features, output_dir, workers=1):
    """Download all features for a given source, saving as JSONL."""
    output_file = output_dir / f"{model_id}_{source}_features.jsonl"

    # Check for existing partial download
    existing_count = 0
    existing_indices = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    feat = json.loads(line)
                    existing_indices.add(feat["index"])
                    existing_count += 1
                except json.JSONDecodeError:
                    continue
        if existing_count > 0:
            print(f"  Resuming: {existing_count} features already downloaded")

    indices_to_download = [i for i in range(max_features) if i not in existing_indices]
    if not indices_to_download:
        print(f"  All {max_features} features already downloaded.")
        return existing_count

    downloaded = existing_count
    failed = 0

    # Progress tracking
    if tqdm:
        pbar = tqdm(total=len(indices_to_download), desc=f"  {source}", unit="feat",
                     initial=0, ncols=80)
    else:
        pbar = None

    def fetch_one(idx):
        raw = client.get_feature(model_id, source, idx)
        return idx, extract_autointerp_data(raw)

    write_lock = threading.Lock()

    with open(output_file, "a") as f:
        def _write_result(data):
            nonlocal downloaded, failed
            if data is not None:
                line = json.dumps(data) + "\n"
                with write_lock:
                    f.write(line)
                    f.flush()
                downloaded += 1
            else:
                failed += 1
            if pbar:
                pbar.update(1)
            elif (downloaded + failed) % 500 == 0:
                print(f"    Progress: {downloaded} downloaded, {failed} failed")

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(fetch_one, idx): idx for idx in indices_to_download}
                for future in as_completed(futures):
                    _, data = future.result()
                    _write_result(data)
        else:
            for idx in indices_to_download:
                _, data = fetch_one(idx)
                _write_result(data)

    if pbar:
        pbar.close()

    print(f"  Done: {downloaded} features saved, {failed} failed → {output_file.name}")
    return downloaded


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_available_sources(client, model_id, hooks, widths, layers):
    """Probe the API to find which sources actually exist."""
    print(f"\nDiscovering available SAE sources for {model_id}...")
    available = []

    # First try listing from the API
    api_sources = client.list_model_sources(model_id)
    if api_sources:
        print(f"  Found {len(api_sources)} sources from API.")
        return api_sources

    # Otherwise, probe common patterns
    candidates = []
    for hook in hooks:
        for width in widths:
            for layer in layers:
                candidates.append(build_source_id(layer, hook, width))

    print(f"  Probing {len(candidates)} candidate sources...")
    for source in candidates:
        exists = client.probe_source(model_id, source)
        status = "✓" if exists else "✗"
        print(f"    {status} {source}")
        if exists:
            available.append(source)

    return available


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Gemma 3 4B autointerpreter features from Neuronpedia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download residual stream features for layers 0, 15, 33:
  python %(prog)s --layers 0 15 33 --hook res --width 16k

  # Download all hooks for layer 10:
  python %(prog)s --layers 10 --hook res att mlp

  # Download first 100 features only (quick test):
  python %(prog)s --layers 0 --hook res --max-features 100

  # Discover what sources are available:
  python %(prog)s --discover

  # Download a specific source by name:
  python %(prog)s --sources 10-gemmascope-2-res-16k 15-gemmascope-2-res-16k
        """,
    )

    parser.add_argument("--api-key", type=str, default=None,
                        help="Neuronpedia API key (or set NEURONPEDIA_API_KEY env var)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")

    # Source selection
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer numbers to download (default: all 34 layers)")
    parser.add_argument("--hook", type=str, nargs="+", default=["res"],
                        choices=HOOKS,
                        help="Hook types: res (residual), att (attention), mlp (default: res)")
    parser.add_argument("--width", type=str, nargs="+", default=["16k"],
                        help="SAE widths (default: 16k)")
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="Explicit source IDs to download (overrides --layers/--hook/--width)")

    # Download options
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES,
                        help=f"Max feature index to download per source (default: {DEFAULT_MAX_FEATURES})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel download workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT_DELAY,
                        help=f"Seconds between API requests (default: {DEFAULT_RATE_LIMIT_DELAY})")

    # Modes
    parser.add_argument("--discover", action="store_true",
                        help="Only discover available sources, don't download")
    parser.add_argument("--raw", action="store_true",
                        help="Save raw API responses instead of extracted autointerp data")

    args = parser.parse_args()

    # Setup
    api_key = args.api_key or os.getenv("NEURONPEDIA_API_KEY")
    client = NeuronpediaClient(api_key=api_key, rate_limit_delay=args.rate_limit)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Neuronpedia Gemma 3 4B Feature Downloader")
    print("=" * 60)
    print(f"Model:       {MODEL_ID}")
    print(f"API key:     {'set' if api_key else 'not set (may be rate limited)'}")
    print(f"Output dir:  {output_dir}")
    print()

    # Determine layers
    layers = args.layers if args.layers is not None else list(range(NUM_LAYERS))

    # Discovery mode
    if args.discover:
        available = discover_available_sources(client, MODEL_ID, args.hook, args.width, layers)
        print(f"\nFound {len(available)} available sources:")
        for s in available:
            if isinstance(s, dict):
                print(f"  - {s.get('id', s)}")
            else:
                print(f"  - {s}")
        return

    # Build source list
    if args.sources:
        sources = args.sources
    else:
        sources = []
        for hook in args.hook:
            for width in args.width:
                for layer in layers:
                    sources.append(build_source_id(layer, hook, width))

    print(f"Sources to download: {len(sources)}")
    print(f"Max features/source: {args.max_features}")
    print(f"Workers:             {args.workers}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta = {
        "model_id": MODEL_ID,
        "sources": sources,
        "max_features": args.max_features,
        "download_started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "download_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Download
    total_features = 0
    for i, source in enumerate(sources):
        print(f"\n[{i+1}/{len(sources)}] Downloading {source}...")

        # Quick probe to check source exists
        if not client.probe_source(MODEL_ID, source):
            print(f"  Source not found, skipping.")
            continue

        count = download_source_features(
            client=client,
            model_id=MODEL_ID,
            source=source,
            max_features=args.max_features,
            output_dir=output_dir,
            workers=args.workers,
        )
        total_features += count

    # Summary
    print("\n" + "=" * 60)
    print(f"Download complete!")
    print(f"Total features: {total_features}")
    print(f"Output dir:     {output_dir}")
    print(f"Files:")
    for f in sorted(output_dir.glob("*.jsonl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
