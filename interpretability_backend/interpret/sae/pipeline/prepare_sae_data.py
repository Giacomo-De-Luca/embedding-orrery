"""Orchestrate the full SAE data pipeline for one or more Neuronpedia sources.

Runs three stages end-to-end, each a thin wrapper around an existing script:

1. **Download** — ``download_neuronpedia_s3.download_source`` pulls features,
   explanations, and raw activation batches from the Neuronpedia S3 bucket.
2. **Merge activations** — ``merge_activations.merge_source`` decompresses
   and concatenates the batch-*.jsonl.gz files into a single sorted JSONL.
3. **Extract vectors** — ``extract_decoder_vectors.extract_and_merge`` loads
   the SAE weights from HuggingFace and merges decoder directions with the
   downloaded labels into a parquet file.

Each entry in ``PrepareSAEConfig.items`` is a ``(source_id, sae_config)`` pair:

- ``source_id`` is the Neuronpedia source string exactly as it appears on S3
  (e.g. ``"22-gemmascope-2-res-65k"``). Passed verbatim to the download +
  merge stages, so no string manipulation happens here.
- ``sae_config`` is the matching ``SAEConfig`` for the extract stage
  (it needs layer/hook/width/l0_size/variant to locate the weights on HF).

Currently supports ``gemma-3-4b-it`` only — the download script has its
model ID hardcoded.

Usage:
    uv run python -m interpret.sae.pipeline.prepare_sae_data
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running directly (`uv run python interpret/sae/pipeline/prepare_sae_data.py`)
# in addition to `uv run python -m interpret.sae.pipeline.prepare_sae_data`.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import requests

from interpret.download.download_neuronpedia_s3 import (
    DEFAULT_OUTPUT_DIR as S3_LABELS_DIR,
    MODEL_ID as S3_MODEL_ID,
    download_source as s3_download_source,
)
from interpret.download.merge_activations import merge_source as merge_activation_batches
from interpret.sae import SAEConfig
from interpret.sae.pipeline.extract_decoder_vectors import (
    OUTPUT_DIR as VECTORS_DIR,
    extract_and_merge,
)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PrepareSAEItem:
    """One SAE to prepare: a Neuronpedia source string and the matching SAEConfig."""

    source: str           # e.g. "22-gemmascope-2-res-65k"
    sae: SAEConfig        # used only by the extract-vectors stage


@dataclass
class PrepareSAEConfig:
    """Configuration for the SAE data preparation pipeline."""

    items: list[PrepareSAEItem] = field(default_factory=list)
    labels_dir: Path = S3_LABELS_DIR
    vectors_dir: Path = VECTORS_DIR

    # Per-stage skip flags — applied uniformly to every item
    skip_download: bool = True
    skip_activations: bool = True      # skip downloading activations/
    skip_merge_activations: bool = False  # skip the batch-merge step
    skip_vectors: bool = False           # skip the parquet extraction stage


# ── Runner ───────────────────────────────────────────────────────────────────

class PrepareSAERunner:
    """Runs the three-stage data pipeline for each item in the config."""

    def __init__(self, config: PrepareSAEConfig) -> None:
        self.config = config
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["Accept-Encoding"] = "gzip"
        return self._session

    def run(self) -> None:
        if not self.config.items:
            print("No items to prepare — PrepareSAEConfig.items is empty.")
            return
        for item in self.config.items:
            self.prepare_one(item)
        print("\nAll done.")

    def prepare_one(self, item: PrepareSAEItem) -> None:
        print(f"\n{'=' * 60}")
        print(f"SAE: {item.source}")
        print(f"{'=' * 60}")

        self._stage_download(item.source)
        self._stage_merge_activations(item.source)
        self._stage_extract_vectors(item)

    def _stage_download(self, source: str) -> None:
        if self.config.skip_download:
            print("\n[1/3] Skipping download")
            return
        print("\n[1/3] Downloading features, explanations, activations from S3")
        s3_download_source(
            self._get_session(),
            source,
            self.config.labels_dir,
            skip_activations=self.config.skip_activations,
        )

    def _stage_merge_activations(self, source: str) -> None:
        if self.config.skip_merge_activations:
            print("\n[2/3] Skipping activation merge")
            return

        print("\n[2/3] Merging activation batches")
        act_source_dir = self.config.labels_dir / "activations" / source
        if not act_source_dir.is_dir():
            print(f"  No activation batches at {act_source_dir}, skipping")
            return

        merged_path = self.config.labels_dir / f"{S3_MODEL_ID}_{source}_activations.jsonl"
        count = merge_activation_batches(act_source_dir, merged_path)
        if count == 0 or not merged_path.exists():
            print(f"  No activation records merged for {source}")
            return
        size_mb = merged_path.stat().st_size / 1024**2
        print(f"  {count} records -> {merged_path.name} ({size_mb:.0f} MB)")

    def _stage_extract_vectors(self, item: PrepareSAEItem) -> None:
        if self.config.skip_vectors:
            print("\n[3/3] Skipping decoder vector extraction")
            return
        print("\n[3/3] Extracting decoder vectors + merging with labels")
        sae = item.sae
        filename = f"w_dec_layer{sae.layer_index}_{sae.hook_type.value}_w{sae.width}.parquet"
        extract_and_merge(sae, self.config.vectors_dir / filename)


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Default run. Edit the items list to change which SAEs are prepared."""
    config = PrepareSAEConfig(
        items=[
            PrepareSAEItem(
                source="9-gemmascope-2-res-65k",
                sae=SAEConfig(layer_index=9, width="65k", device="cpu"),
            ),
        ],
        skip_download=True,      # features + explanations + activations already downloaded
        skip_activations=False,   # also skips the merge step (already merged)
    )
    PrepareSAERunner(config).run()


if __name__ == "__main__":
    main()
