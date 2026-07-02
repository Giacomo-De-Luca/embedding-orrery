"""Unified SAE data pipeline: download, merge, extract, all from a single config.

Takes a ``GemmaScopeSAEConfig`` and runs up to three stages:

1. **Download** — pull features + explanations (+ optionally activations)
   from the Neuronpedia S3 bucket.
2. **Merge activations** — decompress and concatenate raw activation
   batches into a single sorted JSONL.
3. **Extract vectors** — load SAE weights from HuggingFace, extract
   decoder directions, merge with downloaded labels into a parquet file.

Everything is derived from the SAE config — model ID, source string,
directory paths, filenames — no hardcoded values.

This module does **not** import from ``backend/`` and produces files on
disk that the backend can ingest separately.

Usage::

    uv run python -m interpret.sae.pipeline.prepare_sae_data --layer 9 --width 65k
    uv run python -m interpret.sae.pipeline.prepare_sae_data --layer 9 --width 16k --with-activations
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from interpret.sae import paths as sae_paths
from interpret.sae.sae_config import GemmaScopeSAEConfig, HookType
from interpret.sae.source_ids import neuronpedia_source_id

# Type alias for progress callbacks: (stage_name, done, total)
ProgressCallback = Callable[[str, int, int], None] | None


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class SAEPipelineConfig:
    """Configuration for the full SAE data pipeline.

    Only ``sae`` is required — all paths and identifiers are derived from it.
    """

    sae: GemmaScopeSAEConfig

    # Directory overrides (default: derived from sae config via paths.py)
    labels_dir: Path | None = None
    vectors_dir: Path | None = None

    # Stage control
    skip_download: bool = False
    skip_activations: bool = True  # activations are ~336 MB per source
    skip_merge_activations: bool = False
    skip_extract: bool = False

    @property
    def resolved_labels_dir(self) -> Path:
        return self.labels_dir or sae_paths.labels_dir(self.sae)

    @property
    def resolved_vectors_dir(self) -> Path:
        return self.vectors_dir or sae_paths.vectors_dir()

    @property
    def source_id(self) -> str:
        return neuronpedia_source_id(self.sae)

    @property
    def model_id(self) -> str:
        return self.sae.neuronpedia_model_id


# ── Result ───────────────────────────────────────────────────────────────────


@dataclass
class SAEPipelineResult:
    """Output paths and status from a pipeline run."""

    features_parquet: Path | None = None
    activations_jsonl: Path | None = None
    features_jsonl: Path | None = None
    model_id: str = ""
    sae_id: str = ""  # the neuronpedia source string
    error: str | None = None


# ── Runner ───────────────────────────────────────────────────────────────────


class SAEPipelineRunner:
    """Run the download → merge → extract pipeline for one SAE config."""

    def __init__(self, config: SAEPipelineConfig) -> None:
        self.config = config
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["Accept-Encoding"] = "gzip"
        return self._session

    def run(self, progress_callback: ProgressCallback = None) -> SAEPipelineResult:
        """Run all enabled stages, returning paths to output files."""
        cfg = self.config
        result = SAEPipelineResult(
            model_id=cfg.model_id,
            sae_id=cfg.source_id,
        )

        print(f"\n{'=' * 60}")
        print(f"SAE Pipeline: {cfg.source_id}  (model: {cfg.model_id})")
        print(f"{'=' * 60}")

        try:
            self._stage_download(result, progress_callback)
            self._stage_merge_activations(result, progress_callback)
            self._stage_extract_vectors(result, progress_callback)
        except Exception as e:
            result.error = str(e)
            print(f"\nPipeline failed: {e}")

        print(f"\n{'=' * 60}")
        if result.error:
            print(f"Pipeline finished with error: {result.error}")
        else:
            print("Pipeline finished successfully.")
            if result.features_parquet:
                print(f"  Features parquet:  {result.features_parquet}")
            if result.features_jsonl:
                print(f"  Features JSONL:    {result.features_jsonl}")
            if result.activations_jsonl:
                print(f"  Activations JSONL: {result.activations_jsonl}")
        print(f"{'=' * 60}\n")

        return result

    # ── Stage 1: Download ────────────────────────────────────────────────

    def _stage_download(
        self,
        result: SAEPipelineResult,
        progress_callback: ProgressCallback,
    ) -> None:
        if self.config.skip_download:
            print("\n[1/3] Skipping download")
            # Still populate features_jsonl path if the file already exists
            features_path = sae_paths.features_jsonl_path(self.config.sae)
            if features_path.exists():
                result.features_jsonl = features_path
            return

        print("\n[1/3] Downloading features + explanations from Neuronpedia S3")

        from interpret.download.download_neuronpedia_s3 import download_source

        cfg = self.config
        output_dir = cfg.resolved_labels_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Thread per-stage download progress to the pipeline callback
        def _download_progress(stage: str, done: int, total: int) -> None:
            if progress_callback:
                progress_callback(f"download:{stage}", done, total)

        download_source(
            self._get_session(),
            cfg.source_id,
            output_dir,
            skip_activations=cfg.skip_activations,
            model_id=cfg.model_id,
            progress_callback=_download_progress,
        )

        # Record output paths
        features_path = sae_paths.features_jsonl_path(cfg.sae)
        if features_path.exists():
            result.features_jsonl = features_path

        if progress_callback:
            progress_callback("download", 1, 1)

    # ── Stage 2: Merge activations ───────────────────────────────────────

    def _stage_merge_activations(
        self,
        result: SAEPipelineResult,
        progress_callback: ProgressCallback,
    ) -> None:
        if self.config.skip_merge_activations or self.config.skip_activations:
            print("\n[2/3] Skipping activation merge")
            # Check if merged file already exists
            act_path = sae_paths.activations_jsonl_path(self.config.sae)
            if act_path.exists():
                result.activations_jsonl = act_path
            return

        print("\n[2/3] Merging activation batches")

        from interpret.download.merge_activations import merge_source

        act_source_dir = sae_paths.activation_batches_dir(self.config.sae)
        if not act_source_dir.is_dir():
            print(f"  No activation batches at {act_source_dir}, skipping")
            return

        if progress_callback:
            progress_callback("merge_activations", 0, 1)

        merged_path = sae_paths.activations_jsonl_path(self.config.sae)
        count = merge_source(act_source_dir, merged_path)

        if count == 0 or not merged_path.exists():
            print(f"  No activation records merged for {self.config.source_id}")
            return

        size_mb = merged_path.stat().st_size / 1024**2
        print(f"  {count} records -> {merged_path.name} ({size_mb:.0f} MB)")
        result.activations_jsonl = merged_path

        if progress_callback:
            progress_callback("merge_activations", 1, 1)

    # ── Stage 3: Extract decoder vectors ─────────────────────────────────

    def _stage_extract_vectors(
        self,
        result: SAEPipelineResult,
        progress_callback: ProgressCallback,
    ) -> None:
        output_path = sae_paths.vectors_parquet_path(self.config.sae)

        if self.config.skip_extract:
            print("\n[3/3] Skipping decoder vector extraction")
            # Check if parquet already exists
            if output_path.exists():
                result.features_parquet = output_path
            return

        # Skip re-extraction if the parquet already exists — avoids a costly
        # re-download + re-extract of the SAE weights.
        if output_path.exists():
            print(f"\n[3/3] Decoder vectors parquet already exists: {output_path}")
            result.features_parquet = output_path
            if progress_callback:
                progress_callback("extract_vectors", 1, 1)
            return

        print("\n[3/3] Extracting decoder vectors + merging with labels")

        from interpret.sae.extract_decoder_vectors import extract_and_merge

        if progress_callback:
            progress_callback("extract_vectors", 0, 1)

        extract_and_merge(
            self.config.sae,
            output_path,
            resolved_labels_dir=self.config.resolved_labels_dir,
        )
        result.features_parquet = output_path

        if progress_callback:
            progress_callback("extract_vectors", 1, 1)


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAE data pipeline: download → merge → extract decoder vectors",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index (e.g. 9, 17, 22, 29)",
    )
    parser.add_argument(
        "--width",
        type=str,
        default="16k",
        help="SAE width (default: 16k). Options: 16k, 65k, 262k",
    )
    parser.add_argument(
        "--hook",
        type=str,
        default="resid_post",
        help="Hook type (default: resid_post). Options: resid_post, mlp_out, attn_out",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the S3 download stage (assumes labels already exist)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip the decoder vector extraction stage",
    )
    parser.add_argument(
        "--with-activations",
        action="store_true",
        help="Also download and merge activation examples (~336 MB per source)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading SAE weights (default: cpu)",
    )
    args = parser.parse_args()

    sae_config = GemmaScopeSAEConfig(
        layer_index=args.layer,
        width=args.width,
        hook_type=HookType(args.hook),
        device=args.device,
    )
    pipeline_config = SAEPipelineConfig(
        sae=sae_config,
        skip_download=args.skip_download,
        skip_activations=not args.with_activations,
        skip_extract=args.skip_extract,
    )

    result = SAEPipelineRunner(pipeline_config).run()
    if result.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
