"""Extract SAE decoder vectors and merge with Neuronpedia feature labels.

Produces one parquet file per layer containing:
- **index**:         feature index (int32)
- **vector**:        2560-dim decoder direction, list<float32>
- **density**:       activation frequency (float32)
- **label**:         autointerpreter description (string)
- **top_logits**:    list of {token, score} structs
- **bottom_logits**: list of {token, score} structs

The decoder vectors come from the pretrained Gemma-Scope SAE weights
(google/gemma-scope-2-4b-it on HuggingFace). The labels come from
Neuronpedia S3 feature data downloaded by ``download_neuronpedia_s3.py``.

Configuration-driven (no CLI) per the project's "prefer config files over
command-line flags" rule. Edit :func:`main` or instantiate
:class:`ExtractDecoderConfig` / :class:`ExtractDecoderRunner` directly to
run a subset of layers.

Usage::

    uv run python -m interpret.sae.extract_decoder_vectors
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from interpret.sae import SAEConfig, load_sae

OUTPUT_DIR = Path("resources/sae_vectors")
LABELS_DIR = Path("resources/sae_labels/neuronpedia_gemma-3-4b-it")
DEFAULT_LAYERS = [9, 17, 22, 29]

# Parquet schema for the logits struct
_LOGIT_STRUCT = pa.struct([("token", pa.string()), ("score", pa.float32())])


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class ExtractDecoderItem:
    """One decoder-vector extraction: the SAEConfig + output filename."""

    sae: SAEConfig

    def output_filename(self) -> str:
        hook = self.sae.hook_type.value
        return (
            f"w_dec_layer{self.sae.layer_index}_{hook}_w{self.sae.width}.parquet"
        )


@dataclass
class ExtractDecoderConfig:
    """Configuration for the decoder-vector extraction stage."""

    items: list[ExtractDecoderItem] = field(default_factory=list)
    output_dir: Path = OUTPUT_DIR
    labels_dir: Path = LABELS_DIR
    skip_labels: bool = False

    @classmethod
    def for_layers(
        cls,
        layers: list[int] = None,
        *,
        width: str = "16k",
        device: str = "cpu",
        output_dir: Path = OUTPUT_DIR,
        labels_dir: Path = LABELS_DIR,
        skip_labels: bool = False,
    ) -> "ExtractDecoderConfig":
        """Build a config covering one SAE per layer with matching width."""
        layers = layers if layers is not None else DEFAULT_LAYERS
        items = [
            ExtractDecoderItem(
                sae=SAEConfig(layer_index=layer, width=width, device=device),
            )
            for layer in layers
        ]
        return cls(
            items=items,
            output_dir=output_dir,
            labels_dir=labels_dir,
            skip_labels=skip_labels,
        )


# ── Label loading ────────────────────────────────────────────────────────────

def load_feature_labels(config: SAEConfig, labels_dir: Path = LABELS_DIR) -> dict[int, dict]:
    """Load Neuronpedia feature labels from the downloaded JSONL.

    Returns a dict mapping feature index to {density, label, top_logits, bottom_logits}.
    """
    source = f"{config.layer_index}-gemmascope-2-res-{config.width}"
    jsonl_path = labels_dir / f"{config.neuronpedia_model_id}_{source}_features.jsonl"

    if not jsonl_path.exists():
        print(f"  Warning: no labels file at {jsonl_path}")
        return {}

    features: dict[int, dict] = {}
    seen: set[int] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            idx = int(entry["index"])
            if idx in seen:
                continue
            seen.add(idx)

            label = ""
            explanations = entry.get("explanations", [])
            if explanations:
                label = explanations[0].get("text", "")

            top_logits = entry.get("top_logits") or []
            bottom_logits = entry.get("bottom_logits") or []

            features[idx] = {
                "density": entry.get("density", 0.0),
                "label": label,
                "top_logits": [
                    {"token": tok, "score": float(score)}
                    for tok, score in top_logits
                ],
                "bottom_logits": [
                    {"token": tok, "score": float(score)}
                    for tok, score in bottom_logits
                ],
            }

    print(f"  Loaded {len(features)} feature labels from {jsonl_path.name}")
    return features


def extract_and_merge(
    config: SAEConfig,
    output_path: Path,
    skip_labels: bool = False,
    labels_dir: Path = LABELS_DIR,
) -> None:
    """Load SAE decoder vectors, merge with labels, write parquet."""
    sae = load_sae(config)
    w_dec = sae.w_dec.data.float().cpu()  # (d_sae, d_in)
    d_sae, d_in = w_dec.shape
    print(f"  Loaded SAE: {d_sae} features, {d_in}-dim vectors")

    labels = {} if skip_labels else load_feature_labels(config, labels_dir)

    indices = list(range(d_sae))
    vectors = [w_dec[i].tolist() for i in range(d_sae)]
    densities = [labels.get(i, {}).get("density", 0.0) for i in range(d_sae)]
    label_texts = [labels.get(i, {}).get("label", "") for i in range(d_sae)]
    top_logits = [labels.get(i, {}).get("top_logits", []) for i in range(d_sae)]
    bottom_logits = [labels.get(i, {}).get("bottom_logits", []) for i in range(d_sae)]

    table = pa.table(
        {
            "index": pa.array(indices, type=pa.int32()),
            "vector": pa.array(vectors, type=pa.list_(pa.float32())),
            "density": pa.array(densities, type=pa.float32()),
            "label": pa.array(label_texts, type=pa.string()),
            "top_logits": pa.array(top_logits, type=pa.list_(_LOGIT_STRUCT)),
            "bottom_logits": pa.array(bottom_logits, type=pa.list_(_LOGIT_STRUCT)),
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")

    size_mb = output_path.stat().st_size / 1024**2
    print(f"  Saved to {output_path} ({size_mb:.0f} MB)")


# ── Runner ───────────────────────────────────────────────────────────────────

class ExtractDecoderRunner:
    """Run decoder-vector extraction for each item in the config."""

    def __init__(self, config: ExtractDecoderConfig) -> None:
        self.config = config

    def run(self) -> None:
        if not self.config.items:
            print("No items to extract — ExtractDecoderConfig.items is empty.")
            return
        for item in self.config.items:
            print(f"\nLayer {item.sae.layer_index}:")
            extract_and_merge(
                item.sae,
                self.config.output_dir / item.output_filename(),
                skip_labels=self.config.skip_labels,
                labels_dir=self.config.labels_dir,
            )


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Default run. Edit the config below to extract different SAEs."""
    config = ExtractDecoderConfig.for_layers(DEFAULT_LAYERS, width="16k")
    ExtractDecoderRunner(config).run()


if __name__ == "__main__":
    main()
