"""Top-K SAE features ranked by |coef| of a trained linear probe.

Reads the `.npz` direction files written by `sklearn_probes` (when
`save_directions=True`), maps filtered column indices back to original
SAE feature indices via `kept_by_layer`, looks up Neuronpedia labels,
and writes a JSON ranking per layer.

This is the HypotheSAE-style "what features did the probe rely on"
analysis. It runs after the linear probe has been fit + saved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.sae_analysis import (
    TopFeaturesConfig,
)
from interpret.probing.sae_analysis.labels import (
    load_feature_labels,
)

_SAE_INTERMEDIATE = "sae_feat"


def run_top_features(
    sae_dataset: ActivationDataset,
    directions_dir: Path,
    config: TopFeaturesConfig,
    output_dir: Path,
    *,
    width: str,
) -> dict[int, list[dict]]:
    """Surface the top-K features by |coef| from a saved linear probe.

    Args:
        sae_dataset: SAE-encoded dataset whose `metadata["kept_by_layer"]`
            maps filtered column index -> original SAE feature index for
            each layer.
        directions_dir: Path to a probe's `directions/` folder containing
            `L{layer}_{intermediate}_{kind}.npz` files (typically
            `<output>/probes/<source_probe>/directions/`).
        config: top_k + sae_vectors_dir for label lookup.
        output_dir: Where `top_features.json` is written.
        width: SAE width string (e.g. "16k") for label lookup.

    Returns:
        Dict layer -> list of `{feature_idx, coef, label}` dicts (descending |coef|).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directions_dir = Path(directions_dir)

    kept_by_layer = sae_dataset.metadata.get("kept_by_layer", {})

    result: dict[int, list[dict]] = {}
    for layer, intermediate in sorted(sae_dataset.layer_intermediate_keys()):
        if intermediate != _SAE_INTERMEDIATE:
            continue
        npz_path = (
            directions_dir
            / f"L{layer}_{intermediate}_{config.source_probe}.npz"
        )
        if not npz_path.exists():
            print(
                f"  layer {layer}: no directions at {npz_path}, skipping",
            )
            continue
        data = np.load(str(npz_path))
        raw_coef = data["coef"]
        # Single-output regression -> 1-D; classification -> 2-D.
        coef = (
            raw_coef.reshape(-1)
            if raw_coef.ndim <= 2
            else raw_coef.ravel()
        )
        kept = np.asarray(kept_by_layer.get(layer, []), dtype=np.int64)
        if kept.size == 0:
            kept = np.arange(coef.shape[0], dtype=np.int64)
        if coef.shape[0] != kept.shape[0]:
            print(
                f"  layer {layer}: coef shape {coef.shape[0]} != kept "
                f"{kept.shape[0]}, skipping label mapping",
            )
            continue

        labels = load_feature_labels(
            config.sae_vectors_dir, layer, width,
        )
        order = np.argsort(np.abs(coef))[::-1][: config.top_k]
        result[layer] = [
            {
                "feature_idx": int(kept[i]),
                "coef": float(coef[i]),
                "label": labels.get(int(kept[i]), ""),
            }
            for i in order
            if coef[i] != 0.0
        ]

    out_path = output_dir / "top_features.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    _print_summary(result)
    return result


def _print_summary(result: dict[int, list[dict]]) -> None:
    for layer, feats in sorted(result.items()):
        print(f"\nLayer {layer} — top features by |coef|:")
        for feat in feats[:10]:
            lbl = f" [{feat['label'][:55]}]" if feat["label"] else ""
            print(
                f"  F{feat['feature_idx']:>5d}  coef={feat['coef']:+.4f}{lbl}",
            )
