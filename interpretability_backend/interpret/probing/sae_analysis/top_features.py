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
from interpret.probing.sae_analysis.constants import SAE_INTERMEDIATES
from interpret.probing.sae_analysis.labels import (
    load_feature_labels,
)


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
        if intermediate not in SAE_INTERMEDIATES:
            continue
        raw_coef = _load_direction_coef(
            directions_dir,
            layer,
            intermediate,
            config.source_probe,
        )
        if raw_coef is None:
            continue
        # Regression / binary logreg -> one row; multiclass -> [C, d].
        # Multiclass ranks by the strongest class's |coef| and records
        # which class that was.
        if raw_coef.ndim == 2 and raw_coef.shape[0] > 1:
            top_class = np.abs(raw_coef).argmax(axis=0)
            coef = raw_coef[top_class, np.arange(raw_coef.shape[1])]
        else:
            top_class = None
            coef = raw_coef.reshape(-1)
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
            config.sae_vectors_dir,
            layer,
            width,
        )
        order = np.argsort(np.abs(coef))[::-1][: config.top_k]
        result[layer] = [
            {
                "feature_idx": int(kept[i]),
                "coef": float(coef[i]),
                "label": labels.get(int(kept[i]), ""),
                **({"top_class": int(top_class[i])} if top_class is not None else {}),
            }
            for i in order
            if coef[i] != 0.0
        ]

    out_path = output_dir / "top_features.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    _print_summary(result)
    return result


def _load_direction_coef(
    directions_dir: Path,
    layer: int,
    intermediate: str,
    source_probe: str,
) -> np.ndarray | None:
    """Load a probe's coef for one (layer, intermediate).

    Single-split runs write `L{layer}_{intermediate}_{kind}.npz`; k-fold
    runs write `..._fold_{i}.npz` instead. When only fold files exist,
    the per-fold coefs are averaged (signed mean — folds share the class
    ordering, so the mean keeps meaningful magnitude AND sign).
    """
    stem = f"L{layer}_{intermediate}_{source_probe}"
    npz_path = directions_dir / f"{stem}.npz"
    if npz_path.exists():
        return np.asarray(np.load(str(npz_path))["coef"])

    fold_paths = sorted(directions_dir.glob(f"{stem}_fold_*.npz"))
    if not fold_paths:
        print(
            f"  layer {layer}: no directions at {npz_path} (nor {stem}_fold_*.npz), skipping",
        )
        return None
    coefs = [np.asarray(np.load(str(p))["coef"]) for p in fold_paths]
    shapes = {c.shape for c in coefs}
    if len(shapes) != 1:
        print(
            f"  layer {layer}: fold coef shapes differ ({shapes}), skipping",
        )
        return None
    return np.mean(np.stack(coefs, axis=0), axis=0)


def _print_summary(result: dict[int, list[dict]]) -> None:
    for layer, feats in sorted(result.items()):
        print(f"\nLayer {layer} — top features by |coef|:")
        for feat in feats[:10]:
            lbl = f" [{feat['label'][:55]}]" if feat["label"] else ""
            print(
                f"  F{feat['feature_idx']:>5d}  coef={feat['coef']:+.4f}{lbl}",
            )
