"""Neuronpedia label loader for SAE features.

Reads `(index, label)` columns from the decoder-vector parquet files
under `resources/sae_vectors/`. Used by both `correlation_map` and
`top_features` to enrich feature reports with human-readable labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_feature_labels(
    sae_vectors_dir: Path | str,
    layer: int,
    width: str,
) -> dict[int, str]:
    """Load Neuronpedia labels for one (layer, width) SAE.

    Looks for parquet files in two naming conventions:
      * Current: `w_dec_layer{layer}_resid_post_w{width}.parquet`
        (where `width` is "16k", "65k", etc.)
      * Legacy: same path but with the numeric d_sae instead of width
        (e.g. `w16384.parquet`).

    Args:
        sae_vectors_dir: Directory containing decoder-vector parquet files.
        layer: Decoder layer index.
        width: SAE width string ("16k", "65k", ...).

    Returns:
        Dict mapping feature index -> label string.
        Empty dict if no parquet found (caller should treat missing
        labels as empty strings, not raise).
    """
    sae_vectors_dir = Path(sae_vectors_dir)
    candidates = [
        sae_vectors_dir / f"w_dec_layer{layer}_resid_post_w{width}.parquet",
    ]
    if width.endswith("k"):
        numeric = int(width[:-1]) * 1024
        candidates.append(
            sae_vectors_dir / f"w_dec_layer{layer}_resid_post_w{numeric}.parquet",
        )

    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path, columns=["index", "label"])
            return dict(zip(df["index"], df["label"].fillna("")))
    return {}


def load_all_labels(
    sae_vectors_dir: Path | str,
    layers: list[int],
    width: str,
) -> dict[int, dict[int, str]]:
    """Convenience: load labels for many layers at once."""
    return {layer: load_feature_labels(sae_vectors_dir, layer, width) for layer in layers}
