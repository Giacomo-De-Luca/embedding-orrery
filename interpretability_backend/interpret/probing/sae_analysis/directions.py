"""Shared loader for probe direction `.npz` files.

Single-split probe runs write `L{layer}_{intermediate}_{name}.npz`; k-fold
runs write `..._fold_{i}.npz` per fold instead. Every analysis that reads
probe coefficients (top_features, feature_sweep's lasso ranking) resolves
them through this loader so both layouts work everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_direction_coef(
    directions_dir: Path,
    layer: int,
    intermediate: str,
    source_probe: str,
) -> np.ndarray | None:
    """Load a probe's coef for one (layer, intermediate), fold-aware.

    When only fold files exist, the per-fold coefs are averaged (signed
    mean — folds share the class ordering, so the mean keeps meaningful
    magnitude AND sign). Returns None (with a message) when nothing is
    found or fold shapes disagree.
    """
    directions_dir = Path(directions_dir)
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
