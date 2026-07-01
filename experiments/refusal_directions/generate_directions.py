"""Mean-of-difference candidate refusal directions per (site, position, layer).

Adapted from `references/refusal_direction/pipeline/submodules/generate_directions.py`.
The reference batches via HF tokenizer + ``register_forward_pre_hook``; we use
the ``DirectionModel`` adapter's ``capture_means`` to capture per-layer prefill
activations at the post-instruction positions and accumulate the mean in
``float64`` to match the reference's numerical care. The adapter abstracts the
backend (Gemma fork cache vs Qwen forward hooks).
"""

from __future__ import annotations

import json

import torch

from interpret.experiments.directions_common import CaptureSite, DirectionModel
from interpret.experiments.refusal_directions.config import RefusalConfig


def generate_directions(
    model: DirectionModel,
    harmful: list[str],
    harmless: list[str],
    config: RefusalConfig,
    n_eoi: int,
) -> dict[str, torch.Tensor]:
    """Compute candidate directions per `intermediate` from mean activation diffs.

    Returns ``{intermediate: Tensor(n_eoi, n_layers, d_model)}`` and writes
    each tensor to ``config.generate_dir`` as ``mean_diffs_<intermediate>.pt``.
    Cached ``.pt`` files are reused; any missing intermediates are captured in a
    single pass per class.
    """
    out_dir = config.generate_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    directions: dict[str, torch.Tensor] = {}
    to_extract: list[str] = []
    for intermediate in config.intermediates:
        path = out_dir / f"mean_diffs_{intermediate}.pt"
        if path.exists():
            print(f"[generate_directions] reusing cached {path}")
            directions[intermediate] = torch.load(path, map_location="cpu")
        else:
            to_extract.append(intermediate)

    if to_extract:
        sites = tuple(CaptureSite.from_name(name) for name in to_extract)
        mean_h = model.capture_means(harmful, sites, n_eoi)
        mean_b = model.capture_means(harmless, sites, n_eoi)
        for intermediate in to_extract:
            diff = mean_h[intermediate] - mean_b[intermediate]
            if not torch.isfinite(diff).all():
                raise RuntimeError(
                    f"Non-finite values in mean_diffs[{intermediate}]"
                )
            torch.save(diff, out_dir / f"mean_diffs_{intermediate}.pt")
            directions[intermediate] = diff

    metadata_path = out_dir / "metadata.json"
    metadata = {
        "intermediates": list(config.intermediates),
        "n_train": config.n_train,
        "n_eoi": n_eoi,
        "n_layers": config.n_layers,
        "d_model": config.d_model,
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return directions
