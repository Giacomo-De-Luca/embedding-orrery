"""Per-(intermediate, position, layer) mean-diff extraction.

Same residual-stream extraction as the refusal pipeline's
``generate_directions``: for each class, capture per-layer prefill activations
at the EOI positions, accumulate in fp64 on CPU, then subtract per-class means.
Capture goes through the ``DirectionModel`` adapter's ``capture_means``, so the
same logic runs on Gemma or Qwen.

Output tensor shape ``(n_eoi, n_layers, d_model)`` matches the refusal
pipeline's ``mean_diffs.pt`` so the same notebook indexing (``[pos_idx, layer]``)
works without modification.
"""

from __future__ import annotations

import json

import torch

from interpret.experiments.directions_common import CaptureSite, DirectionModel
from interpret.experiments.poetry_directions.config import PoetryConfig
from interpret.experiments.poetry_directions.data import load_classes_for_experiment


def extract_direction(
    model: DirectionModel,
    cfg: PoetryConfig,
    n_eoi: int,
) -> dict[str, torch.Tensor]:
    """Compute per-intermediate `mean(class_a) - mean(class_b)` and save.

    Returns ``{intermediate: Tensor(n_eoi, n_layers, d_model)}`` and writes
    each tensor to ``cfg.extract_dir`` as ``mean_diffs_<intermediate>.pt``.
    Idempotent: cached `.pt` files are reused on subsequent calls. When some
    intermediates are cached and some aren't, the missing ones are captured in a
    single forward-pass batch per class (no redundant per-intermediate passes).
    """
    out_dir = cfg.extract_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    class_a, class_b = load_classes_for_experiment(
        cfg.name,
        cfg.experiment_spec,
        poems_csv=cfg.poems_csv,
        prompts_csv=cfg.prompts_csv,
        max_per_class=cfg.max_per_class,
        seed=cfg.seed,
    )
    print(
        f"[extract:{cfg.name}] "
        f"class_a={len(class_a)} class_b={len(class_b)} "
        f"intermediates={list(cfg.intermediates)}"
    )

    directions: dict[str, torch.Tensor] = {}
    to_extract: list[str] = []
    for intermediate in cfg.intermediates:
        path = out_dir / f"mean_diffs_{intermediate}.pt"
        if path.exists():
            print(f"[extract:{cfg.name}] reusing cached {path}")
            directions[intermediate] = torch.load(path, map_location="cpu")
        else:
            to_extract.append(intermediate)

    if to_extract:
        sites = tuple(CaptureSite.from_name(name) for name in to_extract)
        means_a = model.capture_means(class_a, sites, n_eoi)
        means_b = model.capture_means(class_b, sites, n_eoi)
        for intermediate in to_extract:
            diff = means_a[intermediate] - means_b[intermediate]
            if not torch.isfinite(diff).all():
                raise RuntimeError(
                    f"Non-finite values in mean_diffs[{intermediate}] for {cfg.name}"
                )
            path = out_dir / f"mean_diffs_{intermediate}.pt"
            torch.save(diff, path)
            directions[intermediate] = diff

    metadata = {
        "name": cfg.name,
        "intermediates": list(cfg.intermediates),
        "n_a": len(class_a),
        "n_b": len(class_b),
        "n_eoi": n_eoi,
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    return directions
