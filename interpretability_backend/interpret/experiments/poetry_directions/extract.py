"""Per-(intermediate, position, layer) mean-diff extraction.

Same residual-stream extraction logic as
``interpret.experiments.refusal_directions.generate_directions._accumulate_mean``: forward
pass each prompt through the chat-templated chat-formatter, capture per-layer
prefill activations at the EOI positions, accumulate in fp64 on CPU, then
subtract per-class means.

Output tensor shape ``(n_eoi, n_layers, d_model)`` matches the refusal
pipeline's ``mean_diffs.pt`` so the same notebook indexing (``[pos_idx, layer]``)
works without modification.

Multi-intermediate optimisation: ``cache_activations`` captures any subset of
``{pre_attn, post_attn, mlp_out, post_mlp}`` in a single forward pass, so we
batch all requested intermediates into one accumulation loop. Going from one
to four intermediates is therefore free in compute.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import torch
from tqdm import tqdm

from interpret.experiments.poetry_directions.config import PoetryConfig
from interpret.experiments.poetry_directions.data import load_classes_for_experiment
from interpret.experiments.refusal_directions.tokens import format_chat


def _accumulate_means(
    wrapper,
    instructions: list[str],
    intermediates: Sequence[str],
    n_layers: int,
    d_model: int,
    n_eoi: int,
    desc: str,
) -> dict[str, torch.Tensor]:
    """Average post-instruction activations over `instructions` for each intermediate.

    All intermediates are captured in a single forward pass per prompt
    (``cache_activations`` supports multi-intermediate capture natively).
    Returns ``{intermediate: Tensor(n_eoi, n_layers, d_model)}`` in fp64 on CPU.
    """
    layers = set(range(n_layers))
    intermediates_set = set(intermediates)
    n_samples = len(instructions)
    if n_samples == 0:
        raise ValueError(f"empty class for {desc!r}")
    if not intermediates_set:
        raise ValueError(f"no intermediates requested for {desc!r}")

    accumulators: dict[str, torch.Tensor] = {
        name: torch.zeros((n_eoi, n_layers, d_model), dtype=torch.float64, device="cpu")
        for name in intermediates_set
    }

    with wrapper.cache_activations(
        layers=layers, intermediates=intermediates_set, prefill=True
    ) as get_cache:
        for instruction in tqdm(instructions, desc=desc):
            wrapper.reset_prefill_cache()
            wrapper.generate_from_template(
                format_chat(wrapper, instruction), output_len=1
            )
            cache = get_cache()
            prefill = cache["prefill"]
            for layer_idx in range(n_layers):
                layer_cache = prefill[layer_idx]
                for name in intermediates_set:
                    acts = layer_cache[name]  # (1, seq_len, d_model)
                    tail = acts[0, -n_eoi:, :].to(torch.float64).cpu()
                    accumulators[name][:, layer_idx, :] += tail / n_samples

    return accumulators


def extract_direction(
    wrapper,
    cfg: PoetryConfig,
    n_eoi: int,
) -> dict[str, torch.Tensor]:
    """Compute per-intermediate `mean(class_a) - mean(class_b)` and save.

    Returns ``{intermediate: Tensor(n_eoi, n_layers, d_model)}`` and writes
    each tensor to ``cfg.extract_dir`` as ``mean_diffs_<intermediate>.pt``.
    Idempotent: cached `.pt` files are reused on subsequent calls. When some
    intermediates are cached and some aren't, the missing ones are extracted
    in a single forward-pass batch (no redundant per-intermediate passes).
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
        means_a = _accumulate_means(
            wrapper, class_a, to_extract, cfg.n_layers, cfg.d_model, n_eoi,
            desc=f"{cfg.name}/A {to_extract}",
        )
        means_b = _accumulate_means(
            wrapper, class_b, to_extract, cfg.n_layers, cfg.d_model, n_eoi,
            desc=f"{cfg.name}/B {to_extract}",
        )
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
