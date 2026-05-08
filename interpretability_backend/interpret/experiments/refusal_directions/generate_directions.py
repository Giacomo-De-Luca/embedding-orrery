"""Mean-of-difference candidate refusal directions per (intermediate, position, layer).

Adapted from `references/refusal_direction/pipeline/submodules/generate_directions.py`.
The reference batches via HF tokenizer + ``register_forward_pre_hook``; we use
`GemmaPytorchInference.cache_activations` to capture per-layer prefill
activations at the post-instruction positions and accumulate the mean in
``float64`` to match the reference's numerical care.
"""

from __future__ import annotations

import json

import torch
from tqdm import tqdm

from interpret.experiments.refusal_directions.config import RefusalConfig
from interpret.experiments.refusal_directions.tokens import format_chat


def _accumulate_mean(
    wrapper,
    instructions: list[str],
    intermediate: str,
    n_layers: int,
    d_model: int,
    n_eoi: int,
) -> torch.Tensor:
    """Average post-instruction activations over a list of prompts.

    Returns a tensor of shape ``(n_eoi, n_layers, d_model)`` in fp64 on CPU.
    """
    layers = set(range(n_layers))
    intermediates = {intermediate}
    n_samples = len(instructions)

    accumulator = torch.zeros(
        (n_eoi, n_layers, d_model), dtype=torch.float64, device="cpu"
    )

    with wrapper.cache_activations(
        layers=layers, intermediates=intermediates, prefill=True
    ) as get_cache:
        for instruction in tqdm(instructions, desc=f"prefill {intermediate}"):
            wrapper.reset_prefill_cache()
            wrapper.generate_from_template(
                format_chat(wrapper, instruction), output_len=1
            )
            cache = get_cache()
            prefill = cache["prefill"]
            for layer_idx in range(n_layers):
                acts = prefill[layer_idx][intermediate]  # (1, seq_len, d_model)
                tail = acts[0, -n_eoi:, :].to(torch.float64).cpu()
                accumulator[:, layer_idx, :] += tail / n_samples

    return accumulator


def generate_directions(
    wrapper,
    harmful: list[str],
    harmless: list[str],
    config: RefusalConfig,
    n_eoi: int,
) -> dict[str, torch.Tensor]:
    """Compute candidate directions per `intermediate` from mean activation diffs.

    Returns ``{intermediate: Tensor(n_eoi, n_layers, d_model)}`` and writes
    each tensor to ``config.generate_dir`` as ``mean_diffs_<intermediate>.pt``.
    """
    out_dir = config.generate_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    directions: dict[str, torch.Tensor] = {}
    for intermediate in config.intermediates:
        path = out_dir / f"mean_diffs_{intermediate}.pt"
        if path.exists():
            print(f"[generate_directions] reusing cached {path}")
            directions[intermediate] = torch.load(path, map_location="cpu")
            continue

        mean_h = _accumulate_mean(
            wrapper, harmful, intermediate, config.n_layers, config.d_model, n_eoi,
        )
        mean_b = _accumulate_mean(
            wrapper, harmless, intermediate, config.n_layers, config.d_model, n_eoi,
        )
        diff = mean_h - mean_b
        if not torch.isfinite(diff).all():
            raise RuntimeError(
                f"Non-finite values in mean_diffs[{intermediate}]"
            )

        torch.save(diff, path)
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
