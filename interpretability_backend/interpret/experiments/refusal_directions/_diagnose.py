"""Diagnostic: sanity-check that direction ablation behaves correctly.

Loads the pre-computed mean diffs, picks one direction, and measures the KL
divergence on a few harmless prompts as we add ablation sites one at a time.
If KL grows roughly linearly with sites, the math is fine and the model is
just sensitive. If KL is already huge with a single site, there's a bug.

Run: ``uv run python -m interpret.experiments.refusal_directions._diagnose``.
"""

from __future__ import annotations

import torch

from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.experiments.refusal_directions import data
from interpret.experiments.refusal_directions.config import RefusalConfig
from interpret.experiments.refusal_directions.select_direction import (
    _kl_div,
    _last_position_logits,
    _make_manager,
    _refusal_score,
)
from interpret.sae import HookType, SteeringMode, SteeringOp


def _ablation_ops_subset(
    direction: torch.Tensor,
    layers: list[int],
    hook_types: list[HookType],
) -> list[SteeringOp]:
    return [
        SteeringOp(
            layer_index=l,
            mode=SteeringMode.ABLATION,
            vector=direction.detach().clone(),
            strength=0.0,
            hook_type=ht,
        )
        for l in layers
        for ht in hook_types
    ]


def main() -> None:
    cfg = RefusalConfig()
    print(f"loading model: {cfg.model_name}")
    wrapper = GemmaPytorchInference(cfg.model_name)

    candidates = torch.load(
        cfg.generate_dir / "mean_diffs_pre_attn.pt", map_location="cpu"
    )
    print(f"candidate tensor shape: {tuple(candidates.shape)}")

    # Pick the candidate the sweep flagged as "best so far" (lowest KL):
    # pos=-3, layer=14 had KL=3.54 in the full run.
    pos_idx = candidates.shape[0] - 3  # -3 from end
    layer_idx = 14
    direction = candidates[pos_idx, layer_idx].to(torch.float32)
    print(
        f"chosen candidate: pos=-3 layer={layer_idx}, "
        f"|v|={direction.norm().item():.4f}"
    )

    harmless = data.instructions_only(
        data.sample(
            data.load_split("harmless", "val", cfg.splits_dir),
            8,
            cfg.seed,
        )
    )

    # Baseline last-position logits.
    print("\ncomputing baseline logits …")
    baseline_logits = []
    for p in harmless:
        baseline_logits.append(_last_position_logits(wrapper, p))

    # Plain RESID_POST ablations at increasing layer counts.
    print("\nKL vs. number of ablation sites (RESID_POST only):")
    print(f"  {'layers ablated':>20s} | {'mean KL':>10s} | {'median KL':>10s}")
    print(f"  {'-' * 20} | {'-' * 10} | {'-' * 10}")
    for n in [1, 4, 8, 16, 34]:
        ops = _ablation_ops_subset(
            direction, list(range(n)), [HookType.RESID_POST]
        )
        manager = _make_manager(ops)
        with manager.session(wrapper.model.model.layers):
            kls = []
            for p, base in zip(harmless, baseline_logits):
                ablated = _last_position_logits(wrapper, p)
                kls.append(_kl_div(base, ablated))
        kls_t = torch.tensor(kls)
        print(
            f"  {n:>20d} | {kls_t.mean().item():>10.4f} | "
            f"{kls_t.median().item():>10.4f}"
        )

    # Three-site, all 34 layers (the actual sweep configuration).
    print("\nKL with three-site ablation at all 34 layers:")
    ops = _ablation_ops_subset(
        direction,
        list(range(34)),
        [HookType.RESID_POST, HookType.ATTN_OUT, HookType.MLP_OUT],
    )
    manager = _make_manager(ops)
    with manager.session(wrapper.model.model.layers):
        kls = []
        for p, base in zip(harmless, baseline_logits):
            ablated = _last_position_logits(wrapper, p)
            kls.append(_kl_div(base, ablated))
    kls_t = torch.tensor(kls)
    print(
        f"  three-site, 34 layers | mean KL={kls_t.mean().item():.4f}, "
        f"median KL={kls_t.median().item():.4f}"
    )


if __name__ == "__main__":
    main()
