"""Verify that steering ops actually mutate the residual stream.

Patches `apply_steering` to log (h_in, h_out) snapshots from inside the
SAE hook, runs a single forward pass, then prints diagnostics:

- L2 norm of h_in
- L2 norm of (h_out - h_in)
- cosine similarity between (h_out - h_in) and the steering direction
- max abs delta

If steering is correctly applied, the delta norm should be close to
``strength * ||v_unit||`` and the delta should align with the steering
vector direction.
"""

import torch

import interpret.sae.hook_manager as hm
from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.sae import HookManager, SAEConfig, SteeringMode, SteeringOp

LAYER = 29
FEATURE = 9217
STRENGTH = 500.0


def main() -> None:
    wrapper = GemmaPytorchInference("google/gemma-3-4b-it")

    manager = HookManager()
    manager.add_sae(SAEConfig(layer_index=LAYER, device="mps"))
    manager.add_steering(
        SteeringOp(
            layer_index=LAYER,
            mode=SteeringMode.ADDITIVE,
            feature_index=FEATURE,
            strength=STRENGTH,
            normalise=True,
        )
    )

    log: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    original = hm.apply_steering

    def logging_apply_steering(hidden_states, ops, strength_multiplier=1.0):
        out = original(hidden_states, ops, strength_multiplier)
        log.append(
            (
                hidden_states.detach().float().cpu().clone(),
                out.detach().float().cpu().clone(),
                ops[0].v.detach().float().cpu().clone(),
            )
        )
        return out

    hm.apply_steering = logging_apply_steering

    try:
        with manager.session(wrapper.model.model.layers):
            wrapper.generate(
                "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
                output_len=2,
            )
    finally:
        hm.apply_steering = original

    print(f"Captured {len(log)} steering invocations\n")
    if not log:
        print("ERROR: apply_steering was never called.")
        return

    h_in, h_out, v = log[0]
    print(f"h_in.shape  = {tuple(h_in.shape)}")
    print(f"h_out.shape = {tuple(h_out.shape)}")
    print(f"v.shape     = {tuple(v.shape)}")
    print()
    print(f"||h_in||             = {h_in.norm().item():.3f}")
    print(f"||h_out||            = {h_out.norm().item():.3f}")
    delta = h_out - h_in
    print(f"||h_out - h_in||     = {delta.norm().item():.3f}")
    print(f"max |h_out - h_in|   = {delta.abs().max().item():.4f}")
    print()
    print(f"||v||                = {v.norm().item():.4f}  (expected ~1 after normalise)")
    print(f"expected delta norm  = strength * sqrt(seq_len) = "
          f"{STRENGTH * (h_in.shape[1] ** 0.5):.3f}")
    print()

    # Per-token: each row of delta should equal strength * v
    if delta.abs().sum() > 0:
        per_token = delta[0, 0]
        cos = torch.nn.functional.cosine_similarity(
            per_token.unsqueeze(0), v.unsqueeze(0)
        ).item()
        print(f"cosine(delta_token0, v) = {cos:.4f}  (expected ~1.0)")
        print(f"||delta_token0|| / strength = {per_token.norm().item() / STRENGTH:.4f}  "
              "(expected ~1.0)")
    else:
        print("ERROR: delta is all zeros — steering had no effect on the tensor.")


if __name__ == "__main__":
    main()
