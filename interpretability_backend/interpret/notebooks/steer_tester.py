"""Quick visual smoke test for SAE additive steering on Gemma3-4b-it.

Runs baseline + several additive-steering generations on a chosen Neuronpedia
feature at increasing strengths. Useful for eyeballing whether the steering
pipeline still works end-to-end after refactors.

Gemma3's post-norm architecture pushes per-token residual norms into the
thousands, so steering strengths are on the same scale (`strength=1600` is
strong; above ~2000 the model breaks down).

Run with:

    uv run python -m interpret.notebooks.steer_tester --prompt "What is your favorite job?" --feature 3289 --layer 9 --site resid_post
"""

import argparse

from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.sae import HookManager, HookType, SAEConfig, SteeringMode, SteeringOp

STRENGTHS = [0.0, 800.0, 1200.0, 1600.0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAE additive-steering smoke test on Gemma3-4b-it.",
    )
    parser.add_argument("--prompt", default="What is your favorite job?")
    parser.add_argument("--feature", type=int, default=3289,
                        help="Neuronpedia feature index (default 3289 = poetry).")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--site", default="resid_post",
                        choices=[h.value for h in HookType],
                        help="Decoder-layer site to steer at.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_len", type=int, default=200)
    args = parser.parse_args()
    site = HookType(args.site)

    wrapper = GemmaPytorchInference("google/gemma-3-4b-it")

    manager = HookManager()
    manager.add_sae(SAEConfig(layer_index=args.layer, hook_type=site, device="mps"))

    for strength in STRENGTHS:
        manager.clear_steering()
        if strength != 0.0:
            manager.add_steering(
                SteeringOp(
                    layer_index=args.layer,
                    mode=SteeringMode.ADDITIVE,
                    feature_index=args.feature,
                    strength=strength,
                    normalise=False,
                    hook_type=site,
                )
            )

        with manager.session(wrapper.model.model.layers):
            output = wrapper.generate(args.prompt, output_len=args.output_len, temperature=args.temperature)

        label = "BASELINE" if strength == 0.0 else f"STRENGTH={strength:g}"
        print(f"\n===== {label} =====")
        print(output.strip())


if __name__ == "__main__":
    main()
