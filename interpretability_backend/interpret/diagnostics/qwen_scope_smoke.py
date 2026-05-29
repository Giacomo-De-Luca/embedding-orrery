"""End-to-end smoke test for Qwen-scope TopK SAE attachment.

Loads a Qwen model, attaches a Qwen-scope SAE at a chosen layer via
``HookManager``, runs a prompt through the model, and asserts:

1. captured shapes (feature_acts, reconstruction);
2. dtype matches the model's dtype;
3. exact L0 sparsity (==k per token, since TopK is hard);
4. no NaN / Inf in either captured tensor;
5. non-trivial reconstruction — cosine similarity vs. the raw residual
   stream (captured separately via ``Qwen3Inference.cache_activations``)
   is above ``--cosine-threshold``;
6. additive steering on the top-firing feature actually mutates the
   residual stream (delta L2 norm > a tolerance).

The model and the SAE ``--size`` must agree (guarded by a layer-count check).

Run with::

    uv run python -m interpret.diagnostics.qwen_scope_smoke --model Qwen/Qwen3-1.7B --size 1.7B
    uv run python -m interpret.diagnostics.qwen_scope_smoke --model Qwen/Qwen3-8B   --size 8B
    uv run python -m interpret.diagnostics.qwen_scope_smoke --model Qwen/Qwen3.5-27B --size 27B
"""

import argparse

import torch

from interpret.inference.qwen3_transformers import Qwen3Inference
from interpret.sae import (
    QWEN_SCOPE_MODELS,
    HookManager,
    HookType,
    QwenScopeSAEConfig,
    SteeringMode,
    SteeringOp,
)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.flatten().float()
    b32 = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0)).item()


def run(
    model: str,
    size: str,
    k: int,
    width: str | None,
    layer: int | None,
    dtype: str,
    prompt: str,
    cosine_threshold: float,
) -> None:
    print(f"Loading {model}...")
    wrapper = Qwen3Inference(model, dtype=dtype)
    print(f"  device={wrapper.device}, dtype={wrapper.dtype}")

    n_layers = len(wrapper.decoder_layers)
    expected_layers = QWEN_SCOPE_MODELS[size].n_layers
    assert n_layers == expected_layers, (
        f"base model has {n_layers} layers but Qwen-scope size {size!r} expects "
        f"{expected_layers} — do --model and --size match?"
    )
    if layer is None:
        layer = n_layers // 2
    assert 0 <= layer < n_layers, f"layer {layer} out of range (0..{n_layers - 1})"

    config = QwenScopeSAEConfig(
        layer_index=layer,
        model_size=size,
        k=k,
        width=width,
        device=str(wrapper.device),
        dtype=dtype,
    )
    print(f"Loading SAE: {config.repo_id} / {config.weights_filename()}")
    manager = HookManager()
    manager.add_sae(config)

    # First pass: capture SAE feature_acts + reconstruction via the hook.
    with manager.session(wrapper.decoder_layers) as store:
        wrapper.generate(prompt, output_len=1)
        rec = store.prefill(layer=layer)

    assert rec is not None, "no prefill record stored — hook never fired"
    acts = rec.feature_acts
    recon = rec.reconstruction
    assert recon is not None, "reconstruction is None — read_only path issue"

    prompt_len = acts.shape[1]
    print(f"\nprompt_len={prompt_len}")
    print(f"acts:  shape={tuple(acts.shape)} dtype={acts.dtype}")
    print(f"recon: shape={tuple(recon.shape)} dtype={recon.dtype}")

    # 1. shape checks
    assert acts.shape == (1, prompt_len, config.d_sae), (
        f"acts shape mismatch: {tuple(acts.shape)} vs (1, {prompt_len}, {config.d_sae})"
    )
    assert recon.shape == (1, prompt_len, config.d_in), (
        f"recon shape mismatch: {tuple(recon.shape)} vs (1, {prompt_len}, {config.d_in})"
    )

    # 2. dtype
    assert acts.dtype == wrapper.dtype, f"acts dtype {acts.dtype} != model dtype {wrapper.dtype}"
    assert recon.dtype == wrapper.dtype

    # 3. L0 sparsity — hard TopK, so per-token L0 must equal k exactly.
    per_token_l0 = (acts != 0).sum(dim=-1)  # (1, prompt_len)
    print(
        f"L0 per token: min={per_token_l0.min().item()}, "
        f"max={per_token_l0.max().item()}, mean={per_token_l0.float().mean().item():.2f}"
    )
    assert (per_token_l0 == k).all(), (
        f"L0 sparsity violation — expected exactly {k} active features per token, "
        f"got min={per_token_l0.min().item()} max={per_token_l0.max().item()}"
    )

    # 4. finite
    assert torch.isfinite(acts).all(), "acts contains NaN/Inf"
    assert torch.isfinite(recon).all(), "recon contains NaN/Inf"

    # 5. non-trivial reconstruction — capture the raw residual via the qwen
    # wrapper (no SAE in the loop) and compare.
    with wrapper.cache_activations(layers={layer}, hook_types={HookType.RESID_POST}) as get_cache:
        wrapper.generate(prompt, output_len=1)
        cache = get_cache()
    raw_resid = cache[layer][HookType.RESID_POST]
    print(f"raw_resid shape={tuple(raw_resid.shape)} dtype={raw_resid.dtype}")

    assert raw_resid.shape == recon.shape, (
        f"raw_resid {tuple(raw_resid.shape)} vs recon {tuple(recon.shape)} shape "
        "mismatch — the SAE hook may have captured a different layer"
    )
    cos = _cosine(raw_resid, recon)
    print(f"cosine(raw_resid, recon) = {cos:.4f}")
    assert cos > cosine_threshold, (
        f"reconstruction cosine {cos:.4f} <= {cosine_threshold} — SAE is "
        "reconstructing something very different from the residual stream"
    )

    # 6. steering smoke — pick the top-firing feature on the last token, apply
    # additive steering, confirm the residual changes.
    last_token_acts = acts[0, -1]
    top_feat = int(torch.argmax(last_token_acts).item())
    top_val = float(last_token_acts[top_feat].item())
    print(f"\nSteering smoke: feature {top_feat} (act={top_val:.3f}) += 10.0 * w_dec[{top_feat}]")

    manager.clear_steering()
    manager.add_steering(
        SteeringOp(
            layer_index=layer,
            mode=SteeringMode.ADDITIVE,
            feature_index=top_feat,
            strength=10.0,
            normalise=True,
        )
    )

    with manager.session(wrapper.decoder_layers) as store:
        wrapper.generate(prompt, output_len=1)
        rec_steered = store.prefill(layer=layer)

    assert rec_steered is not None
    delta = (rec_steered.reconstruction - recon).float().norm().item()
    baseline = recon.float().norm().item()
    print(
        f"||recon_steered - recon|| = {delta:.3f}   ||recon|| = {baseline:.3f}   "
        f"relative = {delta / baseline:.4f}"
    )
    assert delta > 0.1, (
        f"steering had no measurable effect on the residual stream (delta L2 = {delta:.6f})"
    )

    print("\nAll checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-scope TopK SAE end-to-end smoke test")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HuggingFace model id")
    parser.add_argument(
        "--size",
        default="1.7B",
        choices=sorted(QWEN_SCOPE_MODELS),
        help="Qwen-scope SAE size key (must match the base model)",
    )
    parser.add_argument("--k", type=int, default=50, help="SAE L0 (50 or 100)")
    parser.add_argument("--width", default=None, help="SAE width (default: the model's only width)")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="layer to probe (default: n_layers // 2)",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--prompt", default="The colour of the sky is")
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.5,
        help="minimum acceptable cosine(raw_resid, recon)",
    )
    args = parser.parse_args()

    run(
        model=args.model,
        size=args.size,
        k=args.k,
        width=args.width,
        layer=args.layer,
        dtype=args.dtype,
        prompt=args.prompt,
        cosine_threshold=args.cosine_threshold,
    )


if __name__ == "__main__":
    main()
