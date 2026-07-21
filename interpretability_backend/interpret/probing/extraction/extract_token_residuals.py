"""Token-level residual extraction: keep every token position, pool later.

Stage 1 of the two-stage SAE-probing pipeline. One forward pass per sample
captures all requested layers × sites; the output keeps the full sequence
so downstream stages (`extract_sae_pooled`, `extract_residual_pooled`)
choose their own pooling — the token-level extension the pooled extractors'
docstrings have always pointed at.

Storage is RAGGED: per `(layer, canonical_site)` key one
``Tensor[total_tokens, hidden]`` (samples concatenated along dim 0), with a
shared ``metadata["token_offsets"]`` list (length N+1) delimiting each
sample's rows. This dataset is NOT probeable (dim 0 is tokens, not
samples) — the orchestrator skips it via the config's ``skip_probes``.

Family reconciliation mirrors ``interpret.inference.residual_norm_profiler``:
Gemma uses configure_cache/reset/get_cached_activations with string
intermediates and returns CPU tensors; Qwen uses the ``cache_activations``
context manager with ``HookType`` keys and returns on-device bf16 tensors
(moved to CPU per sample here, before the next forward).
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.token_extraction import (
    TokenLevelExtractionConfig,
)
from interpret.sae.sae_config import HookType

# Canonical site name -> Gemma fork cache intermediate string.
# "resid_post" (residual stream after the full layer) is the fork's
# "post_mlp"; "mlp_out"/"attn_out" are the raw pre-residual-add sub-block
# outputs (the sites gemma-scope mlp/attn SAEs are trained on).
GEMMA_SITE_MAP: dict[str, str] = {
    "resid_post": "post_mlp",
    "mlp_out": "mlp_out",
    "attn_out": "attn_out",
}

QWEN_SITE_MAP: dict[str, HookType] = {
    "resid_post": HookType.RESID_POST,
    "mlp_out": HookType.MLP_OUT,
    "attn_out": HookType.ATTN_OUT,
}

_STORAGE_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def extract_token_residuals(
    config: TokenLevelExtractionConfig,
    samples: list[str],
    wrapper,
) -> ActivationDataset:
    """Run one prefill per sample and collect ragged token-level residuals.

    Args:
        config: Layers, canonical sites, family, storage dtype.
        samples: Ordered prompt strings; order defines `sample_ids` and the
            offset table.
        wrapper: Loaded `GemmaPytorchInference` or `Qwen3Inference`.

    Returns:
        `ActivationDataset` with `(layer, site)` keys of shape
        `[total_tokens, hidden]`, `metadata["token_offsets"]` of length
        N+1, and `metadata["token_level"] = True`.
    """
    if not samples:
        raise ValueError("samples is empty.")
    dtype = _STORAGE_DTYPES[config.storage_dtype]

    if config.family == "gemma":
        collected, lengths = _collect_gemma(config, samples, wrapper, dtype)
    else:
        collected, lengths = _collect_qwen(config, samples, wrapper, dtype)

    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)

    activations = {key: torch.cat(chunks, dim=0) for key, chunks in collected.items()}
    hidden_size = next(iter(activations.values())).shape[1]
    metadata = {
        "extraction_type": "token_residuals",
        "token_level": True,
        "family": config.family,
        "checkpoint": config.checkpoint,
        "layers": list(config.layers),
        "sites": list(config.sites),
        "intermediates": list(config.sites),
        "token_offsets": offsets,
        "n_tokens": offsets[-1],
        "prepends_bos": bool(wrapper.prepends_bos),
        "hidden_size": hidden_size,
        "storage_dtype": config.storage_dtype,
        "num_samples": len(samples),
    }
    return ActivationDataset(
        activations=activations,
        targets=torch.empty(0),
        sample_ids=list(samples),
        metadata=metadata,
    )


def _collect_gemma(
    config: TokenLevelExtractionConfig,
    samples: list[str],
    wrapper,
    dtype: torch.dtype,
) -> tuple[dict[tuple[int, str], list[torch.Tensor]], list[int]]:
    """Gemma path: configure once, reset + prefill + read per sample."""
    if not hasattr(wrapper, "configure_cache"):
        raise TypeError(
            "family='gemma' expects a GemmaPytorchInference-style wrapper "
            f"with configure_cache(); got {type(wrapper).__name__}",
        )
    intermediates = {GEMMA_SITE_MAP[s] for s in config.sites}
    collected: dict[tuple[int, str], list[torch.Tensor]] = {
        (layer, site): [] for layer in config.layers for site in config.sites
    }
    lengths: list[int] = []

    try:
        wrapper.configure_cache(
            layers=set(config.layers),
            intermediates=intermediates,
            prefill=True,
            last=False,
        )
        for sample in tqdm(samples, desc=f"{config.name} (gemma tokens)"):
            wrapper.reset_prefill_cache()
            # generate_from_template = raw pass-through (BOS + text tokens
            # only). generate() would wrap the sample in the chat template,
            # whose constant tokens would pollute downstream max pooling
            # and break symmetry with the raw Qwen path.
            wrapper.generate_from_template(sample, output_len=1)
            phase_cache = wrapper.get_cached_activations().get("prefill", {})
            lengths.append(
                _collect_one(config, phase_cache, collected, dtype, _gemma_site_tensor),
            )
    finally:
        wrapper.clear_cache()
    return collected, lengths


def _collect_qwen(
    config: TokenLevelExtractionConfig,
    samples: list[str],
    wrapper,
    dtype: torch.dtype,
) -> tuple[dict[tuple[int, str], list[torch.Tensor]], list[int]]:
    """Qwen path: fresh cache context per sample; tensors moved to CPU
    immediately so device memory doesn't accumulate across samples."""
    if hasattr(wrapper, "configure_cache"):
        raise TypeError(
            "family='qwen' expects a Qwen3Inference-style wrapper "
            f"(no configure_cache); got {type(wrapper).__name__}",
        )
    hook_types = {QWEN_SITE_MAP[s] for s in config.sites}
    collected: dict[tuple[int, str], list[torch.Tensor]] = {
        (layer, site): [] for layer in config.layers for site in config.sites
    }
    lengths: list[int] = []

    for sample in tqdm(samples, desc=f"{config.name} (qwen tokens)"):
        with wrapper.cache_activations(
            layers=set(config.layers),
            hook_types=hook_types,
            prefill_only=True,
        ) as get_cache:
            wrapper.generate_from_template(sample, output_len=1, add_bos=True)
            cache = get_cache()
        lengths.append(
            _collect_one(config, cache, collected, dtype, _qwen_site_tensor),
        )
    return collected, lengths


def _collect_one(
    config: TokenLevelExtractionConfig,
    cache: dict,
    collected: dict[tuple[int, str], list[torch.Tensor]],
    dtype: torch.dtype,
    site_tensor,
) -> int:
    """Append one sample's `[T, hidden]` slice per key; return T."""
    sample_len: int | None = None
    for layer in config.layers:
        layer_cache = cache.get(layer)
        if layer_cache is None:
            raise RuntimeError(
                f"No cached activations for layer {layer}. Cache layers: {sorted(cache.keys())}",
            )
        for site in config.sites:
            act = site_tensor(layer_cache, site)
            if act is None:
                raise RuntimeError(
                    f"No activation for layer {layer} site {site!r}. "
                    f"Available: {list(layer_cache.keys())}",
                )
            tokens = act.squeeze(0).to(device="cpu", dtype=dtype)
            if sample_len is None:
                sample_len = tokens.shape[0]
            elif tokens.shape[0] != sample_len:
                raise RuntimeError(
                    f"Sequence-length mismatch at layer {layer} site {site!r}: "
                    f"{tokens.shape[0]} vs {sample_len} — all keys of one "
                    f"sample must share the tokenization.",
                )
            collected[(layer, site)].append(tokens)
    assert sample_len is not None
    return sample_len


def _gemma_site_tensor(layer_cache: dict, site: str) -> torch.Tensor | None:
    return layer_cache.get(GEMMA_SITE_MAP[site])


def _qwen_site_tensor(layer_cache: dict, site: str) -> torch.Tensor | None:
    """Qwen cache keys may be HookType members or their string values."""
    hook = QWEN_SITE_MAP[site]
    if hook in layer_cache:
        return layer_cache[hook]
    return layer_cache.get(hook.value)
