"""Pooled-SAE stage: token-level residuals -> SAE features -> one row/sample.

Stage 2 of the two-stage pipeline. For each layer: load the family-correct
SAE (JumpReLU for gemma-scope, TopK for qwen-scope), encode the source's
ragged `[total_tokens, hidden]` tensor for the configured site, pool per
sample, and dead-filter — producing a probeable `[N, d_kept]` dataset with
`(layer, "sae_max"|"sae_last")` keys and the `kept_by_layer` index map the
sae_analysis stage depends on.

Pooling semantics mirror the production document-activation path
(`interpret.sae.activation_store.max_pool_feature_acts`), replicated as
dense tensor ops so encoding can batch across all samples:

- ``max``: per-feature max over the sample's tokens. The BOS position is
  masked when the source model prepends BOS (the BOS activation sink
  otherwise tops every sample identically), except for samples whose only
  token IS the BOS — those fall back to full-range pooling, like the
  interactive path's ``fallback_to_full``.
- ``last``: encode only the last-token residual.

A feature is kept iff its pooled activation is > 0 in at least
``min_active_samples`` samples — matching the production convention that
non-positive activations count as inactive (TopK can keep negative
pre-activations; those never survive this filter).
"""

from __future__ import annotations

import torch

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.sae_pooled_extraction import (
    SAEPooledExtractionConfig,
)
from interpret.sae import (
    GemmaScopeSAEConfig,
    QwenScopeSAEConfig,
    clear_sae_cache,
    load_sae,
)
from interpret.sae.sae_config import HOOK_TYPE_FROM_STR


def extract_sae_pooled(
    source: ActivationDataset,
    config: SAEPooledExtractionConfig,
) -> ActivationDataset:
    """Encode + pool a token-level source through per-layer SAEs."""
    if not source.metadata.get("token_level"):
        raise ValueError(
            f"sae_pooled source must be a token_residuals extraction "
            f"(metadata['token_level'] missing); got extraction_type="
            f"{source.metadata.get('extraction_type')!r}",
        )
    family = source.metadata["family"]
    if family == "qwen" and config.site != "resid_post":
        raise ValueError(
            "qwen-scope SAEs are residual-stream only; "
            f"site must be 'resid_post', got {config.site!r}.",
        )
    offsets = torch.tensor(source.metadata["token_offsets"], dtype=torch.long)
    n_samples = len(source.sample_ids)
    if offsets.numel() != n_samples + 1:
        raise ValueError(
            f"token_offsets has {offsets.numel()} entries for {n_samples} samples (expected N+1).",
        )
    prepends_bos = bool(source.metadata.get("prepends_bos", False))

    layers = config.layers or sorted(
        layer for (layer, site) in source.activations if site == config.site
    )
    if not layers:
        raise ValueError(
            f"No source keys for site {config.site!r}. Available: {sorted(source.activations)}",
        )

    keep_mask, sample_idx = _token_masks(
        offsets,
        exclude_bos=prepends_bos and config.exclude_bos and config.pooling == "max",
    )

    activations: dict[tuple[int, str], torch.Tensor] = {}
    kept_by_layer: dict[int, list[int]] = {}
    key_name = config.intermediate_key

    for layer in layers:
        source_key = (layer, config.site)
        if source_key not in source.activations:
            raise KeyError(
                f"Layer {layer}: missing source key {source_key}. "
                f"Available: {sorted(source.activations)}",
            )
        residual = source.activations[source_key]

        sae = load_sae(_build_sae_config(config, family, layer))
        try:
            if config.pooling == "last":
                pooled = _encode_last(residual, offsets, sae, config)
            else:
                pooled = _encode_max(
                    residual,
                    sample_idx,
                    keep_mask,
                    n_samples,
                    sae,
                    config,
                )
        finally:
            # One fp32 SAE per layer is fine; 28-34 cached ones are not —
            # the module-level _SAE_CACHE never evicts on its own.
            clear_sae_cache()

        if config.drop_dead_features:
            alive = (pooled > 0).sum(dim=0) >= config.min_active_samples
        else:
            alive = torch.ones(
                pooled.shape[1],
                dtype=torch.bool,
                device=pooled.device,
            )
        kept = torch.nonzero(alive, as_tuple=True)[0]
        kept_by_layer[layer] = kept.tolist()
        activations[(layer, key_name)] = pooled[:, alive].cpu().float()
        print(
            f"  {config.name} layer {layer}: kept {len(kept)}/{pooled.shape[1]} "
            f"features ({len(kept) / pooled.shape[1]:.1%} alive, "
            f"pooling={config.pooling})",
        )

    metadata = {
        k: v
        for k, v in source.metadata.items()
        if k
        not in (
            "token_offsets",
            "token_level",
            "n_tokens",
            "intermediates",
            "storage_dtype",
            "sites",
        )
    }
    metadata.update(
        {
            "extraction_type": "sae_pooled",
            "sae_family": family,
            "sae_site": config.site,
            "sae_width": config.width,
            "pooling": config.pooling,
            "kept_by_layer": kept_by_layer,
            "layers": list(layers),
            "intermediates": [key_name],
        },
    )
    return ActivationDataset(
        activations=activations,
        targets=source.targets,
        sample_ids=list(source.sample_ids),
        metadata=metadata,
    )


def _build_sae_config(
    config: SAEPooledExtractionConfig,
    family: str,
    layer: int,
):
    if family == "gemma":
        return GemmaScopeSAEConfig(
            layer_index=layer,
            hook_type=HOOK_TYPE_FROM_STR[config.site],
            model_size=config.model_size,
            variant=config.variant,
            width=config.width,
            l0_size=config.l0_size,
            dtype="float32",
            device=config.device,
        )
    if family == "qwen":
        return QwenScopeSAEConfig(
            layer_index=layer,
            model_size=config.model_size,
            k=config.k,
            width=config.width,
            dtype="float32",
            device=config.device,
        )
    raise ValueError(f"Unknown SAE family {family!r} (expected gemma|qwen).")


def _token_masks(
    offsets: torch.Tensor,
    exclude_bos: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (keep, sample_index) vectors for segment pooling.

    When `exclude_bos`, each sample's first position is dropped — unless
    it is the sample's ONLY token, which falls back to keeping it
    (mirroring `max_pool_feature_acts`' degenerate-range fallback).
    """
    lengths = offsets[1:] - offsets[:-1]
    if (lengths <= 0).any():
        raise ValueError("token_offsets contains an empty sample.")
    total = int(offsets[-1])
    sample_idx = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.long),
        lengths,
    )
    keep = torch.ones(total, dtype=torch.bool)
    if exclude_bos:
        keep[offsets[:-1]] = False
        single = lengths == 1
        if single.any():
            keep[offsets[:-1][single]] = True
    return keep, sample_idx


def _encode_last(
    residual: torch.Tensor,
    offsets: torch.Tensor,
    sae,
    config: SAEPooledExtractionConfig,
) -> torch.Tensor:
    """Encode each sample's last-token residual in one batch. [N, d_sae]."""
    rows = residual[offsets[1:] - 1].to(device=config.device, dtype=torch.float32)
    with torch.no_grad():
        return sae.encode(rows)


def _encode_max(
    residual: torch.Tensor,
    sample_idx: torch.Tensor,
    keep_mask: torch.Tensor,
    n_samples: int,
    sae,
    config: SAEPooledExtractionConfig,
) -> torch.Tensor:
    """Encode all tokens chunk-wise and segment-max per sample. [N, d_sae].

    `out` starts at -inf with `include_self=True` so the running max is
    correct across chunks (`include_self=False` would discard earlier
    chunks' results); every sample has >= 1 kept token, so no -inf survives.
    """
    d_sae = sae.w_dec.shape[0]
    device = torch.device(config.device)
    out = torch.full((n_samples, d_sae), -torch.inf, device=device)

    total = residual.shape[0]
    chunk = config.batch_size_tokens
    with torch.no_grad():
        for start in range(0, total, chunk):
            sl = slice(start, min(start + chunk, total))
            kept = keep_mask[sl]
            if not kept.any():
                continue
            x = residual[sl][kept].to(device=device, dtype=torch.float32)
            feats = sae.encode(x)
            idx = sample_idx[sl][kept].to(device)
            out.scatter_reduce_(
                0,
                idx.unsqueeze(1).expand(-1, d_sae),
                feats,
                reduce="amax",
                include_self=True,
            )
    if not torch.isfinite(out).all():
        raise RuntimeError(
            "Max pooling left unwritten samples — token masks are "
            "inconsistent with the offset table.",
        )
    return out
