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

Storage: dense fp32 CPU tensors by default; ``sparse: true`` stores scipy
CSR instead (pooled rows are ~1-2% nonzero). Pooling always iterates
sample blocks whose dense device buffer stays under ``_POOL_BLOCK_BYTES``,
so peak memory never scales with ``N * d_sae`` — the sparse path
sparsifies each block as it completes.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
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

# Cap on the per-block dense [rows, d_sae] fp32 pooling buffer. Derived,
# not configurable: an execution-only knob would needlessly enter the
# cache-identity dict (same reason batch_size_tokens shouldn't — see the
# deferred follow-ups in documentation/TREC_SAE_PROBING.md).
_POOL_BLOCK_BYTES = 1 << 30


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
    if n_samples == 0:
        raise ValueError("sae_pooled source has zero samples.")
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

    activations: dict[tuple[int, str], torch.Tensor | sp.spmatrix] = {}
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
                    offsets,
                    sample_idx,
                    keep_mask,
                    sae,
                    config,
                )
        finally:
            # One fp32 SAE per layer is fine; 28-34 cached ones are not —
            # the module-level _SAE_CACHE never evicts on its own.
            clear_sae_cache()

        if sp.issparse(pooled):
            matrix, kept = _filter_sparse(pooled, config)
        else:
            matrix, kept = _filter_dense(pooled, config)
        kept_by_layer[layer] = kept
        activations[(layer, key_name)] = matrix
        nnz_note = (
            f", {matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1]):.1%} nnz (CSR)"
            if sp.issparse(matrix)
            else ""
        )
        print(
            f"  {config.name} layer {layer}: kept {len(kept)}/{pooled.shape[1]} "
            f"features ({len(kept) / pooled.shape[1]:.1%} alive, "
            f"pooling={config.pooling}{nnz_note})",
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
            "sparse": bool(config.sparse),
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


def _block_rows(d_sae: int) -> int:
    """Samples per pooling block, keeping the dense buffer under the cap."""
    return max(1, _POOL_BLOCK_BYTES // (4 * d_sae))


def _finish_block(block: torch.Tensor, sparse: bool) -> torch.Tensor | sp.csr_matrix:
    block = block.cpu().float()
    if sparse:
        return sp.csr_matrix(block.numpy())
    return block


def _stack_blocks(
    blocks: list[torch.Tensor | sp.csr_matrix],
    sparse: bool,
) -> torch.Tensor | sp.csr_matrix:
    if len(blocks) == 1:  # common case — avoid a full-matrix copy
        return blocks[0]
    if sparse:
        return sp.vstack(blocks, format="csr")
    return torch.cat(blocks, dim=0)


def _encode_last(
    residual: torch.Tensor,
    offsets: torch.Tensor,
    sae,
    config: SAEPooledExtractionConfig,
) -> torch.Tensor | sp.csr_matrix:
    """Encode each sample's last-token residual, chunk-wise. [N, d_sae]."""
    rows = residual[offsets[1:] - 1]
    blocks: list[torch.Tensor | sp.csr_matrix] = []
    with torch.no_grad():
        for start in range(0, rows.shape[0], config.batch_size_tokens):
            x = rows[start : start + config.batch_size_tokens].to(
                device=config.device,
                dtype=torch.float32,
            )
            blocks.append(_finish_block(sae.encode(x), config.sparse))
    return _stack_blocks(blocks, config.sparse)


def _encode_max(
    residual: torch.Tensor,
    offsets: torch.Tensor,
    sample_idx: torch.Tensor,
    keep_mask: torch.Tensor,
    sae,
    config: SAEPooledExtractionConfig,
) -> torch.Tensor | sp.csr_matrix:
    """Encode all tokens chunk-wise and segment-max per sample. [N, d_sae].

    Iterates blocks of samples (dense device buffer capped at
    `_POOL_BLOCK_BYTES`); within a block, tokens are encoded in
    `batch_size_tokens` chunks. Each block buffer starts at -inf with
    `include_self=True` so the running max is correct across chunks
    (`include_self=False` would discard earlier chunks' results); every
    sample has >= 1 kept token, so no -inf survives.
    """
    d_sae = sae.w_dec.shape[0]
    device = torch.device(config.device)
    n_samples = offsets.numel() - 1
    block_rows = _block_rows(d_sae)
    blocks: list[torch.Tensor | sp.csr_matrix] = []

    with torch.no_grad():
        for s0 in range(0, n_samples, block_rows):
            s1 = min(s0 + block_rows, n_samples)
            t0, t1 = int(offsets[s0]), int(offsets[s1])
            out = torch.full((s1 - s0, d_sae), -torch.inf, device=device)
            for start in range(t0, t1, config.batch_size_tokens):
                sl = slice(start, min(start + config.batch_size_tokens, t1))
                kept = keep_mask[sl]
                if not kept.any():
                    continue
                x = residual[sl][kept].to(device=device, dtype=torch.float32)
                feats = sae.encode(x)
                idx = (sample_idx[sl][kept] - s0).to(device)
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
            blocks.append(_finish_block(out, config.sparse))
    return _stack_blocks(blocks, config.sparse)


def _filter_dense(
    pooled: torch.Tensor,
    config: SAEPooledExtractionConfig,
) -> tuple[torch.Tensor, list[int]]:
    """Apply the dead-feature filter to a dense CPU fp32 matrix."""
    if config.drop_dead_features:
        alive = (pooled > 0).sum(dim=0) >= config.min_active_samples
    else:
        alive = torch.ones(pooled.shape[1], dtype=torch.bool)
    kept = torch.nonzero(alive, as_tuple=True)[0]
    return pooled[:, alive], kept.tolist()


def _filter_sparse(
    pooled: sp.csr_matrix,
    config: SAEPooledExtractionConfig,
) -> tuple[sp.csr_matrix, list[int]]:
    """Apply the dead-feature filter to a CSR matrix (column slice)."""
    if config.drop_dead_features:
        counts = np.asarray((pooled > 0).sum(axis=0)).ravel()
        alive_idx = np.nonzero(counts >= config.min_active_samples)[0]
    else:
        alive_idx = np.arange(pooled.shape[1])
    if len(alive_idx) == pooled.shape[1]:
        return pooled, [int(i) for i in alive_idx]
    return pooled[:, alive_idx].tocsr(), [int(i) for i in alive_idx]
