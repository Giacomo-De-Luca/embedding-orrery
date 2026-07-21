"""Residual-pooled stage: token-level residuals -> one pooled row per sample.

The no-SAE counterpart of `extract_sae_pooled`: pools a `token_residuals`
extraction's ragged tensors per sample, giving the raw-residual baseline
probes are compared against — for both families, from the same forward
pass that fed the SAEs. Keys: `(layer, "res_last"|"res_max"|"res_mean")`,
fp32 `[N, hidden]`.

BOS handling matches the SAE stage: for max/mean pooling the BOS position
is dropped when the source prepends BOS (single-token samples keep it).
`last` pooling needs no mask — the last token is never the BOS for a
non-empty prompt.
"""

from __future__ import annotations

import torch

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.residual_pooled_extraction import (
    ResidualPooledExtractionConfig,
)


def extract_residual_pooled(
    source: ActivationDataset,
    config: ResidualPooledExtractionConfig,
) -> ActivationDataset:
    """Pool one site's token-level residuals per sample (no SAE)."""
    if not source.metadata.get("token_level"):
        raise ValueError(
            f"residual_pooled source must be a token_residuals extraction "
            f"(metadata['token_level'] missing); got extraction_type="
            f"{source.metadata.get('extraction_type')!r}",
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

    drop_bos = prepends_bos and config.exclude_bos and config.pooling != "last"
    key_name = config.intermediate_key

    activations: dict[tuple[int, str], torch.Tensor] = {}
    for layer in layers:
        source_key = (layer, config.site)
        if source_key not in source.activations:
            raise KeyError(
                f"Layer {layer}: missing source key {source_key}. "
                f"Available: {sorted(source.activations)}",
            )
        tokens = source.activations[source_key].float()
        activations[(layer, key_name)] = _pool(
            tokens,
            offsets,
            config.pooling,
            drop_bos,
        )

    metadata = {
        k: v
        for k, v in source.metadata.items()
        if k not in ("token_offsets", "token_level", "n_tokens", "intermediates")
    }
    metadata.update(
        {
            "extraction_type": "residual_pooled",
            "site": config.site,
            "pooling": config.pooling,
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


def _pool(
    tokens: torch.Tensor,
    offsets: torch.Tensor,
    pooling: str,
    drop_bos: bool,
) -> torch.Tensor:
    """Reduce ragged `[total_tokens, H]` to `[N, H]` per the pooling mode."""
    if pooling == "last":
        return tokens[offsets[1:] - 1].clone()

    rows = []
    for i in range(offsets.numel() - 1):
        start, end = int(offsets[i]), int(offsets[i + 1])
        if drop_bos and end - start > 1:
            start += 1
        segment = tokens[start:end]
        if pooling == "max":
            rows.append(segment.max(dim=0).values)
        elif pooling == "mean":
            rows.append(segment.mean(dim=0))
        else:
            raise ValueError(f"Unknown pooling {pooling!r}.")
    return torch.stack(rows)
