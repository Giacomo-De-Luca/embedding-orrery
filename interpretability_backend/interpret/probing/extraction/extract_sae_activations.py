"""SAE encoding stage: residual activations -> sparse SAE feature activations.

Consumes a Gemma `ActivationDataset` (residual stream per layer) and
produces a new `ActivationDataset` keyed by `(layer, "sae_feat")` with
optionally-filtered sparse feature activations.

Critical: `metadata["kept_by_layer"]` records the mapping from filtered
column index -> original SAE feature index. Downstream label lookup and
top-K feature reporting depend on this surviving save/load.
"""

from __future__ import annotations

import numpy as np
import torch

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.sae_extraction import (
    SAEExtractionConfig,
)
from interpret.sae import SAEConfig, load_sae

INTERMEDIATE_KEY = "sae_feat"


def extract_sae_activations(
    residuals: ActivationDataset,
    config: SAEExtractionConfig,
) -> ActivationDataset:
    """Encode per-layer residuals through Gemma-Scope JumpReLU SAEs.

    Args:
        residuals: Gemma activation dataset. Must contain
            `(layer, config.source_intermediate)` keys for every layer in
            `config.layers`.
        config: SAE width, layers, device, dead-feature filtering flag.

    Returns:
        New `ActivationDataset` with `(layer, "sae_feat")` keys, the same
        `sample_ids` as the input, and `metadata["kept_by_layer"]`
        recording the surviving feature indices per layer.
    """
    activations: dict[tuple[int, str], torch.Tensor] = {}
    kept_by_layer: dict[int, list[int]] = {}

    for layer in config.layers:
        key = (layer, config.source_intermediate)
        if key not in residuals.activations:
            raise KeyError(
                f"Layer {layer}: missing residual activation at {key}. "
                f"Available keys: {list(residuals.activations)}",
            )
        residual_tensor = residuals.activations[key]
        feat_tensor, kept = _encode_and_filter(
            residual_tensor,
            layer=layer,
            width=config.width,
            device=config.device,
            drop_dead=config.drop_dead_features,
        )
        activations[(layer, INTERMEDIATE_KEY)] = feat_tensor
        kept_by_layer[layer] = kept.tolist()

    metadata = dict(residuals.metadata)
    metadata.update(
        {
            "extraction_type": "sae",
            "sae_width": config.width,
            "sae_source_intermediate": config.source_intermediate,
            "drop_dead_features": config.drop_dead_features,
            "layers": list(config.layers),
            "intermediates": [INTERMEDIATE_KEY],
            "kept_by_layer": kept_by_layer,
        },
    )

    return ActivationDataset(
        activations=activations,
        targets=residuals.targets,  # propagate (typically empty at this stage)
        sample_ids=list(residuals.sample_ids),
        metadata=metadata,
    )


def _encode_and_filter(
    residual: torch.Tensor,
    layer: int,
    width: str,
    device: str,
    drop_dead: bool,
) -> tuple[torch.Tensor, np.ndarray]:
    """Encode residuals through one SAE; optionally drop dead features.

    Returns:
        (feat_tensor [N, d_kept], kept_indices [d_kept]).
        When `drop_dead=False`, `kept_indices` is `arange(d_sae)`.
    """
    sae = load_sae(
        SAEConfig(
            layer_index=layer, width=width, device=device, dtype="float32",
        ),
    )
    x = residual.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        feat = sae.encode(x)  # [N, d_sae]
    feat_np = feat.cpu().numpy()  # cheap to keep one big tensor as numpy here

    if not drop_dead:
        kept = np.arange(feat_np.shape[1], dtype=np.int64)
        return torch.from_numpy(feat_np), kept

    alive_mask = (feat_np > 0).any(axis=0)
    kept = np.nonzero(alive_mask)[0].astype(np.int64)
    feat_filtered = feat_np[:, alive_mask]
    print(
        f"  layer {layer}: kept {len(kept)}/{feat_np.shape[1]} features "
        f"({len(kept) / feat_np.shape[1]:.1%} alive)",
    )
    return torch.from_numpy(feat_filtered), kept
