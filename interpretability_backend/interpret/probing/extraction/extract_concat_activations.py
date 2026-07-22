"""Concat stage: stack pooled extractions' keys into one wide feature matrix.

Concatenates every `(layer, intermediate)` key of one or more pooled
source datasets (typically `sae_pooled`) along the feature axis, producing
a single `(0, "concat")` key of shape `[N, sum(d_kept)]` so a probe can
select task-relevant features jointly across the model's depth.

Column identity is preserved in `metadata["feature_names"]` as
`"L{layer}_{site}_f{true_index}"`, where the true SAE feature index comes
from the source's `kept_by_layer` map (falling back to positional indices
for non-SAE sources). The probe stage prefers these names over the
manifest's when writing `feature_importance.csv`.

When any source stores scipy CSR matrices (`sparse: true`), the concat is
built with `scipy.sparse.hstack` and stays CSR; dense blocks mixed in are
sparsified (note a truly dense block stored as CSR costs ~2x its dense
size — concat sparse sources with sparse sources where possible).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.concat_extraction import ConcatExtractionConfig
from interpret.probing.utils.matrix_ops import FeatureMatrix

CONCAT_KEY = (0, "concat")


def extract_concat_activations(
    sources: list[tuple[str, ActivationDataset]],
    config: ConcatExtractionConfig,
) -> ActivationDataset:
    """Concatenate `(name, dataset)` sources' keys along the feature axis."""
    if not sources:
        raise ValueError("extract_concat_activations: no sources given.")

    reference_ids = sources[0][1].sample_ids
    for name, dataset in sources[1:]:
        if dataset.sample_ids != reference_ids:
            raise ValueError(
                f"Concat source {name!r} has different sample_ids than "
                f"{sources[0][0]!r} — all sources must descend from the "
                f"same manifest.",
            )

    blocks: list[FeatureMatrix] = []
    feature_names: list[str] = []
    spans: list[tuple[str, int, str, int, int]] = []
    col = 0
    for name, dataset in sources:
        site = dataset.metadata.get("sae_site") or dataset.metadata.get("site")
        kept_by_layer = dataset.metadata.get("kept_by_layer", {})
        for layer, intermediate in sorted(dataset.activations):
            if config.layers is not None and layer not in config.layers:
                continue
            raw = dataset.activations[(layer, intermediate)]
            if sp.issparse(raw):
                block = raw.tocsr().astype(np.float32, copy=False)
            else:
                block = raw.float()
            if block.shape[0] != len(reference_ids):
                raise ValueError(
                    f"Concat source {name!r} key ({layer}, {intermediate!r}) "
                    f"has {block.shape[0]} rows for {len(reference_ids)} "
                    f"samples — is it token-level?",
                )
            true_indices = kept_by_layer.get(layer)
            if true_indices is None:
                true_indices = list(range(block.shape[1]))
            if len(true_indices) != block.shape[1]:
                raise ValueError(
                    f"Concat source {name!r} layer {layer}: kept_by_layer "
                    f"has {len(true_indices)} entries for {block.shape[1]} "
                    f"columns.",
                )
            label_site = site or intermediate
            feature_names.extend(f"L{layer}_{label_site}_f{idx}" for idx in true_indices)
            blocks.append(block)
            spans.append((name, layer, label_site, col, col + block.shape[1]))
            col += block.shape[1]

    if not blocks:
        raise ValueError(
            f"Concat {config.name!r}: no keys survived the layer filter {config.layers}.",
        )

    if any(sp.issparse(b) for b in blocks):
        matrix = sp.hstack(
            [b if sp.issparse(b) else sp.csr_matrix(b.numpy()) for b in blocks],
            format="csr",
        )
    else:
        matrix = torch.cat(blocks, dim=1)
    metadata = {
        "extraction_type": "concat",
        "sparse": bool(sp.issparse(matrix)),
        "feature_names": feature_names,
        "concat_spans": spans,
        "source_extractions": [name for name, _ in sources],
        "layers_filter": list(config.layers) if config.layers else None,
        "num_samples": len(reference_ids),
        "intermediates": [CONCAT_KEY[1]],
    }
    print(
        f"  {config.name}: concatenated {len(spans)} keys from "
        f"{len(sources)} source(s) -> [{matrix.shape[0]}, {matrix.shape[1]}]",
    )
    return ActivationDataset(
        activations={CONCAT_KEY: matrix},
        targets=sources[0][1].targets,
        sample_ids=list(reference_ids),
        metadata=metadata,
    )
