"""Delta extraction: per-pair activation subtraction.

Reads a precomputed source `ActivationDataset` and the manifest, pairs
each non-baseline row with its baseline counterpart (matched on
`pairing_column`, with `baseline_filter` selecting baselines), and
emits the difference. Output dataset has the same `(layer, intermediate)`
keys as the source; only baseline rows are dropped.

The subtraction is applied uniformly to every layer/intermediate, so the
function works identically for Gemma residual sources and SAE-feature
sources.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.delta_extraction import (
    DeltaExtractionConfig,
)
from interpret.probing.manifests.manifest_base import ManifestBuilder


def extract_delta_activations(
    source: ActivationDataset,
    manifest: ManifestBuilder,
    config: DeltaExtractionConfig,
) -> ActivationDataset:
    """Subtract baseline activations from non-baseline activations pairwise.

    Args:
        source: The upstream `ActivationDataset` whose activations are to
            be paired and subtracted. Must contain BOTH baseline and
            non-baseline rows.
        manifest: The experiment's manifest builder. Provides the
            DataFrame whose rows are aligned with `source.sample_ids` via
            `manifest.get_sample_indices`.
        config: Pairing column, baseline filter, source name.

    Returns:
        New `ActivationDataset`:
          * activations: same keys as `source`, restricted to non-baseline
            rows, values = `source[row] - source[matched_baseline_row]`.
          * sample_ids: the non-baseline rows' sample_ids, in the order
            they appear in `source`.
          * targets: empty (probes pull targets from the manifest).
          * metadata: copy of source.metadata + delta-specific fields.

    Raises:
        ValueError: pairing column missing, baseline filter empty,
            duplicate baseline for a key, missing baseline for a key.
    """
    if source.metadata.get("extraction_type") == "delta":
        raise ValueError(
            f"DeltaExtraction '{config.name}': source "
            f"'{config.source_extraction}' is itself a delta extraction "
            f"(its baseline rows have already been removed, so this "
            f"chained delta has no baselines to subtract). Point at the "
            f"original gemma/sae extraction instead.",
        )

    df = manifest.build_dataframe()

    if config.pairing_column not in df.columns:
        raise ValueError(
            f"DeltaExtraction '{config.name}': pairing_column "
            f"{config.pairing_column!r} not in manifest columns "
            f"{df.columns.tolist()}",
        )
    for col in config.baseline_filter:
        if col not in df.columns:
            raise ValueError(
                f"DeltaExtraction '{config.name}': baseline_filter column "
                f"{col!r} not in manifest columns {df.columns.tolist()}",
            )

    # Re-index manifest so row i == source.sample_ids[i].
    indices = manifest.get_sample_indices(source.sample_ids)
    aligned = df.iloc[indices].reset_index(drop=True)

    baseline_mask = _build_baseline_mask(aligned, config.baseline_filter)
    n_baseline = int(baseline_mask.sum())
    n_non_baseline = int((~baseline_mask).sum())
    if n_baseline == 0:
        raise ValueError(
            f"DeltaExtraction '{config.name}': baseline_filter "
            f"{config.baseline_filter!r} matched zero rows.",
        )
    if n_non_baseline == 0:
        raise ValueError(
            f"DeltaExtraction '{config.name}': all rows are baselines — nothing to subtract.",
        )

    # Map pairing-key → baseline row index in source.
    baseline_lookup: dict[Any, int] = {}
    for i, key in zip(
        np.flatnonzero(baseline_mask.to_numpy()),
        aligned.loc[baseline_mask, config.pairing_column].tolist(),
    ):
        if key in baseline_lookup:
            raise ValueError(
                f"DeltaExtraction '{config.name}': duplicate baseline for "
                f"pairing_column={config.pairing_column!r} value {key!r} "
                f"(rows {baseline_lookup[key]} and {i}). Each pairing key "
                f"may have at most one baseline row.",
            )
        baseline_lookup[key] = int(i)

    # For each non-baseline row, look up its baseline.
    non_baseline_rows = np.flatnonzero((~baseline_mask).to_numpy())
    baseline_for_each = np.empty(len(non_baseline_rows), dtype=np.int64)
    missing: list[Any] = []
    for j, row_i in enumerate(non_baseline_rows):
        key = aligned.loc[row_i, config.pairing_column]
        if key not in baseline_lookup:
            missing.append(key)
            continue
        baseline_for_each[j] = baseline_lookup[key]
    if missing:
        head = missing[:5]
        raise ValueError(
            f"DeltaExtraction '{config.name}': {len(missing)} non-baseline "
            f"rows have no matching baseline for "
            f"pairing_column={config.pairing_column!r}. First 5 missing "
            f"keys: {head}",
        )

    non_baseline_idx_t = torch.from_numpy(non_baseline_rows).long()
    baseline_idx_t = torch.from_numpy(baseline_for_each).long()

    new_activations: dict[tuple[int, str], torch.Tensor] = {}
    for key, tensor in source.activations.items():
        new_activations[key] = tensor[non_baseline_idx_t] - tensor[baseline_idx_t]

    new_sample_ids = [source.sample_ids[i] for i in non_baseline_rows]

    metadata = dict(source.metadata)
    metadata.update(
        {
            "extraction_type": "delta",
            "delta_source": config.source_extraction,
            "delta_pairing_column": config.pairing_column,
            "delta_baseline_filter": dict(config.baseline_filter),
            "delta_n_baseline": n_baseline,
            "delta_n_non_baseline": int(len(new_sample_ids)),
        },
    )

    return ActivationDataset(
        activations=new_activations,
        targets=torch.empty(0),
        sample_ids=new_sample_ids,
        metadata=metadata,
    )


def _build_baseline_mask(
    df: pd.DataFrame,
    baseline_filter: dict[str, Any],
) -> pd.Series:
    """Combine equality filters with logical AND."""
    mask = pd.Series(True, index=df.index)
    for col, val in baseline_filter.items():
        mask &= df[col] == val
    return mask
