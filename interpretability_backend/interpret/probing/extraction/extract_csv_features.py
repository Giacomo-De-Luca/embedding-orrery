"""Extractor for pre-computed feature vectors stored in the manifest.

Reads ``feature_columns`` from the manifest's DataFrame in the order of
``manifest.samples`` and returns an ``ActivationDataset`` with a single
``(0, intermediate_label)`` key. Targets are left empty — the orchestrator
joins them in via ``manifest.get_rated_samples`` like for every other
extraction type.
"""

from __future__ import annotations

import torch

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.csv_features_extraction import (
    CSVFeaturesExtractionConfig,
)
from interpret.probing.manifests.manifest_base import ManifestBuilder


def extract_csv_features(
    config: CSVFeaturesExtractionConfig,
    manifest: ManifestBuilder,
) -> ActivationDataset:
    """Stack pre-computed feature columns into a single activation tensor."""
    feature_columns = config.feature_columns
    if feature_columns is None:
        feature_columns = getattr(manifest, "feature_columns", None)
        if feature_columns is None:
            raise ValueError(
                f"CSVFeaturesExtractionConfig {config.name!r}: "
                f"feature_columns is None and the manifest "
                f"{type(manifest).__name__} does not expose "
                f"`feature_columns`.",
            )
    feature_columns = list(feature_columns)

    df = manifest.build_dataframe()
    prompt_col = manifest.prompt_column
    if prompt_col not in df.columns:
        raise ValueError(
            f"manifest.prompt_column={prompt_col!r} not in DataFrame columns {df.columns.tolist()}",
        )

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature columns {missing} missing from manifest DataFrame.",
        )

    aligned = df.set_index(prompt_col).loc[manifest.samples, feature_columns]
    tensor = torch.tensor(aligned.to_numpy(), dtype=torch.float32)

    metadata = {
        "extraction_type": "csv_features",
        "extraction_name": config.name,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "n_samples": tensor.shape[0],
    }
    return ActivationDataset(
        activations={(0, config.intermediate_label): tensor},
        sample_ids=list(manifest.samples),
        metadata=metadata,
    )
