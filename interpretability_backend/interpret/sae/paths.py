"""Centralized path derivation for SAE data files.

All paths are derived from a ``GemmaScopeSAEConfig`` so there are no
hardcoded model IDs or directory names scattered across the codebase.

The resource root defaults to ``interpretability_backend/resources/`` — two
levels above this file's package directory — but can be overridden via the
``ORRERY_RESOURCE_DIR`` environment variable. Docker sets this so downloaded SAE
artifacts land in a named volume rather than being baked into the image/repo.
"""

import os
from pathlib import Path

from interpret.sae.sae_config import GemmaScopeSAEConfig
from interpret.sae.source_ids import neuronpedia_source_id

_DEFAULT_RESOURCES = Path(__file__).resolve().parents[2] / "resources"
_RESOURCES = Path(os.getenv("ORRERY_RESOURCE_DIR", str(_DEFAULT_RESOURCES))).expanduser()


def labels_dir(config: GemmaScopeSAEConfig) -> Path:
    """Directory holding downloaded Neuronpedia JSONL label files."""
    return _RESOURCES / "sae_labels" / f"neuronpedia_{config.neuronpedia_model_id}"


def vectors_dir() -> Path:
    """Directory holding extracted decoder-vector parquet files."""
    return _RESOURCES / "sae_vectors"


def features_jsonl_path(config: GemmaScopeSAEConfig) -> Path:
    """Path to the merged features + explanations JSONL for one source."""
    source = neuronpedia_source_id(config)
    return labels_dir(config) / f"{config.neuronpedia_model_id}_{source}_features.jsonl"


def activations_jsonl_path(config: GemmaScopeSAEConfig) -> Path:
    """Path to the merged activations JSONL for one source."""
    source = neuronpedia_source_id(config)
    return labels_dir(config) / f"{config.neuronpedia_model_id}_{source}_activations.jsonl"


def activation_batches_dir(config: GemmaScopeSAEConfig) -> Path:
    """Directory holding raw activation batch ``*.jsonl.gz`` files."""
    source = neuronpedia_source_id(config)
    return labels_dir(config) / "activations" / source


def vectors_parquet_path(config: GemmaScopeSAEConfig) -> Path:
    """Path to the decoder-vector + labels parquet for one source."""
    hook = config.hook_type.value
    model = config.neuronpedia_model_id
    return vectors_dir() / f"w_dec_{model}_layer{config.layer_index}_{hook}_w{config.width}.parquet"
