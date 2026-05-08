"""Sparse Autoencoder (SAE) support for interpretability work.

Two pretrained-SAE families are supported:

- Gemma-scope JumpReLU SAEs on Gemma3 (full feature set: capture +
  steering + Neuronpedia labels + decoder-vector pipeline).
- Qwen-scope TopK SAEs on Qwen3 (capture + steering only; no labels
  yet — Qwen-scope is not indexed by Neuronpedia).

Both families share the same hook / activation-store / steering
machinery — the only difference is the SAE class and how its weights
are downloaded and loaded.
"""

from interpret.sae.activation_store import ActivationRecord, ActivationStore
from interpret.sae.exploration.prompt_explorer import PromptExplorer, PromptExplorerConfig
from interpret.sae.feature_labels import FeatureLabelStore
from interpret.sae.hook_manager import HookManager
from interpret.sae.loading import load_sae
from interpret.sae.sae_config import (
    GemmaScopeSAEConfig,
    HookType,
    QwenScopeSAEConfig,
    SAEConfig,
)
from interpret.sae.sae_model import JumpReLUSAE, SAEBase, TopKSAE
from interpret.sae.source_ids import (
    HOOK_TO_NEURONPEDIA,
    neuronpedia_source_id,
    neuronpedia_source_id_prefixed,
)
from interpret.sae.steering import (
    SteeringMode,
    SteeringOp,
    apply_steering,
    resolve_op,
)

__all__ = [
    "ActivationRecord",
    "ActivationStore",
    "FeatureLabelStore",
    "GemmaScopeSAEConfig",
    "HOOK_TO_NEURONPEDIA",
    "HookManager",
    "HookType",
    "JumpReLUSAE",
    "PromptExplorer",
    "PromptExplorerConfig",
    "QwenScopeSAEConfig",
    "SAEBase",
    "SAEConfig",
    "SteeringMode",
    "SteeringOp",
    "TopKSAE",
    "apply_steering",
    "load_sae",
    "neuronpedia_source_id",
    "neuronpedia_source_id_prefixed",
    "resolve_op",
]
