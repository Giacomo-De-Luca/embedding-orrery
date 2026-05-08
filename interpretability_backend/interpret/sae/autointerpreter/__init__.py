"""SAE autointerpreter pipeline.

See ``README.md`` for the end-to-end data flow.
"""

from interpret.sae.autointerpreter.collect_activations import ActivationCollector
from interpret.sae.autointerpreter.config import (
    AgentStageConfig,
    AutoInterpretCollectConfig,
    AutoInterpretConfig,
    AutoInterpretScoreConfig,
    StageFlags,
    TopKExtractConfig,
    load_experiments,
)
from interpret.sae.autointerpreter.extract_top_k import TopKFeatureExtractor
from interpret.sae.autointerpreter.prepare_agent_inputs import AgentInputWriter
from interpret.sae.autointerpreter.run_autointerpret import (
    AutoInterpretRunner,
    run_from_yaml,
)
from interpret.sae.autointerpreter.score_autointerpret import AutoInterpretScorer
from interpret.sae.autointerpreter.sparse_activation_store import SparseActivationStore

__all__ = [
    "ActivationCollector",
    "AgentInputWriter",
    "AgentStageConfig",
    "AutoInterpretCollectConfig",
    "AutoInterpretConfig",
    "AutoInterpretRunner",
    "AutoInterpretScoreConfig",
    "AutoInterpretScorer",
    "SparseActivationStore",
    "StageFlags",
    "TopKExtractConfig",
    "TopKFeatureExtractor",
    "load_experiments",
    "run_from_yaml",
]
