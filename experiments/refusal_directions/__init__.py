"""Replicate Arditi et al., 'Refusal is Mediated by a Single Direction' on Gemma-3-4b-it.

Adapts the reference pipeline at ``references/refusal_direction/`` to the project's
own ``GemmaPytorchInference`` + ``HookManager`` + ``SteeringOp`` infrastructure.
"""

from interpret.experiments.refusal_directions.config import RefusalConfig
from interpret.experiments.refusal_directions.runner import RefusalRunner

__all__ = ["RefusalConfig", "RefusalRunner"]
