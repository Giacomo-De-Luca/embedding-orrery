"""Steering-based autointerpreter: interpret SAE features by intervention.

Steer each feature into Gemma-3's residual stream during generation, collect the
model's answers to fixed probe questions across a strength sweep, and have an LLM
judge name and rate the resulting behaviour. Complements the read-the-activations
autointerpreter in the parent package. See ``README.md`` in this folder.
"""

from interpret.sae.autointerpreter.steering.config import (
    SteeringInterpretConfig,
    load_steering_experiments,
)
from interpret.sae.autointerpreter.steering.generate import SteeringGenerator
from interpret.sae.autointerpreter.steering.judge import (
    AgentQueueDriver,
    SteeringAggregator,
    SteeringJudgeInputWriter,
)
from interpret.sae.autointerpreter.steering.run import (
    SteeringInterpretRunner,
    run_from_yaml,
)

__all__ = [
    "SteeringInterpretConfig",
    "load_steering_experiments",
    "SteeringGenerator",
    "SteeringJudgeInputWriter",
    "SteeringAggregator",
    "AgentQueueDriver",
    "SteeringInterpretRunner",
    "run_from_yaml",
]
