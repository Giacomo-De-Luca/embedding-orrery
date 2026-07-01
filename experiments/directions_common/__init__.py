"""Shared infrastructure for activation-direction experiments (refusal, poetry).

A small backend-agnostic layer so the direction pipelines run on either
``GemmaPytorchInference`` (Gemma-3) or ``Qwen3Inference`` (Qwen3 / Qwen3.5)
without per-model forks:

- ``model_adapter`` — ``DirectionModel`` surface + Gemma/Qwen implementations
  and the ``build_direction_model`` factory.
- ``sites`` — ``CaptureSite`` (canonical residual-stream points) + Qwen
  resolution.
- ``steering_ops`` — ``_additive_op`` / ``_ablation_ops`` / ``_make_manager``.
- ``scoring`` — ``_refusal_score`` / ``_kl_div`` / ``_score_dataset``.

See README.md for details.
"""

from interpret.experiments.directions_common.model_adapter import (
    DirectionModel,
    GemmaDirectionModel,
    QwenDirectionModel,
    build_direction_model,
)
from interpret.experiments.directions_common.scoring import (
    _kl_div,
    _refusal_score,
    _score_dataset,
)
from interpret.experiments.directions_common.sites import (
    QWEN_SITE_MAP,
    CaptureSite,
)
from interpret.experiments.directions_common.steering_ops import (
    _ablation_ops,
    _additive_op,
    _bypass_ops,
    _make_manager,
)

__all__ = [
    "DirectionModel",
    "GemmaDirectionModel",
    "QwenDirectionModel",
    "build_direction_model",
    "CaptureSite",
    "QWEN_SITE_MAP",
    "_additive_op",
    "_ablation_ops",
    "_bypass_ops",
    "_make_manager",
    "_refusal_score",
    "_kl_div",
    "_score_dataset",
]
