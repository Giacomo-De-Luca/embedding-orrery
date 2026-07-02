"""Evaluation package.

Standalone metrics + config-driven runners: topic-quality scoring
(:class:`TopicQualityEvaluator`) and projection fidelity via the Mantel test
(:class:`ProjectionFidelityEvaluator`). See ``README.md`` for usage and the
meaning of each metric.
"""

from .projection_fidelity import ProjectionFidelityEvaluator
from .quality_metrics import TopicQualityEvaluator

__all__ = ["ProjectionFidelityEvaluator", "TopicQualityEvaluator"]
