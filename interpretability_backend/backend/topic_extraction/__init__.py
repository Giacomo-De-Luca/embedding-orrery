"""Topic extraction module for clustering embeddings and generating labels."""

from typing import TYPE_CHECKING

from .llm_labeling import generate_llm_labels

if TYPE_CHECKING:
    from .cluster_and_label import ClassTfidfTransformer, GenerateTopics

_LAZY = ("ClassTfidfTransformer", "GenerateTopics")

__all__ = [*_LAZY, "generate_llm_labels"]


def __getattr__(name: str):
    # Lazy re-export: cluster_and_label pulls in hdbscan/sklearn/scipy (~94 MB RSS),
    # which must stay out of server startup (see topic_extraction_service.py).
    if name in _LAZY:
        from . import cluster_and_label

        return getattr(cluster_and_label, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
