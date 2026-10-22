"""Legacy semantic topic reduction falls back to c-TF-IDF when centroids are unavailable.

Covers ``TopicReductionService``'s ``legacy_semantic`` strategy: on success it
reduces with the raw centroid matrix (rows in sorted topic order, noise
first); on failure the plan reports the fallback, so the job's progress
stream can say so, and reduces with c-TF-IDF instead of failing (or, as
before the original fix, silently merging zero vectors).
"""

import numpy as np

from backend.services.topic_reduction_service import BaseTopics, TopicReductionService
from backend.services.topic_reduction_types import ReductionRequest


class _Centroids:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        self.calls = []

    def centroids_for_assignments(self, collection_name, item_topics, order, preloaded=None):
        self.calls.append({"order": order, "preloaded": preloaded})
        if self.error:
            raise self.error
        return self.result


class _Trees:
    def __init__(self, centroids):
        self.centroids = centroids


def _base() -> BaseTopics:
    topics = [-1, -1, 0, 0, 0, 1, 1, 1, 2, 2]
    documents = [f"word{t} shared text {i}" for i, t in enumerate(topics)]
    keywords = {t: [(f"word{t}", 1.0)] for t in (-1, 0, 1, 2)}
    ids = [f"i{i}" for i in range(len(topics))]
    return BaseTopics.from_clustering(
        "col", ids, documents, topics, keywords, config={}, preloaded=(ids, np.ones((10, 3)))
    )


def _plan(centroids):
    request = ReductionRequest(strategy="legacy_semantic", n_topics=2)
    return TopicReductionService(duckdb=None, trees=_Trees(centroids)).plan(_base(), request)


def test_reduces_with_centroids_in_sorted_topic_order():
    # Topics 0 and 1 point the same way; 2 elsewhere. Noise row first.
    matrix = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 1.0, 0.0]])
    centroids = _Centroids(result=matrix)

    plan = _plan(centroids)

    assert centroids.calls[0]["order"] == [-1, 0, 1, 2]
    assert centroids.calls[0]["preloaded"] is not None
    assert plan.fallback is None
    assert plan.result.topic_mappings[0] == plan.result.topic_mappings[1]
    assert plan.result.num_topics_after == 2


def test_failure_reports_the_fallback_and_reduces_with_ctfidf():
    centroids = _Centroids(error=ValueError("Could not open ChromaDB collection 'col'"))

    plan = _plan(centroids)

    assert "c-TF-IDF" in plan.fallback
    assert plan.result.num_topics_after == 2
