"""Semantic topic reduction falls back to c-TF-IDF when centroids are unavailable.

Covers ``topic_extraction_service._semantic_topic_centroids``: on success it
returns the centroid matrix; on failure it reports the fallback on the job's
progress stream and returns None, so the caller reduces with c-TF-IDF instead
of failing (or, as before the fix, silently merging zero vectors).
"""

import numpy as np

from backend.services import topic_extraction_service


class _Service:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        self.calls = []

    def __call__(self, duckdb):
        return self

    def centroids_for_assignments(self, collection_name, item_topics, order, preloaded=None):
        self.calls.append({"order": order, "preloaded": preloaded})
        if self.error:
            raise self.error
        return self.result


def _call(monkeypatch, service, **overrides):
    events = []
    monkeypatch.setattr(topic_extraction_service, "TopicCentroidService", service)
    monkeypatch.setattr(topic_extraction_service, "_get_duckdb", lambda: None)
    monkeypatch.setattr(
        topic_extraction_service, "emit_progress", lambda **kwargs: events.append(kwargs)
    )
    kwargs = {
        "job_id": "col",
        "total_items": 10,
        "current_batch": 2,
        "total_batches": 5,
        **overrides,
    }
    result = topic_extraction_service._semantic_topic_centroids(
        "col", {"a": 0, "b": 1}, {1: [], -1: [], 0: []}, **kwargs
    )
    return result, events


def test_returns_centroids_in_sorted_topic_order(monkeypatch):
    matrix = np.eye(3)
    service = _Service(result=matrix)

    result, events = _call(monkeypatch, service, preloaded=(["a", "b"], np.ones((2, 3))))

    assert result is matrix
    assert service.calls[0]["order"] == [-1, 0, 1]
    assert service.calls[0]["preloaded"] is not None
    assert events == []


def test_failure_reports_the_fallback_and_returns_none(monkeypatch):
    service = _Service(error=ValueError("Could not open ChromaDB collection 'col'"))

    result, events = _call(monkeypatch, service)

    assert result is None
    assert len(events) == 1
    assert events[0]["job_id"] == "col"
    assert events[0]["current_batch"] == 2
    assert "c-TF-IDF" in events[0]["message"]
