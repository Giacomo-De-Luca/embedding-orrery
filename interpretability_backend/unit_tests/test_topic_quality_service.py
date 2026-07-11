"""Unit tests for topic-quality scoring: service orchestration + DuckDB persistence.

Service tests use a duck-typed fake DuckDB client (no real DB); persistence
tests use the shared in-memory DuckDBClient fixture from conftest.
"""

import json

import pytest

from backend.services.topic_quality_service import score_topic_quality

COLLECTION = "tq_test_collection"
DATASET = "tq_test_dataset"


# ---------------------------------------------------------------------------
# DuckDB persistence (in-memory client via the shared `db` fixture)
# ---------------------------------------------------------------------------


def _seed_extraction(db) -> str:
    db._conn.execute(
        "INSERT OR REPLACE INTO datasets (name, item_count) VALUES (?, ?)", [DATASET, 3]
    )
    db._conn.execute(
        "INSERT OR REPLACE INTO vector_collections "
        "(collection_name, dataset_name, backend, vector_type) VALUES (?, ?, 'chroma', 'dense')",
        [COLLECTION, DATASET],
    )
    return db.create_topic_extraction(COLLECTION, DATASET, config={"projection_type": "umap_2d"})


def test_quality_metrics_roundtrip_and_level_merge(db):
    extraction_id = _seed_extraction(db)

    db.update_topic_quality_metrics(extraction_id, "topic", {"topic_diversity": 0.9})
    db.update_topic_quality_metrics(extraction_id, "subtopic", {"topic_diversity": 0.7})

    stored = db.get_active_topics(COLLECTION)["quality_metrics"]
    assert stored["topic"]["topic_diversity"] == 0.9
    assert stored["subtopic"]["topic_diversity"] == 0.7

    # Re-scoring one level replaces it but preserves the other.
    db.update_topic_quality_metrics(extraction_id, "topic", {"topic_diversity": 0.5})
    stored = db.get_active_topics(COLLECTION)["quality_metrics"]
    assert stored["topic"]["topic_diversity"] == 0.5
    assert stored["subtopic"]["topic_diversity"] == 0.7


def test_update_quality_metrics_unknown_extraction_raises(db):
    with pytest.raises(ValueError):
        db.update_topic_quality_metrics("no-such-id", "topic", {})


# ---------------------------------------------------------------------------
# Service orchestration (fake duck-typed client)
# ---------------------------------------------------------------------------


class FakeDuckDB:
    """Minimal stand-in exposing the four methods the service touches."""

    def __init__(self, config=None, n_items=40, n_clusters=2, subtopics=False):
        self.config = config
        self.saved = []  # (extraction_id, level, metrics)
        self.projection_requests = []
        ids = [f"item_{i}" for i in range(n_items)]
        self._ids = ids
        # Two square blobs in 2-D so silhouette is well defined.
        self._coords = [
            [i % 5, i // 5] if i < n_items // 2 else [100 + i % 5, 100 + i // 5]
            for i in range(n_items)
        ]
        self._docs = [f"apple banana doc {i}" if i % 3 else None for i in range(n_items)]
        half = n_items // 2
        self._assignments = [
            (item_id, 0 if i < half else (1 if n_clusters >= 2 else 0))
            for i, item_id in enumerate(ids)
        ]
        self._subtopics = subtopics

    def get_active_topics(self, collection_name):
        return {"id": "ext-1", "config": self.config, "topics": [{"topic_id": 0}]}

    def get_projection_data(self, collection_name, projection_type):
        self.projection_requests.append(projection_type)
        return {"ids": self._ids, "documents": self._docs, "coordinates": self._coords}

    def get_topic_assignments_raw(self, extraction_id, columns):
        if columns[1] == "subtopic_id" and not self._subtopics:
            return [(item_id, None) for item_id, _ in self._assignments]
        return self._assignments

    def update_topic_quality_metrics(self, extraction_id, level, metrics):
        self.saved.append((extraction_id, level, metrics))


def test_projection_type_read_from_config_snapshot():
    fake = FakeDuckDB(config=json.dumps({"projection_type": "pca_3d", "cluster_on": "projection"}))
    result = score_topic_quality(COLLECTION, metrics={"silhouette"}, duckdb=fake)
    assert result.get("error") is None
    assert fake.projection_requests == ["pca_3d"]
    assert result["projection_type"] == "pca_3d"
    assert result["cluster_space"] == "projection/pca_3d"
    assert result["silhouette_cluster_space"] is not None


def test_missing_config_defaults_to_umap_2d():
    fake = FakeDuckDB(config=None)
    result = score_topic_quality(COLLECTION, metrics={"silhouette"}, duckdb=fake)
    assert fake.projection_requests == ["umap_2d"]
    assert result.get("error") is None


def test_persists_result_keyed_by_level():
    fake = FakeDuckDB(config=None)
    result = score_topic_quality(COLLECTION, metrics={"silhouette", "diversity"}, duckdb=fake)
    assert result.get("error") is None
    assert len(fake.saved) == 1
    extraction_id, level, metrics = fake.saved[0]
    assert (extraction_id, level) == ("ext-1", "topic")
    assert metrics["silhouette_cluster_space"] == result["silhouette_cluster_space"]


def test_persist_false_skips_write():
    fake = FakeDuckDB(config=None)
    result = score_topic_quality(COLLECTION, metrics={"silhouette"}, persist=False, duckdb=fake)
    assert result.get("error") is None
    assert fake.saved == []


def test_subtopic_level_without_subtopics_errors():
    fake = FakeDuckDB(config=None, subtopics=False)
    result = score_topic_quality(COLLECTION, level="subtopic", duckdb=fake)
    assert "subtopic" in result["error"]
    assert fake.saved == []


def test_no_active_topics_errors_never_raises():
    class Empty:
        def get_active_topics(self, name):
            return None

    result = score_topic_quality(COLLECTION, duckdb=Empty())
    assert "no topics" in result["error"].lower()


def test_invalid_level_errors():
    result = score_topic_quality(COLLECTION, level="bogus", duckdb=FakeDuckDB())
    assert "Invalid level" in result["error"]


def test_mutation_metric_names_in_sync_with_evaluator():
    # The evaluateTopics mutation validates against a literal copy of
    # METRIC_NAMES (it cannot import the evaluation package at module level —
    # lean-import boundary). Keep the two in sync.
    import ast
    from pathlib import Path

    from evaluation.quality_metrics import METRIC_NAMES

    mutations_src = (
        Path(__file__).parents[1] / "backend" / "API" / "mutations.py"
    ).read_text()
    literals = [
        set(ast.literal_eval(node))
        for node in ast.walk(ast.parse(mutations_src))
        if isinstance(node, ast.Set)
        and all(isinstance(el, ast.Constant) for el in node.elts)
        and {getattr(el, "value", None) for el in node.elts} & {"dbcv", "silhouette"}
    ]
    assert literals, "evaluate_topics metric-name literal not found in mutations.py"
    for literal in literals:
        assert literal == set(METRIC_NAMES)
