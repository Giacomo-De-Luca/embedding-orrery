"""TopicCentroidService: centroid refresh from DuckDB assignments + topic search.

Runs against the in-memory DuckDB fixture with injected vector batches and a
stub query embedder, so no ChromaDB or embedding model is touched.

Fixture geometry (2-d). Member vectors are L2-normalised before averaging,
so only their directions matter:

    a [2, 0]   -> [1, 0]       topic 0, subtopic 10 "Lions"
    b [0, 2]   -> [0, 1]       topic 0, subtopic 11 "Tabbies"
    c [-2, -2] -> [-r, -r]     topic 1, subtopic 12 "Pugs"   (r = 1/sqrt(2))
    d [0, 0]   -> [0, 0]       noise (-1)
"""

import numpy as np
import pytest

from backend.clients.duckdb_client import DuckDBClient
from backend.services import topic_centroid_service
from backend.services.topic_centroid_service import TopicCentroidService, TopicMatch
from backend.utils.topic_centroids import CENTER_BY_DEFAULT

COLLECTION = "ds_dense"
R = 1 / np.sqrt(2)
TOPIC_MEANS = [[0, 0], [0.5, 0.5], [-R, -R]]
VECTORS = {
    "a": [2.0, 0.0],
    "b": [0.0, 2.0],
    "c": [-2.0, -2.0],
    "d": [0.0, 0.0],
    "z": [9.0, 9.0],  # in ChromaDB but never assigned to a topic: must be skipped
}


def _seed(db: DuckDBClient, with_subtopics: bool = True) -> str:
    db.create_dataset("ds")
    db.insert_items_batch("ds", list("abcd"), list("abcd"), [{}, {}, {}, {}])
    db.register_vector_collection("ds", "chromadb", COLLECTION, "dense")
    db.register_vector_collection("ds", "chromadb", "ds_no_topics", "dense")
    ext = db.create_topic_extraction(COLLECTION, "ds")
    rows = [
        ("a", 0, "Cats", 10, "Lions"),
        ("b", 0, "Cats", 11, "Tabbies"),
        ("c", 1, "Dogs", 12, "Pugs"),
        ("d", -1, "Unclustered", -1, "Unclustered"),
    ]
    db.insert_topic_assignments_batch(
        ext,
        [
            {
                "item_id": item,
                "topic_id": topic,
                "topic_label": label,
                **({"subtopic_id": sub, "subtopic_label": sub_label} if with_subtopics else {}),
            }
            for item, topic, label, sub, sub_label in rows
        ],
    )
    return ext


def _batches(collection_name: str):
    assert collection_name == COLLECTION
    ids = list(VECTORS)
    matrix = np.array([VECTORS[i] for i in ids])
    yield ids[:2], matrix[:2]
    yield ids[2:], matrix[2:]


def _no_batches(collection_name: str):
    raise AssertionError("vector store must not be read when vectors are preloaded")


def _service(db, embed=None, batches=_batches) -> TopicCentroidService:
    return TopicCentroidService(duckdb=db, embed_query=embed, vector_batches=batches)


class TestRefresh:
    def test_writes_topic_and_subtopic_levels(self, db):
        ext = _seed(db)
        written = _service(db).refresh(COLLECTION, ext)

        topics = db.get_topic_centroids(ext, "topic")
        assert topics["ids"].tolist() == [-1, 0, 1]
        assert topics["counts"].tolist() == [1, 2, 1]
        np.testing.assert_allclose(topics["means"], TOPIC_MEANS, atol=1e-6)

        subtopics = db.get_topic_centroids(ext, "subtopic")
        assert subtopics["ids"].tolist() == [-1, 10, 11, 12]
        assert subtopics["parents"].tolist() == [-1, 0, 0, 1]
        np.testing.assert_allclose(subtopics["means"][1:], [[1, 0], [0, 1], [-R, -R]], atol=1e-6)

        assert written == 7

    def test_without_subtopics_writes_topic_level_only(self, db):
        ext = _seed(db, with_subtopics=False)
        _service(db).refresh(COLLECTION, ext)

        assert db.get_topic_centroids(ext, "topic")["ids"].tolist() == [-1, 0, 1]
        assert db.get_topic_centroids(ext, "subtopic") is None

    def test_preloaded_vectors_skip_the_vector_store(self, db):
        ext = _seed(db)
        ids = list("abcd")
        matrix = np.array([VECTORS[i] for i in ids])

        _service(db, batches=_no_batches).refresh(COLLECTION, ext, preloaded=(ids, matrix))

        topics = db.get_topic_centroids(ext, "topic")
        np.testing.assert_allclose(topics["means"], TOPIC_MEANS, atol=1e-6)

    def test_preloaded_vectors_are_processed_in_chunks(self, db, monkeypatch):
        # Chunking bounds the per-batch copies; results must not depend on it.
        monkeypatch.setattr(topic_centroid_service, "PRELOADED_BATCH_SIZE", 1)
        ext = _seed(db)
        ids = list("abcd")
        matrix = np.array([VECTORS[i] for i in ids])

        _service(db, batches=_no_batches).refresh(COLLECTION, ext, preloaded=(ids, matrix))

        topics = db.get_topic_centroids(ext, "topic")
        np.testing.assert_allclose(topics["means"], TOPIC_MEANS, atol=1e-6)

    def test_extraction_without_assignments_raises(self, db):
        _seed(db)
        empty = db.create_topic_extraction(COLLECTION, "ds")
        with pytest.raises(ValueError, match="no topic assignments"):
            _service(db).refresh(COLLECTION, empty)


class TestCentroidsForAssignments:
    def test_rows_follow_requested_order(self, db):
        means = _service(db).centroids_for_assignments(
            COLLECTION, {"a": 0, "b": 0, "c": 1, "d": -1}, order=[-1, 0, 1]
        )
        np.testing.assert_allclose(means, TOPIC_MEANS, atol=1e-6)

    def test_label_without_members_raises(self, db):
        with pytest.raises(ValueError, match="no members"):
            _service(db).centroids_for_assignments(COLLECTION, {"a": 0}, order=[0, 1])


class TestActiveCentroids:
    def test_returns_both_levels_of_the_active_extraction(self, db):
        ext = _seed(db)
        service = _service(db)
        service.refresh(COLLECTION, ext)

        topics, subtopics = service.active_centroids(COLLECTION)

        assert topics.ids.tolist() == [-1, 0, 1]
        np.testing.assert_allclose(topics.means, TOPIC_MEANS, atol=1e-6)
        assert subtopics.parents.tolist() == [-1, 0, 0, 1]

    def test_missing_centroids_raise(self, db):
        _seed(db)
        with pytest.raises(ValueError, match="not been computed"):
            _service(db).active_centroids(COLLECTION)

    def test_default_ranker_uses_the_shared_centring_default(self, db):
        assert _service(db).ranker.center is CENTER_BY_DEFAULT


class TestSearch:
    def test_ranks_topics_and_names_the_matching_subtopic(self, db):
        ext = _seed(db)
        calls = []

        def embed(collection_name, text, query_prompt=None):
            calls.append((collection_name, text, query_prompt))
            # Not unit length: the query is normalised like the members were.
            return np.array([0.0, 3.0])

        service = _service(db, embed=embed)
        service.refresh(COLLECTION, ext)
        matches = service.search(COLLECTION, "tabby cats", limit=5, query_prompt="Retrieval-query")

        assert calls == [(COLLECTION, "tabby cats", "Retrieval-query")]
        assert [m.topic_id for m in matches] == [0, 1]
        assert matches[0] == TopicMatch(
            topic_id=0,
            similarity=pytest.approx(1.0),
            matched_subtopic_id=11,
            matched_subtopic_label="Tabbies",
        )
        assert matches[1].matched_subtopic_id is None
        assert matches[1].matched_subtopic_label is None

    def test_missing_centroids_raise_a_clear_error(self, db):
        _seed(db)
        service = _service(db, embed=lambda *a, **k: np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="not been computed"):
            service.search(COLLECTION, "anything")

    def test_collection_without_topics_raises(self, db):
        _seed(db)
        service = _service(db, embed=lambda *a, **k: np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="no topics"):
            service.search("ds_no_topics", "anything")

    def test_query_dimension_mismatch_raises(self, db):
        ext = _seed(db)
        service = _service(db, embed=lambda *a, **k: np.zeros(3))
        service.refresh(COLLECTION, ext)
        with pytest.raises(ValueError, match="dimension"):
            service.search(COLLECTION, "anything")
