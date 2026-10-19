"""TopicOrderService: topic and subtopic labels ordered by stored centroid similarity.

Runs against the in-memory DuckDB fixture with centroid rows written directly,
so the geometry is exact. Topic ids are assigned so that id order interleaves
the two themes: 0 Cats, 1 Cars, 2 Dogs, 3 Trucks (animals near [1, 0, 0],
vehicles near [0, 0, 1]). A similarity order must pair Cats with Dogs and
Cars with Trucks.
"""

import asyncio

import pytest

from backend.API import queries, schema
from backend.API.read_only import READ_ONLY_ENV
from backend.clients.duckdb_client import DuckDBClient
from backend.services.topic_centroid_service import TopicCentroidService
from backend.services.topic_order_service import TopicOrderService

COLLECTION = "ds_dense"
TOPICS = {  # topic_id: (label, centroid)
    0: ("Cats", [1.0, 0.0, 0.0]),
    1: ("Cars", [0.0, 0.0, 1.0]),
    2: ("Dogs", [0.9, 0.1, 0.0]),
    3: ("Trucks", [0.0, 0.1, 0.9]),
    -1: ("Unclustered", [0.3, 0.3, 0.3]),
}
SUBTOPICS = {  # subtopic_id: (label, parent topic, centroid)
    10: ("Lions", 0, [1.0, 0.0, 0.1]),
    11: ("Tabbies", 0, [1.0, 0.1, 0.0]),
    12: ("Pugs", 2, [0.9, 0.1, 0.0]),
    13: ("Sedans", 1, [0.0, 0.0, 1.0]),
    14: ("Lorries", 3, [0.0, 0.1, 0.9]),
    15: ("Pickups", 3, [0.1, 0.1, 0.9]),
}


def _seed(db: DuckDBClient, centroids: bool = True) -> str:
    db.create_dataset("ds")
    item_ids = [f"i{sub}" for sub in SUBTOPICS] + ["noise"]
    db.insert_items_batch("ds", item_ids, item_ids, [{} for _ in item_ids])
    db.register_vector_collection("ds", "chromadb", COLLECTION, "dense")
    db.register_vector_collection("ds", "chromadb", "ds_no_topics", "dense")
    ext = db.create_topic_extraction(COLLECTION, "ds")
    assignments = [
        {
            "item_id": f"i{sub}",
            "topic_id": parent,
            "topic_label": TOPICS[parent][0],
            "subtopic_id": sub,
            "subtopic_label": label,
        }
        for sub, (label, parent, _) in SUBTOPICS.items()
    ]
    assignments.append(
        {
            "item_id": "noise",
            "topic_id": -1,
            "topic_label": "Unclustered",
            "subtopic_id": -1,
            "subtopic_label": "Unclustered",
        }
    )
    db.insert_topic_assignments_batch(ext, assignments)
    if centroids:
        rows = [
            {"level": "topic", "topic_id": tid, "member_count": 1, "centroid": vec}
            for tid, (_, vec) in TOPICS.items()
        ]
        rows += [
            {
                "level": "subtopic",
                "topic_id": sub,
                "parent_topic_id": parent,
                "member_count": 1,
                "centroid": vec,
            }
            for sub, (_, parent, vec) in SUBTOPICS.items()
        ]
        db.replace_topic_centroids(ext, rows)
    return ext


def _adjacent(order: list[str], a: str, b: str) -> bool:
    return abs(order.index(a) - order.index(b)) == 1


class CountingCentroidService(TopicCentroidService):
    """Counts centroid loads, to observe the order cache."""

    def __init__(self, duckdb):
        super().__init__(duckdb=duckdb)
        self.loads = 0

    def active_centroids(self, collection_name):
        self.loads += 1
        return super().active_centroids(collection_name)


class TestTopicOrder:
    def test_orders_topic_labels_by_similarity(self, db):
        _seed(db)
        order = TopicOrderService(duckdb=db).value_order(COLLECTION, "topic_label")
        assert sorted(order) == ["Cars", "Cats", "Dogs", "Trucks"]
        assert _adjacent(order, "Cats", "Dogs")
        assert _adjacent(order, "Cars", "Trucks")

    def test_orders_topic_ids_as_the_strings_items_carry(self, db):
        _seed(db)
        service = TopicOrderService(duckdb=db)
        labels = service.value_order(COLLECTION, "topic_label")
        ids = service.value_order(COLLECTION, "topic_id")
        by_label = {label: str(tid) for tid, (label, _) in TOPICS.items()}
        assert ids == [by_label[label] for label in labels]

    def test_groups_subtopics_under_their_topics_in_topic_order(self, db):
        _seed(db)
        service = TopicOrderService(duckdb=db)
        topic_order = service.value_order(COLLECTION, "topic_label")
        sub_order = service.value_order(COLLECTION, "subtopic_label")
        parent_label = {label: TOPICS[parent][0] for label, parent, _ in SUBTOPICS.values()}
        parents = [parent_label[label] for label in sub_order]
        assert sorted(sub_order) == sorted(label for label, _, _ in SUBTOPICS.values())
        # One contiguous block per parent, blocks in topic order.
        assert list(dict.fromkeys(parents)) == topic_order
        assert len(parents) - sum(a == b for a, b in zip(parents, parents[1:], strict=False)) == 4

    def test_labels_follow_renames_without_recomputing(self, db):
        ext = _seed(db)
        centroids = CountingCentroidService(db)
        service = TopicOrderService(duckdb=db, centroid_service=centroids)
        before = service.value_order(COLLECTION, "topic_label")
        db.update_topic_label(ext, 0, "Felines")
        after = service.value_order(COLLECTION, "topic_label")
        assert after == ["Felines" if label == "Cats" else label for label in before]
        assert centroids.loads == 1

    def test_a_new_extraction_is_recomputed(self, db):
        ext = _seed(db)
        centroids = CountingCentroidService(db)
        service = TopicOrderService(duckdb=db, centroid_service=centroids)
        service.value_order(COLLECTION, "topic_label")
        new_ext = db.create_topic_extraction(COLLECTION, "ds")
        assert new_ext != ext
        assert service.value_order(COLLECTION, "topic_label") is None  # no centroids yet
        assert centroids.loads == 2


def test_an_extraction_switching_mid_call_is_not_cached(db, monkeypatch):
    """The centroids loaded may belong to a newer extraction than the id read first."""
    ext = _seed(db)
    # Reads: value_order, then active_centroids (still `ext`, valid centroids), then the re-check.
    ids = iter([ext, ext, "newer-extraction"])
    real = db.get_active_extraction_id
    monkeypatch.setattr(db, "get_active_extraction_id", lambda name: next(ids, real(name)))
    centroids = CountingCentroidService(db)
    service = TopicOrderService(duckdb=db, centroid_service=centroids)
    assert service.value_order(COLLECTION, "topic_label") is None
    # Nothing was cached under the stale id: the next call loads the centroids again.
    assert service.value_order(COLLECTION, "topic_label") is not None
    assert centroids.loads == 2


class TestUnavailable:
    def test_non_topic_field_has_no_order(self, db):
        _seed(db)
        assert TopicOrderService(duckdb=db).value_order(COLLECTION, "source") is None

    def test_collection_without_topics_has_no_order(self, db):
        _seed(db)
        assert TopicOrderService(duckdb=db).value_order("ds_no_topics", "topic_label") is None

    def test_missing_centroids_give_no_order(self, db):
        _seed(db, centroids=False)
        assert TopicOrderService(duckdb=db).value_order(COLLECTION, "topic_label") is None

    def test_subtopic_field_without_subtopic_centroids_has_no_order(self, db):
        ext = _seed(db)
        db.replace_topic_centroids(
            ext,
            [
                {"level": "topic", "topic_id": tid, "member_count": 1, "centroid": vec}
                for tid, (_, vec) in TOPICS.items()
            ],
        )
        assert TopicOrderService(duckdb=db).value_order(COLLECTION, "subtopic_label") is None


class TestShared:
    def test_reuses_one_instance_per_database_client(self, db):
        assert TopicOrderService.shared(db) is TopicOrderService.shared(db)

    def test_replaces_the_instance_for_another_client(self, db):
        first = TopicOrderService.shared(db)
        other = DuckDBClient(db_path=":memory:")
        try:
            assert TopicOrderService.shared(other) is not first
        finally:
            other.close()


@pytest.mark.parametrize("field", ["topic_label", "subtopic_label"])
def test_orders_exclude_the_noise_topic(db, field):
    _seed(db)
    assert "Unclustered" not in TopicOrderService(duckdb=db).value_order(COLLECTION, field)


ORDER_QUERY = """
query ($c: String!, $f: String!) { topicSimilarityOrder(collectionName: $c, field: $f) }
"""


class TestGraphQL:
    @pytest.fixture(autouse=True)
    def seeded(self, db, monkeypatch):
        _seed(db)
        monkeypatch.setattr(queries, "get_duckdb_client", lambda: db)

    def _execute(self, **variables):
        result = asyncio.run(schema.execute(ORDER_QUERY, variable_values=variables))
        assert result.errors is None, result.errors
        return result.data["topicSimilarityOrder"]

    def test_returns_the_order_of_a_topic_field(self):
        order = self._execute(c=COLLECTION, f="topic_label")
        assert sorted(order) == ["Cars", "Cats", "Dogs", "Trucks"]

    def test_returns_null_for_other_fields(self):
        assert self._execute(c=COLLECTION, f="source") is None

    def test_works_in_read_only_mode(self, monkeypatch):
        monkeypatch.setenv(READ_ONLY_ENV, "1")
        assert self._execute(c=COLLECTION, f="topic_label") is not None
