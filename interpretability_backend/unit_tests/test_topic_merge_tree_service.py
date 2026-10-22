"""TopicMergeTreeService: merge trees over an extraction's base topics.

Runs against the in-memory DuckDB fixture with centroid rows written directly
(or, for the no-centroids fallback, vectors from an injected batch source).
Topic ids interleave two themes, as in test_topic_order_service: 0 Cats,
1 Cars, 2 Dogs, 3 Trucks (animals near [1, 0, 0], vehicles near [0, 0, 1]);
a cut at two topics must pair Cats with Dogs and Cars with Trucks.
"""

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from backend.API import queries, schema
from backend.API.read_only import READ_ONLY_ENV
from backend.services.topic_centroid_service import TopicCentroidService
from backend.services.topic_merge_tree_service import TopicMergeTreeService
from backend.services.topic_reduction_types import SETTINGS
from backend.utils.topic_merge_tree import TopicTreeCut

COLLECTION = "ds_dense"
TOPICS = {  # topic_id: (label, centroid, members)
    0: ("Cats", [1.0, 0.0, 0.0], 3),
    1: ("Cars", [0.0, 0.0, 1.0], 2),
    2: ("Dogs", [0.9, 0.1, 0.0], 2),
    3: ("Trucks", [0.0, 0.1, 0.9], 1),
    -1: ("Unclustered", [0.3, 0.3, 0.3], 2),
}


def _items():
    """(item_id, topic_id) for every member of TOPICS."""
    return [(f"t{tid}_{i}", tid) for tid, (_, _, n) in TOPICS.items() for i in range(n)]


def _seed(db, centroids: bool = True, subtopics: bool = False) -> str:
    items = _items()
    db.create_dataset("ds")
    ids = [item for item, _ in items]
    db.insert_items_batch("ds", ids, ids, [{} for _ in ids])
    db.register_vector_collection("ds", "chromadb", COLLECTION, "dense")
    ext = db.create_topic_extraction(COLLECTION, "ds")
    rows = []
    for item, tid in items:
        row = {"item_id": item, "topic_id": tid, "topic_label": TOPICS[tid][0]}
        if subtopics:  # a reduced extraction: everything in topic 0, base topics as subtopics
            row = {
                "item_id": item,
                "topic_id": -1 if tid == -1 else 0,
                "topic_label": "Unclustered" if tid == -1 else "All",
                "subtopic_id": tid,
                "subtopic_label": TOPICS[tid][0],
            }
        rows.append(row)
    db.insert_topic_assignments_batch(ext, rows)
    if centroids:
        level = "subtopic" if subtopics else "topic"
        records = [
            {"level": level, "topic_id": tid, "member_count": n, "centroid": vec}
            for tid, (_, vec, n) in TOPICS.items()
        ]
        if subtopics:
            records += [
                {"level": "topic", "topic_id": 0, "member_count": 8, "centroid": [0.5, 0.0, 0.5]},
                {"level": "topic", "topic_id": -1, "member_count": 2, "centroid": [0.3, 0.3, 0.3]},
            ]
        db.replace_topic_centroids(ext, records)
    return ext


def _vector_batches(calls):
    def batches(collection_name, batch_size=5000):
        calls.append(collection_name)
        items = _items()
        yield [i for i, _ in items], np.array([TOPICS[t][1] for _, t in items])

    return batches


def _service(db, calls=None, **settings) -> TopicMergeTreeService:
    centroids = TopicCentroidService(
        duckdb=db, vector_batches=_vector_batches([] if calls is None else calls)
    )
    return TopicMergeTreeService(
        duckdb=db, centroid_service=centroids, settings=replace(SETTINGS, **settings)
    )


def _pairs_at_two(view) -> set[frozenset[str]]:
    cut = TopicTreeCut(view.tree)
    groups = cut.groups(cut.merges_for(2)).tolist()
    labels = [view.labels[t] for t in view.tree.leaf_ids.tolist()]
    return {
        frozenset(lab for lab, g in zip(labels, groups, strict=True) if g == group)
        for group in set(groups)
    }


THEMES = {frozenset({"Cats", "Dogs"}), frozenset({"Cars", "Trucks"})}


class TestSemanticTrees:
    @pytest.mark.parametrize("linkage", ["average", "ward"])
    def test_tree_over_the_topics_of_an_unreduced_extraction(self, db, linkage):
        ext = _seed(db)
        view = _service(db).view(COLLECTION, linkage)

        assert view.extraction_id == ext
        assert view.base_level == "topic"
        assert view.tree.linkage == linkage
        assert view.tree.leaf_ids.tolist() == [0, 1, 2, 3]
        assert view.tree.leaf_sizes.tolist() == [3, 2, 2, 1]
        assert view.noise_count == 2
        assert view.available_linkages == ("average", "ward")
        assert _pairs_at_two(view) == THEMES

    def test_a_reduced_extraction_uses_its_subtopics_as_base(self, db):
        _seed(db, subtopics=True)
        view = _service(db).view(COLLECTION, "average")

        assert view.base_level == "subtopic"
        assert view.labels == {0: "Cats", 1: "Cars", 2: "Dogs", 3: "Trucks"}
        assert _pairs_at_two(view) == THEMES

    def test_labels_follow_renames(self, db):
        ext = _seed(db)
        service = _service(db)
        service.view(COLLECTION, "average")
        db.update_topic_label(ext, 0, "Felines")
        assert service.view(COLLECTION, "average").labels[0] == "Felines"


class TestFallback:
    def test_without_stored_centroids_the_vectors_are_read_once(self, db):
        _seed(db, centroids=False)
        calls: list[str] = []
        service = _service(db, calls)

        first = service.view(COLLECTION, "average")
        second = service.view(COLLECTION, "ward")

        assert _pairs_at_two(first) == THEMES
        assert _pairs_at_two(second) == THEMES
        assert calls == [COLLECTION]

    def test_fallback_is_capped(self, db):
        _seed(db, centroids=False)
        assert _service(db, fallback_centroid_max_items=3).view(COLLECTION, "average") is None


class TestDensity:
    def _store(self, db, ext, leaf_ids=(0, 1, 2, 3)):
        db.replace_topic_merge_tree(
            ext,
            "density",
            leaf_ids=list(leaf_ids),
            leaf_sizes=[3, 2, 2, 1][: len(leaf_ids)],
            children=[[0, 2], [1, 3], [4, 5]][: len(leaf_ids) - 1],
            heights=[0.5, 0.7, 2.0][: len(leaf_ids) - 1],
            noise_item_ids=["t-1_0", "t-1_1"],
            noise_merges=[0, 2][: len(leaf_ids) - 1],
        )

    def test_listed_and_served_when_its_leaves_match(self, db):
        ext = _seed(db)
        self._store(db, ext)
        service = _service(db)

        assert service.view(COLLECTION, "average").available_linkages == (
            "average",
            "ward",
            "density",
        )
        view = service.view(COLLECTION, "density")
        assert view.tree.linkage == "density"
        assert view.tree.noise_merges == {"t-1_0": 0, "t-1_1": 2}
        assert _pairs_at_two(view) == THEMES

    def test_the_stored_tree_is_read_once(self, db, monkeypatch):
        ext = _seed(db)
        self._store(db, ext)
        reads: list[str] = []
        real = db.get_topic_merge_tree
        monkeypatch.setattr(
            db, "get_topic_merge_tree", lambda e, linkage: reads.append(e) or real(e, linkage)
        )
        service = _service(db)

        service.view(COLLECTION, "average")
        service.view(COLLECTION, "ward")
        service.view(COLLECTION, "density")

        assert reads == [ext]

    def test_a_tree_over_other_topics_is_ignored(self, db):
        ext = _seed(db)
        self._store(db, ext, leaf_ids=(0, 1))
        service = _service(db)

        assert "density" not in service.view(COLLECTION, "average").available_linkages
        assert service.view(COLLECTION, "density") is None


class TestUnavailable:
    def test_collection_without_topics(self, db):
        _seed(db)
        assert _service(db).view("elsewhere", "average") is None

    def test_too_many_topics(self, db):
        _seed(db)
        assert _service(db, max_tree_topics=3).view(COLLECTION, "average") is None

    def test_too_few_topics(self, db):
        ext = _seed(db)
        db.replace_topic_centroids(
            ext,
            [
                {"level": "topic", "topic_id": t, "member_count": 1, "centroid": TOPICS[t][1]}
                for t in (0, 1)
            ],
        )
        assert _service(db).view(COLLECTION, "average") is None

    def test_unknown_linkage_raises(self, db):
        _seed(db)
        with pytest.raises(ValueError, match="linkage"):
            _service(db).view(COLLECTION, "single")


def test_an_extraction_switching_mid_call_is_not_cached(db, monkeypatch):
    ext = _seed(db)
    calls: list[str] = []
    service = _service(db, calls)
    ids = iter([ext, "newer-extraction"])
    real = db.get_active_extraction_id
    monkeypatch.setattr(db, "get_active_extraction_id", lambda name: next(ids, real(name)))

    assert service.view(COLLECTION, "average") is None
    assert service.view(COLLECTION, "average") is not None


class TestTreeForAssignments:
    def test_matches_the_served_tree(self, db):
        ext = _seed(db)
        service = _service(db)
        served = service.view(COLLECTION, "ward").tree
        built = service.tree_for_assignments(COLLECTION, ext, dict(_items()), "ward")

        assert built.children.tolist() == served.children.tolist()
        np.testing.assert_allclose(built.heights, served.heights)

    def test_inline_assignments_use_the_given_vectors(self, db):
        items = _items()
        ids = [i for i, _ in items]
        vectors = np.array([TOPICS[t][1] for _, t in items])
        service = _service(db)

        tree = service.tree_for_assignments(
            COLLECTION, None, dict(items), "average", preloaded=(ids, vectors)
        )
        assert tree.leaf_ids.tolist() == [0, 1, 2, 3]
        cut = TopicTreeCut(tree)
        groups = cut.groups(cut.merges_for(2)).tolist()
        assert groups[0] == groups[2] and groups[1] == groups[3]

    def test_topics_without_vectors_raise(self, db):
        ext = _seed(db)
        assignments = dict(_items())
        assignments["extra"] = 7  # assigned, but no centroid row
        with pytest.raises(ValueError, match="without vectors"):
            _service(db).tree_for_assignments(COLLECTION, ext, assignments, "average")


TREE_QUERY = """
query ($c: String!, $l: String!, $n: Boolean!) {
  topicMergeTree(collectionName: $c, linkage: $l, includeNoise: $n) {
    extractionId linkage baseLevel noiseCount availableLinkages
    leaves { topicId label size }
    merges { left right height size }
    noiseItemIds noiseMerges
  }
}
"""


class TestGraphQL:
    @pytest.fixture(autouse=True)
    def seeded(self, db, monkeypatch):
        self.ext = _seed(db)
        TestDensity()._store(db, self.ext)
        monkeypatch.setattr(queries, "get_duckdb_client", lambda: db)
        monkeypatch.setattr(TopicMergeTreeService, "_shared", _service(db))

    def _execute(self, **variables):
        result = asyncio.run(schema.execute(TREE_QUERY, variable_values=variables))
        assert result.errors is None, result.errors
        return result.data["topicMergeTree"]

    def test_returns_the_tree(self):
        tree = self._execute(c=COLLECTION, l="average", n=False)

        assert tree["extractionId"] == self.ext
        assert tree["baseLevel"] == "topic"
        assert [leaf["label"] for leaf in tree["leaves"]] == ["Cats", "Cars", "Dogs", "Trucks"]
        assert [leaf["size"] for leaf in tree["leaves"]] == [3, 2, 2, 1]
        assert len(tree["merges"]) == 3
        assert tree["merges"][-1]["size"] == 8
        assert tree["noiseCount"] == 2
        assert tree["noiseItemIds"] is None

    def test_density_noise_on_request(self):
        tree = self._execute(c=COLLECTION, l="density", n=True)
        assert tree["noiseItemIds"] == ["t-1_0", "t-1_1"]
        assert tree["noiseMerges"] == [0, 2]

    def test_noise_is_only_sent_for_density(self):
        tree = self._execute(c=COLLECTION, l="average", n=True)
        assert tree["noiseItemIds"] is None and tree["noiseMerges"] is None

    def test_null_without_topics(self):
        assert self._execute(c="elsewhere", l="average", n=False) is None

    def test_unknown_linkage_is_a_graphql_error(self):
        result = asyncio.run(
            schema.execute(TREE_QUERY, variable_values={"c": COLLECTION, "l": "x", "n": False})
        )
        assert result.errors and "linkage" in str(result.errors[0])

    def test_works_in_read_only_mode(self, monkeypatch):
        monkeypatch.setenv(READ_ONLY_ENV, "1")
        assert self._execute(c=COLLECTION, l="average", n=False) is not None
