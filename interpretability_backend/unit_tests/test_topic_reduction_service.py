"""Topic reduction end to end: BaseTopics, TopicReductionService and reduce_existing_topics.

Runs against the in-memory DuckDB fixture with an injected vector source
(no ChromaDB). Four base topics in 3-d: Cats [1, 0, 0] and Dogs (close to
Cats) are animals, Cars [0, 0, 1] and Trucks are vehicles; two noise items.
"""

import json

import numpy as np
import pytest

from backend.services import topic_extraction_service as topic_service
from backend.services.topic_centroid_service import TopicCentroidService
from backend.services.topic_merge_tree_service import TopicMergeTreeService
from backend.services.topic_reduction_service import BaseTopics, TopicReductionService
from backend.services.topic_reduction_types import ReductionRequest
from backend.topic_extraction import llm_labeling
from backend.utils.topic_merge_tree import TopicTreeCut

COLLECTION = "ds_dense"
TOPICS = {  # topic id: (label, direction, members, keyword)
    0: ("Cats", [1.0, 0.0, 0.0], 4, "cats"),
    1: ("Cars", [0.0, 0.0, 1.0], 3, "cars"),
    2: ("Dogs", [0.95, 0.1, 0.0], 3, "dogs"),
    3: ("Trucks", [0.0, 0.4, 0.9], 2, "trucks"),
}
NOISE = ["n0", "n1"]


def _items():
    items = [(f"t{t}_{i}", t) for t, (_, _, n, _) in TOPICS.items() for i in range(n)]
    return items + [(item, -1) for item in NOISE]


def _vector(item: str, topic: int) -> list[float]:
    if topic == -1:
        return [0.3, 0.3, 0.3]
    return TOPICS[topic][1]


def _items_with_noise():
    return [(f"t{t}_{i}", t) for t, (_, _, n, _) in TOPICS.items() for i in range(n)] + [
        (item, -1) for item in NOISE
    ]


def _batches(collection_name, batch_size=5000):
    items = _items()
    yield [i for i, _ in items], np.array([_vector(i, t) for i, t in items])


def _seed(db, collection: str = COLLECTION, topics=TOPICS, config=None) -> str:
    items = [(f"t{t}_{i}", t) for t, (_, _, n, _) in topics.items() for i in range(n)]
    items += [(item, -1) for item in NOISE]
    if db.get_dataset("ds") is None:
        db.create_dataset("ds")
        ids = [i for i, _ in items]
        docs = [
            f"{topics[t][3]} {topics[t][3]} shared text" if t != -1 else "random words"
            for _, t in items
        ]
        db.insert_items_batch("ds", ids, docs, [{} for _ in ids])
    db.register_vector_collection("ds", "chromadb", collection, "dense")
    ext = db.create_topic_extraction(
        collection, "ds", config=config or {"projection_type": "umap_2d", "language": "english"}
    )
    db.insert_topic_info_batch(
        ext,
        [
            {"topic_id": t, "label": label, "count": n, "keywords": [{"word": kw, "score": 1.0}]}
            for t, (label, _, n, kw) in topics.items()
        ]
        + [{"topic_id": -1, "label": "Unclustered", "count": len(NOISE)}],
    )
    db.insert_topic_assignments_batch(
        ext,
        [
            {
                "item_id": item,
                "topic_id": t,
                "topic_label": topics[t][0] if t != -1 else "Unclustered",
            }
            for item, t in items
        ],
    )
    return ext


@pytest.fixture
def wired(db, monkeypatch):
    """reduce_existing_topics on the in-memory DB with the fake vector source."""
    events: list[dict] = []
    monkeypatch.setattr(topic_service, "_get_duckdb", lambda: db)
    monkeypatch.setattr(topic_service, "cancellable_embedding_batches", lambda event: _batches)
    monkeypatch.setattr(topic_service, "emit_progress", lambda **kw: events.append(kw))
    monkeypatch.setattr(
        topic_service,
        "_refresh_topic_centroids",
        lambda collection, ext, preloaded=None: TopicCentroidService(
            duckdb=db, vector_batches=_batches
        ).refresh(collection, ext),
    )
    return events


def _reduce(strategy="average", n_topics=2, **kwargs):
    request = ReductionRequest(strategy=strategy, n_topics=n_topics, **kwargs)
    return topic_service.reduce_existing_topics(COLLECTION, request=request)


def _assignments(db) -> dict[str, tuple]:
    ext = db.get_active_extraction_id(COLLECTION)
    rows = db.get_topic_assignments_raw(
        ext, ["item_id", "topic_id", "topic_label", "subtopic_id", "subtopic_label"]
    )
    return {r[0]: r[1:] for r in rows}


def _trees(db) -> TopicMergeTreeService:
    return TopicMergeTreeService(
        db, centroid_service=TopicCentroidService(db, vector_batches=_batches)
    )


class TestBaseTopics:
    def test_an_unreduced_extraction_is_its_own_base(self, db):
        ext = _seed(db)
        base = BaseTopics.load(db, COLLECTION, _trees(db))

        assert base.extraction_id == ext
        assert base.level == "topic"
        assert base.topic_ids == [0, 1, 2, 3]
        assert base.labels[0] == "Cats" and base.labels[-1] == "Unclustered"
        assert base.keywords[3] == [("trucks", 1.0)]
        assert base.config["projection_type"] == "umap_2d"
        assert base.item_ids[0] == "t0_0"  # item order

    def test_a_reduced_extraction_has_its_subtopics_as_base(self, db, wired):
        _seed(db)
        _reduce(n_topics=2)
        base = BaseTopics.load(db, COLLECTION, _trees(db))

        assert base.level == "subtopic"
        assert base.topic_ids == [0, 1, 2, 3]
        assert base.labels[3] == "Trucks"

    def test_items_of_a_re_embedded_collection_come_from_its_dataset(self, db):
        _seed(db)
        _seed(db, collection="ds_reembedded")  # same dataset "ds", other collection
        base = BaseTopics.load(db, "ds_reembedded", _trees(db))

        assert len(base.item_ids) == 14
        assert base.documents[0].startswith("cats")


class TestReduceExisting:
    def test_every_reduction_starts_from_the_original_clusters(self, db, wired):
        _seed(db)

        first = _reduce(n_topics=2)
        assert first.error is None
        assert first.num_topics == 2 and first.reduction_applied
        assert {row[2] for row in _assignments(db).values()} == {-1, 0, 1, 2, 3}

        second = _reduce(n_topics=3)
        assert second.num_topics_before_reduction == 4
        assert second.num_topics == 3
        assert {row[2] for row in _assignments(db).values()} == {-1, 0, 1, 2, 3}

        back = _reduce(n_topics=4)
        assert back.num_topics == 4 and not back.reduction_applied
        assert {row[2] for row in _assignments(db).values()} == {None}

    def test_the_closest_topics_merge_and_unmerged_ones_keep_their_labels(self, db, wired):
        ext = _seed(db)
        db.update_topic_label(ext, 3, "Lorries")

        result = _reduce(n_topics=3)

        rows = _assignments(db)
        assert rows["t0_0"][0] == rows["t2_0"][0]  # Cats + Dogs
        labels = {info.topic_id: info.label for info in result.topics}
        assert labels[rows["t3_0"][0]] == "Lorries"
        assert labels[rows["t1_0"][0]] == "Cars"
        merged = labels[rows["t0_0"][0]]
        assert "cats" in merged and "dogs" in merged
        info = next(i for i in result.topics if i.topic_id == rows["t0_0"][0])
        assert info.subtopics == ["Cats", "Dogs"]

    def test_llm_relabels_only_merged_topics(self, db, wired, monkeypatch):
        _seed(db)
        labelled: list[int] = []
        monkeypatch.setattr(llm_labeling, "_create_labeler", lambda provider, model: object())

        def _label(topic_id, keywords, sample_documents, labeler):
            labelled.append(topic_id)
            return f"LLM {topic_id}"

        monkeypatch.setattr(llm_labeling, "generate_llm_label_for_topic", _label)

        result = _reduce(n_topics=3, regenerate_labels=True)

        rows = _assignments(db)
        merged = rows["t0_0"][0]
        assert labelled == [merged]
        assert {i.topic_id: i.label for i in result.topics}[merged] == f"LLM {merged}"

    def test_the_config_snapshot_records_the_cut(self, db, wired):
        _seed(db, config={"projection_type": "umap_3d", "cluster_on": "embedding"})
        _reduce(n_topics=2)

        config = db.get_active_topics(COLLECTION)["config"]
        assert '"umap_3d"' in config and '"embedding"' in config
        assert '"strategy": "average"' in config
        assert '"language": "english"' in config

    def test_new_topics_drop_the_cached_colour_options(self, db, wired):
        _seed(db)
        db.update_collection_extra_metadata(COLLECTION, {"field_analysis": "{}", "keep": 1})

        _reduce(n_topics=2)

        extra = db.get_dataset("ds")["extra_metadata"]
        extra = json.loads(extra) if isinstance(extra, str) else extra
        assert "field_analysis" not in extra and extra["keep"] == 1  # None removes the key

    def test_a_failed_sync_keeps_the_previous_extraction(self, db, wired, monkeypatch):
        ext = _seed(db)

        def _fail(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(db, "insert_topic_assignments_batch", _fail)
        result = _reduce(n_topics=2)

        assert "disk full" in result.error
        assert db.get_active_extraction_id(COLLECTION) == ext
        count = db._exec("SELECT count(*) FROM topic_extractions").fetchone()[0]
        assert count == 1

    def test_progress_stages_are_integers(self, db, wired):
        _seed(db)
        _reduce(n_topics=2)

        assert wired and all(isinstance(e["current_batch"], int) for e in wired)
        assert wired[-1]["status"] == "completed"

    def test_legacy_fixed_n_counts_topics_without_noise(self, db, wired):
        _seed(db)
        result = _reduce(strategy="legacy_ctfidf", n_topics=2)

        assert result.error is None
        assert result.num_topics == 2

    def test_legacy_auto_that_merges_nothing_is_not_a_reduction(self, db, wired):
        _seed(db)
        request = ReductionRequest(strategy="legacy_ctfidf", method="auto")
        result = topic_service.reduce_existing_topics(COLLECTION, request=request)

        assert result.error is None
        assert result.num_topics == 4 and not result.reduction_applied
        assert result.num_topics_before_reduction == 4  # noise not counted
        assert {row[2] for row in _assignments(db).values()} == {None}

    def test_a_legacy_no_op_on_a_reduced_base_keeps_labels_and_keywords(self, db, wired):
        _seed(db)
        _reduce(n_topics=2)

        result = _reduce(strategy="legacy_ctfidf", n_topics=4)

        assert not result.reduction_applied
        labels = {info.topic_id: info.label for info in result.topics if info.topic_id != -1}
        assert labels == {0: "Cats", 1: "Cars", 2: "Dogs", 3: "Trucks"}
        assert all(info.keywords for info in result.topics if info.topic_id != -1)

    def test_fewer_than_two_topics_cannot_be_reduced(self, db, wired):
        _seed(db, topics={0: TOPICS[0]})
        assert "at least two topics" in _reduce(n_topics=1).error


class TestDensity:
    def _store_tree(self, db, ext):
        db.replace_topic_merge_tree(
            ext,
            "density",
            leaf_ids=[0, 1, 2, 3],
            leaf_sizes=[4, 3, 3, 2],
            children=[[0, 2], [1, 3], [4, 5]],
            heights=[0.5, 0.7, 2.0],
            noise_item_ids=NOISE,
            noise_merges=[0, 2],  # n0 fell out of Cats+Dogs, n1 of the root
        )

    def test_a_density_cut_absorbs_noise_and_is_reversible(self, db, wired):
        ext = _seed(db)
        self._store_tree(db, ext)

        result = _reduce(strategy="density", n_topics=2, absorb_noise=True)

        assert result.error is None and result.absorbed_noise == 1
        rows = _assignments(db)
        assert rows["n0"][0] == rows["t0_0"][0]  # joined Cats + Dogs
        assert rows["n0"][2] == -1  # still noise at the base level
        assert rows["n1"][0] == -1
        new_ext = db.get_active_extraction_id(COLLECTION)
        assert db.get_topic_merge_tree(new_ext, "density") is not None

        _reduce(strategy="average", n_topics=4)
        rows = _assignments(db)
        assert rows["n0"][0] == -1
        # The reset keeps the base ids, so the copied density tree still lines up.
        assert {item: row[0] for item, row in rows.items()} == {
            item: topic for item, topic in _items_with_noise()
        }
        tree = db.get_topic_merge_tree(db.get_active_extraction_id(COLLECTION), "density")
        assert tree["leaf_sizes"].tolist() == [TOPICS[t][2] for t in (0, 1, 2, 3)]

    def test_density_without_a_tree_fails_clearly(self, db, wired):
        _seed(db)
        assert "re-extract" in _reduce(strategy="density", n_topics=2).error


def test_the_saved_cut_is_the_previewed_one(db, wired):
    _seed(db)
    view = _trees(db).view(COLLECTION, "ward")
    cut = TopicTreeCut(view.tree)
    preview = dict(
        zip(view.tree.leaf_ids.tolist(), cut.groups(cut.merges_for(2)).tolist(), strict=True)
    )

    _reduce(strategy="ward", n_topics=2)

    saved = {row[2]: row[0] for row in _assignments(db).values() if row[2] != -1}
    pairs = set(zip(preview.values(), (saved[t] for t in preview), strict=True))
    assert len(pairs) == len(set(preview.values())) == len(set(saved.values()))


def test_plan_writes_nothing(db):
    _seed(db)
    trees = _trees(db)
    base = BaseTopics.load(db, COLLECTION, trees)

    plan = TopicReductionService(db, trees=trees).plan(
        base, ReductionRequest(strategy="average", n_topics=2)
    )

    assert plan.result.num_topics_after == 2 and plan.changed
    assert db._exec("SELECT count(*) FROM topic_extractions").fetchone()[0] == 1
