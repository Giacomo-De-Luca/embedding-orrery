"""ReductionComparison: every strategy x cut scored on one collection, nothing stored.

In-memory DuckDB with projections and a ground-truth metadata field, vectors
from an injected batch source. Four base topics: Cats and Dogs (animals),
Cars and Trucks (vehicles); cut at two topics, the semantic strategies must
recover the animal / vehicle split exactly.
"""

import numpy as np
import pytest

from interpretability_backend.backend.services.topic_centroid_service import TopicCentroidService
from interpretability_backend.backend.services.topic_merge_tree_service import (
    TopicMergeTreeService,
)
from interpretability_backend.backend.services.topic_reduction_service import (
    TopicReductionService,
)
from interpretability_backend.evaluation.reduction_comparison import (
    ComparisonCase,
    PartitionStats,
    ReductionComparison,
)
from interpretability_backend.evaluation.run_reduction_comparison import markdown_report

COLLECTION = "ds_dense"
TOPICS = {  # id: (label, direction, members, kind)
    0: ("Cats", [1.0, 0.0, 0.0], 4, "animal"),
    1: ("Cars", [0.0, 0.0, 1.0], 3, "vehicle"),
    2: ("Dogs", [0.95, 0.1, 0.0], 3, "animal"),
    3: ("Trucks", [0.0, 0.4, 0.9], 2, "vehicle"),
}


def _items():
    items = [(f"t{t}_{i}", t) for t, (_, _, n, _) in TOPICS.items() for i in range(n)]
    return items + [("noise0", -1), ("noise1", -1)]


def _batches(collection_name, batch_size=5000):
    items = _items()
    vectors = [TOPICS[t][1] if t != -1 else [0.3, 0.3, 0.3] for _, t in items]
    yield [i for i, _ in items], np.array(vectors)


@pytest.fixture
def seeded(db):
    items = _items()
    ids = [i for i, _ in items]
    words = {0: "cats", 1: "cars", 2: "dogs", 3: "trucks", -1: "random"}
    db.create_dataset("ds")
    db.insert_items_batch(
        "ds",
        ids,
        [f"{words[t]} {words[t]} text" for _, t in items],
        [{"kind": TOPICS[t][3]} if t != -1 else {} for _, t in items],
    )
    db.register_vector_collection("ds", "chromadb", COLLECTION, "dense")
    db.insert_projections_batch(
        COLLECTION, ids, "umap_2d", [[float(t), float(i % 3)] for i, (_, t) in enumerate(items)]
    )
    ext = db.create_topic_extraction(COLLECTION, "ds", config={"projection_type": "umap_2d"})
    db.insert_topic_assignments_batch(
        ext,
        [
            {
                "item_id": item,
                "topic_id": t,
                "topic_label": TOPICS[t][0] if t != -1 else "Unclustered",
            }
            for item, t in items
        ],
    )
    return db


def _comparison(db) -> ReductionComparison:
    trees = TopicMergeTreeService(
        db, centroid_service=TopicCentroidService(duckdb=db, vector_batches=_batches)
    )
    return ReductionComparison(
        db, TopicReductionService(db, trees=trees), metrics={"diversity"}, n_keywords=3
    )


CASES = [
    ComparisonCase("average", n_topics=2),
    ComparisonCase("ward", n_topics=2),
    ComparisonCase("density", n_topics=2),
    ComparisonCase("average", method="auto", distance_threshold=0.001),
    ComparisonCase("average", method="auto", distance_threshold=2.0),
    ComparisonCase("legacy_ctfidf", method="auto"),
    ComparisonCase("average", n_topics=9),
]


class TestCompare:
    def test_scores_every_case(self, seeded):
        result = _comparison(seeded).compare(COLLECTION, CASES, ground_truth_field="kind")

        rows = {row["case"]: row for row in result["rows"]}
        assert result["base_topics"] == 4
        assert rows["base topics"]["n_topics"] == 4
        for name in ("average N=2", "ward N=2"):
            assert rows[name]["n_topics"] == 2
            assert rows[name]["nmi"] == pytest.approx(1.0)
            assert rows[name]["ari"] == pytest.approx(1.0)
            assert rows[name]["topic_diversity"] is not None
        assert rows["average d<=0.001"]["n_topics"] == 4  # nothing that close
        assert rows["average d<=2.0"]["n_topics"] == 1  # the largest cosine distance
        assert "n_topics" in rows["legacy_ctfidf auto"]

    def test_reports_why_a_case_was_skipped(self, seeded):
        result = _comparison(seeded).compare(COLLECTION, CASES)

        rows = {row["case"]: row for row in result["rows"]}
        assert "density tree" in rows["density N=2"]["skipped"]
        assert "base topics" in rows["average N=9"]["skipped"]

    def test_agreement_between_strategies_at_the_same_cut(self, seeded):
        result = _comparison(seeded).compare(COLLECTION, CASES)

        [pair] = result["agreement"]
        assert (pair["n_topics"], pair["a"], pair["b"]) == (2, "average N=2", "ward N=2")
        assert pair["ari"] == pytest.approx(1.0)

    def test_markdown_report_has_a_row_per_case(self, seeded):
        report = markdown_report([_comparison(seeded).compare(COLLECTION, CASES)])

        assert "## ds_dense" in report
        assert "| average N=2 | 2 |" in report
        assert "skipped:" in report


def test_cases_cover_every_strategy_and_cut():
    names = [
        case.name
        for case in ReductionComparison.cases(
            ["average", "density", "legacy_ctfidf"], [3, 5], [0.7], legacy_auto=True
        )
    ]
    assert names == [
        "average N=3",
        "average N=5",
        "average d<=0.7",
        "density N=3",
        "density N=5",
        "density N=3 +noise",
        "density N=5 +noise",
        "legacy_ctfidf N=3",
        "legacy_ctfidf N=5",
        "legacy_ctfidf auto",
    ]


class TestPartitionStats:
    def test_sizes(self):
        stats = PartitionStats.sizes(np.array([0, 0, 0, 1, -1]))
        assert stats["n_topics"] == 2
        assert stats["largest_share"] == pytest.approx(0.75)
        assert stats["noise_share"] == pytest.approx(0.2)
        assert 0 < stats["size_entropy"] < 1

    def test_equal_sizes_have_entropy_one(self):
        assert PartitionStats.sizes(np.array([0, 1, 2, 0, 1, 2]))["size_entropy"] == pytest.approx(
            1.0
        )

    def test_agreement_ignores_noise(self):
        assert PartitionStats.agreement([0, 0, 1, -1], [5, 5, 7, 7]) == pytest.approx(1.0)
