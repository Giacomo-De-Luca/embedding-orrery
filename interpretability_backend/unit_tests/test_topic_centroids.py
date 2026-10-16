"""Unit tests for the topic-centroid utility (accumulation + query ranking).

Pure numpy: no DuckDB, no ChromaDB. The accumulator streams (labels, vectors)
batches into per-topic sums; the ranker compares a query vector against the
resulting centroids (mean-centred cosine, best-subtopic roll-up).
"""

import numpy as np
import pytest

from backend.utils.topic_centroids import (
    CENTER_BY_DEFAULT,
    CentroidTable,
    TopicCentroidAccumulator,
    TopicCentroidRanker,
    unit_rows,
)

RNG = np.random.default_rng(7)


def _vectors(n: int, dim: int = 4) -> np.ndarray:
    return RNG.normal(size=(n, dim))


class TestTopicCentroidAccumulator:
    def test_means_equal_member_means(self):
        vectors = _vectors(6)
        labels = [0, 1, 0, 2, 1, 0]
        acc = TopicCentroidAccumulator()
        acc.add(labels, vectors)

        table = acc.table()
        assert table.ids.tolist() == [0, 1, 2]
        assert table.counts.tolist() == [3, 2, 1]
        np.testing.assert_allclose(table.means[0], vectors[[0, 2, 5]].mean(axis=0))
        np.testing.assert_allclose(table.means[1], vectors[[1, 4]].mean(axis=0))
        np.testing.assert_allclose(table.means[2], vectors[3])

    def test_batched_accumulation_matches_one_shot(self):
        vectors = _vectors(50)
        labels = RNG.integers(-1, 5, size=50)

        one_shot = TopicCentroidAccumulator()
        one_shot.add(labels, vectors)
        batched = TopicCentroidAccumulator()
        for start in range(0, 50, 7):
            batched.add(labels[start : start + 7], vectors[start : start + 7])

        a, b = one_shot.table(), batched.table()
        assert a.ids.tolist() == b.ids.tolist()
        assert a.counts.tolist() == b.counts.tolist()
        np.testing.assert_allclose(a.means, b.means)

    def test_add_mapped_skips_unassigned_ids(self):
        vectors = _vectors(4)
        acc = TopicCentroidAccumulator()
        acc.add_mapped(["a", "b", "c", "d"], vectors, {"a": 0, "c": 0, "d": 1})

        table = acc.table()
        assert table.ids.tolist() == [0, 1]
        assert table.counts.tolist() == [2, 1]
        np.testing.assert_allclose(table.means[0], vectors[[0, 2]].mean(axis=0))

    def test_table_follows_requested_order(self):
        vectors = _vectors(3)
        acc = TopicCentroidAccumulator()
        acc.add([2, -1, 0], vectors)

        table = acc.table(order=[-1, 0, 2])
        assert table.ids.tolist() == [-1, 0, 2]
        np.testing.assert_allclose(table.means[0], vectors[1])

    def test_table_rejects_order_with_unknown_label(self):
        acc = TopicCentroidAccumulator()
        acc.add([0], _vectors(1))
        with pytest.raises(ValueError, match="no members"):
            acc.table(order=[0, 1])

    def test_empty_accumulator_table_raises(self):
        with pytest.raises(ValueError, match="empty"):
            TopicCentroidAccumulator().table()

    def test_normalize_averages_unit_vectors(self):
        # Directions matter, not lengths: [4, 0] and [0, 1] average to [.5, .5].
        acc = TopicCentroidAccumulator(normalize=True)
        acc.add([0, 0, 1], np.array([[4.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

        table = acc.table()
        np.testing.assert_allclose(table.means[0], [0.5, 0.5])
        np.testing.assert_allclose(table.means[1], [0.0, 0.0])  # zero vector stays zero

    def test_add_rejects_dimension_change(self):
        acc = TopicCentroidAccumulator()
        acc.add([0], _vectors(1, dim=4))
        with pytest.raises(ValueError, match="dimension"):
            acc.add([0], _vectors(1, dim=3))


def _table(ids, means, counts, parents=None) -> CentroidTable:
    return CentroidTable(
        ids=np.asarray(ids, dtype=np.int64),
        means=np.asarray(means, dtype=np.float64),
        counts=np.asarray(counts, dtype=np.int64),
        parents=None if parents is None else np.asarray(parents, dtype=np.int64),
    )


# Three topics around a shared offset (the "dataset mean" every centroid drifts
# toward). Raw cosine sees them all as nearly identical; centring separates them.
OFFSET = np.array([10.0, 10.0, 10.0])
TOPICS = _table(
    ids=[-1, 0, 1, 2],
    means=[OFFSET, OFFSET + [1, 0, 0], OFFSET + [0, 1, 0], OFFSET + [0, 0, 1]],
    counts=[10, 10, 10, 10],
)


class TestCentroidTableGeometry:
    def test_mean_is_count_weighted_and_includes_noise(self):
        table = _table(ids=[-1, 0], means=[[0.0, 0.0], [4.0, 0.0]], counts=[3, 1])
        np.testing.assert_allclose(table.mean(), [1.0, 0.0])

    def test_unit_directions_are_centred_unit_rows(self):
        table = _table(ids=[0, 1, 2], means=[[3.0, 1.0], [1.0, 1.0], [1.0, 4.0]], counts=[1, 1, 1])
        directions = table.unit_directions(np.array([1.0, 1.0]))

        np.testing.assert_allclose(directions, [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])


class TestTopicCentroidRanker:
    def test_origin_follows_the_center_flag(self):
        np.testing.assert_allclose(TopicCentroidRanker(center=True).origin(TOPICS), TOPICS.mean())
        np.testing.assert_allclose(TopicCentroidRanker(center=False).origin(TOPICS), np.zeros(3))

    def test_default_centring_is_the_shared_constant(self):
        assert TopicCentroidRanker().center is CENTER_BY_DEFAULT

    def test_similarity_equals_dot_of_centred_directions(self):
        # Consumers comparing centroids with each other (e.g. topic ordering)
        # reuse origin() + unit_directions(); they must agree with rank().
        ranker = TopicCentroidRanker()
        query = OFFSET + [0, 1.0, 0.1]
        origin = ranker.origin(TOPICS)
        expected = TOPICS.unit_directions(origin) @ unit_rows(query - origin)

        ranked = {r.topic_id: r.similarity for r in ranker.rank(query, TOPICS, limit=3)}
        for row, topic_id in enumerate(TOPICS.ids.tolist()):
            if topic_id != -1:
                assert ranked[topic_id] == pytest.approx(expected[row])

    def test_centred_ranking_picks_the_matching_topic(self):
        query = OFFSET + [0, 1.0, 0.1]
        ranked = TopicCentroidRanker(center=True).rank(query, TOPICS, limit=3)

        assert [r.topic_id for r in ranked] == [1, 2, 0]
        assert ranked[0].similarity > 0.9
        assert ranked[0].matched_subtopic_id is None

    def test_centring_spreads_scores(self):
        query = OFFSET + [0, 1.0, 0.1]
        raw = TopicCentroidRanker(center=False).rank(query, TOPICS, limit=3)
        centred = TopicCentroidRanker(center=True).rank(query, TOPICS, limit=3)

        raw_spread = raw[0].similarity - raw[-1].similarity
        centred_spread = centred[0].similarity - centred[-1].similarity
        assert centred_spread > raw_spread * 10

    def test_noise_topic_is_never_returned(self):
        ranked = TopicCentroidRanker().rank(OFFSET.copy(), TOPICS, limit=10)
        assert -1 not in [r.topic_id for r in ranked]
        assert len(ranked) == 3

    def test_limit_is_respected(self):
        ranked = TopicCentroidRanker().rank(OFFSET + [1, 0, 0], TOPICS, limit=1)
        assert [r.topic_id for r in ranked] == [0]

    def test_best_subtopic_lifts_its_parent(self):
        # Topic 0 is the mean of two subtopics; the query matches subtopic 11
        # exactly, which scores higher than topic 0's blurred centroid.
        topics = _table(
            ids=[-1, 0, 1],
            means=[[0, 0, 0], [1, 1, 0], [-1, -1, 0]],
            counts=[2, 2, 2],
        )
        subtopics = _table(
            ids=[-1, 10, 11, 12],
            means=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [-1, -1, 0]],
            counts=[2, 1, 1, 2],
            parents=[-1, 0, 0, 1],
        )
        query = np.array([0.0, 1.0, 0.0])

        ranked = TopicCentroidRanker(center=False).rank(query, topics, subtopics, limit=2)

        assert ranked[0].topic_id == 0
        assert ranked[0].matched_subtopic_id == 11
        assert ranked[0].similarity == pytest.approx(1.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            TopicCentroidRanker().rank(np.zeros(5), TOPICS)
