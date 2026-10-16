"""SimilarityOrdering: average-linkage + optimal-leaf ordering of vectors by cosine similarity."""

import numpy as np

from backend.utils.similarity_ordering import SimilarityOrdering


def _runs(labels: list[str]) -> int:
    """Number of contiguous runs in a label sequence (2 for "aaabbb")."""
    return 1 + sum(a != b for a, b in zip(labels, labels[1:], strict=False))


def _adjacent_similarity(vectors: np.ndarray, order: list[int]) -> float:
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return float(sum(unit[a] @ unit[b] for a, b in zip(order, order[1:], strict=False)))


def _two_clusters(rng: np.random.Generator, per_cluster: int = 6) -> tuple[np.ndarray, list[str]]:
    """Interleaved rows of two well-separated directions."""
    rows, labels = [], []
    for _ in range(per_cluster):
        rows.append([1.0, 0.0, 0.0] + rng.normal(0, 0.05, 3))
        labels.append("a")
        rows.append([0.0, 0.0, 1.0] + rng.normal(0, 0.05, 3))
        labels.append("b")
    return np.array(rows), labels


class TestOrder:
    def test_keeps_separated_clusters_contiguous(self):
        vectors, labels = _two_clusters(np.random.default_rng(0))
        order = SimilarityOrdering().order(vectors).order
        assert sorted(order) == list(range(len(vectors)))
        assert _runs([labels[i] for i in order]) == 2

    def test_optimal_leaf_ordering_places_similar_rows_side_by_side(self):
        vectors = np.random.default_rng(1).normal(size=(30, 8))
        optimal = SimilarityOrdering().order(vectors).order
        plain = SimilarityOrdering(optimal_leaf_limit=0).order(vectors).order
        assert _adjacent_similarity(vectors, optimal) >= _adjacent_similarity(vectors, plain) - 1e-9

    def test_scaling_a_vector_does_not_change_the_order(self):
        vectors, _ = _two_clusters(np.random.default_rng(2))
        scaled = vectors.copy()
        scaled[3] *= 7.0
        assert SimilarityOrdering().order(scaled).order == SimilarityOrdering().order(vectors).order

    def test_returns_the_linkage_tree(self):
        vectors, _ = _two_clusters(np.random.default_rng(3))
        result = SimilarityOrdering().order(vectors)
        assert result.linkage is not None
        assert result.linkage.shape == (len(vectors) - 1, 4)

    def test_handles_tiny_inputs(self):
        ordering = SimilarityOrdering()
        assert ordering.order(np.zeros((0, 3))).order == []
        assert ordering.order(np.ones((1, 3))).order == [0]
        assert ordering.order(np.ones((1, 3))).linkage is None
        assert sorted(ordering.order(np.eye(2)).order) == [0, 1]

    def test_zero_vectors_still_get_a_place(self):
        vectors = np.array([[1.0, 0.0], [0.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        assert sorted(SimilarityOrdering().order(vectors).order) == [0, 1, 2, 3]

    def test_skips_leaf_ordering_above_the_limit_but_still_orders(self):
        vectors = np.random.default_rng(4).normal(size=(12, 4))
        order = SimilarityOrdering(optimal_leaf_limit=5).order(vectors).order
        assert sorted(order) == list(range(12))


class TestGrouped:
    def test_orders_groups_in_the_given_order_and_rows_within_each(self):
        rng = np.random.default_rng(5)
        vectors, _ = _two_clusters(rng, per_cluster=3)
        groups = np.array([20, 10, 20, 10, 20, 10])
        order = SimilarityOrdering().grouped(vectors, groups, [10, 20])
        assert sorted(order) == list(range(6))
        assert [int(groups[i]) for i in order] == [10, 10, 10, 20, 20, 20]

    def test_appends_rows_of_unlisted_groups_last(self):
        vectors = np.eye(3)
        order = SimilarityOrdering().grouped(vectors, np.array([1, 2, 3]), [2])
        assert order[0] == 1
        assert sorted(order[1:]) == [0, 2]
