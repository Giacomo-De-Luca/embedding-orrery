"""Merge trees over base topics: cutting them, and the three builders.

The literal cut examples (``LITERAL_TREE`` and the ``test_literal_*`` cases)
are mirrored in embedding_visualization/lib/utils/__tests__/topicTreeCut.test.ts,
so the backend cut and the frontend's live slider cannot drift apart.
"""

import hdbscan
import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.datasets import make_blobs

from backend.utils.topic_centroids import unit_rows
from backend.utils.topic_merge_tree import (
    CentroidLinkage,
    DensityTreeBuilder,
    SizeWeightedWard,
    TopicMergeTree,
    TopicTreeCut,
)

# Five base topics (ids 10..14, sizes 5, 3, 8, 2, 1):
#   row 0: leaves 0 + 1 -> node 5 at 0.1
#   row 1: leaves 3 + 4 -> node 6 at 0.2
#   row 2: leaf 2 + node 6 -> node 7 at 0.4
#   row 3: node 5 + node 7 -> node 8 at 0.9
LITERAL_TREE = TopicMergeTree(
    linkage="average",
    leaf_ids=[10, 11, 12, 13, 14],
    leaf_sizes=[5, 3, 8, 2, 1],
    children=[[0, 1], [3, 4], [2, 6], [5, 7]],
    heights=[0.1, 0.2, 0.4, 0.9],
    noise_merges={"n-early": 0, "n-mid": 1, "n-root": 3},
)


def _same_partition(a, b) -> bool:
    return len(set(zip(a, b, strict=True))) == len(set(a)) == len(set(b))


class TestLiteralCut:
    @pytest.mark.parametrize(
        ("n_topics", "groups"),
        [
            (5, [0, 1, 2, 3, 4]),
            (4, [5, 5, 2, 3, 4]),
            (3, [5, 5, 2, 6, 6]),
            (2, [5, 5, 7, 7, 7]),
            (1, [8, 8, 8, 8, 8]),
        ],
    )
    def test_literal_groups_at_n(self, n_topics, groups):
        cut = TopicTreeCut(LITERAL_TREE)
        assert cut.groups(cut.merges_for(n_topics)).tolist() == groups

    @pytest.mark.parametrize(("height", "merges"), [(0.05, 0), (0.1, 1), (0.25, 2), (5.0, 4)])
    def test_literal_merges_within_height(self, height, merges):
        assert TopicTreeCut(LITERAL_TREE).merges_within(height) == merges

    def test_literal_representatives_are_the_largest_leaf(self):
        cut = TopicTreeCut(LITERAL_TREE)
        assert cut.representatives(cut.groups(cut.merges_for(2))) == {5: 10, 7: 12}

    def test_literal_representative_ties_go_to_the_lowest_id(self):
        tree = TopicMergeTree(
            linkage="average",
            leaf_ids=[4, 2],
            leaf_sizes=[3, 3],
            children=[[0, 1]],
            heights=[0.5],
        )
        cut = TopicTreeCut(tree)
        assert cut.representatives(cut.groups(1)) == {2: 2}

    @pytest.mark.parametrize(
        ("n_topics", "absorbed"),
        [
            (5, {}),
            (4, {"n-early": 5}),
            (3, {"n-early": 5, "n-mid": 6}),
            (2, {"n-early": 5, "n-mid": 7}),
            (1, {"n-early": 8, "n-mid": 8, "n-root": 8}),
        ],
    )
    def test_literal_absorbed_noise(self, n_topics, absorbed):
        cut = TopicTreeCut(LITERAL_TREE)
        assert cut.absorbed_noise(cut.merges_for(n_topics)) == absorbed

    @pytest.mark.parametrize("n_topics", [0, 6, -1])
    def test_topic_count_out_of_range_raises(self, n_topics):
        with pytest.raises(ValueError, match="between 1 and 5"):
            TopicTreeCut(LITERAL_TREE).merges_for(n_topics)

    def test_node_sizes_count_documents_under_each_node(self):
        assert LITERAL_TREE.node_sizes().tolist() == [5, 3, 8, 2, 1, 8, 3, 11, 19]


class TestTreeValidation:
    def test_rejects_a_child_made_by_a_later_row(self):
        with pytest.raises(ValueError, match="before"):
            TopicMergeTree(
                linkage="average",
                leaf_ids=[0, 1, 2],
                leaf_sizes=[1, 1, 1],
                children=[[0, 4], [1, 2]],
                heights=[0.1, 0.2],
            )

    def test_rejects_decreasing_heights(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            TopicMergeTree(
                linkage="average",
                leaf_ids=[0, 1, 2],
                leaf_sizes=[1, 1, 1],
                children=[[0, 1], [2, 3]],
                heights=[0.5, 0.2],
            )

    def test_rejects_a_node_merged_twice(self):
        with pytest.raises(ValueError, match="once"):
            TopicMergeTree(
                linkage="average",
                leaf_ids=[0, 1, 2],
                leaf_sizes=[1, 1, 1],
                children=[[0, 1], [0, 2]],
                heights=[0.1, 0.2],
            )

    def test_trees_compare_by_identity(self):
        assert LITERAL_TREE == LITERAL_TREE
        assert LITERAL_TREE != LITERAL_TREE.with_noise({})
        assert {LITERAL_TREE: 1}[LITERAL_TREE] == 1

    def test_with_noise_keeps_the_merges(self):
        tree = LITERAL_TREE.with_noise({"x": 2})
        assert tree.noise_merges == {"x": 2}
        assert tree.children.tolist() == LITERAL_TREE.children.tolist()


def _random_directions(rng, k: int, dim: int = 8) -> np.ndarray:
    return unit_rows(rng.normal(size=(k, dim)))


class TestAverageLinkage:
    def test_partitions_match_scipy_at_every_n(self):
        rng = np.random.default_rng(0)
        directions = _random_directions(rng, 25)
        tree = CentroidLinkage("average").build(np.arange(25), directions, np.ones(25))

        distances = np.clip(1 - directions @ directions.T, 0, 2)
        np.fill_diagonal(distances, 0)
        reference = linkage(squareform(distances, checks=False), "average")
        cut = TopicTreeCut(tree)
        for n in range(1, 26):
            expected = fcluster(reference, n, criterion="maxclust")
            assert _same_partition(cut.groups(cut.merges_for(n)).tolist(), expected.tolist())
        assert np.all(np.diff(tree.heights) >= 0)

    def test_centring_separates_directions_that_share_an_offset(self):
        # Two themes, both dominated by a large shared offset (the anisotropy
        # of averaged embeddings): raw cosines all look alike, centred ones do not.
        rng = np.random.default_rng(3)
        offset = np.zeros(16)
        offset[0] = 10.0
        theme_a, theme_b = np.zeros(16), np.zeros(16)
        theme_a[1], theme_b[2] = 1.0, 1.0
        means = np.vstack(
            [offset + theme_a + rng.normal(scale=0.2, size=16) for _ in range(4)]
            + [offset + theme_b + rng.normal(scale=0.2, size=16) for _ in range(4)]
        )
        centred = unit_rows(means - means.mean(axis=0))
        tree = CentroidLinkage("average").build(np.arange(8), centred, np.ones(8))
        cut = TopicTreeCut(tree)
        assert _same_partition(cut.groups(cut.merges_for(2)).tolist(), [0] * 4 + [1] * 4)

    def test_single_topic_has_no_merges(self):
        tree = CentroidLinkage("average").build([7], np.ones((1, 3)) / np.sqrt(3), [4])
        assert tree.children.shape == (0, 2)
        assert TopicTreeCut(tree).groups(0).tolist() == [0]

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown linkage"):
            CentroidLinkage("single")


def _brute_force_ward(points, sizes, n_clusters):
    """Greedy size-weighted Ward, recomputing every merge cost from centroids."""
    clusters = {i: ([i], points[i].astype(float), float(sizes[i])) for i in range(len(points))}
    while len(clusters) > n_clusters:
        keys = list(clusters)
        best = None
        for x, a in enumerate(keys):
            for b in keys[x + 1 :]:
                _, ca, na = clusters[a]
                _, cb, nb = clusters[b]
                cost = na * nb / (na + nb) * float(np.sum((ca - cb) ** 2))
                if best is None or cost < best[0]:
                    best = (cost, a, b)
        _, a, b = best
        ma, ca, na = clusters.pop(a)
        mb, cb, nb = clusters.pop(b)
        clusters[a] = (ma + mb, (ca * na + cb * nb) / (na + nb), na + nb)
    labels = np.zeros(len(points), dtype=int)
    for label, (members, _, _) in enumerate(clusters.values()):
        labels[members] = label
    return labels


class TestSizeWeightedWard:
    @pytest.mark.parametrize("seed", range(6))
    def test_matches_brute_force_ward_at_every_n(self, seed):
        rng = np.random.default_rng(seed)
        k = int(rng.integers(5, 12))
        points = _random_directions(rng, k, dim=6)
        sizes = rng.integers(3, 200, size=k)
        tree = CentroidLinkage("ward").build(np.arange(k), points, sizes)
        cut = TopicTreeCut(tree)
        for n in range(1, k + 1):
            expected = _brute_force_ward(points, sizes, n)
            assert _same_partition(cut.groups(cut.merges_for(n)).tolist(), expected.tolist())

    def test_heights_are_non_decreasing(self):
        rng = np.random.default_rng(11)
        points = _random_directions(rng, 40)
        children, heights = SizeWeightedWard().linkage(points, rng.integers(1, 50, size=40))
        assert children.shape == (39, 2)
        assert np.all(np.diff(heights) >= 0)

    def test_size_pulls_small_topics_into_large_ones_last(self):
        # A tiny topic sits exactly between two big ones; Ward merges the two
        # equally sized small topics first because merging anything into a big
        # topic is expensive.
        points = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.6], [0.62, 0.58]])
        sizes = np.array([500, 500, 2, 2])
        tree = CentroidLinkage("ward").build(np.arange(4), unit_rows(points), sizes)
        assert sorted(tree.children[0].tolist()) == [2, 3]


def _hand_built_condensed():
    """Condensed tree over 10 points (ids 0-9): root 10 splits into 11 and 12,
    12 splits into 13 and 14. Points 0-2 fall out of 11, 3-4 of 13, 5-6 of 14,
    7 of 12 (noise at the leaf level), 8 and 9 of the root (noise)."""
    rows = [
        (10, 11, 0.5, 3),
        (10, 12, 0.5, 5),
        (12, 13, 2.0, 2),
        (12, 14, 2.0, 2),
        (11, 0, 1.0, 1),
        (11, 1, 1.0, 1),
        (11, 2, 1.0, 1),
        (13, 3, 3.0, 1),
        (13, 4, 3.0, 1),
        (14, 5, 3.0, 1),
        (14, 6, 3.0, 1),
        (12, 7, 1.5, 1),
        (10, 8, 0.4, 1),
        (10, 9, 0.4, 1),
    ]
    dtype = [
        ("parent", np.intp),
        ("child", np.intp),
        ("lambda_val", float),
        ("child_size", np.intp),
    ]
    labels = np.array([0, 0, 0, 1, 1, 2, 2, -1, -1, -1])
    return np.array(rows, dtype=dtype), labels


class TestDensityTree:
    def test_hand_built_tree(self):
        condensed, labels = _hand_built_condensed()
        result = DensityTreeBuilder().build(condensed, labels)
        tree = result.tree

        assert tree.leaf_ids.tolist() == [0, 1, 2]
        assert tree.leaf_sizes.tolist() == [3, 2, 2]
        # Node 12 splits deeper (lambda 2.0) than the root (0.5), so it merges first.
        assert tree.children.tolist() == [[1, 2], [0, 3]]
        np.testing.assert_allclose(tree.heights, [0.5, 2.0])
        # Point 7 fell out of node 12 (merge row 0); 8 and 9 out of the root (row 1).
        assert result.noise_rows.tolist() == [-1] * 7 + [0, 1, 1]

    def test_mismatched_labels_raise(self):
        condensed, labels = _hand_built_condensed()
        labels = labels.copy()
        labels[3] = 0  # cluster 0 now spans two branches: no longer an HDBSCAN selection
        with pytest.raises(ValueError, match="condensed tree"):
            DensityTreeBuilder().build(condensed, labels)

    def test_fewer_than_two_clusters_raise(self):
        condensed, _ = _hand_built_condensed()
        with pytest.raises(ValueError, match="two clusters"):
            DensityTreeBuilder().build(condensed, np.array([0] * 7 + [-1] * 3))

    def test_real_hdbscan_fit(self):
        # Two tight pairs of blobs far apart, plus scattered noise.
        centers = [[0, 0], [1.5, 0], [12, 12], [13.5, 12]]
        points, _ = make_blobs(n_samples=400, centers=centers, cluster_std=0.3, random_state=4)
        noise = np.random.default_rng(4).uniform(-4, 18, size=(40, 2))
        points = np.vstack([points, noise])
        model = hdbscan.HDBSCAN(min_cluster_size=15, cluster_selection_method="leaf").fit(points)
        labels = model.labels_
        k = labels.max() + 1
        assert k >= 3

        result = DensityTreeBuilder().build(model.condensed_tree_.to_numpy(), labels)
        tree = result.tree
        assert tree.children.shape == (k - 1, 2)
        assert np.all(np.diff(tree.heights) >= 0)

        # Every noise point maps to a merge; clustered points to none.
        noise_rows = result.noise_rows
        assert np.all(noise_rows[labels == -1] >= 0)
        assert np.all(noise_rows[labels >= 0] == -1)

        # Cut at two topics: each group stays on one side of the gap.
        cut = TopicTreeCut(tree)
        groups = cut.groups(cut.merges_for(2))
        side = {label: points[labels == label, 0].mean() > 6 for label in range(k)}
        for group in set(groups.tolist()):
            members = [label for label in range(k) if groups[label] == group]
            assert len({side[m] for m in members}) == 1

        # Nothing is absorbed before any merge.
        noise_tree = tree.with_noise({str(i): int(r) for i, r in enumerate(noise_rows) if r >= 0})
        assert TopicTreeCut(noise_tree).absorbed_noise(0) == {}
