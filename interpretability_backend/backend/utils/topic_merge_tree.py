"""Merge trees over an extraction's base topics, and cutting them.

A ``TopicMergeTree`` is a dendrogram whose leaves are the base topics of a
topic extraction (its original clusters; noise is never a leaf), in scipy's
linkage convention: row ``i`` merges the nodes ``children[i]`` at
``heights[i]``, where an index below ``k`` is a leaf and ``k + j`` is the node
made by row ``j``. Rows are in merge order, so a cut at N topics applies the
first ``k - N`` rows. Three builders:

- ``CentroidLinkage("average")``: average linkage on the cosine distance
  between (centred) centroid directions, the tree ``SimilarityOrdering``
  builds for the heatmap order.
- ``CentroidLinkage("ward")``: Ward linkage weighted by topic size
  (``SizeWeightedWard``), so merging into a large topic costs more.
- ``DensityTreeBuilder``: HDBSCAN's own condensed tree above the clusters it
  selected. Noise points map to the merge they fell out of, so a cut can
  absorb them.

``TopicTreeCut`` applies a tree: groups at N topics or at a height, the noise
absorbed, and each group's representative (largest) leaf. The frontend
mirrors it in ``embedding_visualization/lib/utils/topicTreeCut.ts``, tested on
the same literal examples.

Imports scipy (through ``similarity_ordering``) at module level, so modules
in the GraphQL schema's import graph import this one lazily.
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TypeVar

import numpy as np

from .similarity_ordering import SimilarityOrdering

K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True, eq=False)  # ndarray fields: compare by identity
class TopicMergeTree:
    """A dendrogram over base topics; see the module docstring for the layout."""

    linkage: str
    leaf_ids: np.ndarray  # (k,) base topic ids
    leaf_sizes: np.ndarray  # (k,) documents per base topic
    children: np.ndarray  # (k - 1, 2) node indices merged by each row
    heights: np.ndarray  # (k - 1,) non-decreasing merge heights
    # Density trees only: noise item id -> the merge row whose branch it fell out of.
    noise_merges: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "leaf_ids", np.asarray(self.leaf_ids, dtype=np.int64))
        object.__setattr__(self, "leaf_sizes", np.asarray(self.leaf_sizes, dtype=np.int64))
        children = np.asarray(self.children, dtype=np.int64).reshape(-1, 2)
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "heights", np.asarray(self.heights, dtype=np.float64))
        object.__setattr__(self, "noise_merges", dict(self.noise_merges))
        self._validate()

    @property
    def n_leaves(self) -> int:
        return int(self.leaf_ids.shape[0])

    def node_sizes(self) -> np.ndarray:
        """Documents under every node: leaves first, then one entry per merge row."""
        k = self.n_leaves
        sizes = np.zeros(2 * k - 1 if k else 0, dtype=np.int64)
        sizes[:k] = self.leaf_sizes
        for row, (a, b) in enumerate(self.children):
            sizes[k + row] = sizes[a] + sizes[b]
        return sizes

    def with_noise(self, noise_merges: Mapping[str, int]) -> "TopicMergeTree":
        """The same tree with noise items mapped to merge rows."""
        return replace(self, noise_merges=noise_merges)

    def _validate(self) -> None:
        k = self.n_leaves
        if k == 0:
            raise ValueError("A merge tree needs at least one leaf")
        if self.leaf_sizes.shape != (k,):
            raise ValueError(f"Expected {k} leaf sizes, got {self.leaf_sizes.shape[0]}")
        if self.children.shape != (k - 1, 2) or self.heights.shape != (k - 1,):
            raise ValueError(f"A tree over {k} leaves needs {k - 1} merge rows")
        used = set()
        for row, pair in enumerate(self.children.tolist()):
            for node in pair:
                if not 0 <= node < k + row:
                    raise ValueError(f"Merge row {row} uses node {node} before it exists")
                if node in used:
                    raise ValueError(f"Node {node} can be merged only once")
                used.add(node)
        if np.any(np.diff(self.heights) < 0):
            raise ValueError("Merge heights must be non-decreasing")
        for item, row in self.noise_merges.items():
            if not 0 <= row < k - 1:
                raise ValueError(f"Noise item {item!r} maps to missing merge row {row}")


class TopicTreeCut:
    """Groups of a merge tree after its first ``n_merges`` rows are applied.

    Groups are named by their root node index (a leaf index for a topic not
    merged yet), so a group keeps its name while it does not change.
    """

    def __init__(self, tree: TopicMergeTree) -> None:
        self.tree = tree

    def merges_for(self, n_topics: int) -> int:
        """Rows to apply to leave ``n_topics`` groups."""
        k = self.tree.n_leaves
        if not 1 <= n_topics <= k:
            raise ValueError(f"Topic count must be between 1 and {k}, got {n_topics}")
        return k - n_topics

    def merges_within(self, height: float) -> int:
        """Rows whose height is at most ``height``."""
        return int(np.searchsorted(self.tree.heights, height, side="right"))

    def roots(self, n_merges: int) -> np.ndarray:
        """Root node of every node once the first ``n_merges`` rows are applied."""
        k = self.tree.n_leaves
        if not 0 <= n_merges <= k - 1:
            raise ValueError(f"Merge count must be between 0 and {k - 1}, got {n_merges}")
        parent = np.arange(2 * k - 1)
        for row in range(n_merges):
            parent[self.tree.children[row]] = k + row
        # A parent's index is always larger than its children's, so resolving
        # from the top down leaves every node pointing at its final root.
        roots = parent.copy()
        for node in range(2 * k - 2, -1, -1):
            roots[node] = roots[parent[node]]
        return roots

    def groups(self, n_merges: int) -> np.ndarray:
        """Group (root node) of each leaf."""
        return self.roots(n_merges)[: self.tree.n_leaves]

    def absorbed_noise(self, n_merges: int) -> dict[str, int]:
        """The tree's noise items whose merge row has been applied, with the group they join."""
        return self.absorbed(self.tree.noise_merges, n_merges)

    def absorbed(self, noise_merges: Mapping[K, int], n_merges: int) -> dict[K, int]:
        """``absorbed_noise`` for any noise keys (e.g. document positions)."""
        k = self.tree.n_leaves
        roots = self.roots(n_merges)
        return {key: int(roots[k + row]) for key, row in noise_merges.items() if row < n_merges}

    def representatives(self, groups: np.ndarray) -> dict[int, int]:
        """Each group's largest leaf, as a base topic id (ties: the lowest id)."""
        best: dict[int, tuple[int, int]] = {}
        for group, topic_id, size in zip(
            groups.tolist(), self.tree.leaf_ids.tolist(), self.tree.leaf_sizes.tolist(), strict=True
        ):
            current = best.get(group)
            if current is None or (-size, topic_id) < (-current[1], current[0]):
                best[group] = (topic_id, size)
        return {group: topic_id for group, (topic_id, _) in best.items()}


class SizeWeightedWard:
    """Ward linkage on points that each stand for a cluster of ``sizes[i]`` members.

    Merging clusters ``a`` and ``b`` costs ``n_a * n_b / (n_a + n_b) * |c_a - c_b|^2``
    (the increase in within-cluster sum of squares). The Lance-Williams update
    keeps those costs exact from any initial sizes. Merges are found with the
    nearest-neighbour chain (Ward is reducible), then sorted into merge order.
    Heights are ``sqrt(2 * cost)``, scipy's Ward distance.
    """

    def linkage(self, points: np.ndarray, sizes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        weights = np.asarray(sizes, dtype=np.float64).copy()
        k = points.shape[0]
        if k < 2:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0)

        squared = np.einsum("ij,ij->i", points, points)
        distances = np.maximum(squared[:, None] + squared[None, :] - 2 * points @ points.T, 0.0)
        cost = (
            weights[:, None] * weights[None, :] / (weights[:, None] + weights[None, :]) * distances
        )
        np.fill_diagonal(cost, np.inf)

        active = np.ones(k, dtype=bool)
        merges: list[tuple[int, int, float]] = []
        chain: list[int] = []
        while len(merges) < k - 1:
            if not chain:
                chain.append(int(np.flatnonzero(active)[0]))
            a = chain[-1]
            row = cost[a]
            b = int(np.argmin(row))
            # Prefer the previous chain element on ties, so the chain terminates.
            if len(chain) > 1 and row[chain[-2]] <= row[b]:
                b = chain[-2]
            if len(chain) > 1 and b == chain[-2]:
                chain.pop()
                chain.pop()
                merges.append((a, b, float(cost[a, b])))
                self._merge(cost, weights, active, a, b)
            else:
                chain.append(b)
        return self._relabel(merges, k)

    @staticmethod
    def _merge(cost: np.ndarray, weights: np.ndarray, active: np.ndarray, a: int, b: int) -> None:
        """Merge cluster ``b`` into slot ``a`` (Lance-Williams update for Ward)."""
        n_a, n_b, joint = weights[a], weights[b], cost[a, b]
        updated = ((n_a + weights) * cost[a] + (n_b + weights) * cost[b] - weights * joint) / (
            n_a + n_b + weights
        )
        active[b] = False
        updated[~active] = np.inf
        updated[a] = np.inf
        cost[a, :] = updated
        cost[:, a] = updated
        cost[b, :] = np.inf
        cost[:, b] = np.inf
        weights[a] = n_a + n_b

    @staticmethod
    def _relabel(merges: list[tuple[int, int, float]], k: int) -> tuple[np.ndarray, np.ndarray]:
        """Merges named by surviving slot -> scipy-style rows, in height order."""
        order = sorted(range(len(merges)), key=lambda i: merges[i][2])  # stable
        parent = list(range(k))
        node_of = list(range(k))  # current node index of each slot's cluster

        def find(slot: int) -> int:
            while parent[slot] != slot:
                parent[slot] = parent[parent[slot]]
                slot = parent[slot]
            return slot

        children = np.zeros((k - 1, 2), dtype=np.int64)
        heights = np.zeros(k - 1)
        for row, index in enumerate(order):
            a, b, cost = merges[index]
            root_a, root_b = find(a), find(b)
            children[row] = sorted((node_of[root_a], node_of[root_b]))
            heights[row] = np.sqrt(2.0 * max(cost, 0.0))
            parent[root_b] = root_a
            node_of[root_a] = k + row
        return children, heights


class CentroidLinkage:
    """Merge tree over topic centroid directions (rows of unit vectors)."""

    METHODS = ("average", "ward")

    def __init__(self, method: str) -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown linkage {method!r}; expected one of {self.METHODS}")
        self.method = method

    def build(
        self, leaf_ids: Sequence[int], directions: np.ndarray, sizes: Sequence[int]
    ) -> TopicMergeTree:
        directions = np.asarray(directions, dtype=np.float64)
        k = directions.shape[0]
        if k < 2:
            children, heights = np.zeros((0, 2), dtype=np.int64), np.zeros(0)
        elif self.method == "average":
            # Optimal leaf ordering only reorders branches: skip its O(k^3) cost.
            tree = SimilarityOrdering(optimal_leaf_limit=0).order(directions).linkage
            children, heights = tree[:, :2].astype(np.int64), tree[:, 2]
        else:
            children, heights = SizeWeightedWard().linkage(directions, sizes)
        return TopicMergeTree(
            linkage=self.method,
            leaf_ids=leaf_ids,
            leaf_sizes=sizes,
            children=children,
            heights=heights,
        )


@dataclass(frozen=True, eq=False)
class DensityTree:
    """A density merge tree and, per clustered point, the merge row it fell out of."""

    tree: TopicMergeTree  # leaves are the cluster labels 0..k-1
    noise_rows: np.ndarray  # (n_points,): merge row for noise points, -1 otherwise


class DensityTreeBuilder:
    """Merge tree from an HDBSCAN condensed tree, above the clusters it selected.

    ``condensed`` is ``model.condensed_tree_.to_numpy()``: rows of
    (parent, child, lambda_val, child_size), where a child below the number of
    points is a point (``lambda_val``: when it fell out of ``parent``) and any
    other child is a cluster node (``lambda_val``: when it split off). Each
    label's node is the lowest common ancestor of the nodes its points fell
    out of, so no private hdbscan API is needed. The nodes above the selected
    clusters always split in two, giving exactly ``k - 1`` merges, at height
    ``1 / lambda`` of the split (deeper splits first). Every noise point fell
    out of one of those nodes, which is the merge that absorbs it.
    """

    def build(self, condensed: np.ndarray, labels: np.ndarray) -> DensityTree:
        labels = np.asarray(labels, dtype=np.int64)
        n_points = labels.shape[0]
        k = int(labels.max()) + 1 if n_points else 0
        if k < 2:
            raise ValueError("A density tree needs at least two clusters")

        parent_of: dict[int, int] = {}
        birth: dict[int, float] = {}
        children_of: dict[int, list[int]] = {}
        fell_from = np.full(n_points, -1, dtype=np.int64)
        for parent, child, lam, _ in condensed.tolist():
            if child < n_points:
                fell_from[child] = parent
            else:
                parent_of[child] = parent
                birth[child] = lam
                children_of.setdefault(parent, []).append(child)

        depth = self._depths(parent_of)
        selected = [
            self._lowest_common_ancestor(fell_from[labels == c], parent_of, depth) for c in range(k)
        ]
        if len(set(selected)) != k or any(
            self._ancestors(node, parent_of) & set(selected) for node in selected
        ):
            raise ValueError("Cluster labels do not match the condensed tree's selection")

        internal = set().union(*(self._ancestors(node, parent_of) for node in selected))
        if len(internal) != k - 1:
            raise ValueError("Cluster labels do not match the condensed tree's selection")

        nodes = set(selected) | internal
        split: dict[int, tuple[float, list[int]]] = {}
        for node in internal:
            kids = [c for c in children_of.get(node, []) if c in nodes]
            if len(kids) != 2:
                raise ValueError("Cluster labels do not match the condensed tree's selection")
            lam = birth[kids[0]]
            split[node] = (1.0 / lam if lam > 0 else np.inf, kids)

        order = sorted(internal, key=lambda node: (split[node][0], -depth[node], node))
        index = {node: label for label, node in enumerate(selected)}
        index.update({node: k + row for row, node in enumerate(order)})
        children = [sorted(index[kid] for kid in split[node][1]) for node in order]
        heights = np.array([split[node][0] for node in order])
        finite = heights[np.isfinite(heights)]
        # Disconnected components split at lambda 0: place them above every finite merge.
        heights[~np.isfinite(heights)] = 2.0 * (finite.max() if finite.size else 1.0)

        merge_row = {node: row for row, node in enumerate(order)}
        noise_rows = np.full(n_points, -1, dtype=np.int64)
        for point in np.flatnonzero(labels == -1).tolist():
            noise_rows[point] = merge_row.get(int(fell_from[point]), -1)

        tree = TopicMergeTree(
            linkage="density",
            leaf_ids=np.arange(k),
            leaf_sizes=np.bincount(labels[labels >= 0], minlength=k),
            children=children,
            heights=heights,
        )
        return DensityTree(tree=tree, noise_rows=noise_rows)

    @staticmethod
    def _depths(parent_of: Mapping[int, int]) -> dict[int, int]:
        """Depth of every cluster node (the root is 0)."""
        depth: dict[int, int] = {}
        for start in parent_of:
            path = []
            node = start
            while node in parent_of and node not in depth:
                path.append(node)
                node = parent_of[node]
            level = depth.setdefault(node, 0)  # the root, or a node resolved earlier
            for step in reversed(path):
                level += 1
                depth[step] = level
        return depth

    @staticmethod
    def _ancestors(node: int, parent_of: Mapping[int, int]) -> set[int]:
        """Strict ancestors of ``node``."""
        found = set()
        while node in parent_of:
            node = parent_of[node]
            found.add(node)
        return found

    @classmethod
    def _lowest_common_ancestor(
        cls, nodes: np.ndarray, parent_of: Mapping[int, int], depth: Mapping[int, int]
    ) -> int:
        unique = set(nodes.tolist())
        if not unique or -1 in unique:
            raise ValueError("Cluster labels do not match the condensed tree's selection")
        common = None
        for node in unique:
            lineage = cls._ancestors(node, parent_of) | {node}
            common = lineage if common is None else common & lineage
        return max(common, key=lambda node: depth.get(node, 0))
