"""Order vectors so that similar ones sit next to each other.

Average-linkage hierarchical clustering on cosine distance, followed by
optimal leaf ordering, which flips the dendrogram's branches so neighbouring
leaves are as similar as possible. Used to sort heatmap axes (topics by
centroid similarity).

Imports scipy at module level, so this module must stay out of the GraphQL
schema's import graph (``test_torch_free_import``): callers import it lazily.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from .topic_centroids import unit_rows

# Optimal leaf ordering costs O(n^3); above this many rows the plain dendrogram order is kept.
DEFAULT_OPTIMAL_LEAF_LIMIT = 500


@dataclass(frozen=True)
class SimilarityOrder:
    order: list[int]
    """Row indices in similarity order."""
    linkage: np.ndarray | None
    """scipy linkage matrix ``(n - 1, 4)`` over the input rows; None below two rows."""


class SimilarityOrdering:
    def __init__(
        self, optimal_leaf_limit: int = DEFAULT_OPTIMAL_LEAF_LIMIT, method: str = "average"
    ) -> None:
        self.optimal_leaf_limit = optimal_leaf_limit
        self.method = method

    def order(self, vectors: np.ndarray) -> SimilarityOrder:
        """Rows of ``vectors`` ordered by cosine similarity (length is ignored)."""
        vectors = np.asarray(vectors, dtype=np.float64)
        n = vectors.shape[0]
        if n < 2:
            return SimilarityOrder(order=list(range(n)), linkage=None)

        unit = unit_rows(vectors)
        # Zero rows stay zero, so they sit at distance 1 from everything.
        distances = np.clip(1.0 - unit @ unit.T, 0.0, 2.0)
        np.fill_diagonal(distances, 0.0)
        condensed = squareform(distances, checks=False)

        tree = linkage(condensed, method=self.method)
        if n <= self.optimal_leaf_limit:
            tree = optimal_leaf_ordering(tree, condensed)
        return SimilarityOrder(order=leaves_list(tree).tolist(), linkage=tree)

    def grouped(
        self, vectors: np.ndarray, groups: np.ndarray, group_order: Sequence[int]
    ) -> list[int]:
        """Rows block by block: groups in ``group_order``, each block in similarity order.

        Rows whose group is not in ``group_order`` follow, in similarity order.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        groups = np.asarray(groups)
        ordered: list[int] = []
        for group in group_order:
            members = np.flatnonzero(groups == group)
            ordered.extend(int(members[k]) for k in self.order(vectors[members]).order)
        rest = np.setdiff1d(np.arange(len(groups)), ordered)
        ordered.extend(int(rest[k]) for k in self.order(vectors[rest]).order)
        return ordered
