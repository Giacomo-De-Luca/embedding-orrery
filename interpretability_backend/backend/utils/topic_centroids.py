"""Topic centroids in embedding space: accumulation and query ranking.

A topic's centroid is the mean of its members' embedding vectors. Centroids
back two features:

- **Semantic topic reduction**: topics are merged by centroid similarity
  (``TopicReducer`` receives the centroid matrix).
- **Semantic topic search**: a query embedding is compared with every stored
  centroid (``TopicCentroidRanker``).

Pure numpy, no database or vector-store access, so it is safe to import from
the GraphQL schema (torch-free import boundary).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

NOISE_TOPIC_ID = -1

# Whether centroid comparisons subtract the dataset mean (see
# TopicCentroidRanker). The single default for topic search and for anything
# else comparing topic centroids, so the two cannot drift apart.
CENTER_BY_DEFAULT = True


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm; zero rows stay zero."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


@dataclass(frozen=True)
class CentroidTable:
    """Per-topic centroids, row-aligned arrays.

    ``parents`` is only set for subtopic tables: the topic each subtopic
    belongs to.
    """

    ids: np.ndarray  # (k,) int64
    means: np.ndarray  # (k, dim) float64
    counts: np.ndarray  # (k,) int64
    parents: np.ndarray | None = None  # (k,) int64

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    def mean(self) -> np.ndarray:
        """Count-weighted mean of the rows (noise included): the dataset mean."""
        return self.counts @ self.means / self.counts.sum()

    def unit_directions(self, origin: np.ndarray) -> np.ndarray:
        """Unit rows of ``means - origin`` (zero rows stay zero).

        Dot products of these rows are the centred cosines topic search uses.
        """
        return unit_rows(self.means - origin)


@dataclass(frozen=True)
class RankedTopic:
    topic_id: int
    similarity: float
    matched_subtopic_id: int | None = None


class TopicCentroidAccumulator:
    """Streams ``(labels, vectors)`` batches into per-label sums and counts.

    Memory is ``n_labels x dim`` regardless of how many vectors pass through,
    so whole collections can be accumulated batch by batch.

    With ``normalize=True`` each vector is scaled to unit length first, so a
    centroid is the mean *direction* of its members — the right summary when
    similarity is cosine and the embedder does not normalise its output.
    """

    def __init__(self, normalize: bool = False) -> None:
        self.normalize = normalize
        self._sums: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}
        self._dim: int | None = None

    def add(self, labels: Sequence[int] | np.ndarray, vectors: np.ndarray) -> None:
        """Accumulate one batch: ``labels[i]`` is the topic of ``vectors[i]``."""
        labels = np.asarray(labels, dtype=np.int64)
        vectors = np.asarray(vectors, dtype=np.float64)
        if labels.size == 0:
            return
        if vectors.ndim != 2 or vectors.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Expected one vector per label, got {vectors.shape} for {labels.shape[0]} labels"
            )
        if self.normalize:
            vectors = unit_rows(vectors)
        if self._dim is None:
            self._dim = int(vectors.shape[1])
        elif vectors.shape[1] != self._dim:
            raise ValueError(
                f"Vector dimension changed mid-accumulation ({self._dim} -> {vectors.shape[1]})"
            )

        unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind="stable")
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        sums = np.add.reduceat(vectors[order], starts, axis=0)

        for label, label_sum, count in zip(unique.tolist(), sums, counts.tolist(), strict=True):
            if label in self._sums:
                self._sums[label] += label_sum
                self._counts[label] += count
            else:
                self._sums[label] = label_sum.copy()
                self._counts[label] = count

    def add_mapped(
        self, item_ids: Sequence[str], vectors: np.ndarray, label_of: Mapping[str, int]
    ) -> None:
        """Accumulate a batch keyed by item id; ids missing from ``label_of`` are skipped."""
        keep = [i for i, item_id in enumerate(item_ids) if item_id in label_of]
        if not keep:
            return
        labels = [label_of[item_ids[i]] for i in keep]
        vectors = np.asarray(vectors)
        self.add(labels, vectors if len(keep) == len(item_ids) else vectors[keep])

    def table(self, order: Sequence[int] | None = None) -> CentroidTable:
        """Centroids for every accumulated label (sorted), or for ``order`` exactly."""
        if not self._sums:
            raise ValueError("Cannot build centroids from an empty accumulator")
        labels = sorted(self._sums) if order is None else list(order)
        missing = [label for label in labels if label not in self._sums]
        if missing:
            raise ValueError(f"Topics with no members to average: {missing[:5]}")

        counts = np.array([self._counts[label] for label in labels], dtype=np.int64)
        means = np.stack([self._sums[label] for label in labels]) / counts[:, None]
        return CentroidTable(ids=np.array(labels, dtype=np.int64), means=means, counts=counts)


class TopicCentroidRanker:
    """Ranks topics by cosine similarity between a query and topic centroids.

    With ``center=True`` both the query and the centroids are shifted by the
    dataset mean (the count-weighted mean of all topic centroids, noise
    included) before the cosine. Averages of many vectors all drift toward the
    dataset mean, so raw cosines bunch together; centring compares directions
    away from it instead.

    A topic scores the best of its own centroid and its subtopics' centroids,
    so a query matching one strand of a merged topic is not diluted by the
    others.
    """

    def __init__(self, center: bool = CENTER_BY_DEFAULT) -> None:
        self.center = center

    def origin(self, topics: CentroidTable) -> np.ndarray:
        """The point cosines are measured from: the topic-level dataset mean, or zero.

        Use the topic table's origin for subtopic tables too (same dataset).
        """
        return topics.mean() if self.center else np.zeros(topics.dim)

    def rank(
        self,
        query: np.ndarray,
        topics: CentroidTable,
        subtopics: CentroidTable | None = None,
        limit: int = 20,
    ) -> list[RankedTopic]:
        query = np.asarray(query, dtype=np.float64).ravel()
        if query.shape[0] != topics.dim:
            raise ValueError(
                f"Query embedding dimension {query.shape[0]} does not match "
                f"topic centroid dimension {topics.dim}"
            )

        origin = self.origin(topics)
        query_dir = unit_rows(query - origin)

        best: dict[int, RankedTopic] = {}
        for topic_id, sim in zip(
            topics.ids.tolist(), self._cosines(topics, origin, query_dir), strict=True
        ):
            if topic_id != NOISE_TOPIC_ID:
                best[topic_id] = RankedTopic(topic_id=topic_id, similarity=sim)

        if subtopics is not None and subtopics.parents is not None:
            sims = self._cosines(subtopics, origin, query_dir)
            for sub_id, parent, sim in zip(
                subtopics.ids.tolist(), subtopics.parents.tolist(), sims, strict=True
            ):
                current = best.get(parent)
                if sub_id == NOISE_TOPIC_ID or current is None:
                    continue
                if sim > current.similarity:
                    best[parent] = RankedTopic(
                        topic_id=parent, similarity=sim, matched_subtopic_id=sub_id
                    )

        ranked = sorted(best.values(), key=lambda r: (-r.similarity, r.topic_id))
        return ranked[:limit]

    @staticmethod
    def _cosines(table: CentroidTable, origin: np.ndarray, query_dir: np.ndarray) -> list[float]:
        return [float(s) for s in table.unit_directions(origin) @ query_dir]
