"""Topic centroids: computing, storing and searching them.

``TopicCentroidService`` is the single code path for topic centroids:

- ``refresh`` recomputes and stores the centroids of an extraction from its
  ``topic_assignments`` (topic level and, after a reduction, subtopic level).
  Called after topic extraction / reduction and by the backfill script.
- ``centroids_for_assignments`` computes centroids for in-memory assignments,
  in a requested row order (semantic topic reduction).
- ``search`` ranks a collection's topics against a text query.

Member vectors and queries are L2-normalised: centroids are mean directions,
matching cosine similarity even for embedders that do not normalise their
output (e.g. QWEN, which also prefixes queries only).

Dependencies are injected (DuckDB client, query embedder, vector-batch
source) so the GraphQL layer, the extraction pipeline and tests each wire
their own. Imports stay numpy-only (torch-free import boundary).
"""

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..utils.embedding_loader import DEFAULT_LOAD_BATCH_SIZE, iter_embedding_batches
from ..utils.topic_centroids import (
    CentroidTable,
    TopicCentroidAccumulator,
    TopicCentroidRanker,
    unit_rows,
)

logger = logging.getLogger("orrery." + __name__)

TOPIC_LEVEL = "topic"
SUBTOPIC_LEVEL = "subtopic"

# Rows per accumulation step for preloaded vectors: bounds the per-step copies
# (normalisation, label selection) instead of copying the whole matrix.
PRELOADED_BATCH_SIZE = DEFAULT_LOAD_BATCH_SIZE

VectorBatches = Callable[[str], Iterable[tuple[list[str], np.ndarray]]]
QueryEmbedder = Callable[..., np.ndarray]
Preloaded = tuple[Sequence[str], np.ndarray]


@dataclass(frozen=True)
class TopicMatch:
    topic_id: int
    similarity: float
    matched_subtopic_id: int | None = None
    matched_subtopic_label: str | None = None


class TopicCentroidService:
    def __init__(
        self,
        duckdb,
        embed_query: QueryEmbedder | None = None,
        vector_batches: VectorBatches = iter_embedding_batches,
        ranker: TopicCentroidRanker | None = None,
    ) -> None:
        self._db = duckdb
        self._embed_query = embed_query
        self._vector_batches = vector_batches
        # Public so other centroid comparisons share search's centring choice.
        self.ranker = ranker or TopicCentroidRanker()

    # ------------------------------------------------------------------
    # Computing
    # ------------------------------------------------------------------

    def refresh(
        self, collection_name: str, extraction_id: str, preloaded: Preloaded | None = None
    ) -> int:
        """Recompute and store an extraction's centroids. Returns rows written.

        Args:
            preloaded: ``(ids, vectors)`` already in memory (e.g. the raw
                vectors loaded for clustering); skips reading the vector store.
        """
        rows = self._db.get_topic_assignments_raw(
            extraction_id, ["item_id", "topic_id", "subtopic_id"]
        )
        if not rows:
            raise ValueError(f"Extraction {extraction_id!r} has no topic assignments")

        topic_of = {item_id: int(topic_id) for item_id, topic_id, _ in rows}
        subtopic_of = {item_id: int(sub_id) for item_id, _, sub_id in rows if sub_id is not None}

        topic_acc = TopicCentroidAccumulator(normalize=True)
        subtopic_acc = TopicCentroidAccumulator(normalize=True)
        for ids, vectors in self._batches(collection_name, preloaded):
            topic_acc.add_mapped(ids, vectors, topic_of)
            subtopic_acc.add_mapped(ids, vectors, subtopic_of)

        topics = topic_acc.table()
        records = self._records(TOPIC_LEVEL, topics)
        if subtopic_of:
            parents = self._subtopic_parents(rows)
            records += self._records(SUBTOPIC_LEVEL, subtopic_acc.table(), parents)

        covered = int(topics.counts.sum())
        if covered < len(topic_of):
            logger.warning(
                "%d/%d assigned items of %s have no vector; centroids cover the rest",
                len(topic_of) - covered,
                len(topic_of),
                collection_name,
            )
        return self._db.replace_topic_centroids(extraction_id, records)

    def centroids_for_assignments(
        self,
        collection_name: str,
        item_topics: Mapping[str, int],
        order: Sequence[int],
        preloaded: Preloaded | None = None,
    ) -> np.ndarray:
        """Centroid matrix for in-memory assignments, one row per ``order`` entry.

        Raises:
            ValueError: a topic in ``order`` has no member with a vector.
        """
        acc = TopicCentroidAccumulator(normalize=True)
        for ids, vectors in self._batches(collection_name, preloaded):
            acc.add_mapped(ids, vectors, item_topics)
        return acc.table(order=order).means

    def _batches(
        self, collection_name: str, preloaded: Preloaded | None
    ) -> Iterable[tuple[Sequence[str], np.ndarray]]:
        if preloaded is None:
            return self._vector_batches(collection_name)
        ids, vectors = preloaded
        return (
            (
                ids[start : start + PRELOADED_BATCH_SIZE],
                vectors[start : start + PRELOADED_BATCH_SIZE],
            )
            for start in range(0, len(ids), PRELOADED_BATCH_SIZE)
        )

    @staticmethod
    def _subtopic_parents(rows: list[tuple]) -> dict[int, int]:
        """Each subtopic's parent topic (majority vote over its items)."""
        votes: dict[int, Counter] = {}
        for _, topic_id, sub_id in rows:
            if sub_id is not None:
                votes.setdefault(int(sub_id), Counter())[int(topic_id)] += 1
        return {sub_id: counter.most_common(1)[0][0] for sub_id, counter in votes.items()}

    @staticmethod
    def _records(
        level: str, table: CentroidTable, parents: Mapping[int, int] | None = None
    ) -> list[dict]:
        return [
            {
                "level": level,
                "topic_id": topic_id,
                "parent_topic_id": None if parents is None else parents.get(topic_id),
                "member_count": count,
                "centroid": mean.tolist(),
            }
            for topic_id, count, mean in zip(
                table.ids.tolist(), table.counts.tolist(), table.means, strict=True
            )
        ]

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 20,
        query_prompt: str | None = None,
    ) -> list[TopicMatch]:
        """Rank the collection's active topics by similarity to ``query``."""
        if self._embed_query is None:
            raise ValueError("TopicCentroidService.search needs a query embedder")

        extraction_id, topics, subtopics = self._active(collection_name)
        query_vector = unit_rows(
            self._embed_query(collection_name, query, query_prompt=query_prompt)
        )
        ranked = self.ranker.rank(query_vector, topics, subtopics, limit=limit)

        matched = [r.matched_subtopic_id for r in ranked if r.matched_subtopic_id is not None]
        labels = self._db.get_subtopic_labels(extraction_id, matched)
        return [
            TopicMatch(
                topic_id=r.topic_id,
                similarity=r.similarity,
                matched_subtopic_id=r.matched_subtopic_id,
                matched_subtopic_label=labels.get(r.matched_subtopic_id)
                if r.matched_subtopic_id is not None
                else None,
            )
            for r in ranked
        ]

    def active_centroids(self, collection_name: str) -> tuple[CentroidTable, CentroidTable | None]:
        """Topic and (if reduced) subtopic centroids of the active extraction.

        Raises:
            ValueError: the collection has no topics, or no stored centroids.
        """
        _, topics, subtopics = self._active(collection_name)
        return topics, subtopics

    def _active(self, collection_name: str) -> tuple[str, CentroidTable, CentroidTable | None]:
        extraction_id = self._db.get_active_extraction_id(collection_name)
        if extraction_id is None:
            raise ValueError(f"Collection '{collection_name}' has no topics")
        topics = self._table(extraction_id, TOPIC_LEVEL)
        if topics is None:
            raise ValueError(
                f"Topic centroids have not been computed for '{collection_name}'. "
                "Re-run topic extraction or the topic-centroid backfill."
            )
        return extraction_id, topics, self._table(extraction_id, SUBTOPIC_LEVEL)

    def _table(self, extraction_id: str, level: str) -> CentroidTable | None:
        data = self._db.get_topic_centroids(extraction_id, level)
        if data is None:
            return None
        return CentroidTable(
            ids=data["ids"], means=data["means"], counts=data["counts"], parents=data["parents"]
        )
