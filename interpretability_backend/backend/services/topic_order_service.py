"""Topic and subtopic values ordered by embedding-centroid similarity.

Backs the ``topicSimilarityOrder`` GraphQL query, the "Similarity" sort of the
Analytics heatmap. Topics are ordered by average-linkage clustering with
optimal leaf ordering over their stored centroids, in the same centred cosine
space topic search uses (the shared ``TopicCentroidService.ranker``).
Subtopics are grouped under their parent topics, in topic order, and ordered
by similarity within each parent.

Orders are cached as ids per extraction and level; labels are looked up on
every request because they can be renamed. Imports scipy through
``similarity_ordering``, so the GraphQL resolver imports this module lazily.
"""

import logging
import threading
from collections import OrderedDict
from typing import ClassVar

import numpy as np

from ..utils.similarity_ordering import SimilarityOrdering
from ..utils.topic_centroids import NOISE_TOPIC_ID, CentroidTable
from .topic_centroid_service import SUBTOPIC_LEVEL, TOPIC_LEVEL, TopicCentroidService

logger = logging.getLogger("orrery." + __name__)

# Frontend field -> (centroid level, whether its values are labels or ids).
_FIELDS: dict[str, tuple[str, str]] = {
    "topic_label": (TOPIC_LEVEL, "label"),
    "topic_id": (TOPIC_LEVEL, "id"),
    "subtopic_label": (SUBTOPIC_LEVEL, "label"),
    "subtopic_id": (SUBTOPIC_LEVEL, "id"),
}

# The collection resolver shows a topic without a label as "Unclustered"; mirror it.
_UNLABELLED_TOPIC = "Unclustered"

_CACHE_SIZE = 32


class TopicOrderService:
    _shared: ClassVar["TopicOrderService | None"] = None

    def __init__(
        self,
        duckdb,
        centroid_service: TopicCentroidService | None = None,
        ordering: SimilarityOrdering | None = None,
    ) -> None:
        self._db = duckdb
        self._centroids = centroid_service or TopicCentroidService(duckdb=duckdb)
        self._ordering = ordering or SimilarityOrdering()
        self._cache: OrderedDict[tuple[str, str], list[int] | None] = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def shared(cls, duckdb) -> "TopicOrderService":
        """One long-lived instance per DuckDB client, so the cache outlives a request."""
        if cls._shared is None or cls._shared._db is not duckdb:
            cls._shared = cls(duckdb=duckdb)
        return cls._shared

    def value_order(self, collection_name: str, field: str) -> list[str] | None:
        """Values of ``field`` in similarity order, as item metadata carries them.

        None when ``field`` is not a topic field, the collection has no active
        topic extraction, or its centroids have not been computed.
        """
        spec = _FIELDS.get(field)
        if spec is None:
            return None
        level, kind = spec
        extraction_id = self._db.get_active_extraction_id(collection_name)
        if extraction_id is None:
            return None
        ids = self._ids_in_order(collection_name, extraction_id, level)
        if ids is None:
            return None
        if kind == "id":
            return [str(i) for i in ids]
        return self._labels(extraction_id, level, ids)

    def _ids_in_order(
        self, collection_name: str, extraction_id: str, level: str
    ) -> list[int] | None:
        key = (extraction_id, level)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        try:
            topics, subtopics = self._centroids.active_centroids(collection_name)
        except ValueError as exc:
            logger.info("No similarity order for '%s': %s", collection_name, exc)
            return None

        origin = self._centroids.ranker.origin(topics)
        topic_order = self._order_topics(topics, origin)
        orders = {
            TOPIC_LEVEL: topic_order,
            SUBTOPIC_LEVEL: (
                self._order_subtopics(subtopics, origin, topic_order)
                if subtopics is not None and subtopics.parents is not None
                else None
            ),
        }
        with self._lock:
            for each_level, ids in orders.items():
                self._cache[(extraction_id, each_level)] = ids
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        return orders[level]

    def _order_topics(self, topics: CentroidTable, origin: np.ndarray) -> list[int]:
        keep = topics.ids != NOISE_TOPIC_ID
        ids = topics.ids[keep]
        directions = topics.unit_directions(origin)[keep]
        return [int(ids[k]) for k in self._ordering.order(directions).order]

    def _order_subtopics(
        self, subtopics: CentroidTable, origin: np.ndarray, topic_order: list[int]
    ) -> list[int]:
        """Subtopics in parent blocks (topic order); the topic-level origin, as in search."""
        keep = subtopics.ids != NOISE_TOPIC_ID
        ids = subtopics.ids[keep]
        directions = subtopics.unit_directions(origin)[keep]
        order = self._ordering.grouped(directions, subtopics.parents[keep], topic_order)
        return [int(ids[k]) for k in order]

    def _labels(self, extraction_id: str, level: str, ids: list[int]) -> list[str]:
        """Current labels in id order; unlabelled subtopics are skipped, duplicates keep their first place."""
        if level == TOPIC_LEVEL:
            found = self._db.get_topic_labels(extraction_id, ids)
            labels = [found.get(i) or _UNLABELLED_TOPIC for i in ids]
        else:
            found = self._db.get_subtopic_labels(extraction_id, ids)
            labels = [found.get(i) for i in ids]
        return list(dict.fromkeys(label for label in labels if label is not None))
