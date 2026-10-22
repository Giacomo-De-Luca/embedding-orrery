"""Merge trees over a collection's base topics: the live topic slider and tree reductions.

Backs the ``topicMergeTree`` GraphQL query and the tree strategies of topic
reduction, so a saved cut is exactly the previewed one.

- ``average`` and ``ward`` trees are built from the base topics' embedding
  centroids, in the centred cosine space topic search uses
  (``TopicCentroidService.ranker``). Stored centroids are used when the
  extraction has them; otherwise they are computed from the vectors, in
  memory and capped by ``fallback_centroid_max_items``, so extractions older
  than stored centroids (and the read-only demo) get a tree too.
- ``density`` trees come from the HDBSCAN fit and are stored with the
  extraction; one is offered only when its leaves are the base topics.

Base topics are the subtopic level of a reduced extraction (its original
clusters), else the topic level. Base centroids and trees are cached per
extraction (everything is read by extraction id, so a cached entry always
belongs to its key); labels are re-read per request because they can be
renamed. Imports scipy through ``utils/topic_merge_tree``, so the GraphQL
resolver imports this module lazily.
"""

import logging
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..utils.topic_centroids import NOISE_TOPIC_ID, CentroidTable
from ..utils.topic_merge_tree import CentroidLinkage, TopicMergeTree
from .topic_centroid_service import (
    SUBTOPIC_LEVEL,
    TOPIC_LEVEL,
    Preloaded,
    TopicCentroidService,
)
from .topic_reduction_types import SETTINGS, TREE_LINKAGES, TopicReductionSettings

logger = logging.getLogger("orrery." + __name__)

SEMANTIC_LINKAGES = CentroidLinkage.METHODS


@dataclass(frozen=True, eq=False)
class BaseCentroids:
    """Centroids of an extraction's base topics and the origin they are compared from."""

    level: str  # TOPIC_LEVEL or SUBTOPIC_LEVEL
    table: CentroidTable  # base topics, the noise row included when present
    origin: np.ndarray

    @property
    def leaf_ids(self) -> list[int]:
        return [int(t) for t in self.table.ids if t != NOISE_TOPIC_ID]

    @property
    def noise_count(self) -> int:
        return int(self.table.counts[self.table.ids == NOISE_TOPIC_ID].sum())


@dataclass(frozen=True, eq=False)
class TopicTreeView:
    """A merge tree of a collection's active extraction, as the slider shows it."""

    extraction_id: str
    base_level: str
    tree: TopicMergeTree
    labels: dict[int, str]  # base topic id -> current label
    noise_count: int
    available_linkages: tuple[str, ...]


class TopicMergeTreeService:
    _shared: ClassVar["TopicMergeTreeService | None"] = None

    def __init__(
        self,
        duckdb,
        centroid_service: TopicCentroidService | None = None,
        settings: TopicReductionSettings = SETTINGS,
    ) -> None:
        self._db = duckdb
        self._centroids = centroid_service or TopicCentroidService(duckdb=duckdb)
        self._settings = settings
        self._bases: OrderedDict[str, BaseCentroids | None] = OrderedDict()
        self._trees: OrderedDict[tuple[str, str], TopicMergeTree | None] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def centroids(self) -> TopicCentroidService:
        """The centroid service the trees are built from (shared with reductions)."""
        return self._centroids

    @classmethod
    def shared(cls, duckdb) -> "TopicMergeTreeService":
        """One long-lived instance per DuckDB client, so the caches outlive a request."""
        if cls._shared is None or cls._shared._db is not duckdb:
            cls._shared = cls(duckdb=duckdb)
        return cls._shared

    def view(self, collection_name: str, linkage: str) -> TopicTreeView | None:
        """The ``linkage`` tree of the collection's active extraction.

        None when the collection has no topics, fewer than 3 or more than
        ``max_tree_topics`` base topics, no centroids (and too many items to
        compute them), or no density tree for ``linkage="density"``.
        """
        if linkage not in TREE_LINKAGES:
            raise ValueError(f"Unknown linkage {linkage!r}; expected one of {TREE_LINKAGES}")
        extraction_id = self._db.get_active_extraction_id(collection_name)
        if extraction_id is None:
            return None
        base = self._base(collection_name, extraction_id)
        if base is None or not 3 <= len(base.leaf_ids) <= self._settings.max_tree_topics:
            return None

        density = self.density_tree(extraction_id, base.leaf_ids)
        tree = (
            density if linkage == "density" else self._semantic_tree(extraction_id, base, linkage)
        )
        if tree is None:
            return None
        if self._db.get_active_extraction_id(collection_name) != extraction_id:
            return None  # a new extraction became active meanwhile
        available = SEMANTIC_LINKAGES + (("density",) if density is not None else ())
        return TopicTreeView(
            extraction_id=extraction_id,
            base_level=base.level,
            tree=tree,
            labels=self._labels(extraction_id, base.level, base.leaf_ids),
            noise_count=base.noise_count,
            available_linkages=available,
        )

    def tree_for_assignments(
        self,
        collection_name: str,
        extraction_id: str | None,
        item_topics: Mapping[str, int],
        linkage: str,
        preloaded: Preloaded | None = None,
    ) -> TopicMergeTree:
        """The average / ward tree a reduction cuts.

        For a stored extraction it is built exactly as ``view`` builds it
        (same centroids, deterministic linkage), so a saved cut is the
        previewed one even from another instance; for one being made
        (``extraction_id`` None) it is built from the vectors of
        ``item_topics`` (``preloaded`` when given).

        Raises:
            ValueError: a base topic has no centroid, or there are more than
                ``max_tree_topics`` topics.
        """
        if linkage not in SEMANTIC_LINKAGES:
            raise ValueError(f"No centroid tree for linkage {linkage!r}")
        expected = sorted({int(t) for t in item_topics.values() if t != NOISE_TOPIC_ID})
        if len(expected) > self._settings.max_tree_topics:
            raise ValueError(
                f"{len(expected)} topics: merge trees are built for at most "
                f"{self._settings.max_tree_topics} (config/topic_reduction.json)"
            )
        if extraction_id is None:
            table = self._centroids.tables_for_assignments(
                collection_name, [item_topics], preloaded
            )[0]
            base = BaseCentroids(TOPIC_LEVEL, table, self._centroids.ranker.origin(table))
            tree = self.tree_from_base(base, linkage)
        else:
            base = self._base(collection_name, extraction_id)
            if base is None:
                raise ValueError(
                    f"No centroids could be computed for the topics of {collection_name!r}"
                )
            tree = self._semantic_tree(extraction_id, base, linkage)
        if tree.leaf_ids.tolist() != expected:
            missing = sorted(set(expected) - set(tree.leaf_ids.tolist()))
            raise ValueError(
                f"Topics without vectors (or with stale centroids) cannot be merged: {missing[:5]}"
            )
        return tree

    def density_tree(self, extraction_id: str, leaf_ids: list[int]) -> TopicMergeTree | None:
        """The extraction's stored density tree, if its leaves are ``leaf_ids``.

        Read once per extraction (written before the extraction is activated,
        never changed after), including "there is none".
        """
        key = (extraction_id, "density")
        with self._lock:
            cached = key in self._trees
            if cached:
                self._trees.move_to_end(key)
                tree = self._trees[key]
        if not cached:
            tree = self._stored_density_tree(extraction_id)
            with self._lock:
                self._remember(self._trees, key, tree)
        if tree is None or tree.leaf_ids.tolist() != list(leaf_ids):
            return None
        return tree

    def _stored_density_tree(self, extraction_id: str) -> TopicMergeTree | None:
        row = self._db.get_topic_merge_tree(extraction_id, "density")
        if row is None:
            return None
        try:
            return TopicMergeTree(
                linkage="density",
                leaf_ids=row["leaf_ids"],
                leaf_sizes=row["leaf_sizes"],
                children=row["children"],
                heights=row["heights"],
                noise_merges=dict(
                    zip(row["noise_item_ids"], row["noise_merges"].tolist(), strict=True)
                ),
            )
        except ValueError as exc:
            logger.warning("Ignoring the stored density tree of %s: %s", extraction_id, exc)
            return None

    def tree_from_base(self, base: BaseCentroids, linkage: str) -> TopicMergeTree:
        """Average or ward tree over the base topics' centred centroid directions."""
        keep = base.table.ids != NOISE_TOPIC_ID
        directions = base.table.unit_directions(base.origin)[keep]
        return CentroidLinkage(linkage).build(
            base.table.ids[keep], directions, base.table.counts[keep]
        )

    def _semantic_tree(
        self, extraction_id: str, base: BaseCentroids, linkage: str
    ) -> TopicMergeTree:
        key = (extraction_id, linkage)
        with self._lock:
            if key in self._trees:
                self._trees.move_to_end(key)
                return self._trees[key]
        tree = self.tree_from_base(base, linkage)
        with self._lock:
            self._remember(self._trees, key, tree)
        return tree

    def _base(self, collection_name: str, extraction_id: str) -> BaseCentroids | None:
        with self._lock:
            if extraction_id in self._bases:
                self._bases.move_to_end(extraction_id)
                return self._bases[extraction_id]
        topics = self._centroids.stored_table(extraction_id, TOPIC_LEVEL)
        if topics is not None:
            subtopics = self._centroids.stored_table(extraction_id, SUBTOPIC_LEVEL)
            base = BaseCentroids(
                level=SUBTOPIC_LEVEL if subtopics is not None else TOPIC_LEVEL,
                table=subtopics if subtopics is not None else topics,
                origin=self._centroids.ranker.origin(topics),
            )
        else:
            if self._db.count_topic_assignments(extraction_id) > (
                self._settings.fallback_centroid_max_items
            ):
                # Not cached: the extraction's centroids may be stored shortly.
                return None
            base = self._computed_base(collection_name, extraction_id)
        # A failed computation (vectors unreadable) is remembered too, so a
        # request does not re-stream the whole collection each time.
        with self._lock:
            self._remember(self._bases, extraction_id, base)
        return base

    def _computed_base(self, collection_name: str, extraction_id: str) -> BaseCentroids | None:
        """Base centroids from the vectors, for an extraction without stored ones."""
        rows = self._db.get_topic_assignments_raw(
            extraction_id, ["item_id", "topic_id", "subtopic_id"]
        )
        if not rows:
            return None
        topic_of = {item: int(topic) for item, topic, _ in rows}
        subtopic_of = {item: int(sub) for item, _, sub in rows if sub is not None}
        try:
            tables = self._centroids.tables_for_assignments(
                collection_name, [topic_of, subtopic_of] if subtopic_of else [topic_of]
            )
        except ValueError as exc:
            logger.info("No merge tree for %s: %s", collection_name, exc)
            return None
        return BaseCentroids(
            level=SUBTOPIC_LEVEL if subtopic_of else TOPIC_LEVEL,
            table=tables[-1],
            origin=self._centroids.ranker.origin(tables[0]),
        )

    def _labels(self, extraction_id: str, level: str, leaf_ids: list[int]) -> dict[int, str]:
        if level == SUBTOPIC_LEVEL:
            found = self._db.get_subtopic_labels(extraction_id, leaf_ids)
        else:
            found = self._db.get_topic_labels(extraction_id, leaf_ids)
        return {t: found.get(t) or f"Topic {t}" for t in leaf_ids}

    def _remember(self, cache: OrderedDict, key, value) -> None:
        cache[key] = value
        while len(cache) > self._settings.tree_cache_size:
            cache.popitem(last=False)
