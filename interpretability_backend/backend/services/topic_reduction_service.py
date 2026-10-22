"""Topic reduction: from an extraction's base topics to a reduced set, computed but not stored.

``BaseTopics`` are the topics a reduction starts from: the original clusters
of an extraction (its subtopic level once it has been reduced), so
reductions never compound and can always be undone. A cut can also merge
fewer topics than an earlier one.

``TopicReductionService``:

- ``plan`` cuts the base topics with a ``ReductionRequest``: a merge tree
  (average / ward from ``TopicMergeTreeService``, density from the HDBSCAN
  fit) or one of the legacy methods (kept only for comparison);
- ``assemble`` turns a plan into what the extraction tables store: labels
  (unmerged topics keep theirs on a standalone reduction), topic infos,
  assignments, the base topics as subtopics, and an id-keyed hierarchy;
- ``unreduced`` assembles base topics as they are (extraction without
  reduction), with the same labelling.

Persistence is the caller's: ``topic_extraction_service`` syncs the outcome
to DuckDB; the comparison runner only scores plans. Imports the clustering
stack (hdbscan, sklearn, scipy), so modules in the GraphQL schema's import
graph import this one lazily.
"""

import json
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..topic_extraction.cluster_and_label import CtfidfKeywords
from ..topic_extraction.llm_labeling import generate_llm_labels
from ..topic_extraction.topic_reducer import TopicReducer, TopicReductionResult
from ..utils.topic_centroids import NOISE_TOPIC_ID
from ..utils.topic_merge_tree import TopicMergeTree
from .cancel_registry import raise_if_cancelled
from .topic_centroid_service import SUBTOPIC_LEVEL, TOPIC_LEVEL, Preloaded
from .topic_extraction_service import TopicInfoResult
from .topic_merge_tree_service import TopicMergeTreeService
from .topic_reduction_types import ReductionRequest

logger = logging.getLogger("orrery." + __name__)

UNCLUSTERED_LABEL = "Unclustered"
DEFAULT_LANGUAGE = "english"

ProgressCallback = Callable[[int, int], None]


def keyword_label(keywords) -> str:
    """The default label of a topic: its top three keywords."""
    return " | ".join(word for word, _ in keywords[:3])


@dataclass(eq=False)  # ndarray fields: compare by identity
class BaseTopics:
    """The topics a reduction starts from, one entry per assigned item."""

    collection_name: str
    extraction_id: str | None  # None while the extraction is being made
    level: str  # TOPIC_LEVEL, or SUBTOPIC_LEVEL for a reduced extraction
    item_ids: list[str]
    documents: list[str]
    topics: np.ndarray  # base topic of each item (-1 = noise)
    labels: dict[int, str]  # base topic -> current label (noise included)
    keywords: dict[int, list[tuple[str, float]]]  # base topic -> keywords, where known
    config: dict = field(default_factory=dict)  # config snapshot of the source extraction
    preloaded: Preloaded | None = None  # vectors aligned to item_ids (extraction path)
    density_tree: TopicMergeTree | None = None  # noise keyed by item id

    @classmethod
    def load(cls, db, collection_name: str, trees: TopicMergeTreeService) -> "BaseTopics":
        """The base topics of the collection's active extraction.

        Items are the extraction's assignments (not every dataset item), joined
        with their documents in the collection's dataset.
        """
        active = db.get_active_topics(collection_name)
        if not active:
            raise ValueError(
                f"Collection '{collection_name}' has no topics. Run extractTopics first."
            )
        extraction_id = active["id"]
        collection = db.get_vector_collection(collection_name)
        dataset_name = collection["dataset_name"] if collection else collection_name
        rows = db.get_assignment_documents(extraction_id, dataset_name)
        if not rows:
            raise ValueError(f"The topics of '{collection_name}' have no assigned items")

        reduced = any(row[4] is not None for row in rows)
        id_index, label_index = (4, 5) if reduced else (2, 3)
        topics = np.array(
            [NOISE_TOPIC_ID if row[id_index] is None else int(row[id_index]) for row in rows],
            dtype=np.int64,
        )
        labels: dict[int, str] = {}
        for row, topic in zip(rows, topics.tolist(), strict=True):
            if topic not in labels:
                labels[topic] = row[label_index] or f"Topic {topic}"
        if NOISE_TOPIC_ID in labels:
            labels[NOISE_TOPIC_ID] = UNCLUSTERED_LABEL

        keywords = {}
        if not reduced:
            keywords = {
                int(info["topic_id"]): [
                    (kw["word"], kw["score"]) for kw in info.get("keywords") or []
                ]
                for info in active.get("topics", [])
            }
        config = active.get("config")
        config = json.loads(config) if isinstance(config, str) else dict(config or {})
        leaf_ids = sorted(t for t in labels if t != NOISE_TOPIC_ID)
        return cls(
            collection_name=collection_name,
            extraction_id=extraction_id,
            level=SUBTOPIC_LEVEL if reduced else TOPIC_LEVEL,
            item_ids=[row[0] for row in rows],
            documents=[row[1] or "" for row in rows],
            topics=topics,
            labels=labels,
            keywords=keywords,
            config=config,
            density_tree=trees.density_tree(extraction_id, leaf_ids),
        )

    @classmethod
    def from_clustering(
        cls,
        collection_name: str,
        item_ids: list[str],
        documents: list[str],
        labels,
        keywords: Mapping[int, list[tuple[str, float]]],
        config: dict,
        preloaded: Preloaded | None = None,
        density_tree: TopicMergeTree | None = None,
    ) -> "BaseTopics":
        """Base topics of an extraction being made: the clusters with keyword labels."""
        topics = np.asarray(labels, dtype=np.int64)
        names = {
            int(t): UNCLUSTERED_LABEL if t == NOISE_TOPIC_ID else keyword_label(keywords.get(t, []))
            for t in np.unique(topics).tolist()
        }
        return cls(
            collection_name=collection_name,
            extraction_id=None,
            level=TOPIC_LEVEL,
            item_ids=list(item_ids),
            documents=list(documents),
            topics=topics,
            labels=names,
            keywords={int(t): list(kws) for t, kws in keywords.items()},
            config=dict(config),
            preloaded=preloaded,
            density_tree=density_tree,
        )

    @property
    def topic_ids(self) -> list[int]:
        """Base topics, noise excluded."""
        return sorted(t for t in self.labels if t != NOISE_TOPIC_ID)

    @property
    def language(self) -> str | None:
        return self.config.get("language", DEFAULT_LANGUAGE)

    def item_topics(self) -> dict[str, int]:
        return dict(zip(self.item_ids, self.topics.tolist(), strict=True))

    def documents_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Document_ID": range(len(self.item_ids)),
                "Document": self.documents,
                "Topic": self.topics,
            }
        )


@dataclass(eq=False)
class ReductionPlan:
    """A computed (not stored) reduction of some base topics."""

    base: BaseTopics
    request: ReductionRequest
    result: TopicReductionResult
    tree: TopicMergeTree | None = None
    fallback: str | None = None  # set when the request could not run as asked

    @property
    def changed(self) -> bool:
        """Whether the cut merged topics or absorbed noise."""
        return self.result.absorbed_noise > 0 or any(
            len(members) > 1 for members in self.result.topic_hierarchy.values()
        )

    @property
    def num_topics_before(self) -> int:
        """Base topics, noise not counted (the legacy results count it)."""
        return len(self.base.topic_ids)

    def item_topics(self) -> dict[str, int]:
        """Reduced topic of every item."""
        topics = [int(t) for t in self.result.documents_df["Topic"]]
        return dict(zip(self.base.item_ids, topics, strict=True))


@dataclass
class ReductionOutcome:
    """What the extraction tables store for a (possibly unreduced) set of topics."""

    topic_infos: list[TopicInfoResult]
    topic_labels: dict[int, str]
    topic_assignments: dict[str, int]
    subtopic_assignments: dict[str, int] | None = None  # item -> base topic
    subtopic_labels: dict[int, str] | None = None
    topic_hierarchy: dict[str, list[int]] | None = None  # topic id -> base topic ids
    num_noise: int = 0

    @property
    def num_topics(self) -> int:
        return sum(1 for info in self.topic_infos if info.topic_id != NOISE_TOPIC_ID)


class TopicReductionService:
    def __init__(self, duckdb, trees: TopicMergeTreeService | None = None) -> None:
        self._db = duckdb
        self._trees = trees or TopicMergeTreeService(duckdb=duckdb)

    @property
    def trees(self) -> TopicMergeTreeService:
        """The merge trees this service cuts (also what ``BaseTopics.load`` reads)."""
        return self._trees

    def plan(
        self,
        base: BaseTopics,
        request: ReductionRequest,
        cancel_event: threading.Event | None = None,
    ) -> ReductionPlan:
        """Cut ``base`` as ``request`` says. Reads vectors when needed, writes nothing.

        Raises:
            ValueError: fewer than two base topics, or the request cannot run
                (e.g. density without a density tree).
        """
        if len(base.topic_ids) < 2:
            raise ValueError("A reduction needs at least two topics")
        documents_df = base.documents_frame()
        topics_data = {t: base.keywords.get(t, []) for t in sorted(base.labels)}
        if request.is_tree:
            return self._tree_plan(base, request, documents_df, topics_data, cancel_event)
        return self._legacy_plan(base, request, documents_df, topics_data, cancel_event)

    def assemble(
        self,
        plan: ReductionPlan,
        *,
        keep_unmerged_labels: bool,
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ReductionOutcome:
        """Labels, infos and assignments of a plan.

        ``keep_unmerged_labels``: a topic made of one base topic keeps that
        topic's label, and LLM relabelling (``request.regenerate_labels``) is
        limited to merged topics (standalone reductions). Otherwise every
        topic gets a fresh keyword label, or an LLM label (extraction).
        """
        base, result = plan.base, plan.result
        hierarchy = result.topic_hierarchy
        merged = {t for t, members in hierarchy.items() if len(members) > 1}
        labels: dict[int, str] = {}
        to_relabel = {}
        for topic_id, keywords in sorted(result.topics_data.items()):
            if topic_id == NOISE_TOPIC_ID:
                labels[topic_id] = UNCLUSTERED_LABEL
                continue
            keep = keep_unmerged_labels and topic_id not in merged and topic_id in hierarchy
            labels[topic_id] = (
                base.labels[hierarchy[topic_id][0]] if keep else keyword_label(keywords)
            )
            if not keep:
                to_relabel[topic_id] = keywords
        if plan.request.regenerate_labels and to_relabel:
            labels.update(
                generate_llm_labels(
                    topics_data=to_relabel,
                    documents_df=result.documents_df,
                    llm_provider=plan.request.llm_provider,
                    llm_model=plan.request.llm_model,
                    progress_callback=progress,
                    cancel_event=cancel_event,
                )
            )

        counts = result.documents_df["Topic"].value_counts().to_dict()
        infos = [
            TopicInfoResult(
                topic_id=int(topic_id),
                keywords=keywords,
                label=labels[topic_id],
                count=int(counts.get(topic_id, 0)),
                subtopics=(
                    [base.labels[m] for m in hierarchy[topic_id]]
                    if plan.changed and topic_id in hierarchy
                    else None
                ),
            )
            for topic_id, keywords in sorted(result.topics_data.items())
        ]
        outcome = ReductionOutcome(
            topic_infos=infos,
            topic_labels=labels,
            topic_assignments=plan.item_topics(),
            num_noise=int(counts.get(NOISE_TOPIC_ID, 0)),
        )
        if plan.changed:
            outcome.subtopic_assignments = base.item_topics()
            outcome.subtopic_labels = {**base.labels, NOISE_TOPIC_ID: UNCLUSTERED_LABEL}
            outcome.topic_hierarchy = {str(t): members for t, members in hierarchy.items()}
        return outcome

    def unreduced(
        self,
        base: BaseTopics,
        *,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ReductionOutcome:
        """Base topics stored as they are: keyword labels, or LLM labels when
        ``llm_provider`` is given."""
        labels = {
            t: UNCLUSTERED_LABEL if t == NOISE_TOPIC_ID else keyword_label(base.keywords.get(t, []))
            for t in sorted(base.labels)
        }
        if llm_provider is not None:
            labels.update(
                generate_llm_labels(
                    topics_data={t: base.keywords.get(t, []) for t in base.topic_ids},
                    documents_df=base.documents_frame(),
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    progress_callback=progress,
                    cancel_event=cancel_event,
                )
            )
        topics, counts = np.unique(base.topics, return_counts=True)
        count_of = dict(zip(topics.tolist(), counts.tolist(), strict=True))
        infos = [
            TopicInfoResult(
                topic_id=t,
                keywords=base.keywords.get(t, []),
                label=labels[t],
                count=int(count_of.get(t, 0)),
            )
            for t in sorted(base.labels)
        ]
        return ReductionOutcome(
            topic_infos=infos,
            topic_labels=labels,
            topic_assignments=base.item_topics(),
            num_noise=int(count_of.get(NOISE_TOPIC_ID, 0)),
        )

    def config_snapshot(self, plan: ReductionPlan) -> dict:
        """The reduced extraction's config: the source's, plus how it was cut."""
        return {
            **plan.base.config,
            "language": plan.base.language,
            "reduction": {
                **plan.request.snapshot(),
                "base_extraction_id": plan.base.extraction_id,
                "absorbed_noise": plan.result.absorbed_noise,
            },
        }

    def _tree_plan(self, base, request, documents_df, topics_data, cancel_event) -> ReductionPlan:
        k = len(base.topic_ids)
        reducer = TopicReducer(documents_df, topics_data, language=base.language)
        if request.strategy == "density":
            tree = base.density_tree
            if tree is None:
                raise ValueError(
                    "Density merging needs a density tree: re-extract the topics with HDBSCAN"
                )
        else:
            tree = self._trees.tree_for_assignments(
                base.collection_name,
                base.extraction_id,
                base.item_topics(),
                request.strategy,
                preloaded=base.preloaded,
            )
        raise_if_cancelled(cancel_event)

        position = {item: i for i, item in enumerate(base.item_ids)}
        noise_merges = {
            position[item]: row for item, row in tree.noise_merges.items() if item in position
        }
        result = reducer.reduce_with_tree(
            tree,
            n_topics=min(request.n_topics, k) if request.method == "fixed_n" else None,
            distance_threshold=request.distance_threshold if request.method == "auto" else None,
            noise_merges=noise_merges,
            absorb_noise=request.absorb_noise,
        )
        return ReductionPlan(base, request, result, tree=tree)

    def _legacy_plan(self, base, request, documents_df, topics_data, cancel_event) -> ReductionPlan:
        ctfidf = CtfidfKeywords(language=base.language).extract(documents_df)
        embeddings, fallback = None, None
        if request.strategy == "legacy_semantic":
            embeddings, fallback = self._legacy_centroids(base, sorted(topics_data))
        raise_if_cancelled(cancel_event)

        reducer = TopicReducer(
            documents_df,
            topics_data,
            ctfidf_matrix=ctfidf.matrix,
            ctfidf_words=ctfidf.words,
            language=base.language,
            topic_embeddings=embeddings,
        )
        use_ctfidf = embeddings is None
        if request.method == "fixed_n":
            # The legacy method counts the noise "topic" in its target.
            with_noise = request.n_topics + (1 if NOISE_TOPIC_ID in topics_data else 0)
            result = reducer.reduce_to_n_topics(n_topics=with_noise, use_ctfidf=use_ctfidf)
        else:
            result = reducer.auto_reduce_topics(use_ctfidf=use_ctfidf)
        if not result.topic_hierarchy:  # the legacy "nothing to merge" early return
            result = self._identity(base, request, documents_df)
        return ReductionPlan(base, request, result, fallback=fallback)

    @staticmethod
    def _identity(base: BaseTopics, request, documents_df) -> TopicReductionResult:
        """A cut that keeps every base topic, with keywords for each."""
        topics_data = CtfidfKeywords(language=base.language).extract(documents_df).topics_data
        return TopicReductionResult(
            documents_df=documents_df,
            topics_data=topics_data,
            topic_mappings={t: t for t in sorted(base.labels)},
            topic_hierarchy={t: [t] for t in base.topic_ids},
            num_topics_before=len(base.topic_ids),
            num_topics_after=len(base.topic_ids),
            reduction_method=request.method,
        )

    def _legacy_centroids(self, base: BaseTopics, order: list[int]):
        """Raw (uncentred) centroids for legacy semantic reduction; on failure,
        c-TF-IDF is used instead and the reason returned."""
        try:
            return (
                self._trees.centroids.centroids_for_assignments(
                    base.collection_name, base.item_topics(), order=order, preloaded=base.preloaded
                ),
                None,
            )
        except Exception as exc:  # any vector-store failure: JobCancelled is a BaseException
            logger.warning(
                "Semantic topic centroids unavailable for %s (%s); reducing with c-TF-IDF instead.",
                base.collection_name,
                exc,
            )
            return None, "Embeddings unavailable; reducing topics with c-TF-IDF..."
