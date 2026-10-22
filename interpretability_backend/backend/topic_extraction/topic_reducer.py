"""
Topic reduction for merging similar topics after extraction.

Two kinds of reduction:
- ``reduce_with_tree``: cut a merge tree over the topics (strategies average,
  ward and density; see ``utils/topic_merge_tree.py``), at N topics or at a
  distance threshold, optionally absorbing noise the tree assigns to merges.
- Legacy BERTopic-style methods, kept only to compare against the trees:
  ``reduce_to_n_topics`` (AgglomerativeClustering to a target count) and
  ``auto_reduce_topics`` (HDBSCAN run on the topic vectors), on c-TF-IDF or
  raw semantic vectors.
"""

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from ..utils.topic_merge_tree import TopicMergeTree, TopicTreeCut
from .cluster_and_label import CtfidfKeywords

logger = logging.getLogger("orrery." + __name__)


@dataclass
class TopicReductionResult:
    """Result of topic reduction operation."""

    documents_df: pd.DataFrame  # Updated with new topic assignments
    topics_data: dict[int, list[tuple[str, float]]]  # Re-extracted keywords
    topic_mappings: dict[int, int]  # old_topic_id -> new_topic_id
    topic_hierarchy: dict[int, list[int]]  # new_topic_id -> [old_topic_ids that merged into it]
    num_topics_before: int
    num_topics_after: int
    reduction_method: str  # "fixed_n" or "auto"
    absorbed_noise: int = 0  # noise documents a tree cut moved into topics


class TopicReducer:
    """
    Reduces topics by merging similar topics together.

    Supports:
    - Merge-tree cuts at N topics or a distance threshold (``reduce_with_tree``)
    - Legacy: fixed target count (AgglomerativeClustering) and automatic
      reduction (HDBSCAN with min_cluster_size=2), on c-TF-IDF or semantic
      vectors
    """

    def __init__(
        self,
        documents_df: pd.DataFrame,
        topics_data: dict[int, list[tuple[str, float]]],
        ctfidf_matrix: sp.csr_matrix | None = None,
        ctfidf_words: np.ndarray | None = None,
        language: str | None = "english",
        topic_embeddings: np.ndarray | None = None,
    ):
        """Initialize with existing topic extraction results.

        Args:
            documents_df: DataFrame with Document_ID, Document, Topic columns
            topics_data: Dict of topic_id -> [(word, score), ...]
            ctfidf_matrix: c-TF-IDF matrix from keyword extraction (legacy
                c-TF-IDF similarity only)
            ctfidf_words: Feature names from CountVectorizer
            language: Stop words language for re-extraction
            topic_embeddings: Embedding-space topic centroids for legacy semantic
                reduction, one row per ``sorted(topics_data)`` entry (noise
                first when present); see ``TopicCentroidService``.
        """
        self.documents_df = documents_df.copy()
        self.topics_data = topics_data
        self.ctfidf_matrix = ctfidf_matrix
        self.ctfidf_words = ctfidf_words
        self.language = language
        self.topic_embeddings = topic_embeddings

        # Track topic sizes for validation
        self.topic_sizes = documents_df["Topic"].value_counts().to_dict()

    def compute_topic_embeddings(self, use_ctfidf: bool = True) -> np.ndarray:
        """Compute embeddings for each topic.

        Args:
            use_ctfidf: If True, use c-TF-IDF vectors. If False, use the
                       injected semantic centroids (averaged document embeddings).

        Returns:
            topic_embeddings: (n_topics, embedding_dim) array
        """
        if use_ctfidf:
            if self.ctfidf_matrix is None:
                raise ValueError("c-TF-IDF reduction needs the c-TF-IDF matrix")
            logger.info("Computing topic embeddings from c-TF-IDF vectors")
            # Each row in ctfidf_matrix is already a topic embedding
            return self.ctfidf_matrix.toarray()
        logger.info("Using precomputed semantic topic centroids")
        return self._semantic_embeddings()

    def _semantic_embeddings(self) -> np.ndarray:
        """The injected topic centroids, validated against ``topics_data``."""
        if self.topic_embeddings is None:
            raise ValueError("Semantic reduction needs precomputed topic centroids")
        expected = len(self.topics_data)
        if self.topic_embeddings.shape[0] != expected:
            raise ValueError(
                f"Semantic topic centroids have {self.topic_embeddings.shape[0]} rows, "
                f"expected one per topic ({expected})"
            )
        return self.topic_embeddings

    def reduce_with_tree(
        self,
        tree: TopicMergeTree,
        n_topics: int | None = None,
        distance_threshold: float | None = None,
        noise_merges: Mapping[int, int] | None = None,
        absorb_noise: bool = False,
    ) -> TopicReductionResult:
        """Cut a merge tree over this reducer's topics.

        Args:
            tree: Merge tree whose leaves are exactly the non-noise topics
            n_topics: Topics to keep (noise not counted); or
            distance_threshold: Apply every merge at or below this height
            noise_merges: Document_ID of a noise document -> the merge row that
                absorbs it (density trees)
            absorb_noise: Move those documents into their group once its merge
                is applied

        New topic ids are numbered by size (documents, absorbed noise
        included), largest first; ties go to the group with the lower
        representative topic id. A cut that merges and absorbs nothing keeps
        the topic ids.
        """
        if (n_topics is None) == (distance_threshold is None):
            raise ValueError("Give exactly one of n_topics or distance_threshold")
        topics = sorted(t for t in self.topics_data if t != -1)
        if sorted(tree.leaf_ids.tolist()) != topics:
            raise ValueError("Merge tree leaves do not match the topics being reduced")

        cut = TopicTreeCut(tree)
        if n_topics is not None:
            n_merges = cut.merges_for(n_topics)
        else:
            n_merges = cut.merges_within(distance_threshold)
        groups = cut.groups(n_merges)

        doc_topics = self.documents_df["Topic"].tolist()
        position = {doc_id: i for i, doc_id in enumerate(self.documents_df["Document_ID"])}
        absorbed = {}
        if absorb_noise and noise_merges:
            absorbed = {
                position[doc_id]: group
                for doc_id, group in cut.absorbed(noise_merges, n_merges).items()
                if doc_topics[position[doc_id]] == -1
            }

        group_size: dict[int, int] = defaultdict(int)
        for topic_id, group in zip(tree.leaf_ids.tolist(), groups.tolist(), strict=True):
            group_size[group] += self.topic_sizes.get(topic_id, 0)
        for group in absorbed.values():
            group_size[group] += 1
        if n_merges == 0 and not absorbed:
            # Nothing merged or absorbed: keep the ids, so trees stored for
            # these topics (e.g. the density tree) still line up with them.
            new_id = dict(zip(groups.tolist(), tree.leaf_ids.tolist(), strict=True))
        else:
            representatives = cut.representatives(groups)
            order = sorted(group_size, key=lambda g: (-group_size[g], representatives[g]))
            new_id = {group: index for index, group in enumerate(order)}

        mapped_topics = {
            topic_id: new_id[group]
            for topic_id, group in zip(tree.leaf_ids.tolist(), groups.tolist(), strict=True)
        }
        if -1 in self.topics_data:
            mapped_topics[-1] = -1
        new_topics = [mapped_topics[t] for t in doc_topics]
        for pos, group in absorbed.items():
            new_topics[pos] = new_id[group]

        documents_df, topics_data = self._merge_topics(new_topics, mapped_topics)

        sizes = dict(zip(tree.leaf_ids.tolist(), tree.leaf_sizes.tolist(), strict=True))
        topic_hierarchy: dict[int, list[int]] = defaultdict(list)
        for topic_id in sorted(mapped_topics, key=lambda t: (-sizes.get(t, 0), t)):
            if topic_id != -1:
                topic_hierarchy[mapped_topics[topic_id]].append(topic_id)

        logger.info(
            "Cut a %s tree: %d -> %d topics, %d noise documents absorbed",
            tree.linkage,
            len(topics),
            len(group_size),
            len(absorbed),
        )
        return TopicReductionResult(
            documents_df=documents_df,
            topics_data=topics_data,
            topic_mappings=mapped_topics,
            topic_hierarchy=dict(sorted(topic_hierarchy.items())),
            num_topics_before=len(topics),
            num_topics_after=len(group_size),
            reduction_method="fixed_n" if n_topics is not None else "auto",
            absorbed_noise=len(absorbed),
        )

    def reduce_to_n_topics(self, n_topics: int, use_ctfidf: bool = True) -> TopicReductionResult:
        """Legacy: reduce to a fixed number of topics using AgglomerativeClustering.

        Kept only to compare against the merge trees (``reduce_with_tree``).

        Pipeline:
        1. Compute topic embeddings
        2. Create distance matrix (1 - cosine_similarity)
        3. AgglomerativeClustering with n_clusters = n_topics - 1 (exclude -1)
        4. Map old topics to new topics
        5. Re-extract keywords for merged topics

        Args:
            n_topics: Target number of topics (including -1 noise cluster)
            use_ctfidf: Use c-TF-IDF (True) or semantic (False) embeddings

        Returns:
            TopicReductionResult with mappings and new topics
        """
        logger.info(f"Reducing to {n_topics} topics using AgglomerativeClustering")

        topics = self.documents_df["Topic"].tolist()
        unique_topics = sorted(self.topics_data.keys())
        num_topics_before = len(unique_topics)

        # Check if -1 (noise) exists
        has_outliers = -1 in unique_topics
        outliers = 1 if has_outliers else 0

        # Validate n_topics
        if n_topics < 2:
            raise ValueError("n_topics must be >= 2")
        if n_topics >= num_topics_before:
            logger.warning(
                f"Target ({n_topics}) >= extracted ({num_topics_before}), no reduction needed"
            )
            return TopicReductionResult(
                documents_df=self.documents_df,
                topics_data=self.topics_data,
                topic_mappings={t: t for t in unique_topics},
                topic_hierarchy={},
                num_topics_before=num_topics_before,
                num_topics_after=num_topics_before,
                reduction_method="fixed_n",
            )

        # Create topic distance matrix
        topic_embeddings = self.compute_topic_embeddings(use_ctfidf)

        # Filter out noise cluster (-1) if it exists
        if has_outliers:
            topic_embeddings = topic_embeddings[outliers:]  # Skip first row
            non_noise_topics = [t for t in unique_topics if t != -1]
        else:
            non_noise_topics = unique_topics

        # Compute pairwise cosine distances
        distance_matrix = 1 - cosine_similarity(topic_embeddings)
        np.fill_diagonal(distance_matrix, 0)

        # Cluster the topic embeddings
        n_clusters = n_topics - outliers  # Exclude noise cluster from target
        if n_clusters < 1:
            n_clusters = 1

        cluster = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
        cluster.fit(distance_matrix)

        # Create topic mappings (preserve -1 as separate)
        mapped_topics = {}
        for old_topic, new_cluster in zip(non_noise_topics, cluster.labels_, strict=True):
            mapped_topics[old_topic] = int(new_cluster)

        if has_outliers:
            mapped_topics[-1] = -1  # Preserve noise cluster

        # Map topics in documents_df
        new_topics = [mapped_topics[t] for t in topics]

        # Merge topics and re-extract keywords
        documents_df, topics_data = self._merge_topics(new_topics, mapped_topics)

        # Build hierarchy: new_topic_id -> [old_topic_ids that merged into it]
        topic_hierarchy = defaultdict(list)
        for old_id, new_id in mapped_topics.items():
            if old_id != -1:
                topic_hierarchy[new_id].append(old_id)

        num_topics_after = len([t for t in topics_data.keys() if t != -1])

        logger.info(f"Reduced from {num_topics_before} to {num_topics_after} topics")

        return TopicReductionResult(
            documents_df=documents_df,
            topics_data=topics_data,
            topic_mappings=mapped_topics,
            topic_hierarchy=dict(topic_hierarchy),
            num_topics_before=num_topics_before,
            num_topics_after=num_topics_after,
            reduction_method="fixed_n",
        )

    def auto_reduce_topics(self, use_ctfidf: bool = True) -> TopicReductionResult:
        """Legacy: automatically reduce topics using HDBSCAN on the topic vectors.

        Kept only to compare against the merge trees: on near-orthogonal
        c-TF-IDF rows it labels every topic noise and merges nothing.

        Pipeline:
        1. Compute and L2-normalize topic embeddings
        2. Run HDBSCAN (min_cluster_size=2) on topic embeddings
        3. Map topics to lowest topic_id in cluster
        4. Re-extract keywords for merged topics

        Args:
            use_ctfidf: Use c-TF-IDF (True) or semantic (False) embeddings

        Returns:
            TopicReductionResult with mappings and new topics
        """
        logger.info("Auto-reducing topics using HDBSCAN")

        topics = self.documents_df["Topic"].tolist()
        unique_topics = sorted(self.topics_data.keys())
        num_topics_before = len(unique_topics)

        # Check if -1 (noise) exists
        has_outliers = -1 in unique_topics
        outliers = 1 if has_outliers else 0

        # Filter out noise cluster
        if has_outliers:
            non_noise_topics = [t for t in unique_topics if t != -1]
        else:
            non_noise_topics = unique_topics

        if len(non_noise_topics) < 2:
            logger.info("Less than 2 non-noise topics, skipping reduction")
            return TopicReductionResult(
                documents_df=self.documents_df,
                topics_data=self.topics_data,
                topic_mappings={t: t for t in unique_topics},
                topic_hierarchy={},
                num_topics_before=num_topics_before,
                num_topics_after=num_topics_before,
                reduction_method="auto",
            )

        # Compute and normalize topic embeddings
        embeddings = self.compute_topic_embeddings(use_ctfidf)
        if has_outliers:
            embeddings = embeddings[outliers:]  # Skip noise cluster

        norm_embeddings = normalize(embeddings, norm="l2")

        # Run HDBSCAN to find similar topics
        hdbscan = HDBSCAN(
            min_cluster_size=2,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        predictions = hdbscan.fit_predict(norm_embeddings)

        # Map clusters to their lowest topic_id
        cluster_to_lowest = {}
        for cluster, topic_id in zip(predictions, non_noise_topics, strict=True):
            if cluster != -1:  # Ignore unclustered items
                if cluster not in cluster_to_lowest:
                    cluster_to_lowest[cluster] = topic_id
                else:
                    cluster_to_lowest[cluster] = min(cluster_to_lowest[cluster], topic_id)

        # Map each topic_id to the lowest topic_id in its cluster
        mapped_topics = {}
        for cluster, topic_id in zip(predictions, non_noise_topics, strict=True):
            if cluster == -1:
                mapped_topics[topic_id] = topic_id  # No clustering, stays the same
            else:
                mapped_topics[topic_id] = cluster_to_lowest[cluster]

        if has_outliers:
            mapped_topics[-1] = -1  # Preserve noise cluster

        # Map topics in documents
        new_topics = [mapped_topics[t] for t in topics]

        # Merge topics and re-extract keywords
        documents_df, topics_data = self._merge_topics(new_topics, mapped_topics)

        # Build hierarchy: new_topic_id -> [old_topic_ids that merged into it]
        topic_hierarchy = defaultdict(list)
        for old_id, new_id in mapped_topics.items():
            if old_id != -1:
                topic_hierarchy[new_id].append(old_id)

        num_topics_after = len([t for t in topics_data.keys() if t != -1])

        logger.info(f"Auto-reduced from {num_topics_before} to {num_topics_after} topics")

        return TopicReductionResult(
            documents_df=documents_df,
            topics_data=topics_data,
            topic_mappings=mapped_topics,
            topic_hierarchy=dict(topic_hierarchy),
            num_topics_before=num_topics_before,
            num_topics_after=num_topics_after,
            reduction_method="auto",
        )

    def _merge_topics(
        self, new_topics: list[int], topic_mappings: dict[int, int]
    ) -> tuple[pd.DataFrame, dict[int, list[tuple[str, float]]]]:
        """Apply topic mappings and re-extract keywords.

        Args:
            new_topics: New topic assignments for each document
            topic_mappings: Dict mapping old_topic_id -> new_topic_id

        Returns:
            (updated_documents_df, new_topics_data)
        """
        logger.info("Merging topics and re-extracting keywords")

        documents_df = self.documents_df.copy()
        documents_df["Topic"] = new_topics
        topics_data = CtfidfKeywords(language=self.language).extract(documents_df).topics_data

        logger.info(f"Re-extracted keywords for {len(topics_data)} topics")

        return documents_df, topics_data
