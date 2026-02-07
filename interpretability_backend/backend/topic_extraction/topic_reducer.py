"""
Topic reduction for merging similar topics after extraction.

Implements BERTopic-inspired reduction methods:
- Fixed-N reduction: AgglomerativeClustering to target count
- Auto reduction: HDBSCAN to automatically merge similar topics

Supports both c-TF-IDF and semantic embeddings for topic similarity.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import scipy.sparse as sp

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

from .cluster_and_label import ClassTfidfTransformer

logger = logging.getLogger('star_map.' + __name__)


@dataclass
class TopicReductionResult:
    """Result of topic reduction operation."""
    documents_df: pd.DataFrame  # Updated with new topic assignments
    topics_data: Dict[int, List[Tuple[str, float]]]  # Re-extracted keywords
    topic_mappings: Dict[int, int]  # old_topic_id -> new_topic_id
    topic_hierarchy: Dict[int, List[int]]  # new_topic_id -> [old_topic_ids that merged into it]
    num_topics_before: int
    num_topics_after: int
    reduction_method: str  # "fixed_n" or "auto"


class TopicReducer:
    """
    Reduces topics by clustering similar topics together.

    Supports:
    - Fixed target count (AgglomerativeClustering)
    - Automatic reduction (HDBSCAN with min_cluster_size=2)
    - c-TF-IDF or semantic embeddings for similarity
    """

    def __init__(
        self,
        documents_df: pd.DataFrame,
        topics_data: Dict[int, List[Tuple[str, float]]],
        ctfidf_matrix: sp.csr_matrix,
        ctfidf_words: np.ndarray,
        language: Optional[str] = "english",
        collection_name: Optional[str] = None,
        chromadb_client=None
    ):
        """Initialize with existing topic extraction results.

        Args:
            documents_df: DataFrame with Document_ID, Document, Topic columns
            topics_data: Dict of topic_id -> [(word, score), ...]
            ctfidf_matrix: c-TF-IDF matrix from keyword extraction
            ctfidf_words: Feature names from CountVectorizer
            language: Stop words language for re-extraction
            collection_name: ChromaDB collection name (for semantic embeddings)
            chromadb_client: ChromaDB client (for semantic embeddings)
        """
        self.documents_df = documents_df.copy()
        self.topics_data = topics_data
        self.ctfidf_matrix = ctfidf_matrix
        self.ctfidf_words = ctfidf_words
        self.language = language
        self.collection_name = collection_name
        self.chromadb_client = chromadb_client

        # Track topic sizes for validation
        self.topic_sizes = documents_df["Topic"].value_counts().to_dict()

    def compute_topic_embeddings(self, use_ctfidf: bool = True) -> np.ndarray:
        """Compute embeddings for each topic.

        Args:
            use_ctfidf: If True, use c-TF-IDF vectors. If False, use semantic
                       embeddings (averaged document embeddings from ChromaDB).

        Returns:
            topic_embeddings: (n_topics, embedding_dim) array
        """
        if use_ctfidf:
            logger.info("Computing topic embeddings from c-TF-IDF vectors")
            # Each row in ctfidf_matrix is already a topic embedding
            return self.ctfidf_matrix.toarray()
        else:
            logger.info("Computing topic embeddings from semantic embeddings (per-topic loading)")
            return self._compute_semantic_embeddings()

    def _compute_semantic_embeddings(self) -> np.ndarray:
        """Compute topic embeddings by averaging document embeddings per topic.

        Uses per-topic ChromaDB queries to avoid loading all embeddings at once.

        Returns:
            topic_embeddings: (n_topics, embedding_dim) array
        """
        if self.chromadb_client is None or self.collection_name is None:
            raise ValueError("chromadb_client and collection_name required for semantic embeddings")

        # Get collection without loading embedding function
        collection = self.chromadb_client.get_collection(
            name=self.collection_name,
            embedding_function=None
        )

        topic_ids = sorted(self.topics_data.keys())
        topic_embeddings = []

        for topic_id in topic_ids:
            # Load only embeddings for this topic (memory-efficient!)
            try:
                results = collection.get(
                    where={"topic_id": str(topic_id)},
                    include=["embeddings"]
                )

                if results["embeddings"] and len(results["embeddings"]) > 0:
                    # Average embeddings for this topic
                    topic_emb = np.mean(results["embeddings"], axis=0)
                else:
                    # Fallback for empty topic (shouldn't happen but be safe)
                    # Infer dimension from first successful query
                    if len(topic_embeddings) > 0:
                        dim = len(topic_embeddings[0])
                    else:
                        dim = 384  # Default fallback
                    topic_emb = np.zeros(dim)
                    logger.warning(f"Topic {topic_id} has no embeddings, using zero vector")

                topic_embeddings.append(topic_emb)

            except Exception as e:
                logger.error(f"Error loading embeddings for topic {topic_id}: {e}")
                # Use zero vector as fallback
                if len(topic_embeddings) > 0:
                    dim = len(topic_embeddings[0])
                else:
                    dim = 384
                topic_embeddings.append(np.zeros(dim))

        return np.array(topic_embeddings)

    def reduce_to_n_topics(
        self,
        n_topics: int,
        use_ctfidf: bool = True
    ) -> TopicReductionResult:
        """Reduce to a fixed number of topics using AgglomerativeClustering.

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
            logger.warning(f"Target ({n_topics}) >= extracted ({num_topics_before}), no reduction needed")
            return TopicReductionResult(
                documents_df=self.documents_df,
                topics_data=self.topics_data,
                topic_mappings={t: t for t in unique_topics},
                topic_hierarchy={},
                num_topics_before=num_topics_before,
                num_topics_after=num_topics_before,
                reduction_method="fixed_n"
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
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average"
        )
        cluster.fit(distance_matrix)

        # Create topic mappings (preserve -1 as separate)
        mapped_topics = {}
        for old_topic, new_cluster in zip(non_noise_topics, cluster.labels_):
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
            reduction_method="fixed_n"
        )

    def auto_reduce_topics(self, use_ctfidf: bool = True) -> TopicReductionResult:
        """Automatically reduce topics using HDBSCAN.

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
                reduction_method="auto"
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
            prediction_data=True
        )
        predictions = hdbscan.fit_predict(norm_embeddings)

        # Map clusters to their lowest topic_id
        cluster_to_lowest = {}
        for cluster, topic_id in zip(predictions, non_noise_topics):
            if cluster != -1:  # Ignore unclustered items
                if cluster not in cluster_to_lowest:
                    cluster_to_lowest[cluster] = topic_id
                else:
                    cluster_to_lowest[cluster] = min(cluster_to_lowest[cluster], topic_id)

        # Map each topic_id to the lowest topic_id in its cluster
        mapped_topics = {}
        for cluster, topic_id in zip(predictions, non_noise_topics):
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
            reduction_method="auto"
        )

    def _merge_topics(
        self,
        new_topics: List[int],
        topic_mappings: Dict[int, int]
    ) -> Tuple[pd.DataFrame, Dict[int, List[Tuple[str, float]]]]:
        """Apply topic mappings and re-extract keywords.

        Args:
            new_topics: New topic assignments for each document
            topic_mappings: Dict mapping old_topic_id -> new_topic_id

        Returns:
            (updated_documents_df, new_topics_data)
        """
        logger.info("Merging topics and re-extracting keywords")

        # Update documents_df with new topic assignments
        documents_df = self.documents_df.copy()
        documents_df["Topic"] = new_topics

        # Group documents by new topic (mega-document step)
        docs_per_topic = documents_df.groupby(['Topic'], as_index=False).agg({
            'Document': ' '.join
        })

        # Re-run CountVectorizer + c-TF-IDF on merged documents
        count_vectorizer = CountVectorizer(
            stop_words=self.language,
            ngram_range=(1, 1)
        )
        X = count_vectorizer.fit_transform(docs_per_topic.Document.values)
        words = count_vectorizer.get_feature_names_out()

        # c-TF-IDF transformation
        ctfidf = ClassTfidfTransformer()
        ctfidf_matrix = ctfidf.fit_transform(X)

        # Extract top keywords per topic
        topics_data = {}
        n_words = 10  # Match original extraction

        for i, topic_id in enumerate(docs_per_topic.Topic.values):
            row = ctfidf_matrix.getrow(i).toarray()[0]
            top_indices = row.argsort()[-n_words:][::-1]
            topic_keywords = [(words[idx], float(row[idx])) for idx in top_indices]
            topics_data[int(topic_id)] = topic_keywords

        logger.info(f"Re-extracted keywords for {len(topics_data)} topics")

        return documents_df, topics_data
