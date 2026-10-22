import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.utils import check_array

from ..utils.topic_merge_tree import DensityTree, DensityTreeBuilder

logger = logging.getLogger("orrery." + __name__)


class ClassTfidfTransformer(TfidfTransformer):
    """A Class-based TF-IDF procedure using scikit-learns TfidfTransformer as a base.

    ![](../algorithm/c-TF-IDF.svg)

    c-TF-IDF can best be explained as a TF-IDF formula adopted for multiple classes
    by joining all documents per class. Thus, each class is converted to a single document
    instead of set of documents. The frequency of each word **x** is extracted
    for each class **c** and is **l1** normalized. This constitutes the term frequency.

    Then, the term frequency is multiplied with IDF which is the logarithm of 1 plus
    the average number of words per class **A** divided by the frequency of word **x**
    across all classes.

    Arguments:
        bm25_weighting: Uses BM25-inspired idf-weighting procedure instead of the procedure
                        as defined in the c-TF-IDF formula. It uses the following weighting scheme:
                        `log(1+((avg_nr_samples - df + 0.5) / (df+0.5)))`
        reduce_frequent_words: Takes the square root of the bag-of-words after normalizing the matrix.
                               Helps to reduce the impact of words that appear too frequently.


    Examples:
    ```python
    transformer = ClassTfidfTransformer()
    ```
    """

    def __init__(
        self,
        bm25_weighting: bool = False,
        reduce_frequent_words: bool = False,
        use_idf: bool = True,
    ):
        self.bm25_weighting = bm25_weighting
        self.reduce_frequent_words = reduce_frequent_words
        self.use_idf = use_idf
        super().__init__()

    def fit(self, X: sp.csr_matrix, y=None, multiplier: np.ndarray | None = None):
        """Learn the idf vector (global term weights).

        Arguments:
            X: A matrix of term/token counts.
            multiplier: A multiplier for increasing/decreasing certain IDF scores
        """
        X = check_array(X, accept_sparse=("csr", "csc"))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = np.float64

        if self.use_idf:
            _, n_features = X.shape

            # Calculate the frequency of words across all classes
            # (.ravel() keeps df 1-D even for a single-word vocabulary, where
            # np.squeeze would collapse it to a 0-d scalar that sp.diags rejects)
            df = np.asarray(X.sum(axis=0)).ravel()

            # Calculate the average number of samples as regularization
            avg_nr_samples = int(X.sum(axis=1).mean())

            # BM25-inspired weighting procedure
            if self.bm25_weighting:
                idf = np.log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))

            # Divide the average number of samples by the word frequency
            # +1 is added to force values to be positive
            else:
                idf = np.log((avg_nr_samples / df) + 1)

            # Multiplier to increase/decrease certain idf scores
            if multiplier is not None:
                idf = idf * multiplier

            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def transform(self, X: sp.csr_matrix, copy: bool = False):
        """Transform a count-based matrix to c-TF-IDF.

        Arguments:
            X (sparse matrix): A matrix of term/token counts.

        Returns:
            X (sparse matrix): A c-TF-IDF matrix
        """
        if self.use_idf:
            X = normalize(X, axis=1, norm="l1", copy=False)

            if self.reduce_frequent_words:
                X.data = np.sqrt(X.data)

            X = X * self._idf_diag

        return X


@dataclass(frozen=True, eq=False)  # matrix fields: compare by identity
class CtfidfResult:
    """Keywords per topic, plus the c-TF-IDF matrix they came from."""

    topics_data: dict[int, list[tuple[str, float]]]  # topic id -> [(word, score), ...]
    topic_ids: list[int]  # topic of each matrix row (sorted, noise first when present)
    matrix: sp.csr_matrix  # (topics, words)
    words: np.ndarray


class CtfidfKeywords:
    """c-TF-IDF keywords: each topic's documents are joined into one class
    document, counted (CountVectorizer) and weighted (ClassTfidfTransformer).

    The single implementation behind extraction, reduction and re-labelling.
    """

    def __init__(
        self,
        language: str | None = "english",
        ngram_range: tuple[int, int] = (1, 1),
        n_words: int = 10,
    ):
        self.language = language
        self.ngram_range = ngram_range
        self.n_words = n_words

    def extract(self, documents_df: pd.DataFrame) -> CtfidfResult:
        """Keywords for every topic in ``documents_df`` (columns Document and Topic)."""
        docs_per_topic = documents_df.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
        count_vectorizer = CountVectorizer(stop_words=self.language, ngram_range=self.ngram_range)
        counts = count_vectorizer.fit_transform(docs_per_topic.Document.values)
        words = count_vectorizer.get_feature_names_out()
        matrix = ClassTfidfTransformer().fit_transform(counts)

        topic_ids = [int(t) for t in docs_per_topic.Topic.values]
        topics_data = {}
        for row_index, topic_id in enumerate(topic_ids):
            row = matrix.getrow(row_index).toarray()[0]
            top = row.argsort()[-self.n_words :][::-1]
            topics_data[topic_id] = [(words[i], float(row[i])) for i in top]
        return CtfidfResult(
            topics_data=topics_data, topic_ids=topic_ids, matrix=matrix, words=words
        )


@dataclass(frozen=True, eq=False)
class ClusteringOutcome:
    """Per-point cluster labels (``-1`` = noise, HDBSCAN only) and, for HDBSCAN,
    the density merge tree over the clusters (None with fewer than two)."""

    labels: np.ndarray
    density: DensityTree | None = None


def cluster_points(
    points: np.ndarray,
    clustering_method: str = "hdbscan",
    min_topic_size: int = 10,
    n_clusters: int | None = None,
) -> ClusteringOutcome:
    """Cluster ``points`` with the configured method.

    Module-level so it can run as a CancellableTask in a child process; the
    density tree is built there because the fitted model stays in the child.
    """
    logger.info(f"Cluster - Start clustering with method='{clustering_method}'")

    if n_clusters is not None and n_clusters > len(points):
        raise ValueError(f"n_clusters ({n_clusters}) exceeds number of data points ({len(points)})")

    if clustering_method == "hdbscan":
        model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            gen_min_span_tree=True,  # required for relative_validity_ (DBCV) after fit
        )
        labels = model.fit(points).labels_
        return ClusteringOutcome(labels=labels, density=_density_tree(model, labels))
    if clustering_method == "kmeans":
        labels = KMeans(n_clusters=n_clusters, random_state=7, n_init=10).fit_predict(points)
    elif clustering_method == "gmm":
        labels = GaussianMixture(n_components=n_clusters, random_state=7).fit_predict(points)
    elif clustering_method == "spectral":
        labels = SpectralClustering(
            n_clusters=n_clusters, random_state=7, affinity="nearest_neighbors"
        ).fit_predict(points)
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    return ClusteringOutcome(labels=labels)


def _density_tree(model: HDBSCAN, labels: np.ndarray) -> DensityTree | None:
    """The fitted model's density merge tree; None (logged) when it has none."""
    if int(labels.max(initial=-1)) < 1:
        return None
    try:
        return DensityTreeBuilder().build(model.condensed_tree_.to_numpy(), labels)
    except ValueError as exc:
        logger.warning("No density merge tree for this clustering: %s", exc)
        return None


class GenerateTopics:
    SUPPORTED_METHODS = ("hdbscan", "kmeans", "gmm", "spectral")

    def __init__(
        self,
        documents: list[str],
        min_topic_size: int = 10,
        n_gram_range: tuple[int, int] = (1, 1),
        language: str | None = None,
        clustering_method: str = "hdbscan",
        n_clusters: int | None = None,
    ):
        self.documents = documents
        self.min_topic_size = min_topic_size
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters

        if clustering_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown clustering_method '{clustering_method}'. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )
        if clustering_method != "hdbscan" and n_clusters is None:
            raise ValueError(f"n_clusters is required when clustering_method='{clustering_method}'")

        self.docs_id = range(len(self.documents))
        self.n_gram_range = n_gram_range
        self.language = language

        # Store c-TF-IDF matrix and words for topic reduction
        self.ctfidf_matrix: sp.csr_matrix | None = None
        self.ctfidf_words: np.ndarray | None = None

    def generate_clusters(self, reduced_embeddings: np.ndarray) -> pd.DataFrame:
        """Cluster reduced embeddings using the configured clustering method.

        Arguments:
            reduced_embeddings: The reduced sentence embeddings (from UMAP/PCA)

        Returns:
            documents_df: DataFrame with Document_ID, Document, and Topic columns
        """
        outcome = cluster_points(
            reduced_embeddings, self.clustering_method, self.min_topic_size, self.n_clusters
        )
        return self.documents_frame(outcome.labels)

    def documents_frame(self, labels) -> pd.DataFrame:
        """Document_ID / Document / Topic frame for per-document cluster labels
        (computed by ``cluster_points``, possibly in another process)."""
        documents_df = pd.DataFrame(
            {"Document_ID": self.docs_id, "Document": self.documents, "Topic": labels}
        )
        n_clusters = len(set(labels) - {-1})
        n_noise = int((np.asarray(labels) == -1).sum())
        logger.info(f"Cluster - Completed ✓ ({n_clusters} clusters, {n_noise} noise points)")

        return documents_df

    def extract_topics(
        self, documents_df: pd.DataFrame, n_words: int = 10
    ) -> dict[int, list[tuple[str, float]]]:
        """Step 2: Extract keywords using c-TF-IDF.

        Returns:
            topics: Dict where key is Topic ID and value is list of (word, score) tuples.
        """
        logger.info("Topics - Start extracting topic keywords")
        result = CtfidfKeywords(
            language=self.language, ngram_range=self.n_gram_range, n_words=n_words
        ).extract(documents_df)

        # Kept for the legacy c-TF-IDF topic reduction
        self.ctfidf_matrix = result.matrix
        self.ctfidf_words = result.words

        logger.info("Topics - Completed \u2713")
        return result.topics_data

    @property
    def c_tf_idf_matrix(self) -> sp.csr_matrix:
        """Get the c-TF-IDF matrix for topic reduction."""
        return self.ctfidf_matrix

    @property
    def words(self) -> np.ndarray:
        """Get the feature names (words) from CountVectorizer."""
        return self.ctfidf_words
