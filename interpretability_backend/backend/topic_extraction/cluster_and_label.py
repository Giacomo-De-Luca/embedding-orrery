

from typing import List, Union, Tuple, Dict
import numpy as np
import logging
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
import scipy.sparse as sp

logger = logging.getLogger('star_map.'+__name__)


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
        super(ClassTfidfTransformer, self).__init__()

    def fit(
            self, 
            X: sp.csr_matrix, 
            y = None, 
            multiplier: Union[np.ndarray, None] = None
            ):
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
            df = np.squeeze(np.asarray(X.sum(axis=0)))

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


class GenerateTopics:

    def __init__(
        self,
        documents: List[str],
        #reduced_embeddings: np.ndarray = np.array([]),
        min_topic_size: int = 10,
        n_gram_range: Tuple[int, int] = (1, 1),
        language: Union[str, None] = None,

        ):
        self.documents = documents
        #self.embeddings = reduced_embeddings
        self.min_topic_size = min_topic_size
        self.hdbscan_model = HDBSCAN(
                min_cluster_size=self.min_topic_size,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
        self.docs_id = range(len(self.documents))
        self.n_gram_range = n_gram_range
        self.language = language

        # Store c-TF-IDF matrix and words for topic reduction
        self.ctfidf_matrix: Union[sp.csr_matrix, None] = None
        self.ctfidf_words: Union[np.ndarray, None] = None
        
        
    def generate_clusters(self, reduced_embeddings: np.ndarray) -> pd.DataFrame:
        """Cluster reduced embeddings using HDBSCAN.

        Arguments:
            reduced_embeddings: The reduced sentence embeddings with UMAP

        Returns:
            documents: DataFrame with documents and their corresponding IDs and newly added Topics
            probabilities: The distribution of probabilities
        """
        logger.info("Cluster - Start clustering the reduced embeddings")
        self.hdbscan_model.fit(reduced_embeddings)
        labels = self.hdbscan_model.labels_
        documents_df = pd.DataFrame({
            "Document_ID": self.docs_id,
            "Document": self.documents,
            "Topic": labels
        })
        logger.info("Cluster - Completed \u2713")
        
        return documents_df

    def extract_topics(self, documents_df: pd.DataFrame, n_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """Step 2: Extract keywords using c-TF-IDF.
        
        Returns:
            topics: Dict where key is Topic ID and value is list of (word, score) tuples.
        """
        logger.info("Topics - Start extracting topic keywords")

        # A. Group documents by Topic (The "Mega-Document" step)
        docs_per_topic = documents_df.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        
        # B. Count Vectorizer (Bag of Words)
        # We process the grouped text, not the individual documents!
        count_vectorizer = CountVectorizer(stop_words=self.language, ngram_range=self.n_gram_range)
        X = count_vectorizer.fit_transform(docs_per_topic.Document.values)
        words = count_vectorizer.get_feature_names_out()

        # C. c-TF-IDF (Importance Scores)
        ctfidf = ClassTfidfTransformer()
        ctfidf_matrix = ctfidf.fit_transform(X)

        # Store for topic reduction
        self.ctfidf_matrix = ctfidf_matrix
        self.ctfidf_words = words

        # D. Extract Top N Words per Topic
        topics_data = {}
        # Iterate over each topic row in the c-TF-IDF matrix
        for i, topic_id in enumerate(docs_per_topic.Topic.values):
            # Get that topic's row
            row = ctfidf_matrix.getrow(i).toarray()[0]
            # Sort indices by score (descending) and take top N
            top_indices = row.argsort()[-n_words:][::-1]
            
            # Map indices to words and scores
            topic_keywords = [(words[idx], float(row[idx])) for idx in top_indices]
            topics_data[topic_id] = topic_keywords

        logger.info("Topics - Completed \u2713")
        return topics_data
    

    @property
    def c_tf_idf_matrix(self) -> sp.csr_matrix:
        """Get the c-TF-IDF matrix for topic reduction."""
        return self.ctfidf_matrix

    @property
    def words(self) -> np.ndarray:
        """Get the feature names (words) from CountVectorizer."""
        return self.ctfidf_words
