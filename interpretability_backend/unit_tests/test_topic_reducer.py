"""
Unit tests for TopicReducer class.

Tests topic reduction functionality including:
- Fixed-N reduction with AgglomerativeClustering
- Auto reduction with HDBSCAN
- Noise cluster preservation
- Keyword re-extraction after merging
- Edge cases (invalid inputs, single topic, etc.)
"""

import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from unittest.mock import Mock, MagicMock

from interpretability_backend.backend.topic_extraction.topic_reducer import (
    TopicReducer,
    TopicReductionResult
)


@pytest.fixture
def sample_documents_df():
    """Create sample documents DataFrame with topics."""
    return pd.DataFrame({
        "Document_ID": range(100),
        "Document": [f"document {i} with some text content" for i in range(100)],
        "Topic": [i % 10 for i in range(100)]  # 10 topics (0-9)
    })


@pytest.fixture
def sample_topics_data():
    """Create sample topics data with keywords."""
    topics_data = {}
    for topic_id in range(10):
        topics_data[topic_id] = [
            (f"keyword{topic_id}_1", 0.9),
            (f"keyword{topic_id}_2", 0.8),
            (f"keyword{topic_id}_3", 0.7),
            (f"keyword{topic_id}_4", 0.6),
            (f"keyword{topic_id}_5", 0.5),
        ]
    return topics_data


@pytest.fixture
def sample_ctfidf_matrix():
    """Create sample c-TF-IDF sparse matrix."""
    # 10 topics x 50 words vocabulary
    matrix = sp.random(10, 50, density=0.3, format='csr')
    return matrix


@pytest.fixture
def sample_ctfidf_words():
    """Create sample vocabulary."""
    return np.array([f"word_{i}" for i in range(50)])


@pytest.fixture
def reducer(sample_documents_df, sample_topics_data, sample_ctfidf_matrix, sample_ctfidf_words):
    """Create TopicReducer instance with sample data."""
    return TopicReducer(
        documents_df=sample_documents_df,
        topics_data=sample_topics_data,
        ctfidf_matrix=sample_ctfidf_matrix,
        ctfidf_words=sample_ctfidf_words,
        language="english"
    )


class TestTopicReducerInitialization:
    """Test TopicReducer initialization."""

    def test_init_with_valid_data(self, reducer, sample_documents_df, sample_topics_data):
        """Test initialization with valid data."""
        assert reducer.documents_df.shape == sample_documents_df.shape
        assert reducer.topics_data == sample_topics_data
        assert reducer.language == "english"
        assert len(reducer.topic_sizes) == 10

    def test_topic_sizes_calculated(self, reducer):
        """Test that topic sizes are correctly calculated."""
        # Each topic should have 10 documents (100 docs / 10 topics)
        for topic_id in range(10):
            assert reducer.topic_sizes[topic_id] == 10


class TestComputeTopicEmbeddings:
    """Test topic embedding computation."""

    def test_compute_ctfidf_embeddings(self, reducer):
        """Test c-TF-IDF embedding extraction."""
        embeddings = reducer.compute_topic_embeddings(use_ctfidf=True)

        assert embeddings.shape == (10, 50)  # 10 topics x 50 words
        assert isinstance(embeddings, np.ndarray)

    def test_compute_semantic_embeddings_without_chromadb(self, reducer):
        """Test semantic embeddings fail without ChromaDB client."""
        with pytest.raises(ValueError, match="chromadb_client and collection_name required"):
            reducer.compute_topic_embeddings(use_ctfidf=False)


class TestReduceToNTopics:
    """Test fixed-N topic reduction."""

    def test_reduce_10_to_5_topics(self, reducer):
        """Test reducing from 10 topics to 5."""
        result = reducer.reduce_to_n_topics(n_topics=5, use_ctfidf=True)

        assert isinstance(result, TopicReductionResult)
        assert result.num_topics_before == 10
        assert result.num_topics_after <= 5  # May be less if some merge completely
        assert result.reduction_method == "fixed_n"
        assert len(result.topics_data) <= 5
        assert result.documents_df.shape[0] == 100  # Same number of documents

    def test_reduce_validates_n_topics_minimum(self, reducer):
        """Test that n_topics must be >= 2."""
        with pytest.raises(ValueError, match="n_topics must be >= 2"):
            reducer.reduce_to_n_topics(n_topics=1, use_ctfidf=True)

    def test_reduce_skips_if_target_equals_current(self, reducer):
        """Test that reduction is skipped if target >= current."""
        result = reducer.reduce_to_n_topics(n_topics=10, use_ctfidf=True)

        assert result.num_topics_before == result.num_topics_after
        assert len(result.topic_mappings) == 10

    def test_reduce_keywords_reextracted(self, reducer):
        """Test that keywords are re-extracted after merging."""
        result = reducer.reduce_to_n_topics(n_topics=5, use_ctfidf=True)

        # Check that all topics have keywords
        for topic_id, keywords in result.topics_data.items():
            assert len(keywords) > 0
            # Each keyword is a (word, score) tuple
            for word, score in keywords:
                assert isinstance(word, str)
                assert isinstance(score, float)

    def test_reduce_preserves_document_count(self, reducer):
        """Test that all documents are preserved after reduction."""
        result = reducer.reduce_to_n_topics(n_topics=3, use_ctfidf=True)

        assert result.documents_df.shape[0] == 100
        assert "Topic" in result.documents_df.columns


class TestAutoReduceTopics:
    """Test automatic topic reduction."""

    def test_auto_reduce_merges_similar_topics(self, reducer):
        """Test that auto reduction merges similar topics."""
        result = reducer.auto_reduce_topics(use_ctfidf=True)

        assert isinstance(result, TopicReductionResult)
        assert result.num_topics_after <= result.num_topics_before
        assert result.reduction_method == "auto"

    def test_auto_reduce_with_single_topic(self):
        """Test auto reduction with only 1 topic (should skip)."""
        # Create dataset with only 1 topic
        df = pd.DataFrame({
            "Document_ID": range(10),
            "Document": ["text"] * 10,
            "Topic": [0] * 10
        })
        topics_data = {0: [("word", 0.9)]}
        matrix = sp.random(1, 10, density=0.5, format='csr')
        words = np.array([f"w{i}" for i in range(10)])

        reducer = TopicReducer(df, topics_data, matrix, words)
        result = reducer.auto_reduce_topics(use_ctfidf=True)

        # Should not reduce (only 1 topic)
        assert result.num_topics_after == result.num_topics_before


class TestNoiseClusterPreservation:
    """Test that noise cluster (-1) is never merged."""

    @pytest.fixture
    def reducer_with_noise(self):
        """Create reducer with noise cluster."""
        df = pd.DataFrame({
            "Document_ID": range(50),
            "Document": ["text"] * 50,
            "Topic": [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [-1] * 10  # 4 topics + noise
        })
        topics_data = {
            -1: [("unclustered", 0.5)],
            0: [("topic0", 0.9)],
            1: [("topic1", 0.9)],
            2: [("topic2", 0.9)],
            3: [("topic3", 0.9)],
        }
        matrix = sp.random(5, 20, density=0.3, format='csr')
        words = np.array([f"w{i}" for i in range(20)])

        return TopicReducer(df, topics_data, matrix, words)

    def test_noise_cluster_not_merged_fixed_n(self, reducer_with_noise):
        """Test that -1 is preserved in fixed-N reduction."""
        result = reducer_with_noise.reduce_to_n_topics(n_topics=3, use_ctfidf=True)

        # -1 should still exist in topics
        assert -1 in result.topics_data
        assert -1 in result.topic_mappings
        assert result.topic_mappings[-1] == -1  # -1 maps to itself

    def test_noise_cluster_not_merged_auto(self, reducer_with_noise):
        """Test that -1 is preserved in auto reduction."""
        result = reducer_with_noise.auto_reduce_topics(use_ctfidf=True)

        # -1 should still exist
        assert -1 in result.topics_data
        assert result.topic_mappings[-1] == -1


class TestTopicMappings:
    """Test topic mapping creation and application."""

    def test_mappings_created(self, reducer):
        """Test that topic mappings are created."""
        result = reducer.reduce_to_n_topics(n_topics=5, use_ctfidf=True)

        assert isinstance(result.topic_mappings, dict)
        assert len(result.topic_mappings) == 10  # All original topics mapped

    def test_all_original_topics_have_mapping(self, reducer):
        """Test that every original topic has a mapping."""
        result = reducer.reduce_to_n_topics(n_topics=3, use_ctfidf=True)

        for original_topic in range(10):
            assert original_topic in result.topic_mappings

    def test_documents_reassigned_correctly(self, reducer):
        """Test that documents are reassigned to new topics."""
        result = reducer.reduce_to_n_topics(n_topics=5, use_ctfidf=True)

        # Check that all document topics are valid
        unique_topics = result.documents_df["Topic"].unique()
        for topic_id in unique_topics:
            assert topic_id in result.topics_data


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_documents_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"Document_ID": [], "Document": [], "Topic": []})
        topics_data = {}
        matrix = sp.csr_matrix((0, 10))
        words = np.array([f"w{i}" for i in range(10)])

        reducer = TopicReducer(df, topics_data, matrix, words)

        # Should handle gracefully
        assert len(reducer.topic_sizes) == 0

    def test_single_document_per_topic(self):
        """Test with only 1 document per topic."""
        df = pd.DataFrame({
            "Document_ID": range(5),
            "Document": [f"doc {i}" for i in range(5)],
            "Topic": range(5)
        })
        topics_data = {i: [(f"word{i}", 0.9)] for i in range(5)}
        matrix = sp.random(5, 10, density=0.5, format='csr')
        words = np.array([f"w{i}" for i in range(10)])

        reducer = TopicReducer(df, topics_data, matrix, words)
        result = reducer.reduce_to_n_topics(n_topics=3, use_ctfidf=True)

        assert result.num_topics_after <= 3


class TestKeywordReextraction:
    """Test keyword re-extraction after topic merging."""

    def test_merged_topics_have_new_keywords(self, reducer):
        """Test that merged topics get new keywords extracted."""
        result = reducer.reduce_to_n_topics(n_topics=5, use_ctfidf=True)

        # All topics should have keywords
        for topic_id, keywords in result.topics_data.items():
            assert len(keywords) > 0
            # Keywords should be different from originals (merged content)
            # Just verify structure
            for word, score in keywords:
                assert len(word) > 0
                assert 0 <= score <= 1.5  # c-TF-IDF scores can be > 1

    def test_keywords_sorted_by_score(self, reducer):
        """Test that keywords are sorted by score (descending)."""
        result = reducer.reduce_to_n_topics(n_topics=5, use_ctfidf=True)

        for topic_id, keywords in result.topics_data.items():
            scores = [score for _, score in keywords]
            # Check descending order
            assert scores == sorted(scores, reverse=True)


class TestResultDataclass:
    """Test TopicReductionResult dataclass."""

    def test_result_contains_all_fields(self, reducer):
        """Test that result contains all required fields."""
        result = reducer.reduce_to_n_topics(n_topics=5, use_ctfidf=True)

        assert hasattr(result, 'documents_df')
        assert hasattr(result, 'topics_data')
        assert hasattr(result, 'topic_mappings')
        assert hasattr(result, 'num_topics_before')
        assert hasattr(result, 'num_topics_after')
        assert hasattr(result, 'reduction_method')

    def test_result_method_label(self, reducer):
        """Test that reduction method is correctly labeled."""
        result_fixed = reducer.reduce_to_n_topics(n_topics=5, use_ctfidf=True)
        result_auto = reducer.auto_reduce_topics(use_ctfidf=True)

        assert result_fixed.reduction_method == "fixed_n"
        assert result_auto.reduction_method == "auto"
