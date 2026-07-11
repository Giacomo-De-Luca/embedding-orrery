"""Unit tests for clustering method dispatch in GenerateTopics."""

import numpy as np
import pytest

from backend.topic_extraction.cluster_and_label import GenerateTopics


@pytest.fixture
def synthetic_data():
    """Generate synthetic 2D data with 5 known clusters."""
    np.random.seed(42)
    centers = [(-3, 0), (-1.5, 2.5), (0, 0), (1.5, 2.5), (3, 0)]
    points = np.vstack([np.random.randn(100, 2) * 0.4 + c for c in centers])
    documents = [f"doc_{i}" for i in range(len(points))]
    return points, documents


class TestClusteringMethodValidation:
    """Test parameter validation for clustering methods."""

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown clustering_method"):
            GenerateTopics(documents=["a", "b"], clustering_method="invalid")

    def test_kmeans_requires_n_clusters(self):
        with pytest.raises(ValueError, match="n_clusters is required"):
            GenerateTopics(documents=["a", "b"], clustering_method="kmeans")

    def test_gmm_requires_n_clusters(self):
        with pytest.raises(ValueError, match="n_clusters is required"):
            GenerateTopics(documents=["a", "b"], clustering_method="gmm")

    def test_spectral_requires_n_clusters(self):
        with pytest.raises(ValueError, match="n_clusters is required"):
            GenerateTopics(documents=["a", "b"], clustering_method="spectral")

    def test_hdbscan_does_not_require_n_clusters(self):
        gen = GenerateTopics(documents=["a", "b"], clustering_method="hdbscan")
        assert gen.clustering_method == "hdbscan"


class TestHDBSCAN:
    """Test HDBSCAN clustering (default behavior)."""

    def test_default_method_is_hdbscan(self, synthetic_data):
        points, documents = synthetic_data
        gen = GenerateTopics(documents=documents, min_topic_size=10)
        assert gen.clustering_method == "hdbscan"

    def test_hdbscan_produces_clusters(self, synthetic_data):
        points, documents = synthetic_data
        gen = GenerateTopics(documents=documents, min_topic_size=10)
        df = gen.generate_clusters(points)
        assert "Topic" in df.columns
        # HDBSCAN may produce noise (-1) labels
        unique_topics = set(df["Topic"].unique())
        assert len(unique_topics) >= 2  # At least 1 cluster + possibly noise


class TestKMeans:
    """Test KMeans clustering."""

    def test_kmeans_produces_exact_k_clusters(self, synthetic_data):
        points, documents = synthetic_data
        gen = GenerateTopics(
            documents=documents, clustering_method="kmeans", n_clusters=5
        )
        df = gen.generate_clusters(points)
        unique_topics = set(df["Topic"].unique())
        assert len(unique_topics) == 5
        # KMeans should never produce noise labels
        assert -1 not in unique_topics

    def test_kmeans_assigns_all_points(self, synthetic_data):
        points, documents = synthetic_data
        gen = GenerateTopics(
            documents=documents, clustering_method="kmeans", n_clusters=5
        )
        df = gen.generate_clusters(points)
        assert len(df) == len(points)
        assert (df["Topic"] >= 0).all()


class TestGMM:
    """Test Gaussian Mixture Model clustering."""

    def test_gmm_produces_exact_k_clusters(self, synthetic_data):
        points, documents = synthetic_data
        gen = GenerateTopics(
            documents=documents, clustering_method="gmm", n_clusters=5
        )
        df = gen.generate_clusters(points)
        unique_topics = set(df["Topic"].unique())
        assert len(unique_topics) == 5
        assert -1 not in unique_topics

    def test_gmm_assigns_all_points(self, synthetic_data):
        points, documents = synthetic_data
        gen = GenerateTopics(
            documents=documents, clustering_method="gmm", n_clusters=5
        )
        df = gen.generate_clusters(points)
        assert len(df) == len(points)
        assert (df["Topic"] >= 0).all()


class TestSpectral:
    """Test Spectral clustering."""

    def test_spectral_produces_exact_k_clusters(self, synthetic_data):
        points, documents = synthetic_data
        gen = GenerateTopics(
            documents=documents, clustering_method="spectral", n_clusters=5
        )
        df = gen.generate_clusters(points)
        unique_topics = set(df["Topic"].unique())
        assert len(unique_topics) == 5
        assert -1 not in unique_topics

    def test_spectral_assigns_all_points(self, synthetic_data):
        points, documents = synthetic_data
        gen = GenerateTopics(
            documents=documents, clustering_method="spectral", n_clusters=5
        )
        df = gen.generate_clusters(points)
        assert len(df) == len(points)
        assert (df["Topic"] >= 0).all()


class TestCTFIDFIntegration:
    """Test that c-TF-IDF keyword extraction works with all clustering methods."""

    @pytest.fixture
    def text_data(self):
        """Generate data with meaningful text for c-TF-IDF."""
        np.random.seed(42)
        centers = [(-2, 0), (0, 2), (2, 0)]
        points = np.vstack([np.random.randn(50, 2) * 0.3 + c for c in centers])
        documents = (
            ["python machine learning neural network"] * 50
            + ["cooking recipe kitchen food"] * 50
            + ["sports football basketball team"] * 50
        )
        return points, documents

    @pytest.mark.parametrize("method", ["hdbscan", "kmeans", "gmm", "spectral"])
    def test_keyword_extraction_works(self, text_data, method):
        points, documents = text_data
        kwargs = {"documents": documents, "clustering_method": method}
        if method != "hdbscan":
            kwargs["n_clusters"] = 3
        else:
            kwargs["min_topic_size"] = 5

        gen = GenerateTopics(**kwargs)
        df = gen.generate_clusters(points)
        topics_data = gen.extract_topics(df, n_words=5)

        # Should have keywords for each cluster
        assert len(topics_data) >= 1
        for topic_id, keywords in topics_data.items():
            assert len(keywords) > 0
            assert all(isinstance(w, str) and isinstance(s, float) for w, s in keywords)
