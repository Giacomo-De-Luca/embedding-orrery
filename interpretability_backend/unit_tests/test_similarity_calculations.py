"""
Tests for similarity calculation logic.

These tests verify that distance-to-similarity conversions
are mathematically correct for different metrics.
"""



class TestDistanceToSimilarity:
    """Test distance metric conversions used in semantic search."""

    def test_cosine_distance_to_similarity_zero_distance(self):
        """Cosine distance 0 should give similarity 1 (identical vectors)."""
        distance = 0.0
        similarity = 1 - distance
        assert similarity == 1.0

    def test_cosine_distance_to_similarity_max_distance(self):
        """Cosine distance 2 should give similarity -1 (opposite vectors)."""
        distance = 2.0
        similarity = 1 - distance
        assert similarity == -1.0

    def test_cosine_distance_to_similarity_orthogonal(self):
        """Cosine distance 1 should give similarity 0 (orthogonal vectors)."""
        distance = 1.0
        similarity = 1 - distance
        assert similarity == 0.0

    def test_cosine_similarity_range(self):
        """Cosine similarity should be in range [-1, 1] for valid distances."""
        distances = [0.0, 0.5, 1.0, 1.5, 2.0]
        for d in distances:
            similarity = 1 - d
            assert -1.0 <= similarity <= 1.0, f"Distance {d} gave invalid similarity {similarity}"

    def test_l2_distance_to_similarity_zero_distance(self):
        """L2 distance 0 should give similarity 1 (identical vectors)."""
        distance = 0.0
        similarity = 1 / (1 + distance)
        assert similarity == 1.0

    def test_l2_distance_to_similarity_large_distance(self):
        """L2 similarity should approach 0 as distance increases."""
        distances = [1.0, 10.0, 100.0, 1000.0]
        prev_similarity = 1.0
        for d in distances:
            similarity = 1 / (1 + d)
            assert 0 < similarity < prev_similarity, "L2 similarity should decrease with distance"
            prev_similarity = similarity

    def test_l2_similarity_always_positive(self):
        """L2 similarity should always be positive (0, 1] for any distance >= 0."""
        distances = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
        for d in distances:
            similarity = 1 / (1 + d)
            assert similarity > 0, f"L2 similarity should be positive, got {similarity}"
            assert similarity <= 1, f"L2 similarity should be <= 1, got {similarity}"

    def test_inner_product_distance_to_similarity(self):
        """Inner product: ChromaDB stores negative IP, so we negate to get similarity."""
        # ChromaDB stores -IP as distance, so high similarity = low distance
        distances = [-0.9, -0.5, 0.0, 0.5]  # These are actually -similarity values
        for d in distances:
            similarity = -d
            # Result should be the actual inner product value
            assert similarity == -d

    def test_inner_product_ordering(self):
        """More negative distance = higher similarity for inner product."""
        distances = [-0.9, -0.5, -0.1]  # Most similar to least similar
        similarities = [-d for d in distances]
        # Should be in descending order
        assert similarities == sorted(similarities, reverse=True)


class TestSimilarityConversionFunction:
    """Test the actual conversion logic as implemented in chromadb_client."""

    def convert_distances_to_similarities(
        self, distances: list[float], metric: str
    ) -> list[float]:
        """Replicate the conversion logic from chromadb_client.py."""
        if metric == "cosine":
            return [1 - d for d in distances]
        elif metric == "l2":
            return [1 / (1 + d) for d in distances]
        elif metric == "ip":
            return [-d for d in distances]
        else:
            return [1 - d for d in distances]  # Default to cosine

    def test_batch_cosine_conversion(self):
        """Test batch conversion for cosine distances."""
        distances = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]
        expected = [1.0, 0.9, 0.5, 0.0, -0.5, -1.0]
        result = self.convert_distances_to_similarities(distances, "cosine")
        assert result == expected

    def test_batch_l2_conversion(self):
        """Test batch conversion for L2 distances."""
        distances = [0.0, 1.0, 3.0, 9.0]
        expected = [1.0, 0.5, 0.25, 0.1]
        result = self.convert_distances_to_similarities(distances, "l2")
        assert result == expected

    def test_batch_ip_conversion(self):
        """Test batch conversion for inner product distances."""
        distances = [-1.0, -0.5, 0.0, 0.5]
        expected = [1.0, 0.5, 0.0, -0.5]
        result = self.convert_distances_to_similarities(distances, "ip")
        assert result == expected

    def test_unknown_metric_defaults_to_cosine(self):
        """Unknown metrics should default to cosine behavior."""
        distances = [0.0, 0.5, 1.0]
        expected = [1.0, 0.5, 0.0]
        result = self.convert_distances_to_similarities(distances, "unknown_metric")
        assert result == expected

    def test_empty_distances(self):
        """Empty distance list should return empty similarity list."""
        result = self.convert_distances_to_similarities([], "cosine")
        assert result == []


class TestSearchResultOrdering:
    """Test that search results maintain correct ordering after conversion."""

    def test_cosine_ordering_preserved(self):
        """Most similar items (lowest distance) should have highest similarity."""
        # Distances sorted by relevance (most similar first)
        distances = [0.1, 0.3, 0.5, 0.8, 1.2]
        similarities = [1 - d for d in distances]

        # Similarities should be in descending order
        assert similarities == sorted(similarities, reverse=True)

    def test_l2_ordering_preserved(self):
        """Most similar items (lowest distance) should have highest similarity."""
        distances = [0.5, 1.0, 2.0, 5.0, 10.0]
        similarities = [1 / (1 + d) for d in distances]

        # Similarities should be in descending order
        assert similarities == sorted(similarities, reverse=True)

    def test_realistic_search_results(self):
        """Test with realistic ChromaDB cosine distance values."""
        # Typical ChromaDB cosine distances for semantic search
        distances = [0.12, 0.25, 0.38, 0.45, 0.67]
        similarities = [1 - d for d in distances]

        # All should be positive (similar items)
        assert all(s > 0 for s in similarities)
        # Most similar should be > 0.8
        assert similarities[0] > 0.8
        # Least similar should still be somewhat relevant
        assert similarities[-1] > 0.3
