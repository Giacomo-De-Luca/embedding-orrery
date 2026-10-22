"""
Unit tests for topic-extraction config plumbing.

Covers:
- ``build_topic_extraction_config`` converter: defaults and pass-through of the
  clustering-space fields (``cluster_on`` + BERTopic UMAP params) that were
  previously dropped on the way from GraphQL to the service config.
- ``_reduce_for_clustering`` helper: BERTopic-style UMAP reduction shape.
- Reduction requests: strategy pass-through, validation, legacy defaults.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.API.converters import build_topic_extraction_config, reduction_request_from_input
from backend.services.topic_extraction_service import _reduce_for_clustering
from backend.services.topic_reduction_types import (
    SETTINGS,
    ReductionRequest,
    TopicReductionSettings,
)


def _make_tc(**overrides):
    """Build a stub GraphQL TopicConfigInput-like object with sane defaults."""
    base = {
        "min_topic_size": 10,
        "n_keywords": 10,
        "use_llm_labels": False,
        "llm_provider": "gemini",
        "llm_model": "gemini-3-flash-preview",
        "projection_type": "umap_2d",
        "clustering_method": "hdbscan",
        "n_clusters": None,
        "reduction": None,
        "cluster_on": "cluster_umap",
        "cluster_n_components": 5,
        "cluster_min_dist": 0.0,
        "cluster_n_neighbors": 15,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_converter_defaults_to_cluster_umap():
    """With no input, the converter should default to the BERTopic UMAP space."""
    config = build_topic_extraction_config("my_collection", None)

    assert config.collection_name == "my_collection"
    assert config.cluster_on == "cluster_umap"
    assert config.cluster_n_components == 5
    assert config.cluster_min_dist == 0.0
    assert config.cluster_n_neighbors == 15


def test_converter_passes_through_projection_and_umap_params():
    """Explicit cluster_on + custom UMAP params must reach the service config."""
    tc = _make_tc(
        cluster_on="projection",
        cluster_n_components=8,
        cluster_min_dist=0.05,
        cluster_n_neighbors=30,
    )

    config = build_topic_extraction_config("c", tc)

    assert config.cluster_on == "projection"
    assert config.cluster_n_components == 8
    assert config.cluster_min_dist == 0.05
    assert config.cluster_n_neighbors == 30


def test_converter_passes_through_embedding_mode():
    tc = _make_tc(cluster_on="embedding")
    config = build_topic_extraction_config("c", tc)
    assert config.cluster_on == "embedding"


def test_reduce_for_clustering_output_shape():
    """UMAP reduction should return (n_rows, n_components)."""
    pytest.importorskip("umap")
    rng = np.random.default_rng(7)
    embeddings = rng.standard_normal((60, 16)).astype(np.float32)

    reduced = _reduce_for_clustering(embeddings, n_components=5, min_dist=0.0, n_neighbors=15)

    assert reduced.shape == (60, 5)


def test_reduce_for_clustering_clamps_small_n():
    """n_neighbors/n_components clamp to the sample count so small N doesn't raise."""
    pytest.importorskip("umap")
    rng = np.random.default_rng(7)
    # 8 samples with defaults n_neighbors=15, n_components=5 → must clamp, not raise.
    embeddings = rng.standard_normal((8, 16)).astype(np.float32)

    reduced = _reduce_for_clustering(embeddings, n_components=5, min_dist=0.0, n_neighbors=15)

    assert reduced.shape[0] == 8
    assert reduced.shape[1] <= 5


def _reduction(**overrides):
    base = {
        "enabled": True,
        "method": "fixed_n",
        "n_topics": 5,
        "strategy": None,
        "use_ctfidf": True,
        "distance_threshold": None,
        "absorb_noise": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_converter_passes_the_reduction_strategy_through():
    tc = _make_tc(reduction=_reduction(strategy="density", absorb_noise=True))
    config = build_topic_extraction_config("c", tc)

    assert config.reduce_topics
    assert config.reduction_strategy == "density"
    assert config.reduction_absorb_noise
    assert config.nr_topics == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"strategy": "single"}, "Unknown reduction strategy"),
        ({"strategy": "ward", "method": "auto"}, "topic count"),
        ({"n_topics": None}, "nTopics"),
        ({"strategy": "average", "absorb_noise": True}, "absorbNoise"),
        ({"strategy": "ward", "method": "auto", "distance_threshold": 0.5}, "topic count"),
    ],
)
def test_converter_rejects_invalid_reductions(overrides, message):
    tc = _make_tc(reduction=_reduction(**overrides))
    with pytest.raises(ValueError, match=message):
        build_topic_extraction_config("c", tc)


def test_disabled_reduction_is_not_validated():
    tc = _make_tc(reduction=_reduction(enabled=False, strategy="single"))
    assert not build_topic_extraction_config("c", tc).reduce_topics


class TestReductionRequest:
    def test_null_strategy_is_the_legacy_one_use_ctfidf_picks(self):
        ctfidf = ReductionRequest.create(strategy=None, method="auto", n_topics=None)
        semantic = ReductionRequest.create(
            strategy=None, method="auto", n_topics=None, use_ctfidf=False
        )
        assert ctfidf.strategy == "legacy_ctfidf"
        assert semantic.strategy == "legacy_semantic"

    def test_auto_average_takes_the_configured_threshold(self):
        request = ReductionRequest.create(strategy="average", method="auto", n_topics=9)
        assert request.distance_threshold == SETTINGS.auto_distance_threshold
        assert request.n_topics is None

    def test_from_graphql_input(self):
        request = reduction_request_from_input(
            SimpleNamespace(
                strategy="ward",
                method="fixed_n",
                n_topics=4,
                use_ctfidf=True,
                distance_threshold=None,
                absorb_noise=False,
                regenerate_labels=True,
                llm_provider="openai",
                llm_model="gpt",
            )
        )
        assert (request.strategy, request.n_topics, request.llm_provider) == ("ward", 4, "openai")
        assert request.regenerate_labels

    @pytest.mark.parametrize("threshold", [-0.1, 2.5])
    def test_a_threshold_outside_cosine_distance_is_rejected(self, threshold):
        with pytest.raises(ValueError, match="between 0 and 2"):
            ReductionRequest.create(
                strategy="average", method="auto", n_topics=None, distance_threshold=threshold
            )

    def test_settings_reject_a_threshold_outside_cosine_distance(self):
        with pytest.raises(ValueError, match="auto_distance_threshold"):
            TopicReductionSettings(auto_distance_threshold=3.0)
