"""
Unit tests for topic-extraction config plumbing.

Covers:
- ``build_topic_extraction_config`` converter: defaults and pass-through of the
  clustering-space fields (``cluster_on`` + BERTopic UMAP params) that were
  previously dropped on the way from GraphQL to the service config.
- ``_reduce_for_clustering`` helper: BERTopic-style UMAP reduction shape.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.API.converters import build_topic_extraction_config
from backend.services.topic_extraction_service import _reduce_for_clustering


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
