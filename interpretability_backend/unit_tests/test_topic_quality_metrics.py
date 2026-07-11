"""Unit tests for TopicQualityEvaluator (synthetic data, no DB / no model)."""

import numpy as np
import pytest

from evaluation.quality_metrics import METRIC_NAMES, TopicQualityEvaluator


def _blobs(centers, n_per=40, spread=0.05, dim=8, seed=0):
    """Build (coords, labels) from gaussian blobs around given centers."""
    rng = np.random.default_rng(seed)
    pts, labels = [], []
    for cluster_id, center in enumerate(centers):
        base = np.zeros(dim)
        base[: len(center)] = center
        pts.append(rng.normal(base, spread, size=(n_per, dim)))
        labels.extend([cluster_id] * n_per)
    return np.vstack(pts), np.array(labels)


@pytest.fixture
def evaluator():
    return TopicQualityEvaluator()


def test_well_separated_blobs_high_silhouette(evaluator):
    coords, labels = _blobs([(0, 0), (10, 10), (-10, 8)], spread=0.05)
    res = evaluator.evaluate(labels, projection_coords=coords)
    assert res["num_clusters_evaluated"] == 3
    assert res["silhouette_cluster_space"] is not None
    assert res["silhouette_cluster_space"] > 0.5


def test_overlapping_blobs_lower_silhouette(evaluator):
    sep_coords, sep_labels = _blobs([(0, 0), (10, 10)], spread=0.05, seed=1)
    ov_coords, ov_labels = _blobs([(0, 0), (0.3, 0.3)], spread=1.0, seed=1)
    sep = evaluator.evaluate(sep_labels, projection_coords=sep_coords)
    overlap = evaluator.evaluate(ov_labels, projection_coords=ov_coords)
    assert overlap["silhouette_cluster_space"] < sep["silhouette_cluster_space"]


def test_topic_diversity_disjoint_vs_duplicated(evaluator):
    disjoint = {0: [("a", 1.0), ("b", 1.0)], 1: [("c", 1.0), ("d", 1.0)]}
    duplicated = {0: [("a", 1.0), ("b", 1.0)], 1: [("a", 1.0), ("b", 1.0)]}
    assert evaluator._topic_diversity(disjoint, n_keywords=2) == 1.0
    # 2 unique words out of 4 total -> 0.5 == 1/n_topics
    assert evaluator._topic_diversity(duplicated, n_keywords=2) == pytest.approx(0.5)


def test_coherence_cooccurring_beats_non_cooccurring(evaluator):
    # "apple" & "banana" always co-occur; "car" & "ocean" never co-occur.
    documents = ["apple banana apple banana"] * 20 + ["car only here", "ocean far away"] * 20
    topics_data = {
        0: [("apple", 1.0), ("banana", 1.0)],  # coherent
        1: [("car", 1.0), ("ocean", 1.0)],  # incoherent
    }
    measures = {"coherence_cv", "coherence_umass"}
    # Evaluate each topic alone so the aggregate reflects that single topic.
    coherent = evaluator._coherence(documents, {0: topics_data[0]}, "english", 10, measures)
    incoherent = evaluator._coherence(documents, {1: topics_data[1]}, "english", 10, measures)
    assert coherent["coherence_cv"] is not None and incoherent["coherence_cv"] is not None
    assert coherent["coherence_cv"] > incoherent["coherence_cv"]
    assert coherent["coherence_umass"] > incoherent["coherence_umass"]


def test_fewer_than_two_clusters_returns_none(evaluator):
    coords, _ = _blobs([(0, 0)], n_per=30)
    labels = np.zeros(30, dtype=int)  # single cluster
    res = evaluator.evaluate(labels, projection_coords=coords)
    assert res["num_clusters_evaluated"] == 1
    assert res["silhouette_cluster_space"] is None


def test_noise_points_excluded_from_silhouette(evaluator):
    coords, labels = _blobs([(0, 0), (10, 10)], n_per=30, spread=0.05, seed=2)
    base = evaluator.evaluate(labels, projection_coords=coords, sample_size=100000)

    rng = np.random.default_rng(3)
    noise_pts = rng.normal(5, 5, size=(15, coords.shape[1]))
    coords_with_noise = np.vstack([coords, noise_pts])
    labels_with_noise = np.concatenate([labels, np.full(15, -1)])
    with_noise = evaluator.evaluate(
        labels_with_noise, projection_coords=coords_with_noise, sample_size=100000
    )

    assert with_noise["silhouette_cluster_space"] == pytest.approx(
        base["silhouette_cluster_space"], rel=1e-9
    )


def test_sampling_sets_flag_and_returns_finite(evaluator):
    coords, labels = _blobs([(0, 0), (10, 10), (20, 0)], n_per=200, spread=0.1, seed=4)
    res = evaluator.evaluate(labels, projection_coords=coords, sample_size=50)
    assert res["sampled"] is True
    assert res["silhouette_cluster_space"] is not None
    assert np.isfinite(res["silhouette_cluster_space"])


def test_mismatched_lengths_do_not_raise(evaluator):
    # coords shorter than labels -> ignored, not an IndexError ("never raises").
    coords, labels = _blobs([(0, 0), (10, 10)], n_per=30, seed=6)
    res = evaluator.evaluate(labels, projection_coords=coords[:-5])
    assert res["silhouette_cluster_space"] is None  # dropped due to mismatch


def test_dbcv_read_from_hdbscan_model(evaluator):
    class _FakeModel:
        relative_validity_ = 0.73

    coords, labels = _blobs([(0, 0), (10, 10)], seed=5)
    res = evaluator.evaluate(labels, projection_coords=coords, hdbscan_model=_FakeModel())
    assert res["dbcv"] == pytest.approx(0.73)


def test_metric_selection_only_requested_keys(evaluator):
    coords, labels = _blobs([(0, 0), (10, 10)], seed=7)
    docs = ["apple banana fruit"] * len(labels)
    topics = {0: [("apple", 1.0), ("banana", 1.0)], 1: [("fruit", 1.0), ("apple", 0.5)]}
    res = evaluator.evaluate(
        labels,
        projection_coords=coords,
        topics_data=topics,
        documents=docs,
        metrics={"silhouette", "diversity"},
    )
    assert "silhouette_cluster_space" in res and res["silhouette_cluster_space"] is not None
    assert "topic_diversity" in res and res["topic_diversity"] is not None
    # Unrequested metric keys are omitted entirely.
    assert "coherence_cv" not in res
    assert "coherence_umass" not in res
    assert "dbcv" not in res
    assert sorted(res["metrics_computed"]) == ["diversity", "silhouette"]
    assert res["computed_at"]


def test_metric_selection_unknown_names_ignored(evaluator):
    coords, labels = _blobs([(0, 0), (10, 10)], seed=8)
    res = evaluator.evaluate(
        labels, projection_coords=coords, metrics={"silhouette", "not_a_metric"}
    )
    assert res["silhouette_cluster_space"] is not None
    assert res["metrics_computed"] == ["silhouette"]


def test_default_metrics_include_all_names(evaluator):
    coords, labels = _blobs([(0, 0), (10, 10)], seed=9)
    res = evaluator.evaluate(labels, projection_coords=coords)
    assert set(res["metrics_computed"]) == set(METRIC_NAMES)
    # All metric result keys present (values may be None without docs/topics).
    for key in ("dbcv", "silhouette_cluster_space", "topic_diversity", "coherence_cv"):
        assert key in res
