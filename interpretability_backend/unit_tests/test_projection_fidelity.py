"""Unit tests for ProjectionFidelityEvaluator (synthetic data, no DB/model)."""

import numpy as np
import pytest

from evaluation.projection_fidelity import (
    ProjectionFidelityEvaluator,
    _n_items_from_pairs,
)


def _find(result, reference, target):
    for c in result["comparisons"]:
        if c["reference"] == reference and c["target"] == target:
            return c
    return None


def test_n_items_inference():
    # N=5 -> 10 pairs; N=60 -> 1770 pairs.
    assert _n_items_from_pairs(10) == 5
    assert _n_items_from_pairs(1770) == 60


def test_identical_structure_gives_rho_one():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(40, 3))
    d = ProjectionFidelityEvaluator.projection_distances(coords)
    ev = ProjectionFidelityEvaluator(k=5, n_perms=50, seed=1)
    res = ev.evaluate(references={"a": d}, targets={"a_copy": d.copy()})
    comp = _find(res, "a", "a_copy")
    assert comp is not None
    assert comp["global_rho"] == pytest.approx(1.0, abs=1e-9)
    assert comp["knn_rho"] == pytest.approx(1.0, abs=1e-6)
    # Observed rho of 1.0 is far above the permutation null.
    assert comp["perm_empirical_p"] == pytest.approx(0.0, abs=1e-9)
    assert res["n_items"] == 40


def test_unrelated_structure_gives_low_rho():
    rng = np.random.default_rng(1)
    a = ProjectionFidelityEvaluator.projection_distances(rng.normal(size=(60, 4)))
    b = ProjectionFidelityEvaluator.projection_distances(rng.normal(size=(60, 4)))
    ev = ProjectionFidelityEvaluator(k=10, n_perms=200, seed=2)
    res = ev.evaluate(references={"a": a}, targets={"b": b})
    comp = _find(res, "a", "b")
    assert abs(comp["global_rho"]) < 0.2
    # Not significant: empirical p should be far from 0.
    assert comp["perm_empirical_p"] > 0.01


def test_fidelity_and_baseline_pairs_enumerated():
    rng = np.random.default_rng(3)
    n = 30
    emb = ProjectionFidelityEvaluator.embedding_distances(rng.normal(size=(n, 8)))
    col = ProjectionFidelityEvaluator.projection_distances(rng.normal(size=(n, 3)))
    umap = ProjectionFidelityEvaluator.projection_distances(rng.normal(size=(n, 3)))
    pca = ProjectionFidelityEvaluator.projection_distances(rng.normal(size=(n, 3)))
    ev = ProjectionFidelityEvaluator(k=5, n_perms=0)  # perms disabled for speed
    res = ev.evaluate(
        references={"colour": col, "embedding": emb},
        targets={"umap_3d": umap, "pca_3d": pca},
        cross_reference=True,
    )
    kinds = {(c["reference"], c["target"]): c["kind"] for c in res["comparisons"]}
    # 2 references x 2 targets = 4 fidelity comparisons
    assert kinds[("colour", "umap_3d")] == "fidelity"
    assert kinds[("embedding", "pca_3d")] == "fidelity"
    # plus the single reference-vs-reference baseline
    assert kinds[("colour", "embedding")] == "baseline"
    assert sum(1 for v in kinds.values() if v == "fidelity") == 4
    assert sum(1 for v in kinds.values() if v == "baseline") == 1
    # perms disabled -> null-test fields are None, correlations still present
    for c in res["comparisons"]:
        assert c["perm_z"] is None
        assert c["global_rho"] is not None


def test_degenerate_input_returns_none_never_raises():
    # All points identical -> all pairwise distances zero -> Spearman undefined.
    zeros = ProjectionFidelityEvaluator.projection_distances(np.ones((20, 3)))
    other = ProjectionFidelityEvaluator.projection_distances(
        np.random.default_rng(4).normal(size=(20, 3))
    )
    ev = ProjectionFidelityEvaluator(k=5, n_perms=10)
    res = ev.evaluate(references={"flat": zeros}, targets={"real": other})
    comp = _find(res, "flat", "real")
    assert comp["global_rho"] is None  # degraded, not raised
    # Permutation stats must also degrade to None (not a misleading p_emp=0.0).
    assert comp["perm_z"] is None
    assert comp["perm_empirical_p"] is None


def test_malformed_condensed_length_does_not_raise():
    # A length that is not a triangular number (N*(N-1)/2) is not a valid
    # condensed vector; evaluate() must skip it rather than let squareform raise.
    bad = np.zeros(5)  # 5 is not triangular (3->3, 4->6)
    res = ProjectionFidelityEvaluator(k=3, n_perms=0).evaluate(
        references={"bad": bad}, targets={"bad2": np.ones(5)}
    )
    assert res["comparisons"] == []


def test_non_1d_input_does_not_raise():
    # A 2-D matrix (or scalar) is not a condensed vector; must be skipped.
    square = np.zeros((6, 6))
    res = ProjectionFidelityEvaluator(k=3, n_perms=0).evaluate(
        references={"matrix": square}, targets={"scalar": np.float64(1.0)}
    )
    assert res["comparisons"] == []


def test_length_mismatch_is_skipped():
    rng = np.random.default_rng(5)
    big = ProjectionFidelityEvaluator.projection_distances(rng.normal(size=(40, 3)))
    small = ProjectionFidelityEvaluator.projection_distances(rng.normal(size=(20, 3)))
    ev = ProjectionFidelityEvaluator(k=5, n_perms=0)
    res = ev.evaluate(references={"big": big}, targets={"small": small})
    # Mismatched vector dropped -> no comparison produced, no raise.
    assert res["comparisons"] == []
    assert res["n_items"] == 40


def test_colour_distances_shape_and_symmetry():
    # Uses scikit-image if present; skip cleanly otherwise.
    pytest.importorskip("skimage")
    hexes = ["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"]
    d = ProjectionFidelityEvaluator.colour_distances(hexes)
    n = len(hexes)
    assert d.shape == (n * (n - 1) // 2,)
    assert np.all(d >= 0)
    # black vs white should be the largest perceptual gap among these.
    from scipy.spatial.distance import squareform

    sq = squareform(d)
    assert sq[0, 1] == pytest.approx(sq.max())


def test_colour_distances_rejects_bad_hex():
    with pytest.raises(ValueError):
        ProjectionFidelityEvaluator.colour_distances(["#000000", "not-a-hex"])
