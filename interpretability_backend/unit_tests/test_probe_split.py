"""Seeded train/val splits for probes (backend/utils/probe_split.py).

Regression and binary probes keep the original random split bit-for-bit (so
metrics stay comparable with probes trained before multiclass support, and
across kinds on the same field). Multiclass probes get a per-class split that
keeps every class on both sides, and a stratified subsample when the
training cap applies so rare classes survive it.
"""

import numpy as np
import pytest

from backend.utils.probe_split import ProbeSplitter


def _legacy_split(valid_idx: np.ndarray, cap: int, seed: int, train_split: float):
    """The inline split run_probe_core used before ProbeSplitter existed."""
    rng = np.random.default_rng(seed)
    if len(valid_idx) > cap:
        pool_idx = rng.choice(valid_idx, size=cap, replace=False)
        pool_idx.sort()
    else:
        pool_idx = valid_idx
    perm = rng.permutation(len(pool_idx))
    cut = int(len(pool_idx) * train_split)
    return pool_idx, perm[:cut], perm[cut:]


class TestRandomSplit:
    @pytest.mark.parametrize("cap", [10_000, 120])
    def test_matches_legacy_split_exactly(self, cap):
        valid_idx = np.flatnonzero(np.arange(400) % 3 != 0)
        expected = _legacy_split(valid_idx, cap, seed=7, train_split=0.8)
        actual = ProbeSplitter(seed=7, train_split=0.8).split(valid_idx, cap)
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want)


def _labels(counts: list[int]) -> np.ndarray:
    return np.concatenate([np.full(n, k) for k, n in enumerate(counts)]).astype(np.int64)


class TestStratifiedSplit:
    def test_every_class_on_both_sides(self):
        labels = _labels([200, 50, 5, 6])
        valid_idx = np.arange(len(labels))
        pool, train, val = ProbeSplitter(seed=1, train_split=0.8).split(
            valid_idx, 10_000, labels=labels
        )
        np.testing.assert_array_equal(pool, valid_idx)
        for k in range(4):
            assert np.any(labels[pool[train]] == k)
            assert np.any(labels[pool[val]] == k)
        assert len(train) + len(val) == len(labels)
        assert set(train).isdisjoint(val)

    def test_validation_share_is_proportional(self):
        labels = _labels([500, 100])
        _, _, val = ProbeSplitter(seed=1, train_split=0.8).split(
            np.arange(600), 10_000, labels=labels
        )
        assert len(val) == 120

    def test_deterministic_for_a_seed(self):
        labels = _labels([40, 30, 20])
        a = ProbeSplitter(seed=3, train_split=0.8).split(np.arange(90), 10_000, labels=labels)
        b = ProbeSplitter(seed=3, train_split=0.8).split(np.arange(90), 10_000, labels=labels)
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)

    def test_labels_index_the_valid_rows(self):
        # valid_idx is a subset of rows; labels are aligned to valid_idx.
        valid_idx = np.array([3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
        labels = np.array([0, 1] * 6)
        pool, train, val = ProbeSplitter(seed=0, train_split=0.5).split(
            valid_idx, 10_000, labels=labels
        )
        assert set(pool[train]) | set(pool[val]) == set(valid_idx)


class TestStratifiedSubsample:
    def test_cap_keeps_rare_classes(self):
        labels = _labels([5_000, 8, 6])
        valid_idx = np.arange(len(labels))
        pool, train, val = ProbeSplitter(seed=0, train_split=0.8).split(
            valid_idx, 500, labels=labels
        )
        assert len(pool) <= 500
        pool_labels = labels[pool]
        assert np.sum(pool_labels == 1) >= 5
        assert np.sum(pool_labels == 2) >= 5
        for k in range(3):
            assert np.any(pool_labels[val] == k)
        assert np.all(np.diff(pool) > 0)  # sorted, unique

    def test_cap_roughly_preserves_proportions(self):
        labels = _labels([3_000, 1_000])
        pool, _, _ = ProbeSplitter(seed=0, train_split=0.8).split(
            np.arange(4_000), 400, labels=labels
        )
        share = np.mean(labels[pool] == 0)
        assert 0.7 <= share <= 0.8
