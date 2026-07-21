"""Unit tests for the MLP probe trainer's train/dev/val discipline.

Early stopping and best-checkpoint selection must run on a dev subset carved
out of the train side — never on the validation split whose metrics are
reported. These tests pin the `_split_train_dev` helper contract and the
trainer behavior around it (leak fix: previously both patience and best-state
selection read the reporting split).
"""

import numpy as np
import pandas as pd
import torch

import pytest

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.probe import MLPProbeSpec
from interpret.probing.probes.mlp_probe import _DEV_FLOOR, _split_train_dev, train_mlp_probes
from interpret.probing.utils.enums import TaskType


# ------------------------------------------------------------------
# _split_train_dev
# ------------------------------------------------------------------


class TestSplitTrainDev:
    def test_disjoint_and_covering(self):
        fit_idx, dev_idx = _split_train_dev(100, 0.2, seed=7)
        assert len(dev_idx) == 20
        assert len(fit_idx) == 80
        assert len(np.intersect1d(fit_idx, dev_idx)) == 0
        assert np.array_equal(np.sort(np.concatenate([fit_idx, dev_idx])), np.arange(100))

    def test_zero_dev_split_disables(self):
        fit_idx, dev_idx = _split_train_dev(100, 0.0, seed=7)
        assert len(dev_idx) == 0
        assert np.array_equal(fit_idx, np.arange(100))

    def test_floor_falls_back_to_no_dev(self):
        # 30 * 0.2 = 6 < _DEV_FLOOR -> no dev set, all rows train.
        assert 30 * 0.2 < _DEV_FLOOR
        fit_idx, dev_idx = _split_train_dev(30, 0.2, seed=7)
        assert len(dev_idx) == 0
        assert np.array_equal(fit_idx, np.arange(30))

    def test_stratified_split_preserves_class_ratio(self):
        y = np.array([0] * 180 + [1] * 20)
        fit_idx, dev_idx = _split_train_dev(200, 0.2, seed=7, stratify_y=y)
        assert len(dev_idx) == 40
        # The minority class keeps its share of the dev set (4 of 20).
        assert int((y[dev_idx] == 1).sum()) == 4
        assert int((y[fit_idx] == 1).sum()) == 16

    def test_deterministic_per_seed(self):
        a_fit, a_dev = _split_train_dev(100, 0.2, seed=7)
        b_fit, b_dev = _split_train_dev(100, 0.2, seed=7)
        assert np.array_equal(a_fit, b_fit) and np.array_equal(a_dev, b_dev)
        _, c_dev = _split_train_dev(100, 0.2, seed=8)
        assert not np.array_equal(a_dev, c_dev)


# ------------------------------------------------------------------
# Spec validation
# ------------------------------------------------------------------


class TestDevSplitSpec:
    def test_default_dev_split(self):
        assert MLPProbeSpec().dev_split == 0.2

    def test_invalid_dev_split_raises(self):
        with pytest.raises(ValueError, match="dev_split"):
            MLPProbeSpec(dev_split=1.0)
        with pytest.raises(ValueError, match="dev_split"):
            MLPProbeSpec(dev_split=-0.1)


# ------------------------------------------------------------------
# Trainer integration
# ------------------------------------------------------------------


def _linear_setup(n=400, d=16, seed=0):
    """ActivationDataset + targets with a clean linear signal, fixed split."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = (X.astype(np.float64) @ w).astype(np.float32)
    dataset = ActivationDataset(
        activations={(0, "embedding"): torch.from_numpy(X)},
        sample_ids=[f"s{i}" for i in range(n)],
    )
    targets = torch.from_numpy(y).reshape(-1, 1)
    split = (np.arange(0, int(n * 0.8)), np.arange(int(n * 0.8), n))
    return dataset, targets, split


def _run(tmp_path, spec, n=400):
    dataset, targets, split = _linear_setup(n=n)
    train_mlp_probes(
        dataset,
        spec,
        targets,
        tmp_path,
        task_type=TaskType.REGRESSION,
        target_columns=["y"],
        indices_override=split,
    )
    df = pd.read_csv(tmp_path / "probe_results.csv")
    return df.iloc[0]


class TestTrainerDevSplit:
    def test_learns_with_default_dev_split(self, tmp_path):
        row = _run(tmp_path, MLPProbeSpec(hidden_dims=[16], epochs=60, seed=7))
        assert row["val_r2"] > 0.5
        # Early stopping ran on a real dev set.
        assert np.isfinite(row["dev_loss_best"])
        assert (tmp_path / "checkpoints" / "layer_0_embedding.pt").exists()

    def test_zero_dev_split_runs_all_epochs(self, tmp_path):
        row = _run(
            tmp_path,
            MLPProbeSpec(hidden_dims=[16], epochs=12, patience=2, seed=7, dev_split=0.0),
        )
        # No dev set -> early stopping disabled -> every epoch runs and the
        # final weights are the checkpoint.
        assert row["total_epochs"] == 12
        assert row["best_epoch"] == 11
        assert not np.isfinite(row["dev_loss_best"])

    def test_dev_floor_disables_early_stopping(self, tmp_path):
        # Pool of 40 -> train 32 -> dev would be ~6 (< floor) -> disabled.
        row = _run(
            tmp_path,
            MLPProbeSpec(hidden_dims=[8], epochs=8, patience=1, seed=7),
            n=40,
        )
        assert row["total_epochs"] == 8
        assert not np.isfinite(row["dev_loss_best"])

    def test_deterministic_metrics(self, tmp_path):
        spec = MLPProbeSpec(hidden_dims=[16], epochs=20, seed=7)
        row_a = _run(tmp_path / "a", spec)
        row_b = _run(tmp_path / "b", spec)
        assert row_a["val_r2"] == row_b["val_r2"]
        assert row_a["best_epoch"] == row_b["best_epoch"]
