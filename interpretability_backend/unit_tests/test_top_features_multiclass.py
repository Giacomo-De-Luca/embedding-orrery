"""Tests for top_features' fold-glob + multiclass handling, and the
multiclass logreg feature-importance path in train_sklearn_probe."""

import numpy as np
import pandas as pd
import torch

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.probe import SklearnProbeSpec
from interpret.probing.configs.sae_analysis import TopFeaturesConfig
from interpret.probing.probes.sklearn_probes import train_sklearn_probe
from interpret.probing.sae_analysis.top_features import run_top_features


def _sae_dataset(kept: list[int], intermediate: str = "sae_max"):
    return ActivationDataset(
        activations={(0, intermediate): torch.zeros(2, len(kept))},
        sample_ids=["a", "b"],
        metadata={"kept_by_layer": {0: kept}},
    )


def _save_npz(path, coef):
    np.savez(
        str(path),
        coef=np.asarray(coef),
        intercept=np.zeros(1),
        scaler_mean=np.zeros(2),
        scaler_scale=np.ones(2),
    )


class TestTopFeatures:
    def test_multiclass_fold_glob_and_top_class(self, tmp_path):
        directions = tmp_path / "directions"
        directions.mkdir()
        # [3 classes, 2 features]; identical folds -> mean unchanged.
        coef = [[1.0, -4.0], [2.0, 1.0], [-3.0, 0.0]]
        _save_npz(directions / "L0_sae_max_logreg_fold_0.npz", coef)
        _save_npz(directions / "L0_sae_max_logreg_fold_1.npz", coef)

        result = run_top_features(
            _sae_dataset(kept=[5, 9]),
            directions,
            TopFeaturesConfig(
                source_probe="logreg",
                top_k=5,
                sae_vectors_dir=str(tmp_path / "no_labels"),
            ),
            tmp_path / "out",
            width="16k",
        )
        feats = result[0]
        # Feature column 1 (true idx 9) wins via class 0's |-4|; column 0
        # (true idx 5) follows via class 2's |-3|. Signed coefs preserved.
        assert [f["feature_idx"] for f in feats] == [9, 5]
        assert feats[0]["coef"] == -4.0
        assert feats[0]["top_class"] == 0
        assert feats[1]["coef"] == -3.0
        assert feats[1]["top_class"] == 2
        assert (tmp_path / "out" / "top_features.json").exists()

    def test_binary_unsuffixed_no_top_class(self, tmp_path):
        directions = tmp_path / "directions"
        directions.mkdir()
        _save_npz(directions / "L0_sae_max_logreg.npz", [[0.5, -2.0]])

        result = run_top_features(
            _sae_dataset(kept=[3, 4]),
            directions,
            TopFeaturesConfig(
                source_probe="logreg",
                top_k=5,
                sae_vectors_dir=str(tmp_path / "no_labels"),
            ),
            tmp_path / "out",
            width="16k",
        )
        feats = result[0]
        assert [f["feature_idx"] for f in feats] == [4, 3]
        assert all("top_class" not in f for f in feats)

    def test_classic_sae_feat_intermediate_still_handled(self, tmp_path):
        directions = tmp_path / "directions"
        directions.mkdir()
        _save_npz(directions / "L0_sae_feat_logreg.npz", [[1.0, 2.0]])
        result = run_top_features(
            _sae_dataset(kept=[0, 1], intermediate="sae_feat"),
            directions,
            TopFeaturesConfig(
                source_probe="logreg",
                top_k=5,
                sae_vectors_dir=str(tmp_path / "no_labels"),
            ),
            tmp_path / "out",
            width="16k",
        )
        assert [f["feature_idx"] for f in result[0]] == [1, 0]

    def test_missing_directions_skips_layer(self, tmp_path):
        directions = tmp_path / "empty"
        directions.mkdir()
        result = run_top_features(
            _sae_dataset(kept=[0, 1]),
            directions,
            TopFeaturesConfig(
                source_probe="logreg",
                top_k=5,
                sae_vectors_dir=str(tmp_path / "no_labels"),
            ),
            tmp_path / "out",
            width="16k",
        )
        assert result == {}


class TestMulticlassFeatureImportance:
    def test_kfold_multiclass_logreg_writes_importance(self, tmp_path):
        rng = np.random.default_rng(0)
        n, d, classes = 90, 6, 3
        y = np.repeat(np.arange(classes), n // classes).astype(np.int64)
        # Feature 0 correlates with the class; the rest are noise.
        X = rng.normal(size=(n, d)).astype(np.float32)
        X[:, 0] += y * 2.0

        dataset = ActivationDataset(
            activations={(0, "sae_max"): torch.from_numpy(X)},
            sample_ids=[f"s{i}" for i in range(n)],
        )
        spec = SklearnProbeSpec(
            kind="logreg",
            class_weight="balanced",
            n_folds=2,
            save_directions=True,
        )
        out = train_sklearn_probe(
            dataset,
            spec,
            y,
            tmp_path / "probe",
            feature_names=[f"L0_resid_post_f{i}" for i in range(d)],
        )

        importance = pd.read_csv(out / "feature_importance.csv")
        assert len(importance) == d
        assert set(importance["feature_name"]) == {f"L0_resid_post_f{i}" for i in range(d)}
        # Multiclass collapse: |beta| max over classes -> nonnegative means,
        # and the informative feature dominates.
        assert (importance["beta_mean"] >= 0).all()
        top = importance.sort_values("abs_beta_mean", ascending=False).iloc[0]
        assert top["feature_name"] == "L0_resid_post_f0"

        # Per-fold multiclass directions exist for top_features to glob.
        folds = sorted((out / "directions").glob("L0_sae_max_logreg_fold_*.npz"))
        assert len(folds) == 2
        assert np.load(str(folds[0]))["coef"].shape == (classes, d)
