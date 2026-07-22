"""Tests for the sparse (scipy CSR) storage path of pooled SAE activations.

Covers the full chain: `extract_sae_pooled(sparse=True)` equivalence with
the dense path (same stub-SAE trick as `test_sae_pooled_extraction.py`),
`ActivationDataset` row-subset + save/load round-trip on CSR matrices,
`train_sklearn_probe` on sparse input (scale-only standardisation, guards),
sparse `concat`, and `correlation_map` numeric equivalence.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import interpret.probing.extraction.extract_sae_pooled as esp
from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.concat_extraction import ConcatExtractionConfig
from interpret.probing.configs.probe import MLPProbeSpec, SklearnProbeSpec
from interpret.probing.configs.sae_pooled_extraction import (
    SAEPooledExtractionConfig,
)
from interpret.probing.extraction.extract_concat_activations import (
    extract_concat_activations,
)
from interpret.probing.extraction.extract_sae_pooled import extract_sae_pooled
from interpret.probing.probes.mlp_probe import train_mlp_probes
from interpret.probing.probes.sklearn_probes import train_sklearn_probe
from interpret.probing.sae_analysis.correlation_map import _correlate_features
from interpret.probing.utils.cross_validation import resolve_folds
from interpret.probing.utils.enums import TaskType
from unit_tests.sae_pooled_stubs import D_SAE, StubSAE, token_source as _token_source


@pytest.fixture
def patched_sae(monkeypatch):
    monkeypatch.setattr(esp, "load_sae", lambda config: StubSAE())
    monkeypatch.setattr(esp, "clear_sae_cache", lambda: None)


def _config(**overrides) -> SAEPooledExtractionConfig:
    kwargs = dict(
        name="p",
        source_extraction="t",
        device="cpu",
        pooling="max",
        layers=[0, 2],
    )
    kwargs.update(overrides)
    return SAEPooledExtractionConfig(**kwargs)


def _pair(**overrides):
    """Run the extractor twice (dense + sparse) on identical sources."""
    dense = extract_sae_pooled(_token_source(), _config(sparse=False, **overrides))
    sparse = extract_sae_pooled(_token_source(), _config(sparse=True, **overrides))
    return dense, sparse


class TestSparsePooledEquivalence:
    def test_sparse_max_matches_dense(self, patched_sae):
        dense, sparse = _pair()
        for key, dense_mat in dense.activations.items():
            sparse_mat = sparse.activations[key]
            assert sp.issparse(sparse_mat)
            assert sparse_mat.dtype == np.float32
            np.testing.assert_allclose(
                np.asarray(sparse_mat.todense()),
                dense_mat.numpy(),
            )
        assert sparse.metadata["kept_by_layer"] == dense.metadata["kept_by_layer"]
        assert sparse.metadata["sparse"] is True
        assert dense.metadata["sparse"] is False

    def test_sparse_respects_chunk_and_block_boundaries(
        self,
        patched_sae,
        monkeypatch,
    ):
        # Force 1-sample pooling blocks and 2-token encode chunks so both
        # boundary paths are exercised on the 6-token source.
        monkeypatch.setattr(esp, "_POOL_BLOCK_BYTES", 4 * D_SAE)
        dense, sparse = _pair(batch_size_tokens=2)
        for key, dense_mat in dense.activations.items():
            np.testing.assert_allclose(
                np.asarray(sparse.activations[key].todense()),
                dense_mat.numpy(),
            )

    def test_sparse_last_matches_dense_and_keeps_negatives(self, patched_sae):
        dense, sparse = _pair(pooling="last")
        mat = sparse.activations[(0, "sae_last")]
        np.testing.assert_allclose(
            np.asarray(mat.todense()),
            dense.activations[(0, "sae_last")].numpy(),
        )
        # Sample 0's last token has feature 1 = 2.0, sample 1's = -1.0:
        # negative TopK-style values must be stored explicitly, not dropped.
        assert (mat.data < 0).any()

    def test_sparse_min_active_samples_matches_dense(self, patched_sae):
        dense, sparse = _pair(min_active_samples=2)
        assert sparse.metadata["kept_by_layer"] == dense.metadata["kept_by_layer"]
        for key, dense_mat in dense.activations.items():
            np.testing.assert_allclose(
                np.asarray(sparse.activations[key].todense()),
                dense_mat.numpy(),
            )

    def test_sparse_without_dead_feature_filter(self, patched_sae):
        _, sparse = _pair(drop_dead_features=False)
        mat = sparse.activations[(0, "sae_max")]
        assert mat.shape[1] == D_SAE
        assert sparse.metadata["kept_by_layer"][0] == list(range(D_SAE))


def _sparse_dataset(n: int = 4, d: int = 3) -> ActivationDataset:
    rows = np.arange(n * d, dtype=np.float32).reshape(n, d)
    rows[rows % 2 == 0] = 0.0
    return ActivationDataset(
        activations={(0, "sae_max"): sp.csr_matrix(rows)},
        targets=torch.arange(n, dtype=torch.float32),
        sample_ids=[f"s{i}" for i in range(n)],
        metadata={"sparse": True},
    )


class TestSparseDataset:
    def test_subset_reorders_sparse_rows(self):
        ds = _sparse_dataset()
        sub = ds.subset(["s2", "s0"])
        full = np.asarray(ds.activations[(0, "sae_max")].todense())
        got = sub.activations[(0, "sae_max")]
        assert sp.issparse(got)
        np.testing.assert_allclose(np.asarray(got.todense()), full[[2, 0]])
        assert sub.targets.tolist() == [2.0, 0.0]
        assert sub.sample_ids == ["s2", "s0"]

    def test_subset_missing_id_raises(self):
        with pytest.raises(KeyError):
            _sparse_dataset().subset(["s0", "nope"])

    def test_save_load_roundtrip(self, tmp_path):
        ds = _sparse_dataset()
        loaded = ActivationDataset.load(ds.save(tmp_path / "ds.pt"))
        got = loaded.activations[(0, "sae_max")]
        assert sp.issparse(got)
        np.testing.assert_allclose(
            np.asarray(got.todense()),
            np.asarray(ds.activations[(0, "sae_max")].todense()),
        )


def _classification_data(n: int = 90, d: int = 6, classes: int = 3):
    """Zero-inflated non-negative features; feature 0 tracks the class."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(classes), n // classes).astype(np.int64)
    X = rng.random((n, d)).astype(np.float32)
    X[rng.random((n, d)) < 0.6] = 0.0
    X[:, 0] += y * 2.0
    return X, y


def _probe_dataset(matrix) -> ActivationDataset:
    return ActivationDataset(
        activations={(0, "sae_max"): matrix},
        sample_ids=[f"s{i}" for i in range(matrix.shape[0])],
    )


class TestSparseProbes:
    def test_logreg_sparse_matches_scale_only_reference(self, tmp_path):
        X, y = _classification_data()
        spec = SklearnProbeSpec(kind="logreg", save_directions=True)
        out = train_sklearn_probe(
            _probe_dataset(sp.csr_matrix(X)),
            spec,
            y,
            tmp_path / "probe",
        )
        results = pd.read_csv(out / "probe_results.csv")
        assert results["val_accuracy"].notna().all()

        saved = np.load(str(out / "directions" / "L0_sae_max_logreg.npz"))
        folds = resolve_folds(
            n=len(y),
            n_folds=None,
            seed=spec.seed,
            is_classification=True,
            stratify_y=y,
            train_split=spec.train_split,
        )
        _, train_idx, _ = folds[0]
        scaler = StandardScaler(with_mean=False).fit(X[train_idx])
        reference = LogisticRegression(
            C=spec.C,
            max_iter=spec.logreg_max_iter,
        ).fit(scaler.transform(X[train_idx]), y[train_idx])
        np.testing.assert_allclose(
            saved["coef"],
            reference.coef_,
            atol=1e-5,
        )
        np.testing.assert_allclose(saved["scaler_mean"], np.zeros(X.shape[1]))
        np.testing.assert_allclose(saved["scaler_scale"], scaler.scale_)

    def test_linear_svc_sparse_kfold_writes_importance(self, tmp_path):
        X, y = _classification_data()
        spec = SklearnProbeSpec(
            kind="linear_svc",
            class_weight="balanced",
            n_folds=2,
            save_directions=True,
        )
        out = train_sklearn_probe(
            _probe_dataset(sp.csr_matrix(X)),
            spec,
            y,
            tmp_path / "probe",
            feature_names=[f"L0_resid_post_f{i}" for i in range(X.shape[1])],
        )
        importance = pd.read_csv(out / "feature_importance.csv")
        top = importance.sort_values("abs_beta_mean", ascending=False).iloc[0]
        assert top["feature_name"] == "L0_resid_post_f0"

    def test_center_only_sparse_raises(self, tmp_path):
        X, y = _classification_data()
        spec = SklearnProbeSpec(kind="logreg", center_only=True)
        with pytest.raises(ValueError, match="center_only"):
            train_sklearn_probe(
                _probe_dataset(sp.csr_matrix(X)),
                spec,
                y,
                tmp_path / "probe",
            )

    def test_massmean_sparse_raises(self, tmp_path):
        X, y = _classification_data(classes=2)
        spec = SklearnProbeSpec(kind="massmean")
        with pytest.raises(ValueError, match="massmean"):
            train_sklearn_probe(
                _probe_dataset(sp.csr_matrix(X)),
                spec,
                y.astype(np.float64),
                tmp_path / "probe",
            )

    def test_mlp_sparse_raises(self, tmp_path):
        X, y = _classification_data()
        with pytest.raises(ValueError, match="sparse"):
            train_mlp_probes(
                _probe_dataset(sp.csr_matrix(X)),
                MLPProbeSpec(),
                torch.from_numpy(y),
                tmp_path / "probe",
                task_type=TaskType.CLASSIFICATION,
                num_classes=3,
            )


def _pooled_source(matrix, kept: list[int], site: str = "resid_post") -> ActivationDataset:
    return ActivationDataset(
        activations={(0, "sae_max"): matrix},
        targets=torch.zeros(matrix.shape[0]),
        sample_ids=[f"s{i}" for i in range(matrix.shape[0])],
        metadata={"sae_site": site, "kept_by_layer": {0: kept}},
    )


class TestSparseConcat:
    def test_sparse_sources_hstack(self):
        a = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        b = np.array([[3.0], [0.0]], dtype=np.float32)
        config = ConcatExtractionConfig(name="c", source_extractions=["a", "b"])
        ds = extract_concat_activations(
            [
                ("a", _pooled_source(sp.csr_matrix(a), kept=[5, 9])),
                ("b", _pooled_source(sp.csr_matrix(b), kept=[2], site="mlp_out")),
            ],
            config,
        )
        matrix = ds.activations[(0, "concat")]
        assert sp.issparse(matrix)
        np.testing.assert_allclose(
            np.asarray(matrix.todense()),
            np.hstack([a, b]),
        )
        assert ds.metadata["feature_names"] == [
            "L0_resid_post_f5",
            "L0_resid_post_f9",
            "L0_mlp_out_f2",
        ]
        assert ds.metadata["concat_spans"] == [
            ("a", 0, "resid_post", 0, 2),
            ("b", 0, "mlp_out", 2, 3),
        ]

    def test_mixed_dense_and_sparse_sources(self):
        a = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        b = np.array([[3.0], [4.0]], dtype=np.float32)
        config = ConcatExtractionConfig(name="c", source_extractions=["a", "b"])
        ds = extract_concat_activations(
            [
                ("a", _pooled_source(sp.csr_matrix(a), kept=[5, 9])),
                ("b", _pooled_source(torch.from_numpy(b), kept=[0])),
            ],
            config,
        )
        matrix = ds.activations[(0, "concat")]
        assert sp.issparse(matrix)
        np.testing.assert_allclose(
            np.asarray(matrix.todense()),
            np.hstack([a, b]),
        )


class TestSparseCorrelation:
    def test_correlate_features_matches_dense(self):
        X, y = _classification_data(n=60, d=5)
        targets = {"coarse_label": y.astype(np.float64)}
        kept = np.arange(X.shape[1], dtype=np.int64)
        dense_df = _correlate_features(X, targets, max_density=None, kept=kept)
        sparse_df = _correlate_features(
            sp.csr_matrix(X),
            targets,
            max_density=None,
            kept=kept,
        )
        pd.testing.assert_frame_equal(sparse_df, dense_df)

    def test_density_filter_on_sparse(self):
        X, y = _classification_data(n=60, d=5)
        X[:, 1] = 1.0  # fully dense column -> filtered out
        targets = {"coarse_label": y.astype(np.float64)}
        kept = np.arange(X.shape[1], dtype=np.int64)
        df = _correlate_features(
            sp.csr_matrix(X),
            targets,
            max_density=0.99,
            kept=kept,
        )
        assert 1 not in set(df["feature_idx"])
