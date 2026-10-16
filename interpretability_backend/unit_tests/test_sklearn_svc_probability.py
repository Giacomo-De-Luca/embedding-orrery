"""SklearnProbeSpec.probability: RBF SVC with Platt-scaled class probabilities.

The platform's SVC probe scores every item with ``predict_proba``, so the
saved estimator must be fitted with ``probability=True`` — and seeded, since
Platt scaling runs an internal cross-validation that is otherwise random.
"""

import joblib
import numpy as np
import torch

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.probe import SklearnProbeSpec
from interpret.probing.probes.sklearn_probes import train_sklearn_probe


def _three_class_data(n=150, d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n // 3).astype(np.int64)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, 0] += y * 3.0
    return X, y


def _fit_svc(tmp_path, *, probability: bool, seed: int = 7):
    X, y = _three_class_data()
    dataset = ActivationDataset(
        activations={(0, "embedding"): torch.from_numpy(X)},
        sample_ids=[f"s{i}" for i in range(len(y))],
    )
    spec = SklearnProbeSpec(kind="svc", save_models=True, probability=probability, seed=seed)
    perm = np.random.default_rng(0).permutation(len(y))
    train_sklearn_probe(dataset, spec, y, tmp_path, indices_override=(perm[:120], perm[120:]))
    bundle = joblib.load(tmp_path / "models" / "L0_embedding_svc.joblib")
    return bundle["estimator"], X


def test_default_spec_has_no_probabilities(tmp_path):
    estimator, _ = _fit_svc(tmp_path, probability=False)
    assert not hasattr(estimator, "predict_proba")


def test_probability_flag_yields_normalised_class_probabilities(tmp_path):
    estimator, X = _fit_svc(tmp_path, probability=True)
    probs = estimator.predict_proba(X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_probabilities_are_seeded(tmp_path):
    first, X = _fit_svc(tmp_path / "a", probability=True, seed=3)
    second, _ = _fit_svc(tmp_path / "b", probability=True, seed=3)
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))
