"""Classifier probe metrics (backend/utils/classification_metrics.py).

Computed from the same class probabilities the platform persists, so the
reported metrics always agree with the predicted-class field the UI shows
(SVC's own `predict` uses one-vs-one voting and can disagree with argmax).
"""

import json

import numpy as np

from backend.utils.classification_metrics import ClassificationMetrics


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    probs = np.full((len(labels), n_classes), 0.0)
    probs[np.arange(len(labels)), labels] = 1.0
    return probs


LABELS = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1])
TRAIN = np.arange(0, 6)
VAL = np.arange(6, 12)


class TestMulticlass:
    def test_perfect_predictions(self):
        metrics = ClassificationMetrics(_one_hot(LABELS, 3), LABELS, TRAIN, VAL).compute()
        for key in (
            "val_accuracy",
            "val_balanced_accuracy",
            "val_f1_macro",
            "val_f1_weighted",
            "val_auc_ovr",
            "train_accuracy",
        ):
            assert metrics[key] == 1.0, key
        assert "val_auc" not in metrics

    def test_majority_baseline_uses_train_majority_on_val(self):
        # Train majority class is 0 (2 of 6 ties broken by lowest index);
        # val has three 0s out of six.
        metrics = ClassificationMetrics(_one_hot(LABELS, 3), LABELS, TRAIN, VAL).compute()
        assert metrics["val_majority_baseline"] == 0.5

    def test_wrong_predictions_lower_accuracy(self):
        probs = _one_hot((LABELS + 1) % 3, 3)
        metrics = ClassificationMetrics(probs, LABELS, TRAIN, VAL).compute()
        assert metrics["val_accuracy"] == 0.0

    def test_missing_val_class_gives_no_auc(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 1])
        metrics = ClassificationMetrics(
            _one_hot(labels, 3), labels, np.arange(6), np.arange(6, 10)
        ).compute()
        assert metrics["val_auc_ovr"] is None
        assert metrics["val_accuracy"] == 1.0

    def test_metrics_are_json_safe(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(3), size=len(LABELS))
        metrics = ClassificationMetrics(probs, LABELS, TRAIN, VAL).compute()
        dumped = json.dumps(metrics)
        assert "NaN" not in dumped and "Infinity" not in dumped
        assert all(v is None or isinstance(v, float) for v in metrics.values())


class TestBinary:
    def test_binary_reports_auc(self):
        labels = np.array([0, 1] * 10)
        probs = np.column_stack([1 - labels * 0.8, labels * 0.8]).astype(float)
        probs = probs / probs.sum(axis=1, keepdims=True)
        metrics = ClassificationMetrics(probs, labels, np.arange(10), np.arange(10, 20)).compute()
        assert metrics["val_auc"] == 1.0
        assert "val_auc_ovr" not in metrics

    def test_single_class_val_gives_no_auc(self):
        labels = np.array([0, 1, 0, 1, 1, 1])
        metrics = ClassificationMetrics(
            _one_hot(labels, 2), labels, np.arange(4), np.arange(4, 6)
        ).compute()
        assert metrics["val_auc"] is None
