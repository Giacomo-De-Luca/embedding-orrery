"""Metrics for classifier probes, computed from per-item class probabilities.

Every classifier kind (logreg, svc, mlp_clf) reports the same metric set,
derived from the probabilities the platform persists and the argmax it shows
as the predicted class — so reported accuracy always matches the UI (SVC's
own ``predict`` uses one-vs-one voting and can disagree with the argmax).

Uses sklearn: import only from the probing service, never from the GraphQL
schema's import graph (torch/sklearn-free boundary).
"""

import math

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


class ClassificationMetrics:
    """Flat, JSON-safe metric dict for one trained classifier probe.

    Args:
        probs: [N, C] class probabilities (rows sum to 1).
        labels: [N] class indices; only ``train_rows``/``val_rows`` are read.
        train_rows / val_rows: row indices of the training and validation sets.
    """

    def __init__(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        train_rows: np.ndarray,
        val_rows: np.ndarray,
    ):
        self.probs = np.asarray(probs, dtype=np.float64)
        self.labels = np.asarray(labels).astype(np.int64, copy=False)
        self.train_rows = np.asarray(train_rows)
        self.val_rows = np.asarray(val_rows)
        self.n_classes = self.probs.shape[1]

    def compute(self) -> dict[str, float | None]:
        y_val = self.labels[self.val_rows]
        pred_val = self.probs[self.val_rows].argmax(axis=1)
        y_train = self.labels[self.train_rows]
        pred_train = self.probs[self.train_rows].argmax(axis=1)
        class_ids = list(range(self.n_classes))

        metrics = {
            "val_accuracy": accuracy_score(y_val, pred_val),
            "val_balanced_accuracy": balanced_accuracy_score(y_val, pred_val),
            "val_f1_macro": f1_score(
                y_val, pred_val, labels=class_ids, average="macro", zero_division=0
            ),
            "val_f1_weighted": f1_score(
                y_val, pred_val, labels=class_ids, average="weighted", zero_division=0
            ),
            "train_accuracy": accuracy_score(y_train, pred_train),
            "val_majority_baseline": self._majority_baseline(y_train, y_val),
        }
        auc_key = "val_auc" if self.n_classes == 2 else "val_auc_ovr"
        metrics[auc_key] = self._auc(y_val)
        return {key: self._clean(value) for key, value in metrics.items()}

    def _majority_baseline(self, y_train: np.ndarray, y_val: np.ndarray) -> float:
        """Validation accuracy of always predicting the training majority class."""
        majority = np.bincount(y_train, minlength=self.n_classes).argmax()
        return float(np.mean(y_val == majority))

    def _auc(self, y_val: np.ndarray) -> float | None:
        """Binary ROC-AUC, or macro one-vs-rest AUC; None if a class is absent."""
        if len(np.unique(y_val)) < self.n_classes:
            return None
        probs_val = self.probs[self.val_rows]
        try:
            if self.n_classes == 2:
                return roc_auc_score(y_val, probs_val[:, 1])
            return roc_auc_score(
                y_val,
                probs_val,
                multi_class="ovr",
                average="macro",
                labels=list(range(self.n_classes)),
            )
        except ValueError:
            return None

    @staticmethod
    def _clean(value) -> float | None:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
