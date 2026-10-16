"""Categorical probe targets: label canonicalisation, class selection, encoding.

Classifier probes (``CLASSIFICATION_KINDS``) read their target field as
labels. Labels must match how the browser renders the same metadata value
(``String(JSON.parse(..))``) so the predicted-class field lines up with the
target field's categories and colours:

- integral JSON numbers become ``"3"`` (DuckDB returns ``3.0`` as "3.0");
- JSON strings stay verbatim (``"03"``, ``"3.0"``);
- ``""``, null, arrays and objects are unlabelled (the browser shows "" as
  Missing, and stringifies arrays differently from DuckDB).

Mirrored client-side by ``embedding_visualization/lib/utils/probeTargets.ts``
(keep the threshold and numeric-parse rules in sync).

Numpy-only on purpose: no torch/sklearn, so it stays cheap to import.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .probing_types import MAX_PROBE_CLASSES, MIN_CLASS_COUNT_FLOOR, ProbeTask

# Strict decimal-number syntax — rejects what float()/Number() leniently
# accept ("1_000", "nan", "inf", "0x1A", " ").
_NUMERIC_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

# Tolerance so 0.01 * 1400 (== 14.000000000000002) does not round up to 15.
_THRESHOLD_EPSILON = 1e-9


class ProbeTargetError(ValueError):
    """The field cannot serve as a categorical probe target."""


def class_count_threshold(n_labelled: int, min_class_fraction: float) -> int:
    """Minimum items a class needs to be kept for training."""
    scaled = math.ceil(min_class_fraction * n_labelled - _THRESHOLD_EPSILON)
    return max(MIN_CLASS_COUNT_FLOOR, scaled)


class CategoricalLabel:
    """Canonical class labels from DuckDB JSON values (text + ``json_type``)."""

    _UNLABELLED_TYPES = (None, "NULL", "ARRAY", "OBJECT")
    _FLOAT_TYPES = ("DOUBLE",)

    @classmethod
    def canonical(cls, text: str | None, json_type: str | None) -> str | None:
        """Canonical label for one value, or None when it is unlabelled."""
        if text is None or json_type in cls._UNLABELLED_TYPES or text == "":
            return None
        if json_type in cls._FLOAT_TYPES:
            value = float(text)
            if value.is_integer():
                return str(int(value))  # also maps -0.0 -> "0"
        return text

    @staticmethod
    def from_number(value: float) -> str:
        """Canonical label for a numeric target value (integral -> "3")."""
        return str(int(value)) if float(value).is_integer() else repr(float(value))

    @staticmethod
    def is_numeric(label: str) -> bool:
        return bool(_NUMERIC_RE.match(label))

    @classmethod
    def sort(cls, labels) -> list[str]:
        """Numeric ascending when every label is a number, else code-point order."""
        labels = list(labels)
        if labels and all(cls.is_numeric(label) for label in labels):
            return sorted(labels, key=float)
        return sorted(labels)


@dataclass
class CategoricalTarget:
    """A categorical target encoded for training.

    ``codes`` is aligned to the input labels: the kept class index as a
    float, NaN for unlabelled items and items of excluded classes (they are
    not trained on, but still scored).
    """

    task: str
    classes: list[str]
    codes: np.ndarray
    class_counts: dict[str, int]
    excluded: dict[str, int]
    threshold: int
    numeric: bool
    n_labelled: int

    @classmethod
    def from_labels(
        cls,
        labels: list[str | None],
        min_class_fraction: float,
        *,
        max_classes: int = MAX_PROBE_CLASSES,
    ) -> "CategoricalTarget":
        """Build from canonical labels (None = unlabelled; "" is treated as None)."""
        labels = [None if label == "" else label for label in labels]
        counts = Counter(label for label in labels if label is not None)
        n_labelled = sum(counts.values())
        if len(counts) < 2:
            raise ProbeTargetError(
                f"A categorical target needs at least two distinct values; found {len(counts)}."
            )
        numeric = all(CategoricalLabel.is_numeric(label) for label in counts)
        if numeric and len(counts) > 2 and any(not float(label).is_integer() for label in counts):
            raise ProbeTargetError(
                "The field is continuous (non-integer numbers with more than two "
                "values); use a regression probe (ridge, lasso, SVR, mass-mean, MLP)."
            )

        threshold = class_count_threshold(n_labelled, min_class_fraction)
        if len(counts) == 2:
            kept = list(counts)  # binary targets are never thresholded
        else:
            kept = [label for label, count in counts.items() if count >= threshold]
        if len(kept) < 2:
            raise ProbeTargetError(
                f"Only {len(kept)} class(es) have at least {threshold} labelled items; "
                "at least two are required. Lower the Min class % setting."
            )
        if len(kept) > max_classes:
            raise ProbeTargetError(
                f"{len(kept)} classes have at least {threshold} labelled items; at most "
                f"{max_classes} are supported. Raise the Min class % setting."
            )

        classes = CategoricalLabel.sort(kept)
        index = {label: float(i) for i, label in enumerate(classes)}
        codes = np.array(
            [index.get(label, np.nan) if label is not None else np.nan for label in labels],
            dtype=np.float64,
        )
        return cls(
            task=ProbeTask.BINARY if len(classes) == 2 else ProbeTask.MULTICLASS,
            classes=classes,
            codes=codes,
            class_counts={label: counts[label] for label in classes},
            excluded={label: count for label, count in counts.items() if label not in index},
            threshold=threshold,
            numeric=numeric,
            n_labelled=n_labelled,
        )

    @classmethod
    def from_numeric(cls, y: np.ndarray, min_class_fraction: float) -> "CategoricalTarget":
        """Build from numeric targets (NaN = unlabelled)."""
        labels = [None if not np.isfinite(v) else CategoricalLabel.from_number(v) for v in y]
        return cls.from_labels(labels, min_class_fraction)

    @property
    def target_mapping(self) -> dict[str, float] | None:
        """Legacy value->0/1 mapping, kept for binary non-numeric targets
        (surfaced as ``ProbeInfo.targetMapping`` in the UI tooltip)."""
        if self.task != ProbeTask.BINARY or self.numeric:
            return None
        return {label: float(i) for i, label in enumerate(self.classes)}
