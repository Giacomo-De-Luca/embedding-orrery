"""Distance metrics for interpretability experiments.

Two layers, both self-contained (numpy + skimage only — no parent-repo deps):

1. Perceptual LAB distance functions (CIEDE2000 + Euclidean in CIELAB). These are
   the LAB primitives historically housed in ``scripts/utils/colour_distances.py``;
   they live here so the toolkit (Mantel test, probe metrics) does not depend on the
   parent repo. ``scripts.utils.colour_distances.ColourDistance`` now delegates its
   LAB methods here to keep a single source of truth.

2. A generic ``ExperimentalDistance`` plug-in: a probe-evaluation metric that maps a
   ``(pred, target)`` batch to per-sample distances. Resolved from a dotted-path
   string (``"module:Class"``) the same way manifest builders are, so experiment
   configs can name a distance without the engine importing it. ``LabCiede2000Distance``
   is the built-in colour implementation.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

import numpy as np
from skimage.color import deltaE_ciede2000

# ── Perceptual LAB distances ──────────────────────────────────────────────────


def lab_ciede2000(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
    kL: float = 1,
    kC: float = 1,
    kH: float = 1,
) -> float:
    """Perceptual colour difference between two CIELAB colours using CIEDE2000.

    Parameters
    ----------
    lab1, lab2 : (L, a, b) tuples in CIELAB space.
    kL, kC, kH : scale factors (see ``skimage.color.deltaE_ciede2000``).
    """
    a1 = np.array([[lab1]])
    a2 = np.array([[lab2]])
    return float(deltaE_ciede2000(a1, a2, kL=kL, kC=kC, kH=kH)[0, 0])


def lab_euclidean(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
) -> float:
    """Euclidean distance in CIELAB space (Delta E*ab)."""
    return float(np.linalg.norm(np.array(lab1) - np.array(lab2)))


def batch_lab_ciede2000(
    lab1: np.ndarray,
    lab2: np.ndarray,
    kL: float = 1,
    kC: float = 1,
    kH: float = 1,
) -> np.ndarray:
    """Element-wise CIEDE2000 distances between two arrays of LAB colours.

    Parameters
    ----------
    lab1, lab2 : (N, 3) arrays of CIELAB [L, a, b] values.

    Returns
    -------
    (N,) array of CIEDE2000 distances.
    """
    a1 = np.asarray(lab1).reshape(-1, 1, 3)
    a2 = np.asarray(lab2).reshape(-1, 1, 3)
    return deltaE_ciede2000(a1, a2, kL=kL, kC=kC, kH=kH).ravel()


def pairwise_lab_ciede2000(lab: np.ndarray) -> np.ndarray:
    """Pairwise CIEDE2000 distances for an array of LAB colours.

    Parameters
    ----------
    lab : (N, 3) array of CIELAB [L, a, b] values.

    Returns
    -------
    Condensed distance vector of length N*(N-1)/2, compatible with
    ``scipy.spatial.distance.squareform``.
    """
    from scipy.spatial.distance import squareform

    lab_row = lab[:, np.newaxis, :]  # (N, 1, 3)
    lab_col = lab[np.newaxis, :, :]  # (1, N, 3)
    dist_matrix = deltaE_ciede2000(lab_row, lab_col)
    return squareform(dist_matrix, checks=False)


# ── Generic experimental-distance plug-in ─────────────────────────────────────


class ExperimentalDistance(ABC):
    """A probe-evaluation distance metric over (prediction, target) batches.

    Concrete subclasses declare:

    * ``metric_key``   — the column name the probe writes the mean distance under
      (e.g. ``"val_lab_distance"``).
    * ``target_dim``   — required width of the target vector for this metric to
      apply (e.g. ``3`` for LAB); ``None`` means any width.
    * ``higher_is_better`` — whether larger is better (distances: ``False``).

    and implement :meth:`batch`. Experiment configs reference a subclass by
    dotted path (``"module:Class"``); :func:`resolve_distance` instantiates it.
    """

    metric_key: str
    target_dim: int | None = None
    higher_is_better: bool = False

    @abstractmethod
    def batch(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Per-sample distances between predicted and target rows.

        Parameters
        ----------
        pred, target : (N, D) arrays.

        Returns
        -------
        (N,) array of per-sample distances; the probe takes the mean.
        """

    def applies_to(self, target_dim: int) -> bool:
        """Whether this metric should be computed for targets of width ``target_dim``."""
        return self.target_dim is None or target_dim == self.target_dim


def resolve_distance(path: str) -> ExperimentalDistance:
    """Import + instantiate an ``ExperimentalDistance`` from a ``"module:Class"`` path.

    Mirrors ``interpret.probing.configs.experiment.ManifestSpec.resolve``
    so probe configs can name a distance without the engine importing it eagerly.
    """
    if ":" not in path:
        raise ValueError(
            f"distance path must be 'module.path:ClassName', got {path!r}"
        )
    module_path, class_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise ImportError(
            f"Module {module_path!r} has no attribute {class_name!r}"
        ) from e
    return cls()


class LabCiede2000Distance(ExperimentalDistance):
    """Mean CIEDE2000 over 3-D LAB regression targets (the colour default)."""

    metric_key = "val_lab_distance"
    target_dim = 3
    higher_is_better = False

    def batch(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        return batch_lab_ciede2000(pred, target)
