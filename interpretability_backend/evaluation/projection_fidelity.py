"""Projection-fidelity metrics via the Mantel test.

Scores how faithfully a low-dimensional projection (e.g. UMAP-3D, PCA-3D)
preserves a *reference* distance structure — the original embedding geometry,
or, for colour datasets, perceptual colour distance. A Mantel test is a rank
(Spearman) correlation between two pairwise-distance structures, with a
permutation-based significance test.

This module is pure: it takes condensed distance vectors (or coordinate arrays
via the static builders) and returns a plain results dict. It never touches
DuckDB/ChromaDB — the config-driven runner (``run_projection_fidelity.py``) does
the loading. Like :class:`TopicQualityEvaluator`, every metric degrades to
``None`` on degenerate input and :meth:`evaluate` never raises.

The colour builder (:meth:`colour_distances`) lazily imports the toolkit's
CIEDE2000 helper (which needs scikit-image) *inside the method*, so importing
this module — and running embedding-space fidelity — has no scikit-image
dependency; only colour analysis does.
"""

from __future__ import annotations

import logging
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

# The `interpret` toolkit lives one level up (interpretability_backend/interpret);
# put its parent on sys.path so `import interpret...` resolves, mirroring
# backend/services/interpret_service.py.
_INTERPRET_PARENT = str(Path(__file__).resolve().parents[1])
if _INTERPRET_PARENT not in sys.path:
    sys.path.insert(0, _INTERPRET_PARENT)

from interpret.utils.mantel import MantelTest  # noqa: E402

logger = logging.getLogger("orrery." + __name__)


def _clean(value: float | None) -> float | None:
    """Map NaN/inf (degenerate correlations) to ``None``; pass finite floats."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _n_items_from_pairs(n_pairs: int) -> int:
    """Invert ``N*(N-1)/2`` to recover the item count from a condensed length."""
    return int((1 + math.isqrt(1 + 8 * n_pairs)) // 2)


def _condensed_length(d) -> int:
    """Length of ``d`` if it is a valid 1-D condensed distance vector, else ``-1``.

    A condensed vector must be 1-D with a triangular-number length ``N*(N-1)/2``;
    anything else (2-D matrices, scalars, non-triangular lengths) returns ``-1`` so
    the caller can skip it without ``squareform`` ever raising.
    """
    arr = np.asarray(d)
    if arr.ndim != 1 or arr.shape[0] <= 0:
        return -1
    length = int(arr.shape[0])
    n = _n_items_from_pairs(length)
    return length if n * (n - 1) // 2 == length else -1


class ProjectionFidelityEvaluator:
    """Mantel-based fidelity of projections against reference distance structures.

    Parameters
    ----------
    k : neighbourhood size for the local (kNN) Spearman probe.
    n_perms : permutations for the significance test (0 disables it).
    seed : RNG seed for the permutation test.
    """

    def __init__(self, k: int = 10, n_perms: int = 1000, seed: int = 42) -> None:
        self.k = k
        self.n_perms = n_perms
        self.seed = seed

    # ── distance builders (condensed vectors, scipy convention) ────────────────

    @staticmethod
    def embedding_distances(embeddings: np.ndarray) -> np.ndarray:
        """Condensed cosine distances for embedding vectors ``(N, D)``."""
        return pdist(np.asarray(embeddings, dtype=np.float64), metric="cosine")

    @staticmethod
    def projection_distances(coords: np.ndarray) -> np.ndarray:
        """Condensed Euclidean distances for projection coordinates ``(N, d)``."""
        return pdist(np.asarray(coords, dtype=np.float64), metric="euclidean")

    @staticmethod
    def colour_distances(hex_codes: list[str]) -> np.ndarray:
        """Condensed perceptual (CIEDE2000) distances for hex colour strings.

        Lazily imports scikit-image (for sRGB→LAB) and the toolkit's CIEDE2000
        helper, so the rest of the module has no scikit-image dependency.

        Raises ``ValueError`` on a malformed hex code — the runner catches this
        and skips the colour reference.
        """
        from skimage.color import rgb2lab

        from interpret.utils.distances import pairwise_lab_ciede2000

        rgb = np.empty((len(hex_codes), 3), dtype=np.float64)
        for i, code in enumerate(hex_codes):
            if not isinstance(code, str) or len(code.lstrip("#")) != 6:
                raise ValueError(f"malformed hex colour at index {i}: {code!r}")
            h = code.lstrip("#")
            rgb[i] = [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]
        lab = rgb2lab((rgb / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
        return pairwise_lab_ciede2000(lab)

    # ── evaluation ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        references: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        cross_reference: bool = True,
    ) -> dict:
        """Correlate each target projection against each reference structure.

        Parameters
        ----------
        references : name → condensed distance vector for reference structures
            (e.g. ``{"colour": ..., "embedding": ...}``).
        targets : name → condensed distance vector for projections
            (e.g. ``{"umap_3d": ..., "pca_3d": ...}``).
        cross_reference : also correlate reference-vs-reference pairs (baselines,
            e.g. colour ↔ embedding).

        Returns
        -------
        ``{"n_items", "n_pairs", "k", "n_perms", "comparisons": [...]}`` where each
        comparison is ``{"reference", "target", "kind", "global_rho", "global_p",
        "knn_rho", "knn_k", "perm_z", "perm_empirical_p", "perm_n"}``. Never raises;
        degenerate comparisons carry ``None`` metrics.
        """
        all_dists = {**references, **targets}
        # Validate each vector is a proper condensed distance (1-D, triangular
        # length); invalid ones get length -1 and are dropped, so squareform below
        # can never raise. Then keep only those sharing the common (max) length.
        lengths = {name: _condensed_length(d) for name, d in all_dists.items()}
        valid_lengths = [length for length in lengths.values() if length > 0]
        n_pairs = max(valid_lengths) if valid_lengths else 0
        usable = {
            name: np.asarray(d, dtype=np.float64)
            for name, d in all_dists.items()
            if lengths[name] == n_pairs and n_pairs > 0
        }
        for name in lengths:
            if name not in usable:
                logger.warning(
                    "Distance %r is not a usable condensed vector of length %d "
                    "(got %d); skipping",
                    name,
                    n_pairs,
                    lengths[name],
                )

        # Square matrices for the kNN probe (built once per distance vector).
        squares = {name: squareform(d, checks=False) for name, d in usable.items()}

        pairs: list[tuple[str, str, str]] = []
        for ref in references:
            for tgt in targets:
                if ref in usable and tgt in usable:
                    pairs.append((ref, tgt, "fidelity"))
        if cross_reference:
            for a, b in combinations([r for r in references if r in usable], 2):
                pairs.append((a, b, "baseline"))

        comparisons = [
            {
                "reference": ref,
                "target": tgt,
                "kind": kind,
                **self._compare(usable[ref], usable[tgt], squares[ref], squares[tgt]),
            }
            for ref, tgt, kind in pairs
        ]

        return {
            "n_items": _n_items_from_pairs(n_pairs) if n_pairs else 0,
            "n_pairs": int(n_pairs),
            "k": self.k,
            "n_perms": self.n_perms,
            "comparisons": comparisons,
        }

    def _compare(
        self, d_ref: np.ndarray, d_tgt: np.ndarray, sq_ref: np.ndarray, sq_tgt: np.ndarray
    ) -> dict:
        """Global + kNN-local Spearman and a permutation test for one pair."""
        result = {
            "global_rho": None,
            "global_p": None,
            "knn_rho": None,
            "knn_k": self.k,
            "perm_z": None,
            "perm_empirical_p": None,
            "perm_n": self.n_perms,
        }
        try:
            g_rho, g_p = MantelTest.global_spearman(d_ref, d_tgt)
            result["global_rho"] = _clean(g_rho)
            result["global_p"] = _clean(g_p)

            k = min(self.k, sq_ref.shape[0] - 1)
            if k >= 1:
                k_rho, _ = MantelTest.knn_spearman(sq_ref, sq_tgt, k)
                result["knn_rho"] = _clean(k_rho)
                result["knn_k"] = k

            # Only run the null test when the observed correlation is defined:
            # on a constant (degenerate) reference the observed rho is NaN, and
            # `NaN >= NaN` is False, which would report a misleading empirical
            # p = 0.0 ("maximally significant"). Keep the whole result None instead.
            if self.n_perms > 0 and result["global_rho"] is not None:
                perm = MantelTest.permutation_test(d_ref, d_tgt, self.n_perms, seed=self.seed)
                result["perm_z"] = _clean(perm.z_score)
                result["perm_empirical_p"] = _clean(perm.empirical_p)
        except Exception as e:  # never raise from evaluate()
            logger.warning("Mantel comparison failed: %s", e)
        return result
