"""Mantel test: rank correlation between two distance structures.

A Mantel test measures how well one distance matrix preserves the structure of
another by Spearman-correlating their (condensed) pairwise distances. This module
is domain-agnostic — it operates purely on distance vectors/matrices, so it can
compare colour-space vs embedding-space distances, activation vs behaviour
distances, etc. The caller supplies the distances (see
``interpret.utils.distances`` for perceptual-distance helpers).

Conventions follow ``scipy.spatial.distance``: a *condensed* distance vector has
length ``N*(N-1)/2`` and a *square* matrix is ``(N, N)`` with a zero diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


@dataclass
class PermutationResult:
    """Outcome of a permutation-based significance test for a Mantel correlation."""

    observed_rho: float
    null_mean: float
    null_std: float
    z_score: float
    empirical_p: float  # fraction of null rhos >= observed


class MantelTest:
    """Spearman rank correlation between two distance structures.

    All methods are static — ``MantelTest`` is a namespace, not state.
    """

    @staticmethod
    def global_spearman(
        d1_condensed: np.ndarray, d2_condensed: np.ndarray
    ) -> tuple[float, float]:
        """Spearman rho + p-value over all pairwise distances.

        Parameters
        ----------
        d1_condensed, d2_condensed : condensed distance vectors of equal length.

        Returns
        -------
        ``(rho, p_value)``.
        """
        result = spearmanr(d1_condensed, d2_condensed)
        return float(result.statistic), float(result.pvalue)

    @staticmethod
    def knn_spearman(
        d1_square: np.ndarray, d2_square: np.ndarray, k: int
    ) -> tuple[float, float]:
        """Spearman rho restricted to each item's ``k`` nearest neighbours in ``d1``.

        For each item ``i`` take its ``k`` closest neighbours by ``d1`` (excluding
        self), collect the ``(d1, d2)`` distance pairs, and compute a single Spearman
        correlation over all such pairs (pooled across items). This probes whether
        *local* structure is preserved, even when global correlation is weak.

        Parameters
        ----------
        d1_square, d2_square : ``(N, N)`` distance matrices (zero diagonal).
        k : neighbourhood size.
        """
        n = d1_square.shape[0]
        d1_vals: list[float] = []
        d2_vals: list[float] = []
        for i in range(n):
            # k nearest neighbours by d1; index 0 is self (distance 0), so skip it.
            neighbours = np.argsort(d1_square[i])[1 : k + 1]
            d1_vals.extend(d1_square[i, neighbours])
            d2_vals.extend(d2_square[i, neighbours])
        result = spearmanr(d1_vals, d2_vals)
        return float(result.statistic), float(result.pvalue)

    @staticmethod
    def permutation_test(
        d1_condensed: np.ndarray,
        d2_condensed: np.ndarray,
        n_perms: int,
        seed: int = 42,
    ) -> PermutationResult:
        """Estimate the null distribution of the global rho by permuting ``d2``.

        Jointly permutes the rows and columns of ``d2`` (the standard Mantel
        permutation — equivalent to relabelling the items) ``n_perms`` times and
        recomputes the correlation each time, giving a null against which the
        observed rho is scored.

        Parameters
        ----------
        d1_condensed, d2_condensed : condensed distance vectors of equal length.
        n_perms : number of permutations.
        seed : RNG seed for reproducibility.
        """
        observed_rho, _ = MantelTest.global_spearman(d1_condensed, d2_condensed)
        d2_square = squareform(d2_condensed)
        n = d2_square.shape[0]
        rng = np.random.default_rng(seed)

        null_rhos = np.empty(n_perms, dtype=float)
        for i in range(n_perms):
            perm = rng.permutation(n)
            permuted = d2_square[np.ix_(perm, perm)]
            null_rhos[i] = spearmanr(
                d1_condensed, squareform(permuted, checks=False)
            ).statistic

        null_mean = float(null_rhos.mean())
        null_std = float(null_rhos.std())
        z_score = (observed_rho - null_mean) / null_std if null_std > 0 else float("inf")
        empirical_p = float(np.mean(null_rhos >= observed_rho))
        return PermutationResult(
            observed_rho=observed_rho,
            null_mean=null_mean,
            null_std=null_std,
            z_score=z_score,
            empirical_p=empirical_p,
        )
