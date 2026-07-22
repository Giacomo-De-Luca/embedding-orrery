"""Dispatch helpers for activation matrices: dense torch or scipy CSR.

Pooled SAE extractions may store scipy CSR matrices (``sparse: true`` on
`SAEPooledExtractionConfig`) instead of dense torch tensors. These helpers
let downstream consumers accept both without scattering isinstance checks.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

FeatureMatrix = torch.Tensor | sp.spmatrix


def as_feature_matrix(mat: FeatureMatrix) -> np.ndarray | sp.csr_matrix:
    """Tensor -> ndarray; sparse -> CSR passthrough (sklearn-ready)."""
    if sp.issparse(mat):
        return mat.tocsr()
    if isinstance(mat, torch.Tensor):
        return mat.numpy()
    return np.asarray(mat)


def select_rows(mat: FeatureMatrix, indices: list[int]) -> FeatureMatrix:
    """Row-subset (with reordering) preserving the matrix's storage type."""
    if sp.issparse(mat):
        return mat.tocsr()[np.asarray(indices, dtype=np.int64)]
    if isinstance(mat, torch.Tensor):
        return mat[torch.as_tensor(indices, dtype=torch.long)]
    return np.asarray(mat)[np.asarray(indices, dtype=np.int64)]


def to_dense_numpy(mat: FeatureMatrix) -> np.ndarray:
    """Materialise as a dense ndarray — small-data consumers only."""
    if sp.issparse(mat):
        return np.asarray(mat.todense())
    if isinstance(mat, torch.Tensor):
        return mat.numpy()
    return np.asarray(mat)
