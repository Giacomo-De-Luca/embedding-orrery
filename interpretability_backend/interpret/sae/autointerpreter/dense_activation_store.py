"""Dense ``.npy`` shard store for embedding-model dimensions.

Embedding vectors are fully dense and **signed** (no JumpReLU sparsity), so
CSR storage would just be a dense matrix in disguise. This backend stores each
flush as a fresh ``activations_batch_NNNNNN.npy`` shard and reuses the base
class's buffered append, append-only flush, and ``index.parquet`` machinery.

Sizing: ~200k WordNet (word, synset) pairs × 768 dims × float16 ≈ 300 MB —
comfortably loadable in memory by ``load_matrix``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from interpret.sae.autointerpreter.activation_store import ActivationStore


class DenseActivationStore(ActivationStore):
    """Buffered store backed by dense ``.npy`` shards.

    Usage mirrors :class:`SparseActivationStore`::

        store = DenseActivationStore(run_dir, n_features=384, dtype=np.float32)
        for vec, meta in ...:
            store.append(vec, meta)
        store.flush()
    """

    ACTIVATIONS_FILE = "activations.npy"
    SHARD_EXT = "npy"

    # ── Buffer hooks ─────────────────────────────────────────────────────

    def _init_buffers(self) -> None:
        self._rows: list[np.ndarray] = []

    def _reset_buffers(self) -> None:
        self._rows = []

    def _buffer_row(self, vector: np.ndarray) -> None:
        # Keep the full (signed) vector — no thresholding, no nnz packing.
        self._rows.append(np.asarray(vector, dtype=self.dtype))

    def _build_pending(self) -> np.ndarray:
        return np.stack(self._rows).astype(self.dtype, copy=False)

    # ── Shard serialization ──────────────────────────────────────────────

    @classmethod
    def _save_shard(cls, payload: np.ndarray, path: Path) -> None:
        # Atomic: save to a sibling .tmp.npy then rename. The temp must already
        # end in .npy so np.save doesn't append a second extension.
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, payload)
        os.replace(tmp, path)

    @classmethod
    def _load_shard(cls, path: Path) -> np.ndarray:
        return np.load(path)

    @classmethod
    def _concat(cls, payloads: list[np.ndarray]) -> np.ndarray:
        return np.vstack(payloads)

    @classmethod
    def _shard_n_rows(cls, path: Path) -> int:
        # mmap so we read only the header for the row count.
        return int(np.load(path, mmap_mode="r").shape[0])
