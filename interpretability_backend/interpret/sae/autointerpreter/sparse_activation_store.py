"""Sparse CSR shard storage for per-sample SAE feature activations.

Each WordNet sample produces one dense activation vector ``(d_sae,)`` of which
only ~100 entries are nonzero (JumpReLU SAEs have L0 ≈ 50-150). Storing all
samples densely would cost tens of GB; the CSR format keeps it at a few
hundred MB.

The buffered-append / append-only-shard flush / shard discovery /
``index.parquet`` machinery lives in the shared :class:`ActivationStore` base;
this class supplies only the CSR-specific buffering and serialization hooks.
A legacy single-file ``activations.npz`` (from older load-stack-save runs) is
recognised as an immutable base shard. See :class:`DenseActivationStore` for
the embedding-dimension counterpart.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy import sparse

from interpret.sae.autointerpreter.activation_store import ActivationStore


class SparseActivationStore(ActivationStore):
    """Buffered CSR builder with a parallel metadata index (parquet).

    Rows are samples, columns are SAE features. Each ``flush()`` writes one
    new ``activations_batch_NNNNNN.npz`` shard under ``run_dir`` and appends to
    ``index.parquet``.

    Usage::

        store = SparseActivationStore(run_dir, n_features=16384)
        for vec, meta in ...:
            store.append(vec, meta)
        store.flush()
    """

    ACTIVATIONS_FILE = "activations.npz"
    SHARD_EXT = "npz"

    # ── Buffer hooks ─────────────────────────────────────────────────────

    def _init_buffers(self) -> None:
        self._data: list[np.ndarray] = []
        self._indices: list[np.ndarray] = []
        self._indptr: list[int] = [0]

    def _reset_buffers(self) -> None:
        self._data = []
        self._indices = []
        self._indptr = [0]

    def _buffer_row(self, vector: np.ndarray) -> None:
        nz = np.flatnonzero(vector)
        self._data.append(vector[nz].astype(self.dtype, copy=False))
        self._indices.append(nz.astype(np.int32, copy=False))
        self._indptr.append(self._indptr[-1] + len(nz))

    def _build_pending(self) -> sparse.csr_matrix:
        data = (
            np.concatenate(self._data) if self._data else np.empty(0, self.dtype)
        )
        indices = (
            np.concatenate(self._indices)
            if self._indices
            else np.empty(0, np.int32)
        )
        indptr = np.asarray(self._indptr, dtype=np.int64)
        return sparse.csr_matrix(
            (data, indices, indptr),
            shape=(self._rows_pending, self.n_features),
        )

    # ── Shard serialization ──────────────────────────────────────────────

    @classmethod
    def _save_shard(cls, payload: sparse.csr_matrix, path: Path) -> None:
        # Atomic write: save to a sibling .tmp.npz then rename. A SIGKILL
        # mid-save leaves the shard absent rather than partially written.
        # NOTE: scipy.sparse.save_npz auto-appends ".npz" if the path doesn't
        # already end in it — so the temp must end in .npz too, otherwise we'd
        # write to "...tmp.npz" and then try to rename a non-existent "...tmp".
        tmp_npz = path.with_name(path.stem + ".tmp.npz")
        sparse.save_npz(tmp_npz, payload, compressed=True)
        os.replace(tmp_npz, path)

    @classmethod
    def _load_shard(cls, path: Path) -> sparse.csr_matrix:
        return sparse.load_npz(path)

    @classmethod
    def _concat(cls, payloads: list[sparse.csr_matrix]) -> sparse.csr_matrix:
        return sparse.vstack(payloads, format="csr")

    @classmethod
    def _shard_n_rows(cls, path: Path) -> int:
        # Peek the shape array from the npz without decoding data/indices.
        with np.load(path) as npz:
            return int(npz["shape"][0])

    # ── Streaming whole-matrix load ──────────────────────────────────────

    @classmethod
    def load_matrix(cls, run_dir: Path) -> sparse.csr_matrix:
        """Concatenate every shard into one CSR via a streaming two-pass load.

        Overrides the base class's list+``vstack`` path, which holds every
        shard in memory simultaneously and then doubles peak usage by
        re-concatenating their arrays — on a 65k-feature store with 425
        shards this OOMs the process (~16 GB peak). The streaming load
        below holds at most one shard plus the pre-allocated output at
        any moment (~50 MB + ~150 MB → ~200 MB peak for the 65k case).

        Pass 1 peeks each shard's ``shape`` array and the length of its
        ``indices`` array (no data decoded) to learn ``total_rows`` and
        ``total_nnz``. Pass 2 pre-allocates the output ``data``,
        ``indices``, ``indptr`` arrays and copies one shard at a time into
        the right slice, dropping each shard before loading the next.
        """
        run_dir = Path(run_dir)
        paths = cls._discover_shard_paths(run_dir)
        if not paths:
            raise FileNotFoundError(f"No activation shards found under {run_dir}")
        if len(paths) == 1:
            return cls._load_shard(paths[0])

        # Pass 1 — peek dimensions; never decode data.
        n_cols: int | None = None
        data_dtype: np.dtype | None = None
        shard_info: list[tuple[int, int]] = []  # (n_rows, nnz)
        total_rows = 0
        total_nnz = 0
        for p in paths:
            with np.load(p) as npz:
                shape = npz["shape"]
                nrows, ncols = int(shape[0]), int(shape[1])
                nnz = int(npz["indices"].shape[0])
                if data_dtype is None:
                    data_dtype = npz["data"].dtype
            if n_cols is None:
                n_cols = ncols
            elif ncols != n_cols:
                raise ValueError(
                    f"shard column-count mismatch at {p}: {ncols} (expected {n_cols})"
                )
            shard_info.append((nrows, nnz))
            total_rows += nrows
            total_nnz += nnz

        # Pass 2 — pre-allocate output, fill from each shard, drop shard.
        out_data = np.empty(total_nnz, dtype=data_dtype)
        out_indices = np.empty(total_nnz, dtype=np.int32)
        out_indptr = np.empty(total_rows + 1, dtype=np.int64)
        out_indptr[0] = 0
        row_offset = 0
        nnz_offset = 0
        for p, (nr, nz) in zip(paths, shard_info, strict=True):
            m = sparse.load_npz(p)
            if nz:
                out_data[nnz_offset:nnz_offset + nz] = m.data
                out_indices[nnz_offset:nnz_offset + nz] = m.indices
            # m.indptr is length nr+1 and starts at 0; shift and skip leading 0
            # so subsequent shards' indptr continues from this shard's last nnz.
            out_indptr[row_offset + 1:row_offset + 1 + nr] = (
                m.indptr[1:].astype(np.int64, copy=False) + nnz_offset
            )
            row_offset += nr
            nnz_offset += nz
            del m  # drop before loading the next shard

        return sparse.csr_matrix(
            (out_data, out_indices, out_indptr),
            shape=(total_rows, n_cols),
        )
