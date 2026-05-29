"""Incremental sparse CSR storage for per-sample SAE feature activations.

Designed for Stage 1 of the autointerpreter pipeline: each WordNet sample
produces one dense activation vector ``(d_sae,)`` of which only ~100 entries
are nonzero (JumpReLU SAEs have L0 ≈ 50-150). Storing all samples densely
would cost tens of GB; the CSR format keeps it at a few hundred MB.

The store buffers rows in memory and flushes to disk as a single ``.npz``
+ a parallel ``index.parquet`` on demand. It can also reload an existing
run directory so Stage 2 consumers work directly on the sparse matrix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse


class SparseActivationStore:
    """Buffered CSR builder with a parallel metadata index (parquet).

    Rows are samples, columns are SAE features. Writes ``activations.npz``
    (compressed CSR) and ``index.parquet`` under ``run_dir`` on ``flush()``.

    Usage::

        store = SparseActivationStore(run_dir, n_features=16384)
        for vec, meta in ...:
            store.append(vec, meta)
        store.flush()
    """

    ACTIVATIONS_FILE = "activations.npz"
    INDEX_FILE = "index.parquet"

    def __init__(
        self,
        run_dir: Path,
        n_features: int,
        dtype: np.dtype = np.float16,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.n_features = int(n_features)
        self.dtype = np.dtype(dtype)

        # CSR buffers for pending (unflushed) rows.
        self._data: list[np.ndarray] = []
        self._indices: list[np.ndarray] = []
        self._indptr: list[int] = [0]
        self._pending_meta: list[dict[str, Any]] = []
        self._rows_pending = 0
        self._rows_flushed = self._count_existing_rows()

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def n_rows(self) -> int:
        """Total rows: already on disk plus buffered."""
        return self._rows_flushed + self._rows_pending

    @property
    def activations_path(self) -> Path:
        return self.run_dir / self.ACTIVATIONS_FILE

    @property
    def index_path(self) -> Path:
        return self.run_dir / self.INDEX_FILE

    # ── Public API ───────────────────────────────────────────────────────

    def append(self, vector: np.ndarray, meta: dict[str, Any]) -> int:
        """Buffer one sample. Returns the assigned row index."""
        if vector.shape != (self.n_features,):
            raise ValueError(
                f"expected shape ({self.n_features},), got {vector.shape}"
            )
        nz = np.flatnonzero(vector)
        self._data.append(vector[nz].astype(self.dtype, copy=False))
        self._indices.append(nz.astype(np.int32, copy=False))
        self._indptr.append(self._indptr[-1] + len(nz))
        row_idx = self.n_rows
        enriched = dict(meta)
        enriched.setdefault("row_idx", row_idx)
        self._pending_meta.append(enriched)
        self._rows_pending += 1
        return row_idx

    def flush(self) -> None:
        """Persist buffered rows to disk. Safe to call repeatedly."""
        if self._rows_pending == 0:
            return

        new_matrix = self._build_pending_csr()
        if self._rows_flushed > 0 and self.activations_path.exists():
            old = sparse.load_npz(self.activations_path)
            combined = sparse.vstack([old, new_matrix], format="csr")
        else:
            combined = new_matrix
        # Atomic write: save to a sibling .tmp then rename. A SIGKILL during
        # the multi-MB save_npz used to truncate activations.npz and lose
        # every prior flush (the file is rewritten in full each time).
        # NOTE: scipy.sparse.save_npz auto-appends ".npz" if the path
        # doesn't already end in it — so the temp must end in .npz too,
        # otherwise we'd write to "...tmp.npz" and then try to rename a
        # non-existent "...tmp".
        tmp_npz = self.activations_path.with_name(
            self.activations_path.stem + ".tmp.npz",
        )
        sparse.save_npz(tmp_npz, combined, compressed=True)
        os.replace(tmp_npz, self.activations_path)

        self._append_index(self._pending_meta)

        self._rows_flushed += self._rows_pending
        self._rows_pending = 0
        self._data.clear()
        self._indices.clear()
        self._indptr = [0]
        self._pending_meta.clear()

    def existing_row_keys(self) -> set[tuple[str, str]]:
        """Return ``(word, synset_id)`` pairs already stored (for resume)."""
        if not self.index_path.exists():
            return set()
        tbl = pq.read_table(
            self.index_path, columns=["word", "synset_id"],
        ).to_pandas()
        return set(zip(tbl["word"].tolist(), tbl["synset_id"].tolist()))

    # ── Loaders (used by Stage 2 onward) ────────────────────────────────

    @classmethod
    def load_matrix(cls, run_dir: Path) -> sparse.csr_matrix:
        path = Path(run_dir) / cls.ACTIVATIONS_FILE
        return sparse.load_npz(path)

    @classmethod
    def load_index(cls, run_dir: Path) -> pd.DataFrame:
        path = Path(run_dir) / cls.INDEX_FILE
        return pq.read_table(path).to_pandas()

    # ── Internals ────────────────────────────────────────────────────────

    def _build_pending_csr(self) -> sparse.csr_matrix:
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

    def _append_index(self, rows: list[dict[str, Any]]) -> None:
        df_new = pd.DataFrame(rows)
        if self.index_path.exists():
            df_old = pq.read_table(self.index_path).to_pandas()
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        # Atomic write, same reasoning as flush(): a kill mid-write
        # would otherwise leave an empty/partial parquet that desyncs
        # against activations.npz.
        tmp_index = self.index_path.with_name(self.index_path.name + ".tmp")
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False), tmp_index,
        )
        os.replace(tmp_index, self.index_path)

    def _count_existing_rows(self) -> int:
        if not self.activations_path.exists():
            return 0
        mat = sparse.load_npz(self.activations_path)
        return mat.shape[0]
