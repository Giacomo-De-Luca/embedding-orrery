"""Abstract base for the autointerpreter's incremental activation stores.

Holds the source-agnostic machinery shared by every backend: buffered
``append``, append-only shard ``flush``, shard discovery / row counting, the
parallel ``index.parquet`` writer (atomic), and resume-key lookup. Concrete
subclasses supply only the activation-matrix buffering/serialization for their
format:

- :class:`SparseActivationStore` — scipy CSR shards (``.npz``) for SAE features
  (L0 ≈ 50-150 nonzeros per row; storing densely would cost tens of GB).
- :class:`DenseActivationStore` — numpy shards (``.npy``) for embedding-model
  dimensions (fully dense and signed, a few hundred MB at most).

Both expose the same ``append`` / ``flush`` / ``load_matrix`` / ``load_index`` /
``existing_row_keys`` surface so downstream stages (extract) never need to know
which backend produced a run.

Storage model: each ``flush()`` writes the buffered rows to a fresh
``activations_batch_NNNNNN.<ext>`` shard — no rewrite of prior shards, so
per-flush cost is O(rows-in-batch). ``load_matrix`` concatenates the shards on
read. A legacy single-file ``activations.<ext>`` (from older runs) is treated
as an immutable "base shard": loaded first, counted, never modified. Writes are
atomic (sibling ``.tmp`` then ``os.replace``).
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ActivationStore(ABC):
    """Buffered activation store with a parallel metadata index (parquet).

    Rows are samples, columns are features (SAE features or embedding dims).
    Subclasses set :attr:`ACTIVATIONS_FILE` (legacy base name) and
    :attr:`SHARD_EXT`, and implement the buffer/serialize hooks below.
    """

    INDEX_FILE = "index.parquet"
    BATCH_PREFIX = "activations_batch_"
    BATCH_PAD = 6
    # Subclasses override:
    ACTIVATIONS_FILE: str = "activations"  # legacy single-file base shard
    SHARD_EXT: str = ""                    # shard extension without the dot

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

        # Pending (unflushed) per-row metadata, parallel to the format-specific
        # row buffers managed by the subclass via _init_buffers/_buffer_row.
        self._pending_meta: list[dict[str, Any]] = []
        self._rows_pending = 0
        self._init_buffers()
        self._rows_flushed = self._count_existing_rows()
        # Next shard index, picked from existing batch files so a resumed run
        # never overwrites a sibling.
        self._next_batch_idx = self._discover_next_batch_index()

    # ── Properties ─────────────────────────────────────────────────────────

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
        self._buffer_row(vector)
        row_idx = self.n_rows
        enriched = dict(meta)
        enriched.setdefault("row_idx", row_idx)
        self._pending_meta.append(enriched)
        self._rows_pending += 1
        return row_idx

    def flush(self) -> None:
        """Persist buffered rows as a new shard. Safe to call repeatedly.

        Writes pending rows to ``activations_batch_<idx>.<ext>`` — no rewrite
        of prior shards. ``index.parquet`` (written atomically) is the source
        of truth for row counts.
        """
        if self._rows_pending == 0:
            return

        payload = self._build_pending()
        self._save_shard(payload, self._batch_path(self._next_batch_idx))
        self._append_index(self._pending_meta)

        self._next_batch_idx += 1
        self._rows_flushed += self._rows_pending
        self._rows_pending = 0
        self._reset_buffers()
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
    def load_index(cls, run_dir: Path) -> pd.DataFrame:
        path = Path(run_dir) / cls.INDEX_FILE
        return pq.read_table(path).to_pandas()

    @classmethod
    def load_matrix(cls, run_dir: Path):
        """Concatenate every shard under ``run_dir`` into a single matrix.

        Loads the legacy ``activations.<ext>`` (if present) first, then every
        ``activations_batch_NNNNNN.<ext>`` in batch-index order. Returns the
        backend-native matrix type (CSR for sparse, ndarray for dense).
        """
        run_dir = Path(run_dir)
        paths = cls._discover_shard_paths(run_dir)
        if not paths:
            raise FileNotFoundError(f"No activation shards found under {run_dir}")
        if len(paths) == 1:
            return cls._load_shard(paths[0])
        return cls._concat([cls._load_shard(p) for p in paths])

    # ── Shared internals ─────────────────────────────────────────────────

    def _batch_path(self, idx: int) -> Path:
        return self.run_dir / (
            f"{self.BATCH_PREFIX}{idx:0{self.BATCH_PAD}d}.{self.SHARD_EXT}"
        )

    @classmethod
    def _batch_regex(cls) -> re.Pattern[str]:
        return re.compile(
            rf"^{re.escape(cls.BATCH_PREFIX)}(\d+)\.{cls.SHARD_EXT}$"
        )

    @classmethod
    def _discover_shard_paths(cls, run_dir: Path) -> list[Path]:
        """All shards in append order: legacy base first, then numbered batches."""
        run_dir = Path(run_dir)
        out: list[Path] = []
        legacy = run_dir / cls.ACTIVATIONS_FILE
        if legacy.exists():
            out.append(legacy)
        regex = cls._batch_regex()
        batches: list[tuple[int, Path]] = []
        for p in run_dir.glob(f"{cls.BATCH_PREFIX}*.{cls.SHARD_EXT}"):
            m = regex.match(p.name)
            if m:
                batches.append((int(m.group(1)), p))
        batches.sort(key=lambda t: t[0])
        out.extend(p for _, p in batches)
        return out

    def _discover_next_batch_index(self) -> int:
        regex = self._batch_regex()
        existing: list[int] = []
        for p in self.run_dir.glob(f"{self.BATCH_PREFIX}*.{self.SHARD_EXT}"):
            m = regex.match(p.name)
            if m:
                existing.append(int(m.group(1)))
        return (max(existing) + 1) if existing else 1

    def _count_existing_rows(self) -> int:
        return sum(
            self._shard_n_rows(p)
            for p in self._discover_shard_paths(self.run_dir)
        )

    def _append_index(self, rows: list[dict[str, Any]]) -> None:
        df_new = pd.DataFrame(rows)
        if self.index_path.exists():
            df_old = pq.read_table(self.index_path).to_pandas()
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        # Atomic write: a kill mid-write would otherwise leave an empty/partial
        # parquet that desyncs against the shard files.
        tmp_index = self.index_path.with_name(self.index_path.name + ".tmp")
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False), tmp_index,
        )
        os.replace(tmp_index, self.index_path)

    # ── Format-specific hooks (implemented by subclasses) ────────────────

    @abstractmethod
    def _init_buffers(self) -> None:
        """Initialise the empty per-row buffers for pending appends."""

    @abstractmethod
    def _reset_buffers(self) -> None:
        """Clear the per-row buffers after a flush."""

    @abstractmethod
    def _buffer_row(self, vector: np.ndarray) -> None:
        """Append one row to the pending buffers."""

    @abstractmethod
    def _build_pending(self):
        """Materialise the buffered rows into a single shard payload."""

    @classmethod
    @abstractmethod
    def _save_shard(cls, payload, path: Path) -> None:
        """Atomically write a shard payload to ``path``."""

    @classmethod
    @abstractmethod
    def _load_shard(cls, path: Path):
        """Load one shard payload from ``path``."""

    @classmethod
    @abstractmethod
    def _concat(cls, payloads: list):
        """Concatenate loaded shard payloads row-wise."""

    @classmethod
    @abstractmethod
    def _shard_n_rows(cls, path: Path) -> int:
        """Row count of the shard at ``path`` (loaded cheaply where possible)."""
