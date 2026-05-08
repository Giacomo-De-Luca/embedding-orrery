"""Stage 2 — extract per-feature top-k and linspace samples.

For each SAE feature, writes two JSON files under the run directory:

- ``topk/feature_{idx:06d}.json`` — the ``k`` most-strongly-activating samples,
  consumed by the LabelInterpreter agent.
- ``linspace/feature_{idx:06d}.json`` — 50 samples drawn at
  ``np.linspace`` positions along sorted nonzero activations (plus zero-pad
  when needed), shuffled deterministically per-feature. Ground-truth
  activations are recorded under the ``_true_activations`` key so the scorer
  (Stage 5) can compute Pearson/Spearman against the evaluator's predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from tqdm import tqdm

from interpret.sae.autointerpreter.config import (
    AutoInterpretCollectConfig,
    TopKExtractConfig,
)
from interpret.sae.autointerpreter.sparse_activation_store import SparseActivationStore


class TopKFeatureExtractor:
    """Compute and persist top-k / linspace sample sets per SAE feature."""

    FEATURE_STATS_FILE = "feature_stats.parquet"
    TOPK_DIR = "topk"
    LINSPACE_DIR = "linspace"

    def __init__(
        self,
        run_dir: Path,
        extract_config: TopKExtractConfig,
        collect_config: AutoInterpretCollectConfig,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.cfg = extract_config
        self.collect_cfg = collect_config
        self.topk_dir = self.run_dir / self.TOPK_DIR
        self.linspace_dir = self.run_dir / self.LINSPACE_DIR
        self.topk_dir.mkdir(parents=True, exist_ok=True)
        self.linspace_dir.mkdir(parents=True, exist_ok=True)

        self.matrix: sparse.csc_matrix = (
            SparseActivationStore.load_matrix(self.run_dir).tocsc()
        )
        self.index: pd.DataFrame = SparseActivationStore.load_index(self.run_dir)

    # ── Feature selection ───────────────────────────────────────────────

    def compute_feature_stats(self) -> pd.DataFrame:
        """Density / mean_nonzero / nnz per feature; cached to parquet."""
        out = self.run_dir / self.FEATURE_STATS_FILE
        if out.exists():
            return pq.read_table(out).to_pandas()

        n_rows, n_features = self.matrix.shape
        indptr = self.matrix.indptr
        nnz_per_col = np.diff(indptr)
        density = nnz_per_col / max(n_rows, 1)

        data = self.matrix.data.astype(np.float32, copy=False)
        sums = np.add.reduceat(
            np.concatenate([data, [0.0]]),
            indptr[:-1],
        )
        sums[nnz_per_col == 0] = 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_nonzero = np.where(nnz_per_col > 0, sums / nnz_per_col, 0.0)

        df = pd.DataFrame(
            {
                "feature_idx": np.arange(n_features, dtype=np.int32),
                "density": density.astype(np.float32),
                "nnz": nnz_per_col.astype(np.int32),
                "mean_nonzero_activation": mean_nonzero.astype(np.float32),
                "n_rows": np.full(n_features, n_rows, dtype=np.int32),
            },
        )
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
        return df

    def selected_features(self, stats: pd.DataFrame) -> list[int]:
        mask = (
            (stats["density"] >= self.cfg.density_min)
            & (stats["density"] <= self.cfg.density_max)
            & (stats["nnz"] >= self.cfg.require_min_nonzero)
        )
        features = stats.loc[mask, "feature_idx"].astype(int).tolist()
        if self.cfg.feature_indices is not None:
            keep = set(self.cfg.feature_indices)
            features = [f for f in features if f in keep]
        return features

    # ── Per-feature sample extraction ───────────────────────────────────

    def _column_nonzero(self, feature_idx: int) -> tuple[np.ndarray, np.ndarray]:
        indptr = self.matrix.indptr
        start, end = indptr[feature_idx], indptr[feature_idx + 1]
        rows = self.matrix.indices[start:end]
        values = self.matrix.data[start:end].astype(np.float32, copy=False)
        return rows, values

    def _row_meta(self, row_idx: int) -> dict:
        row = self.index.iloc[row_idx]
        return {
            "word": row["word"],
            "synset_id": row["synset_id"],
            "definition": row["definition"],
            "prompt": row["prompt"],
        }

    def _topk_samples(
        self, rows: np.ndarray, values: np.ndarray, k: int,
    ) -> list[dict]:
        if len(values) == 0:
            return []
        order = np.argsort(-values)[:k]
        samples = []
        for rank, i in enumerate(order, 1):
            meta = self._row_meta(int(rows[i]))
            samples.append(
                {
                    "rank": rank,
                    "row_idx": int(rows[i]),
                    "activation": float(values[i]),
                    **meta,
                },
            )
        return samples

    def _linspace_samples(
        self,
        rows: np.ndarray,
        values: np.ndarray,
        feature_idx: int,
        n_rows_total: int,
    ) -> tuple[list[dict], list[float], bool]:
        """Return (samples, true_activations, padded_with_zeros)."""
        n = self.cfg.eval_sample_count
        rng = np.random.default_rng(self.cfg.eval_shuffle_seed + feature_idx)

        if len(values) >= n:
            order = np.argsort(values)          # ascending
            pick = np.linspace(0, len(order) - 1, n).astype(int)
            chosen_local = order[pick]
            chosen_rows = rows[chosen_local]
            chosen_vals = values[chosen_local]
            padded = False
        else:
            # Pad with zero rows sampled uniformly from rows with zero acts
            order = np.argsort(values)
            chosen_rows = list(rows[order])
            chosen_vals = list(values[order])
            zero_pool = np.setdiff1d(
                np.arange(n_rows_total, dtype=np.int64),
                rows,
                assume_unique=False,
            )
            need = n - len(chosen_rows)
            if need > 0 and len(zero_pool) > 0:
                extra = rng.choice(zero_pool, size=need, replace=len(zero_pool) < need)
                chosen_rows.extend(extra.tolist())
                chosen_vals.extend([0.0] * need)
            chosen_rows = np.asarray(chosen_rows, dtype=np.int64)
            chosen_vals = np.asarray(chosen_vals, dtype=np.float32)
            padded = True

        shuffle_order = rng.permutation(len(chosen_rows))
        chosen_rows = chosen_rows[shuffle_order]
        chosen_vals = chosen_vals[shuffle_order]

        samples = []
        for sample_id, row_idx in enumerate(chosen_rows):
            meta = self._row_meta(int(row_idx))
            samples.append({"sample_id": sample_id, "row_idx": int(row_idx), **meta})
        return samples, [float(v) for v in chosen_vals], padded

    # ── Run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        stats = self.compute_feature_stats()
        features = self.selected_features(stats)
        n_rows_total = int(self.matrix.shape[0])
        stats_by_idx = stats.set_index("feature_idx")

        collect_meta = {
            "layer": self.collect_cfg.layer_index,
            "hook": self.collect_cfg.hook_type,
            "width": self.collect_cfg.width,
            "aggregation": self.collect_cfg.aggregation,
        }

        for feature_idx in tqdm(features, desc="extract", unit="feat"):
            rows, values = self._column_nonzero(feature_idx)
            row_stats = stats_by_idx.loc[feature_idx]
            density = float(row_stats["density"])
            zero_fraction = 1.0 - density
            mean_nonzero = float(row_stats["mean_nonzero_activation"])

            top_samples = self._topk_samples(rows, values, self.cfg.top_k)
            lin_samples, lin_truth, padded = self._linspace_samples(
                rows, values, feature_idx, n_rows_total,
            )

            base = {
                "feature_index": int(feature_idx),
                **collect_meta,
                "density": density,
                "zero_fraction": zero_fraction,
                "mean_nonzero_activation": mean_nonzero,
            }
            (self.topk_dir / f"feature_{feature_idx:06d}.json").write_text(
                json.dumps({**base, "samples": top_samples}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (self.linspace_dir / f"feature_{feature_idx:06d}.json").write_text(
                json.dumps(
                    {
                        **base,
                        "max_activation_in_set": (
                            float(max(lin_truth)) if lin_truth else 0.0
                        ),
                        "padded_with_zeros": padded,
                        "samples": lin_samples,
                        "_true_activations": lin_truth,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        return {"n_features": len(features), "n_rows": n_rows_total}
