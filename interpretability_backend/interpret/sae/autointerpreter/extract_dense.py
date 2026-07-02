"""Stage 2 (dense variant) — per-dimension top-k and linspace samples.

Counterpart to :class:`TopKFeatureExtractor` for **dense, signed** activation
matrices — sentence-transformer embeddings (``source_kind: embedding``) or
raw residual-stream dims (``source_kind: residual``); both store via
:class:`DenseActivationStore`. Writes the same ``topk/`` + ``linspace/``
JSON contract so the
downstream label / eval / score stages are unchanged. Two modes
(``extract.dim_mode``):

- ``signed`` — one feature per dimension. ``topk`` shows the ``top_k/2`` most
  positive and ``top_k/2`` most negative samples (each tagged ``pole``);
  ``linspace`` spans the full signed range and ``_true_activations`` stay
  signed. Pair with the signed (-N..+N) eval rubric.
- ``split`` — two features per dimension: ``pos = max(0, x)`` (feature index
  ``2·dim``) and ``neg = max(0, -x)`` (``2·dim + 1``). Each half is
  non-negative and produces SAE-shaped JSON (density / zero_fraction /
  mean_nonzero_activation), so it reuses the 0-10 rubric and is directly
  comparable to SAE features. Tests whether the +dir and -dir of one axis
  encode different things.

Dimension selection: ``select="all"`` labels every dimension (only a few
hundred, vs 16k-262k SAE features — no density filter needed);
``select="top_variance"`` keeps the highest-variance dims (capped by
``max_features``). ``feature_indices`` filters by *dimension* index in both
modes. The activation matrix may be read from another run via
``extract.activations_run_dir`` (run a second ``dim_mode`` without
re-embedding).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from interpret.sae.autointerpreter.config import (
    PROJECT_ROOT,
    AutoInterpretCollectConfig,
    TopKExtractConfig,
)
from interpret.sae.autointerpreter.dense_activation_store import DenseActivationStore


class DenseFeatureExtractor:
    """Compute and persist top-k / linspace sample sets per embedding dimension."""

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

        src = self._resolve_activations_dir()
        self.matrix: np.ndarray = (
            DenseActivationStore.load_matrix(src).astype(np.float32, copy=False)
        )
        self.index: pd.DataFrame = DenseActivationStore.load_index(src)
        self.n_rows, self.n_dims = self.matrix.shape

    def _resolve_activations_dir(self) -> Path:
        if not self.cfg.activations_run_dir:
            return self.run_dir
        src = Path(self.cfg.activations_run_dir)
        if not src.is_absolute():
            src = (PROJECT_ROOT / src).resolve()
        return src

    # ── Dimension stats / selection ─────────────────────────────────────

    def compute_feature_stats(self) -> pd.DataFrame:
        """Per-dimension variance / abs_mean / min / max; cached to parquet."""
        out = self.run_dir / self.FEATURE_STATS_FILE
        if out.exists():
            return pq.read_table(out).to_pandas()

        df = pd.DataFrame(
            {
                "feature_idx": np.arange(self.n_dims, dtype=np.int32),
                "variance": self.matrix.var(axis=0).astype(np.float32),
                "abs_mean": np.abs(self.matrix).mean(axis=0).astype(np.float32),
                "vmin": self.matrix.min(axis=0).astype(np.float32),
                "vmax": self.matrix.max(axis=0).astype(np.float32),
                "n_rows": np.full(self.n_dims, self.n_rows, dtype=np.int32),
            },
        )
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
        return df

    def selected_dims(self, stats: pd.DataFrame) -> list[int]:
        mask = stats["variance"] >= self.cfg.min_variance
        kept = stats.loc[mask]
        if self.cfg.select == "top_variance" and self.cfg.max_features:
            kept = kept.sort_values("variance", ascending=False).head(
                self.cfg.max_features,
            )
        dims = sorted(kept["feature_idx"].astype(int).tolist())
        if self.cfg.feature_indices is not None:
            keep = set(self.cfg.feature_indices)
            dims = [d for d in dims if d in keep]
        return dims

    # ── Per-row metadata ─────────────────────────────────────────────────

    def _row_meta(self, row_idx: int) -> dict:
        # Only what the agent needs to interpret a sample. The embedded text was
        # "{word}: {definition}.", so word + definition fully convey it — echoing
        # the literal `prompt` (= word + definition) or the opaque `synset_id`
        # would just multiply tokens. The full record stays in index.parquet,
        # joinable on row_idx.
        row = self.index.iloc[row_idx]
        return {"word": row["word"], "definition": row["definition"]}

    # ── Sample extraction ────────────────────────────────────────────────

    def _one_sided_topk(
        self, rows: np.ndarray, values: np.ndarray, k: int, pole: str | None = None,
    ) -> list[dict]:
        """Top-k by descending value (most-activating). Optional ``pole`` tag."""
        if len(values) == 0:
            return []
        order = np.argsort(-values)[:k]
        samples = []
        for rank, i in enumerate(order, 1):
            sample = {
                "rank": rank,
                "row_idx": int(rows[i]),
                "activation": float(values[i]),
                **self._row_meta(int(rows[i])),
            }
            if pole is not None:
                sample["pole"] = pole
            samples.append(sample)
        return samples

    def _both_poles_topk(self, col: np.ndarray, k: int) -> list[dict]:
        """``k/2`` most-positive (pole=high) + ``k/2`` most-negative (pole=low)."""
        k_half = max(1, k // 2)
        all_rows = np.arange(len(col), dtype=np.int64)
        high = self._one_sided_topk(all_rows, col, k_half, pole="high")
        high_rows = {s["row_idx"] for s in high}
        # Most-negative = top-k of the negated column; report the signed value.
        low = self._one_sided_topk(all_rows, -col, k_half, pole="low")
        for s in low:
            s["activation"] = -s["activation"]  # restore the signed (negative) value
        # When n_rows < k or the column is near-constant, a row can top both
        # ends; keep it only as the (stronger) positive pole so the interpreter
        # never sees one sample tagged both high and low.
        low = [s for s in low if s["row_idx"] not in high_rows]
        return high + low

    def _linspace_samples(
        self,
        rows: np.ndarray,
        values: np.ndarray,
        feature_idx: int,
        exclude_rows: set[int] | frozenset = frozenset(),
    ) -> tuple[list[dict], list[float], bool]:
        """Return (samples, true_activations, padded_with_zeros).

        The eval set is **held out** from the interpreter: ``exclude_rows`` (the
        top-k samples used to write the label) are dropped from the pool before
        sampling, so the simulation score measures generalisation rather than
        re-recognition of the label's own source words. With ~10^5 rows per
        dimension this barely changes the sampled range.

        Mirrors the SAE extractor otherwise: when ``values`` covers
        ``eval_sample_count`` rows it linspaces over the sorted range with no
        padding; for a sparse ``split`` half it pads with zero-valued rows from
        the complement.
        """
        n = self.cfg.eval_sample_count
        rng = np.random.default_rng(self.cfg.eval_shuffle_seed + feature_idx)

        if exclude_rows:
            excl = np.fromiter(exclude_rows, dtype=np.int64)
            keep = ~np.isin(rows, excl)
            rows = rows[keep]
            values = values[keep]
        else:
            excl = np.empty(0, dtype=np.int64)

        if len(values) >= n:
            order = np.argsort(values)          # ascending (signed-aware)
            pick = np.linspace(0, len(order) - 1, n).astype(int)
            chosen_local = order[pick]
            chosen_rows = rows[chosen_local]
            chosen_vals = values[chosen_local]
            padded = False
        else:
            order = np.argsort(values)
            chosen_rows = list(rows[order])
            chosen_vals = list(values[order])
            # Zero pool excludes both the active rows and the held-out top-k.
            zero_pool = np.setdiff1d(
                np.arange(self.n_rows, dtype=np.int64),
                np.concatenate([rows, excl]),
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
            samples.append(
                {"sample_id": sample_id, "row_idx": int(row_idx),
                 **self._row_meta(int(row_idx))},
            )
        return samples, [float(v) for v in chosen_vals], padded

    # ── JSON writers ─────────────────────────────────────────────────────

    def _collect_meta(self) -> dict:
        if self.collect_cfg.source_kind == "residual":
            site = self.collect_cfg.resolve_residual().sites[0]
            return {
                "source": "residual",
                "model_name": self.collect_cfg.resolve_base_model().checkpoint,
                "layer": (
                    -1 if site.intermediate == "final_norm" else site.layer_index
                ),
                "intermediate": site.intermediate,
                "aggregation": self.collect_cfg.aggregation,
                "n_dims": int(self.n_dims),
                "dim_mode": self.cfg.dim_mode,
            }
        return {
            "source": "embedding",
            "model_name": self.collect_cfg.resolve_embedding().model_name,
            "n_dims": int(self.n_dims),
            "dim_mode": self.cfg.dim_mode,
        }

    def _write_feature(
        self,
        feature_idx: int,
        base_extra: dict,
        top_samples: list[dict],
        lin_samples: list[dict],
        lin_truth: list[float],
        padded: bool,
    ) -> None:
        base = {"feature_index": int(feature_idx), **self._collect_meta(), **base_extra}
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

    # ── Per-dimension emitters ───────────────────────────────────────────

    def _emit_signed(self, dim: int, stats_row: pd.Series) -> None:
        col = self.matrix[:, dim]
        all_rows = np.arange(self.n_rows, dtype=np.int64)
        top_samples = self._both_poles_topk(col, self.cfg.top_k)
        # Hold the label's source samples out of the eval set (non-circular).
        exclude = {s["row_idx"] for s in top_samples}
        lin_samples, lin_truth, padded = self._linspace_samples(
            all_rows, col, dim, exclude,
        )
        base_extra = {
            "variance": float(stats_row["variance"]),
            "vmin": float(stats_row["vmin"]),
            "vmax": float(stats_row["vmax"]),
            "abs_mean": float(stats_row["abs_mean"]),
        }
        self._write_feature(dim, base_extra, top_samples, lin_samples, lin_truth, padded)

    def _emit_split(self, dim: int) -> None:
        col = self.matrix[:, dim]
        for half, feature_idx, half_vals in (
            ("pos", 2 * dim, np.maximum(col, 0.0)),
            ("neg", 2 * dim + 1, np.maximum(-col, 0.0)),
        ):
            nz = np.flatnonzero(half_vals)
            rows = nz.astype(np.int64)
            values = half_vals[nz].astype(np.float32)
            density = float(len(nz)) / max(self.n_rows, 1)
            mean_nonzero = float(values.mean()) if len(values) else 0.0

            top_samples = self._one_sided_topk(rows, values, self.cfg.top_k)
            # Hold the label's source samples out of the eval set (non-circular).
            exclude = {s["row_idx"] for s in top_samples}
            lin_samples, lin_truth, padded = self._linspace_samples(
                rows, values, feature_idx, exclude,
            )
            base_extra = {
                "dim": int(dim),
                "half": half,
                "density": density,
                "zero_fraction": 1.0 - density,
                "mean_nonzero_activation": mean_nonzero,
            }
            self._write_feature(
                feature_idx, base_extra, top_samples, lin_samples, lin_truth, padded,
            )

    # ── Run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        stats = self.compute_feature_stats()
        dims = self.selected_dims(stats)
        stats_by_idx = stats.set_index("feature_idx")

        for dim in tqdm(dims, desc="extract-dense", unit="dim"):
            if self.cfg.dim_mode == "signed":
                self._emit_signed(dim, stats_by_idx.loc[dim])
            else:
                self._emit_split(dim)

        n_features = len(dims) * (2 if self.cfg.dim_mode == "split" else 1)
        return {"n_features": n_features, "n_dims": len(dims), "n_rows": int(self.n_rows)}
