"""Stage 5 — score evaluator predictions vs. true activations, push labels.

Joins each evaluator result file with the held-out ``_true_activations``
from the Stage 2 linspace JSON, computes Pearson + Spearman correlations
per feature, writes a ``scores.parquet`` summary, and (optionally) pushes
labels above a Pearson threshold back into
:class:`FeatureLabelStore` under ``method="autointerpret"``.

Also reports the A/B split (zero-hint on vs. off) from
:class:`AgentInputWriter` so the team can pick a default for future runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats as scipy_stats

from interpret.sae.autointerpreter.config import (
    AgentStageConfig,
    AutoInterpretCollectConfig,
    AutoInterpretScoreConfig,
)
from interpret.sae.autointerpreter.prepare_agent_inputs import AgentInputWriter
from interpret.sae.feature_labels import FeatureLabelStore


class AutoInterpretScorer:
    """Compute per-feature correlations and push labels back to the store."""

    def __init__(
        self,
        run_dir: Path,
        score_config: AutoInterpretScoreConfig,
        collect_config: AutoInterpretCollectConfig,
        agents_config: AgentStageConfig,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.cfg = score_config
        self.collect_cfg = collect_config
        self.agents_cfg = agents_config

    # ── Public API ───────────────────────────────────────────────────────

    def score_all(self) -> pd.DataFrame:
        linspace_dir = self.run_dir / "linspace"
        # Per-store dirs (populated by the runner's tag-stripping sync) —
        # clean filenames, no risk of cross-config collision in the
        # shared global queue.
        labels_dir = self.run_dir / "labels"
        eval_dir = self.run_dir / "evaluator"
        ab = self._load_ab_split()

        rows: list[dict] = []
        for linspace_path in sorted(linspace_dir.glob("feature_*.json")):
            feature_idx = int(linspace_path.stem.split("_")[-1])
            label_path = labels_dir / linspace_path.name
            eval_path = eval_dir / linspace_path.name
            if not (label_path.exists() and eval_path.exists()):
                continue

            linspace = json.loads(linspace_path.read_text(encoding="utf-8"))
            label_data = json.loads(label_path.read_text(encoding="utf-8"))
            eval_data = json.loads(eval_path.read_text(encoding="utf-8"))

            predicted, truth = self._align(eval_data, linspace)
            if len(predicted) < 3:
                continue
            pearson = _safe_corr(predicted, truth, method="pearson")
            spearman = _safe_corr(predicted, truth, method="spearman")

            rows.append(
                {
                    "feature_idx": feature_idx,
                    "pearson": pearson,
                    "spearman": spearman,
                    "n_samples": int(len(predicted)),
                    "short_name": label_data.get("short_name", ""),
                    "explanation": label_data.get("explanation", ""),
                    "polarity": label_data.get("polarity"),
                    "zero_fraction": linspace.get("zero_fraction"),
                    "zero_hint_shown": bool(ab.get(feature_idx, False)),
                },
            )

        df = pd.DataFrame(rows)
        out = self.run_dir / "scores.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
        return df

    def report_ab_split(self, scores: pd.DataFrame) -> pd.DataFrame | None:
        if not self.cfg.report_ab_split or scores.empty:
            return None
        summary = (
            scores.groupby("zero_hint_shown")[["pearson", "spearman", "n_samples"]]
            .agg(["mean", "count"])
        )
        return summary

    def push_to_label_store(self, scores: pd.DataFrame) -> int:
        if not self.cfg.write_to_label_store or scores.empty:
            return 0
        # Embedding dimensions have no SAE config to key the store by, so the
        # push (which calls collect_cfg.to_sae_config()) doesn't apply.
        if getattr(self.collect_cfg, "source_kind", "sae") != "sae":
            return 0
        keep = scores[scores["pearson"] >= self.cfg.min_pearson]
        if keep.empty:
            return 0
        labels = {
            int(row.feature_idx):
            f"{row.short_name} (r={row.pearson:.2f})"
            for row in keep.itertuples(index=False)
        }
        store = FeatureLabelStore(self.cfg.label_store_dir)
        sae_config = self.collect_cfg.to_sae_config()
        model_id, layer, hook, width = FeatureLabelStore.params_from_config(sae_config)
        store.write_labels(
            labels,
            model_id=model_id,
            layer=layer,
            hook=hook,
            width=width,
            method=self.cfg.method_name,
        )
        return len(labels)

    # ── Internals ────────────────────────────────────────────────────────

    def _load_ab_split(self) -> dict[int, bool]:
        path = self.run_dir / AgentInputWriter.AB_SPLIT_FILE
        if not path.exists():
            return {}
        df = pq.read_table(path).to_pandas()
        return dict(zip(df["feature_idx"].astype(int), df["zero_hint_shown"].astype(bool), strict=True))

    @staticmethod
    def _align(eval_data: dict, linspace: dict) -> tuple[np.ndarray, np.ndarray]:
        """Pair predictions and ground-truth by ``sample_id``."""
        truth_by_id = {
            s["sample_id"]: t
            for s, t in zip(linspace["samples"], linspace["_true_activations"], strict=True)
        }
        predicted = []
        truth = []
        for pred in eval_data.get("predictions", []):
            sid = pred.get("sample_id")
            score = pred.get("score")
            if sid in truth_by_id and isinstance(score, (int, float)):
                predicted.append(float(score))
                truth.append(float(truth_by_id[sid]))
        return np.asarray(predicted, dtype=np.float64), np.asarray(truth, dtype=np.float64)


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """Return NaN when the correlation is undefined (zero variance)."""
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if method == "pearson":
        r, _ = scipy_stats.pearsonr(x, y)
    elif method == "spearman":
        r, _ = scipy_stats.spearmanr(x, y)
    else:
        raise ValueError(method)
    return float(r)
