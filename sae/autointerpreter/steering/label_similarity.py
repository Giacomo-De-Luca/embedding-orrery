"""Quantify steering-vs-activation label agreement by embedding cosine.

The steering judge names each feature's *behaviour* (`steering_short_name`); the
activation-based autointerpreter named the same feature's *top activations*
(`activation_short_name`). Whether the two agree is the experiment's key question.
String overlap undercounts agreement (``math`` vs ``numerical`` share no word but
mean the same thing), so we embed both short labels with a sentence-transformer
and take their cosine: high = the behaviour confirms the activation label, low =
the activation view alone would mislead.

Augments ``verdicts.parquet`` and ``summary.csv`` in place with a ``label_cosine``
column and returns the agree/diverge breakdown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Cosine cut points for the (necessarily arbitrary) agree / partial / diverge
# buckets. all-MiniLM cosines for related short phrases typically sit 0.3-0.7.
AGREE_THRESHOLD = 0.5
DIVERGE_THRESHOLD = 0.3


class LabelAgreementAnalyzer:
    """Embed the two label columns of a steering run and score their cosine."""

    def __init__(
        self,
        run_dir: Path | str,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.model_name = model_name
        self.device = device

    def run(self) -> dict:
        df = pd.read_parquet(self.run_dir / "verdicts.parquet")
        df["label_cosine"] = self._cosine(
            df["steering_short_name"], df["activation_short_name"]
        )

        # Persist: full table to parquet, the scannable subset to summary.csv.
        df.to_parquet(self.run_dir / "verdicts.parquet", index=False)
        summary_cols = [
            "feature_index", "working_steering", "steers", "broken",
            "steering_strength_0_10", "confidence", "label_cosine",
            "steering_short_name", "activation_short_name",
        ]
        df[summary_cols].to_csv(self.run_dir / "summary.csv", index=False)

        work = df[df["working_steering"]]
        return {
            "n": len(df),
            "n_working": len(work),
            "mean_cosine_all": _safe_mean(df["label_cosine"]),
            "mean_cosine_working": _safe_mean(work["label_cosine"]),
            "agree": int((work["label_cosine"] >= AGREE_THRESHOLD).sum()),
            "partial": int(
                work["label_cosine"].between(DIVERGE_THRESHOLD, AGREE_THRESHOLD, inclusive="left").sum()
            ),
            "diverge": int((work["label_cosine"] < DIVERGE_THRESHOLD).sum()),
        }

    def _cosine(self, a: pd.Series, b: pd.Series) -> np.ndarray:
        """Row-wise cosine between two label columns (NaN where a label is missing)."""
        from sentence_transformers import SentenceTransformer

        a = a.astype("object")
        b = b.astype("object")
        valid = a.notna() & b.notna()
        out = np.full(len(a), np.nan, dtype=np.float64)
        if not valid.any():
            return out

        model = SentenceTransformer(self.model_name, device=self.device)
        ea = model.encode(a[valid].tolist(), normalize_embeddings=True)
        eb = model.encode(b[valid].tolist(), normalize_embeddings=True)
        out[valid.to_numpy()] = (ea * eb).sum(axis=1)
        return out


def _safe_mean(s: pd.Series) -> float | None:
    s = s.dropna()
    return float(s.mean()) if len(s) else None
