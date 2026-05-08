"""Extract poetry-vs-prose contrast directions from Gemma-3 residual stream.

Three experiments share one extraction primitive (paired or unpaired
mean-of-difference at the EOI positions of the chat template). Each runs an
end-to-end pipeline (extract → layer × coefficient bypass sweep → JailbreakBench
ASR eval) under ``resources/experiments/poetry_directions/<name>/``, mirroring
the structure of ``interpret/experiments/refusal_directions/`` so the two experiments'
summary numbers are directly comparable.

The three named experiments are catalogued in ``EXPERIMENTS``:

- ``poems_paraphrase`` — 1000 paired (poem, paraphrase) from
  ``resources/poetry/paraphrased_poems_aligned.csv``.
- ``poetry_prose`` — 1154 poetry vs 1151 prose prompts from
  ``resources/poetry/alignment_dataset_long_clean.tsv`` (no safety filter:
  the direction is extracted over safe + unsafe rows).
- ``poetry_unsafe_safe`` — 540 unsafe-poetry vs 614 safe-poetry prompts from
  the same corpus, using the verified ``final_safety_status`` label.
"""

from interpret.experiments.poetry_directions.config import EXPERIMENTS, PoetryConfig
from interpret.experiments.poetry_directions.extract import extract_direction
from interpret.experiments.poetry_directions.runner import PoetryRunner

__all__ = ["EXPERIMENTS", "PoetryConfig", "PoetryRunner", "extract_direction"]
