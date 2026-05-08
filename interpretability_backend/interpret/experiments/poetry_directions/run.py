"""Driver for the poetry-direction experiments.

Run all three experiments back-to-back (default), or a single named experiment
by passing its name as an argument::

    uv run python -m interpret.experiments.poetry_directions.run                        # all three
    uv run python -m interpret.experiments.poetry_directions.run poems_paraphrase       # just this one
    uv run python -m interpret.experiments.poetry_directions.run poetry_prose

When run for multiple experiments the model is loaded once and reused.
Each experiment is idempotent (per-phase artifact-skip), so calling this
again after a partial run resumes from the last completed phase.
"""

from __future__ import annotations

import sys

from interpret.experiments.poetry_directions.config import EXPERIMENTS, PoetryConfig
from interpret.experiments.poetry_directions.runner import PoetryRunner
from interpret.inference.gemma_pytorch import GemmaPytorchInference


def main() -> None:
    if len(sys.argv) > 1:
        requested = sys.argv[1]
        if requested not in EXPERIMENTS:
            raise SystemExit(
                f"Unknown experiment {requested!r}. Valid: {list(EXPERIMENTS)}"
            )
        names = [requested]
    else:
        names = list(EXPERIMENTS)

    model_name = "google/gemma-3-4b-it"
    print(f"loading {model_name} for: {names}")
    wrapper = GemmaPytorchInference(model_name)

    for name in names:
        print(f"\n{'=' * 80}\nrunning experiment: {name}\n{'=' * 80}")
        cfg = PoetryConfig(name=name, model_name=model_name)
        out_dir = PoetryRunner(cfg, wrapper=wrapper).run()
        print(f"[{name}] done. artifacts at: {out_dir}")


if __name__ == "__main__":
    main()
