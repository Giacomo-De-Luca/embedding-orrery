"""Driver for the poetry-direction experiments.

Run all three experiments back-to-back (default), or a single named experiment,
and optionally against a Qwen model::

    uv run python -m interpret.experiments.poetry_directions.run                       # all three, Gemma
    uv run python -m interpret.experiments.poetry_directions.run poems_paraphrase      # one, Gemma
    uv run python -m interpret.experiments.poetry_directions.run --model Qwen/Qwen3-1.7B
    uv run python -m interpret.experiments.poetry_directions.run poetry_prose --model Qwen/Qwen3-4B

When run for multiple experiments the model is loaded once and reused.
Each experiment is idempotent (per-phase artifact-skip), so calling this
again after a partial run resumes from the last completed phase.
"""

from __future__ import annotations

import argparse

from interpret.experiments.directions_common import build_direction_model
from interpret.experiments.poetry_directions.config import EXPERIMENTS, PoetryConfig
from interpret.experiments.poetry_directions.runner import PoetryRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run poetry-direction experiments")
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help=f"One of {list(EXPERIMENTS)}; omit to run all three.",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-4b-it",
        help="HuggingFace model id (gemma-* or Qwen*).",
    )
    args = parser.parse_args()

    if args.experiment is not None:
        if args.experiment not in EXPERIMENTS:
            raise SystemExit(
                f"Unknown experiment {args.experiment!r}. Valid: {list(EXPERIMENTS)}"
            )
        names = [args.experiment]
    else:
        names = list(EXPERIMENTS)

    print(f"loading {args.model} for: {names}")
    model = build_direction_model(args.model)

    for name in names:
        print(f"\n{'=' * 80}\nrunning experiment: {name}\n{'=' * 80}")
        cfg = PoetryConfig(name=name, model_name=args.model)
        out_dir = PoetryRunner(cfg, model=model).run()
        print(f"[{name}] done. artifacts at: {out_dir}")


if __name__ == "__main__":
    main()
