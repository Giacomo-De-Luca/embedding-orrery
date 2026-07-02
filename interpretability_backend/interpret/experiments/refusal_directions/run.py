"""Driver for the refusal-direction replication.

Run with the default Gemma model::

    uv run python -m interpret.experiments.refusal_directions.run

or against a Qwen model by passing its HuggingFace id::

    uv run python -m interpret.experiments.refusal_directions.run Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import sys
from pathlib import Path

from interpret.experiments.refusal_directions.config import RefusalConfig
from interpret.experiments.refusal_directions.runner import RefusalRunner

_DEFAULT_GEMMA = "google/gemma-3-4b-it"
_BASE_OUTPUT_DIR = Path("resources/experiments/refusal_directions")


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_GEMMA
    # Non-default models get their own subdir so Gemma artifacts aren't
    # overwritten and per-model runs don't collide on the cached mean_diffs.
    output_dir = (
        _BASE_OUTPUT_DIR
        if model_name == _DEFAULT_GEMMA
        else _BASE_OUTPUT_DIR / model_name.split("/")[-1].lower()
    )
    config = RefusalConfig(model_name=model_name, output_dir=output_dir)
    output_dir = RefusalRunner(config).run()
    print(f"refusal-direction pipeline complete. Artifacts at: {output_dir}")


if __name__ == "__main__":
    main()
