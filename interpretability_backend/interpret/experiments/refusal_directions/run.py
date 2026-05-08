"""Driver for the refusal-direction replication.

Run with::

    uv run python -m interpret.experiments.refusal_directions.run
"""

from __future__ import annotations

from interpret.experiments.refusal_directions.config import RefusalConfig
from interpret.experiments.refusal_directions.runner import RefusalRunner


def main() -> None:
    config = RefusalConfig()
    output_dir = RefusalRunner(config).run()
    print(f"refusal-direction pipeline complete. Artifacts at: {output_dir}")


if __name__ == "__main__":
    main()
