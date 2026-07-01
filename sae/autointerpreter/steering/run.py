"""Orchestrate the steering-autointerpreter: generate → judge → aggregate.

One YAML drives the whole thing; no CLI flags. Stages run sequentially (the 4B
model is fully released before the headless judge workers spawn) and each is
independently skippable via ``stages.skip_*`` so a stage can be re-run alone.

Run with::

    uv run python -m interpret.sae.autointerpreter.steering.run

Edit :func:`main` (or call :func:`run_from_yaml`) to point at a different config.
"""

from __future__ import annotations

from pathlib import Path

from interpret.sae.autointerpreter.config import PROJECT_ROOT, dump_yaml
from interpret.sae.autointerpreter.steering.config import (
    SteeringInterpretConfig,
    load_steering_experiments,
)
from interpret.sae.autointerpreter.steering.generate import SteeringGenerator
from interpret.sae.autointerpreter.steering.judge import (
    AgentQueueDriver,
    SteeringAggregator,
    SteeringJudgeInputWriter,
)


class SteeringInterpretRunner:
    """Run one steering-autointerpret experiment end-to-end."""

    def __init__(self, config: SteeringInterpretConfig) -> None:
        self.config = config
        self.run_dir = config.run_dir()

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(self.config, self.run_dir / "experiment.yaml")
        stages = self.config.stages

        if not stages.skip_generate:
            print(f"[1/3] generate → {self.run_dir}")
            SteeringGenerator(self.config, self.run_dir).run()

        if not stages.skip_judge:
            n = SteeringJudgeInputWriter(self.run_dir, self.config.judge).write()
            print(f"[2/3] judge: queued {n} features")
            judge = self.config.judge
            AgentQueueDriver(
                poll_interval_seconds=judge.poll_interval_seconds,
                model_override=judge.model_override,
                fail_on_queue_errors=judge.fail_on_queue_errors,
            ).run_task(judge.task, judge.workers, judge.reps_per_worker)

        if not stages.skip_aggregate:
            info = SteeringAggregator(self.run_dir, self.config.judge).run()
            print(
                f"[3/3] aggregate: {info['n_scored']} scored, "
                f"{info['n_missing']} missing → {self.run_dir / 'summary.csv'}"
            )


def run_from_yaml(path: Path | str) -> None:
    for cfg in load_steering_experiments(Path(path)):
        print(f"\n=== steering experiment: {cfg.resolved_slug()} ===")
        SteeringInterpretRunner(cfg).run()


def main() -> None:
    """Default run. Edit the path to point at your config."""
    run_from_yaml(PROJECT_ROOT / "configs" / "autointerpret" / "steering_L9_16k.yaml")


if __name__ == "__main__":
    main()
