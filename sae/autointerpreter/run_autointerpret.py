"""Orchestrate the full autointerpreter pipeline from a YAML config.

Entry point for one or more experiments. Stages::

    collect → extract → label-agents → eval-agents → score

Any stage can be skipped via ``stages.skip_*`` flags, and each experiment
runs in its own ``run_dir = output_root/<run_slug>/``.

The agent stages shell out to the existing ``scripts/AgentSystem`` job
queue and launcher, then poll for completion. No CLI flags — edit
:func:`main` or call :func:`run_from_yaml` directly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from interpret.sae.autointerpreter.collect_activations import ActivationCollector
from interpret.sae.autointerpreter.config import (
    AutoInterpretConfig,
    dump_yaml,
    load_experiments,
)
from interpret.sae.autointerpreter.extract_top_k import TopKFeatureExtractor
from interpret.sae.autointerpreter.prepare_agent_inputs import AgentInputWriter
from interpret.sae.autointerpreter.score_autointerpret import AutoInterpretScorer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
JOB_QUEUE = PROJECT_ROOT / "scripts" / "AgentSystem" / "job_queue.py"
LAUNCHER = PROJECT_ROOT / "scripts" / "AgentSystem" / "launch_agents.sh"


@dataclass
class RunnerInputs:
    """Minimal entry-point dataclass — points at the YAML/JSON to run."""

    config_path: Path


class AutoInterpretRunner:
    """Run a single experiment end-to-end."""

    def __init__(self, config: AutoInterpretConfig) -> None:
        self.config = config
        self.run_dir = config.collect.run_dir()

    # ── Stage wrappers ──────────────────────────────────────────────────

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(self.config, self.run_dir / "experiment.yaml")

        if not self.config.stages.skip_collect:
            self._stage_collect()
        if not self.config.stages.skip_topk:
            self._stage_extract()
        writer = AgentInputWriter(self.run_dir, self.config.agents)
        if not self.config.stages.skip_label:
            self._stage_label(writer)
        if not self.config.stages.skip_eval:
            self._stage_eval(writer)
        if not self.config.stages.skip_score:
            self._stage_score()

    # ── Stages ──────────────────────────────────────────────────────────

    def _stage_collect(self) -> None:
        print(f"[1/5] collect → {self.run_dir}")
        ActivationCollector(self.config.collect).run()

    def _stage_extract(self) -> None:
        print("[2/5] extract top-k + linspace")
        TopKFeatureExtractor(
            self.run_dir, self.config.extract, self.config.collect,
        ).run()

    def _stage_label(self, writer: AgentInputWriter) -> None:
        n = writer.write_label_inputs()
        print(f"[3/5] label agents: queued {n} features")
        self._run_agent_task(
            task_name=self.config.agents.label_task,
            workers=self.config.agents.label_workers,
            reps=self.config.agents.label_reps_per_worker,
        )

    def _stage_eval(self, writer: AgentInputWriter) -> None:
        info = writer.write_evaluator_inputs()
        print(
            f"[4/5] eval agents: wrote {info['n_written']} inputs "
            f"(skipped {info['n_skipped_no_label']} with no label)",
        )
        self._run_agent_task(
            task_name=self.config.agents.eval_task,
            workers=self.config.agents.eval_workers,
            reps=self.config.agents.eval_reps_per_worker,
        )

    def _stage_score(self) -> None:
        print("[5/5] score")
        scorer = AutoInterpretScorer(
            self.run_dir,
            self.config.score,
            self.config.collect,
            self.config.agents,
        )
        scores = scorer.score_all()
        if scores.empty:
            print("  no scored features")
            return
        print(
            f"  scored {len(scores)} features "
            f"(mean Pearson = {scores['pearson'].mean():.3f})"
        )
        ab = scorer.report_ab_split(scores)
        if ab is not None:
            print(ab)
        written = scorer.push_to_label_store(scores)
        if written:
            print(f"  pushed {written} labels to FeatureLabelStore")

    # ── Agent subprocess helpers ────────────────────────────────────────

    def _run_agent_task(self, task_name: str, workers: int, reps: int) -> None:
        """Init the queue, launch agents, then poll until complete."""
        self._queue_cmd(task_name, ["init"], check=True)
        subprocess.run(
            ["bash", str(LAUNCHER), "-t", task_name, "-n", str(workers), "-r", str(reps)],
            check=True,
        )
        self._wait_until_done(task_name)

    def _queue_cmd(self, task_name: str, extra: list[str], check: bool = True):
        cmd = [sys.executable, str(JOB_QUEUE), "--task", task_name, *extra]
        return subprocess.run(cmd, check=check, capture_output=True, text=True)

    def _wait_until_done(self, task_name: str) -> None:
        interval = max(1, self.config.agents.poll_interval_seconds)
        while True:
            result = self._queue_cmd(task_name, ["status", "--json"], check=False)
            if result.returncode != 0:
                print(result.stderr.strip())
                time.sleep(interval)
                continue
            status = json.loads(result.stdout)
            pending = int(status.get("items_pending", 0))
            in_progress = int(status.get("items_in_progress", 0))
            completed = int(status.get("items_completed", 0))
            total = int(status.get("items_total", 0))
            print(
                f"  [{task_name}] completed {completed}/{total} "
                f"(in_progress={in_progress}, pending={pending})"
            )
            if pending == 0 and in_progress == 0:
                return
            time.sleep(interval)


# ── Entry point ──────────────────────────────────────────────────────────────

def run_from_yaml(path: Path | str) -> None:
    for cfg in load_experiments(Path(path)):
        print(f"\n=== experiment: {cfg.resolved_slug()} ===")
        AutoInterpretRunner(cfg).run()


def main() -> None:
    """Default run. Edit ``inputs.config_path`` to point at your YAML."""
    inputs = RunnerInputs(
        config_path=PROJECT_ROOT / "configs" / "autointerpret" / "debug_L29_16k.yaml",
    )
    run_from_yaml(inputs.config_path)


if __name__ == "__main__":
    main()
