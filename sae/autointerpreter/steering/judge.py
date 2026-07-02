"""Stages 2 & 3 — drive the LLM judge over the generations, then aggregate.

``AgentQueueDriver`` wraps the ``interpret/agent_system`` job queue + launcher (the
same mechanism ``AutoInterpretRunner`` uses, factored out as a runner-agnostic
class). ``SteeringJudgeInputWriter`` populates the judge's input folder (resetting
the manifest first, the single most common cause of a silent no-op re-run).
``SteeringAggregator`` collects the verdicts back into ``run_dir`` and writes the
side-by-side ``verdicts.parquet`` + ``summary.csv``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from interpret.sae.autointerpreter.config import AGENT_SYSTEM_DIR
from interpret.sae.autointerpreter.steering.config import (
    SteeringJudgeConfig,
    resolve_path,
)

JOB_QUEUE = AGENT_SYSTEM_DIR / "job_queue.py"
LAUNCHER = AGENT_SYSTEM_DIR / "launch_agents.sh"


class AgentQueueDriver:
    """Init a task queue, launch headless agent workers, and poll to completion.

    Runner-agnostic: takes the few knobs it needs directly rather than a config
    object, so any pipeline can reuse it. (Extracted from
    ``AutoInterpretRunner._run_agent_task`` & friends — see the steering README.)
    """

    def __init__(
        self,
        poll_interval_seconds: int = 15,
        model_override: str | None = None,
        fail_on_queue_errors: bool = False,
    ) -> None:
        self.poll_interval_seconds = poll_interval_seconds
        self.model_override = model_override
        self.fail_on_queue_errors = fail_on_queue_errors

    def run_task(self, task_name: str, workers: int, reps: int) -> None:
        """Init the queue, launch ``workers`` agents (``reps`` items each), poll."""
        self._queue_cmd(task_name, ["init"], check=True)
        cmd = [
            "bash", str(LAUNCHER), "-t", task_name,
            "-n", str(workers), "-r", str(reps),
        ]
        if self.model_override:
            cmd += ["-m", self.model_override]
        subprocess.run(cmd, check=True)
        self._wait_until_done(task_name)

    def _queue_cmd(self, task_name: str, extra: list[str], check: bool = True):
        cmd = [sys.executable, str(JOB_QUEUE), "--task", task_name, *extra]
        return subprocess.run(cmd, check=check, capture_output=True, text=True)

    def _wait_until_done(self, task_name: str) -> None:
        interval = max(1, self.poll_interval_seconds)
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
            failed = int(status.get("items_failed", 0))
            total = int(status.get("items_total", 0))
            print(
                f"  [{task_name}] completed {completed}/{total} "
                f"(in_progress={in_progress}, pending={pending}, failed={failed})"
            )
            if pending == 0 and in_progress == 0:
                # `failed` items never re-enter the queue, so the loop would
                # otherwise spin forever — surface them, then raise or fall
                # through depending on `fail_on_queue_errors`.
                if failed > 0:
                    msg = (
                        f"[{task_name}] {failed} item(s) finished in failed status. "
                        f"Inspect resources/jobs/{task_name}/queue/manifest.json, "
                        "reset, and re-run the judge stage (skip_generate: true)."
                    )
                    if self.fail_on_queue_errors:
                        raise RuntimeError(msg)
                    print(f"  {msg}")
                return
            time.sleep(interval)

    # ── Queue-folder hygiene (static so callers can use without an instance) ──

    @staticmethod
    def clear_json_dir(path: Path) -> None:
        """Remove any ``feature_*.json`` files in ``path`` (idempotent)."""
        path = Path(path)
        if not path.is_dir():
            return
        for f in path.glob("feature_*.json"):
            f.unlink()

    @staticmethod
    def reset_queue_manifest(results_dir: Path) -> None:
        """Delete ``<queue>/manifest.json`` so the next ``init`` rebuilds it fresh.

        ``init`` preserves existing item statuses; without this, a re-run whose
        input filenames match a prior run inherits their ``done`` status and the
        agents no-op silently.
        """
        Path(Path(results_dir).parent / "manifest.json").unlink(missing_ok=True)

    @classmethod
    def sync_json_dir(cls, src: Path, dst: Path) -> None:
        """Copy ``feature_*.json`` from ``src`` to ``dst``, clearing ``dst`` first."""
        src, dst = Path(src), Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        cls.clear_json_dir(dst)
        if not src.is_dir():
            return
        for f in src.glob("feature_*.json"):
            shutil.copy2(f, dst / f.name)


class SteeringJudgeInputWriter:
    """Populate the judge task's input folder from ``run_dir/generations``."""

    def __init__(self, run_dir: Path, judge: SteeringJudgeConfig) -> None:
        self.run_dir = Path(run_dir)
        self.input_dir = resolve_path(judge.input_dir)
        self.results_dir = resolve_path(judge.results_dir)

    def write(self) -> int:
        """Reset the queue, clear stale input/results, copy generations in."""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        AgentQueueDriver.reset_queue_manifest(self.results_dir)
        AgentQueueDriver.clear_json_dir(self.input_dir)
        AgentQueueDriver.clear_json_dir(self.results_dir)
        n = 0
        for f in sorted((self.run_dir / "generations").glob("feature_*.json")):
            shutil.copy2(f, self.input_dir / f.name)
            n += 1
        return n


class SteeringAggregator:
    """Collect verdicts into ``run_dir`` and merge them with activation-labels."""

    def __init__(self, run_dir: Path, judge: SteeringJudgeConfig) -> None:
        self.run_dir = Path(run_dir)
        self.results_dir = resolve_path(judge.results_dir)
        self.verdicts_dir = self.run_dir / "verdicts"

    def run(self) -> dict:
        import pandas as pd

        AgentQueueDriver.sync_json_dir(self.results_dir, self.verdicts_dir)
        rows: list[dict] = []
        n_missing = 0
        for gen_file in sorted((self.run_dir / "generations").glob("feature_*.json")):
            gen = json.loads(gen_file.read_text(encoding="utf-8"))
            verdict_path = self.verdicts_dir / gen_file.name
            if not verdict_path.exists():
                n_missing += 1
                continue
            v = json.loads(verdict_path.read_text(encoding="utf-8"))
            rows.append(self._merge_row(gen, v))

        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(self.run_dir / "verdicts.parquet", index=False)
            df[_SUMMARY_COLUMNS].to_csv(self.run_dir / "summary.csv", index=False)
        return {"n_scored": len(rows), "n_missing": n_missing}

    @staticmethod
    def _merge_row(gen: dict, v: dict) -> dict:
        steers = bool(v.get("steers"))
        broken = bool(v.get("broken"))
        act = gen.get("activation_label") or {}
        return {
            "feature_index": gen.get("feature_index"),
            "working_steering": steers and not broken,
            "steers": steers,
            "broken": broken,
            "steering_strength_0_10": v.get("steering_strength_0_10"),
            "confidence": v.get("confidence"),
            "steering_short_name": v.get("short_name"),
            "activation_short_name": act.get("short_name"),
            "steering_explanation": v.get("explanation"),
            "activation_explanation": act.get("explanation"),
            "activation_pearson": act.get("pearson"),
        }


# Scannable subset for summary.csv — both short_names side by side.
_SUMMARY_COLUMNS = [
    "feature_index",
    "working_steering",
    "steers",
    "broken",
    "steering_strength_0_10",
    "confidence",
    "steering_short_name",
    "activation_short_name",
]
