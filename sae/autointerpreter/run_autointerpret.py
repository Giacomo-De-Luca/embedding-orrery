"""Orchestrate the full autointerpreter pipeline from a YAML config.

Entry point for one or more experiments. Stages::

    collect → extract → label-agents → eval-agents → score

Any stage can be skipped via ``stages.skip_*`` flags, and each experiment
runs in its own ``run_dir = output_root/<run_slug>/``.

The agent stages shell out to the existing ``scripts/AgentSystem`` job
queue and launcher, then poll for completion. No CLI flags — edit
:func:`main` or call :func:`run_from_yaml` directly.

Multi-SAE runs:
- ``collect`` runs once and produces one ``SparseActivationStore`` per
  SAE in its own ``<sae_subdir>`` underneath ``run_dir``.
- ``extract / label / eval / score`` iterate per-SAE: each per-SAE call
  uses the subdir as its run_dir and a per-SAE view of the collect
  config (flat fields rewritten to match the spec) so the existing
  single-SAE downstream code paths keep working unchanged.
- The AgentSystem launcher reads from / writes to **global** input /
  results directories (set by the task JSONs); the runner clears these
  before every per-SAE batch and copies the global results into the
  per-SAE subdir afterwards so the scorer reads from
  ``<sae_subdir>/labels`` and ``<sae_subdir>/evaluator``.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from interpret.sae.autointerpreter.collect_activations import ActivationCollector
from interpret.sae.autointerpreter.collect_embeddings import EmbeddingCollector
from interpret.sae.autointerpreter.collect_residuals import ResidualCollector
from interpret.sae.autointerpreter.config import (
    PROJECT_ROOT,
    AgentStageConfig,
    AutoInterpretCollectConfig,
    AutoInterpretConfig,
    dump_yaml,
    load_experiments,
    residual_subdir,
    sae_unit_layout,
)
from interpret.sae.autointerpreter.extract_dense import DenseFeatureExtractor
from interpret.sae.autointerpreter.extract_top_k import TopKFeatureExtractor
from interpret.sae.autointerpreter.prepare_agent_inputs import AgentInputWriter
from interpret.sae.autointerpreter.score_autointerpret import AutoInterpretScorer

JOB_QUEUE = PROJECT_ROOT / "scripts" / "AgentSystem" / "job_queue.py"
LAUNCHER = PROJECT_ROOT / "scripts" / "AgentSystem" / "launch_agents.sh"


@dataclass
class RunnerInputs:
    """Minimal entry-point dataclass — points at the YAML/JSON to run."""

    config_path: Path


@dataclass
class _SaeUnit:
    """Per-SAE view of the run: where its artifacts live and how to find them.

    ``collect_cfg`` is a flat single-SAE collect-config copy so the
    existing downstream code paths (extract / score) see this SAE's
    metadata via the historical flat fields. ``agents_cfg`` overrides
    ``label_results_dir`` and ``eval_results_dir`` to per-SAE paths so
    the scorer reads the right batch.
    """

    sub_run_dir: Path
    collect_cfg: AutoInterpretCollectConfig
    agents_cfg: AgentStageConfig


class AutoInterpretRunner:
    """Run a single experiment end-to-end."""

    def __init__(self, config: AutoInterpretConfig) -> None:
        self.config = config
        # Honour the top-level ``run_slug:`` override when set so the
        # smoke/debug experiments don't collide on the same auto-derived
        # SAE-parameter slug. Falls back to ``collect.run_slug()``.
        slug = config.run_slug or config.collect.run_slug()
        self.run_dir = Path(config.collect.output_root) / slug
        self.sae_units = self._build_sae_units()

    def _build_sae_units(self) -> list[_SaeUnit]:
        """One unit per resolved SAE spec.

        Single-SAE runs keep the legacy layout: ``sub_run_dir == self.run_dir``
        and the global agents config is reused (no per-SAE results
        copies needed). Multi-SAE runs land each SAE under its own
        ``<sae_subdir>`` and override the agents-config results paths to
        point inside that subdir.

        Embedding runs are always a single source: one unit at ``run_dir``
        reusing the global agents config (same shape as a single-SAE run).

        Residual runs mirror the SAE shape — one unit per capture site, each
        with a collect-config narrowed to exactly that site so downstream
        stages read the right store and metadata.
        """
        if self.config.collect.source_kind == "embedding":
            return [
                _SaeUnit(
                    sub_run_dir=self.run_dir,
                    collect_cfg=self.config.collect,
                    agents_cfg=self.config.agents,
                ),
            ]
        if self.config.collect.source_kind == "residual":
            sites = self.config.collect.resolve_residual().sites
            multi = len(sites) > 1
            units = []
            for site in sites:
                sub = self.run_dir / residual_subdir(site) if multi else self.run_dir
                collect_cfg = dataclasses.replace(
                    self.config.collect,
                    residual=dataclasses.replace(
                        self.config.collect.resolve_residual(), sites=[site],
                    ),
                )
                if multi:
                    agents_cfg = dataclasses.replace(
                        self.config.agents,
                        label_results_dir=sub / "labels",
                        eval_results_dir=sub / "evaluator",
                    )
                else:
                    agents_cfg = self.config.agents
                units.append(
                    _SaeUnit(
                        sub_run_dir=sub,
                        collect_cfg=collect_cfg,
                        agents_cfg=agents_cfg,
                    ),
                )
            return units
        specs = self.config.collect.resolve_saes()
        aggregations = self.config.collect.resolve_aggregations()
        layout = sae_unit_layout(specs, aggregations)
        units: list[_SaeUnit] = []
        for spec, agg, sub_name in layout:
            sub = self.run_dir / sub_name if sub_name else self.run_dir
            # Rewrite the flat single-Gemma fields to match this spec
            # (Qwen specs land in the same fields — e.g. width="32k"
            # — so the historical flat-field-driven downstream still
            # reads sensible metadata).
            collect_cfg = dataclasses.replace(
                self.config.collect,
                layer_index=spec.layer_index,
                hook_type=spec.hook_type,
                width=spec.width,
                l0_size=spec.l0_size,
                model_size=spec.model_size,
                variant=spec.variant,
                # Narrow the per-SAE view to exactly this spec and this
                # aggregation. The flat fields carry no ``family`` or
                # ``k``, so letting ``resolve_saes()`` fall back to them
                # would rebuild every spec as Gemma (a Qwen run would then
                # file its labels under a fabricated Gemma model id in the
                # score stage).
                saes=[spec],
                aggregation=agg,
            )
            if sub_name:
                agents_cfg = dataclasses.replace(
                    self.config.agents,
                    label_results_dir=sub / "labels",
                    eval_results_dir=sub / "evaluator",
                )
            else:
                agents_cfg = self.config.agents
            units.append(_SaeUnit(sub_run_dir=sub, collect_cfg=collect_cfg, agents_cfg=agents_cfg))
        return units

    # ── Orchestration ──────────────────────────────────────────────────

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(self.config, self.run_dir / "experiment.yaml")

        if not self.config.stages.skip_collect:
            self._stage_collect()

        multi = len(self.sae_units) > 1
        for unit in self.sae_units:
            if multi:
                print(f"\n--- downstream: {unit.sub_run_dir.name} ---")
            writer = AgentInputWriter(unit.sub_run_dir, self.config.agents)
            if not self.config.stages.skip_topk:
                self._stage_extract(unit)
            if not self.config.stages.skip_label:
                self._stage_label(writer, unit)
            if not self.config.stages.skip_eval:
                self._stage_eval(writer, unit)
            if not self.config.stages.skip_score:
                self._stage_score(unit)

    # ── Stages ─────────────────────────────────────────────────────────

    def _stage_collect(self) -> None:
        kind = self.config.collect.source_kind
        if kind == "embedding":
            print(f"[1/5] collect (embedding) → {self.run_dir}")
            EmbeddingCollector(self.config.collect, run_dir=self.run_dir).run()
            return
        if kind == "residual":
            print(f"[1/5] collect (residual) → {self.run_dir}")
            ResidualCollector(self.config.collect, run_dir=self.run_dir).run()
            return
        print(f"[1/5] collect → {self.run_dir}")
        ActivationCollector(self.config.collect, run_dir=self.run_dir).run()

    def _stage_extract(self, unit: _SaeUnit) -> None:
        if self.config.collect.source_kind in ("embedding", "residual"):
            print(
                f"[2/5] extract dense ({self.config.extract.dim_mode}) "
                f"→ {unit.sub_run_dir}"
            )
            DenseFeatureExtractor(
                unit.sub_run_dir, self.config.extract, unit.collect_cfg,
            ).run()
            return
        print(f"[2/5] extract top-k + linspace → {unit.sub_run_dir}")
        TopKFeatureExtractor(
            unit.sub_run_dir, self.config.extract, unit.collect_cfg,
        ).run()

    def _stage_label(self, writer: AgentInputWriter, unit: _SaeUnit) -> None:
        # Clear and re-populate the global label-input dir so the
        # launcher only sees this SAE's feature files. ``_clear_json_dir``
        # wipes both untagged and tagged feature_*.json files.
        self._clear_json_dir(Path(self.config.agents.label_input_dir))
        n = writer.write_label_inputs()  # writes tagged: {tag}__feature_NNNNNN.json
        print(f"[3/5] label agents: queued {n} features")
        # The launcher writes results into the GLOBAL label_results_dir.
        # Clear it first so a previous SAE's results don't leak in.
        self._clear_json_dir(Path(self.config.agents.label_results_dir))
        self._reset_queue_manifest(self.config.agents.label_results_dir)
        self._run_agent_task(
            task_name=self.config.agents.label_task,
            workers=self.config.agents.label_workers,
            reps=self.config.agents.label_reps_per_worker,
        )
        # ALWAYS sync global → per-store, stripping the tag. Per-store
        # ``labels/`` is the persistent archive (clean filenames) that the
        # eval and score stages read from.
        self._sync_strip_tag(
            Path(self.config.agents.label_results_dir),
            unit.sub_run_dir / "labels",
            writer.tag,
        )

    def _stage_eval(self, writer: AgentInputWriter, unit: _SaeUnit) -> None:
        # Evaluator inputs come from the per-store labels/ dir (clean
        # filenames, populated by _stage_label sync). Writer tag matches
        # the label writer's tag (both default to sub_run_dir.name).
        per_sae_writer = AgentInputWriter(unit.sub_run_dir, unit.agents_cfg)
        self._clear_json_dir(Path(self.config.agents.eval_input_dir))
        info = per_sae_writer.write_evaluator_inputs()  # writes tagged inputs
        print(
            f"[4/5] eval agents: wrote {info['n_written']} inputs "
            f"(skipped {info['n_skipped_no_label']} with no label)",
        )
        self._clear_json_dir(Path(self.config.agents.eval_results_dir))
        self._reset_queue_manifest(self.config.agents.eval_results_dir)
        self._run_agent_task(
            task_name=self.config.agents.eval_task,
            workers=self.config.agents.eval_workers,
            reps=self.config.agents.eval_reps_per_worker,
        )
        # ALWAYS sync global → per-store, stripping the tag.
        self._sync_strip_tag(
            Path(self.config.agents.eval_results_dir),
            unit.sub_run_dir / "evaluator",
            per_sae_writer.tag,
        )

    def _archive_raw_outputs(self, unit: _SaeUnit) -> None:
        """Defensive secondary sync from global → per-store (strip tag).

        ``_stage_label`` and ``_stage_eval`` already sync — this is the
        safety net for when score is invoked alone (e.g. after a stage skip)
        and the per-store dirs need to be (re-)populated from whatever the
        global queue still holds for this unit's tag. Idempotent.
        """
        tag = unit.sub_run_dir.name
        for src_dir, name in (
            (Path(self.config.agents.label_results_dir), "labels"),
            (Path(self.config.agents.eval_results_dir), "evaluator"),
        ):
            self._sync_strip_tag(src_dir, unit.sub_run_dir / name, tag)

    def _stage_score(self, unit: _SaeUnit) -> None:
        print(f"[5/5] score: {unit.sub_run_dir.name}")
        self._archive_raw_outputs(unit)
        scorer = AutoInterpretScorer(
            unit.sub_run_dir,
            self.config.score,
            unit.collect_cfg,
            unit.agents_cfg,
        )
        scores = scorer.score_all()
        if scores.empty:
            print("  no scored features")
            return
        print(
            f"  scored {len(scores)} features "
            f"(mean Pearson = {scores['pearson'].mean():.3f})",
        )
        ab = scorer.report_ab_split(scores)
        if ab is not None:
            print(ab)
        written = scorer.push_to_label_store(scores)
        if written:
            print(f"  pushed {written} labels to FeatureLabelStore")

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _clear_json_dir(path: Path) -> None:
        """Remove feature-input/result files (idempotent).

        Matches both untagged ``feature_*.json`` and tagged
        ``{tag}__feature_*.json`` files so chain transitions don't leave
        stale alien-tag files in the global queue.
        """
        if not path.is_dir():
            return
        for f in path.glob("*feature_*.json"):
            f.unlink()

    @classmethod
    def _sync_strip_tag(cls, src: Path, dst: Path, tag: str) -> None:
        """Copy ``src/{tag}__feature_*.json`` → ``dst/feature_*.json``.

        Strips the tag prefix when copying so per-store dirs hold clean
        filenames. Only this tag's files are copied; alien-tag files in
        ``src`` are left in place (they belong to a different config).
        """
        dst.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            return
        from interpret.sae.autointerpreter.prepare_agent_inputs import TAG_SEP
        prefix = f"{tag}{TAG_SEP}"
        for f in src.glob(f"{prefix}feature_*.json"):
            shutil.copy2(f, dst / f.name[len(prefix):])

    @staticmethod
    def _reset_queue_manifest(results_dir: Path) -> None:
        """Delete the queue manifest so the next ``init`` rebuilds it fresh.

        ``init`` preserves existing item statuses, so a re-run whose input files
        reuse the prior run's filenames (e.g. the same feature indices) would
        inherit their ``done`` status and the agents would no-op. The manifest
        lives next to the results dir at ``<queue>/manifest.json``.
        """
        manifest = Path(results_dir).parent / "manifest.json"
        manifest.unlink(missing_ok=True)

    @classmethod
    def _sync_json_dir(cls, src: Path, dst: Path) -> None:
        """Copy ``feature_*.json`` from ``src`` to ``dst``, clearing ``dst`` first."""
        dst.mkdir(parents=True, exist_ok=True)
        cls._clear_json_dir(dst)
        if not src.is_dir():
            return
        for f in src.glob("feature_*.json"):
            shutil.copy2(f, dst / f.name)

    # ── Agent subprocess helpers ───────────────────────────────────────

    def _run_agent_task(self, task_name: str, workers: int, reps: int) -> None:
        """Init the queue, launch agents, then poll until complete."""
        self._queue_cmd(task_name, ["init"], check=True)
        cmd = [
            "bash", str(LAUNCHER), "-t", task_name,
            "-n", str(workers), "-r", str(reps),
        ]
        # Per-run model override (e.g. sonnet) — wins over the task JSON's model.
        if self.config.agents.model_override:
            cmd += ["-m", self.config.agents.model_override]
        subprocess.run(cmd, check=True)
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
            failed = int(status.get("items_failed", 0))
            total = int(status.get("items_total", 0))
            print(
                f"  [{task_name}] completed {completed}/{total} "
                f"(in_progress={in_progress}, pending={pending}, failed={failed})"
            )
            if pending == 0 and in_progress == 0:
                # The queue auto-resets only ``in_progress`` items past the
                # stale timeout; ``failed`` items keep ``pending`` at zero
                # but never re-enter the queue, so the loop would otherwise
                # spin forever. Surface them, then either raise or fall
                # through depending on ``fail_on_queue_errors``.
                if failed > 0:
                    msg = (
                        f"[{task_name}] {failed} item(s) finished in "
                        f"failed status. Inspect "
                        f"resources/jobs/{task_name}/queue/manifest.json, "
                        f"reset, and re-run the affected stage."
                    )
                    if self.config.agents.fail_on_queue_errors:
                        raise RuntimeError(msg)
                    print(f"  {msg}")
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
