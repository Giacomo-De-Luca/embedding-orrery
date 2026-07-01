"""Stage 1 — steer each feature and generate answers to the probe questions.

For every selected feature we attach an additive steering op at the SAE's layer
and generate the model's answer to each fixed question across the strength sweep
(plus a single shared baseline at strength 0). One JSON per feature lands in
``run_dir/generations/`` for the judge to read and for manual inspection.

The steer-and-generate path is the one validated in
``scripts/scratch/bench_steered_generation.py`` (~9 tok/s on MPS): direction from
the decoder parquet -> ``SteeringOp(vector=…)`` -> ``HookManager.session`` ->
``GemmaPytorchInference.generate``.
"""

from __future__ import annotations

import gc
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.sae import HookManager, HookType, SteeringMode, SteeringOp
from interpret.sae.autointerpreter.steering.config import (
    SteeringInterpretConfig,
    resolve_path,
)

# Degeneracy heuristics (broken-steering signal handed to the judge).
_MIN_UNIQUE_TOKEN_RATIO = 0.2
_MAX_REPEAT_RUN = 10


class SteeringGenerator:
    """Run the steer-and-generate sweep for one experiment into ``run_dir``."""

    def __init__(self, config: SteeringInterpretConfig, run_dir: Path) -> None:
        self.cfg = config
        self.run_dir = Path(run_dir)
        self.gen_dir = self.run_dir / "generations"
        gen = config.generation
        self.questions = list(gen.questions)
        self.strengths = list(gen.strengths)
        self.max_tokens = gen.max_tokens
        self.temperature = gen.temperature
        self.layer = config.sae.layer_index
        self.hook = config.sae.hook_type
        self.width = config.sae.width

    # ── Orchestration ──────────────────────────────────────────────────────

    def run(self) -> None:
        self.gen_dir.mkdir(parents=True, exist_ok=True)
        features = self.cfg.resolve_features()
        todo = [f for f in features if not self._complete(f["feature_index"])]
        print(
            f"[generate] {len(features)} features "
            f"({len(features) - len(todo)} already done, {len(todo)} to generate)"
        )
        if not todo and self._baseline_path().exists():
            return

        wrapper = self._load_model()
        manager = HookManager()
        use_sae = self.cfg.direction.kind == "sae"
        if use_sae:
            sae_cfg = self.cfg.sae.to_sae_config()
            sae_cfg.device = self.cfg.base_model.device
            sae_cfg.dtype = self.cfg.base_model.dtype
            manager.add_sae(sae_cfg)

        try:
            directions = (
                {} if use_sae
                else self._load_directions([f["feature_index"] for f in todo])
            )
            baseline = self._ensure_baseline(wrapper)
            for f in todo:
                idx = f["feature_index"]
                vec = None if use_sae else directions.get(idx)
                if not use_sae and vec is None:
                    print(f"  feature {idx}: no decoder vector in parquet — skipped")
                    continue
                try:
                    self._generate_feature(wrapper, manager, f, vec, baseline)
                    print(f"  feature {idx}: written")
                except Exception as exc:
                    # Isolate per-feature failures: one transient MPS/hook error
                    # must not abort a multi-hour run. The feature is left
                    # incomplete (no file written) and retried on the next resume.
                    print(f"  feature {idx}: FAILED ({exc!r}) — skipped, retried on resume")
        finally:
            del wrapper, manager
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # ── Per-feature ────────────────────────────────────────────────────────

    def _generate_feature(
        self,
        wrapper: GemmaPytorchInference,
        manager: HookManager,
        feature: dict,
        vec: np.ndarray | None,
        baseline: list[dict],
    ) -> None:
        idx = feature["feature_index"]
        steered: list[dict] = []
        for strength in self.strengths:
            manager.clear_steering()
            manager.add_steering(self._make_op(idx, vec, strength))
            answers: list[dict] = []
            with manager.session(wrapper.model.model.layers):
                for q in self.questions:
                    answers.append(self._generate_one(wrapper, q))
            steered.append({"strength": strength, "answers": answers})

        payload = {
            "feature_index": idx,
            "layer": self.layer,
            "hook": self.hook,
            "width": self.width,
            "direction_source": self.cfg.direction.kind,
            "activation_label": feature.get("activation_label"),
            "questions": self.questions,
            "baseline": baseline,
            "steered": steered,
            "meta": self._meta(),
        }
        self._write_json(self._feature_path(idx), payload)

    def _make_op(self, idx: int, vec: np.ndarray | None, strength: float) -> SteeringOp:
        hook_type = HookType(self.hook)
        common = dict(
            layer_index=self.layer,
            mode=SteeringMode.ADDITIVE,
            strength=float(strength),
            normalise=self.cfg.generation.normalise,
            hook_type=hook_type,
        )
        if vec is not None:
            return SteeringOp(vector=torch.tensor(vec, dtype=torch.float32), **common)
        return SteeringOp(
            feature_index=idx, sae_key=(self.layer, hook_type), **common
        )

    def _generate_one(self, wrapper: GemmaPytorchInference, question: str) -> dict:
        try:
            text = wrapper.generate(
                question, output_len=self.max_tokens, temperature=self.temperature
            )
        except Exception as exc:  # one bad (feature, strength) must not abort the run
            return {"question": question, "text": None, "error": repr(exc)}
        tokens = wrapper.tokenize(text)
        return {
            "question": question,
            "text": text,
            "eos_early": len(tokens) < self.max_tokens,
            "degenerate": _is_degenerate(tokens),
        }

    # ── Baseline (strength 0, computed once, shared across features) ─────────

    def _ensure_baseline(self, wrapper: GemmaPytorchInference) -> list[dict]:
        path = self._baseline_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if len(data.get("baseline", [])) == len(self.questions):
                    return data["baseline"]
            except (json.JSONDecodeError, OSError):
                pass
        print("  baseline: generating (no steering)")
        baseline = [self._generate_one(wrapper, q) for q in self.questions]
        self._write_json(
            path,
            {"questions": self.questions, "baseline": baseline, "meta": self._meta()},
        )
        return baseline

    # ── Direction lookup ────────────────────────────────────────────────────

    def _load_directions(self, indices: list[int]) -> dict[int, np.ndarray]:
        import pandas as pd

        path = resolve_path(self.cfg.direction.w_dec_parquet_path)
        df = pd.read_parquet(path)
        need = set(indices)
        out: dict[int, np.ndarray] = {}
        for i, v in zip(df["index"].to_numpy(), df["vector"].to_numpy()):
            ii = int(i)
            if ii in need:
                out[ii] = np.asarray(v, dtype=np.float32)
        return out

    # ── Model ────────────────────────────────────────────────────────────────

    def _load_model(self) -> GemmaPytorchInference:
        bm = self.cfg.base_model
        return GemmaPytorchInference(
            bm.checkpoint,
            model_size=self.cfg.sae.model_size,
            precision=bm.dtype,
        )

    # ── Small helpers ─────────────────────────────────────────────────────────

    def _meta(self) -> dict:
        return {
            "model": self.cfg.base_model.checkpoint,
            "layer": self.layer,
            "hook": self.hook,
            "width": self.width,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "mode": self.cfg.generation.mode,
            "normalise": self.cfg.generation.normalise,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _feature_path(self, idx: int) -> Path:
        return self.gen_dir / f"feature_{idx:06d}.json"

    def _baseline_path(self) -> Path:
        return self.run_dir / "baseline.json"

    def _complete(self, idx: int) -> bool:
        """A feature is done only if its file parses with every strength × question."""
        path = self._feature_path(idx)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        steered = data.get("steered")
        if not isinstance(steered, list) or len(steered) != len(self.strengths):
            return False
        for tier in steered:
            answers = tier.get("answers")
            if not isinstance(answers, list) or len(answers) != len(self.questions):
                return False
        return bool(data.get("baseline"))

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        """Atomic write so a crash never leaves a half-written (resumable) file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def _is_degenerate(tokens: list[int]) -> bool:
    """Cheap broken-steering heuristic: empty, low token diversity, or long repeats."""
    if not tokens:
        return True
    unique_ratio = len(set(tokens)) / len(tokens)
    max_run = run = 1
    for prev, cur in zip(tokens, tokens[1:]):
        run = run + 1 if cur == prev else 1
        max_run = max(max_run, run)
    return unique_ratio < _MIN_UNIQUE_TOKEN_RATIO or max_run >= _MAX_REPEAT_RUN
