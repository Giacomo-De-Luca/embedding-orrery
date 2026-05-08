"""End-to-end orchestrator for one poetry-direction experiment."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from interpret.experiments.poetry_directions.config import PoetryConfig
from interpret.experiments.poetry_directions.evaluate import evaluate_jailbreakbench
from interpret.experiments.poetry_directions.extract import extract_direction
from interpret.experiments.poetry_directions.sweep import sweep_layers_coeffs
from interpret.experiments.refusal_directions import data as refusal_data
from interpret.experiments.refusal_directions.tokens import (
    compute_eoi_token_ids,
    verify_refusal_tokens,
)
from interpret.inference.gemma_pytorch import GemmaPytorchInference


class PoetryRunner:
    """Three-phase orchestrator: extract → sweep → evaluate.

    Each phase is idempotent: artifacts on disk are reused on rerun.
    Optionally accepts a pre-loaded ``GemmaPytorchInference`` so the same
    model instance can be reused across multiple experiments without
    reloading weights.
    """

    def __init__(
        self,
        config: PoetryConfig,
        wrapper: GemmaPytorchInference | None = None,
    ) -> None:
        self.config = config
        self._wrapper = wrapper
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

    def _save_config(self) -> None:
        with (self.config.output_dir / "config.json").open("w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def _ensure_wrapper(self) -> GemmaPytorchInference:
        if self._wrapper is None:
            self._wrapper = GemmaPytorchInference(self.config.model_name)
        return self._wrapper

    def run(self) -> Path:
        cfg = self.config
        wrapper = self._ensure_wrapper()
        verify_refusal_tokens(wrapper, ids=cfg.refusal_token_ids)
        eoi_ids = compute_eoi_token_ids(wrapper)
        n_eoi = len(eoi_ids)
        with (cfg.output_dir / "tokens.json").open("w") as f:
            json.dump({"eoi_token_ids": eoi_ids, "n_eoi": n_eoi}, f, indent=2)

        # 1. Extract per-intermediate mean-diff directions.
        candidates = extract_direction(wrapper, cfg, n_eoi=n_eoi)

        # 2. Sweep (intermediate, position, layer, coefficient) on val splits
        #    borrowed read-only from the refusal pipeline.
        harmful_val = refusal_data.instructions_only(
            refusal_data.sample(
                refusal_data.load_split("harmful", "val", cfg.splits_dir),
                cfg.n_val,
                cfg.seed,
            )
        )
        harmless_val = refusal_data.instructions_only(
            refusal_data.sample(
                refusal_data.load_split("harmless", "val", cfg.splits_dir),
                cfg.n_val,
                cfg.seed,
            )
        )
        pos_labels = [str(tid) for tid in eoi_ids]
        selection = sweep_layers_coeffs(
            wrapper,
            candidates,
            harmful_val,
            harmless_val,
            pos_labels,
            cfg,
        )
        direction = selection["direction"].to(torch.float32)
        selected_layer = int(selection["layer"])
        selected_coeff = float(selection["coefficient"])

        # 3. JailbreakBench substring-ASR eval at the chosen (layer, coeff).
        eval_rates = evaluate_jailbreakbench(
            wrapper,
            direction,
            selected_layer,
            selected_coeff,
            cfg,
        )

        summary = {
            "selection": {k: v for k, v in selection.items() if k != "direction"},
            "jailbreakbench_refusal_rates": eval_rates,
            "n_eoi": n_eoi,
        }
        with (cfg.output_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        return cfg.output_dir
