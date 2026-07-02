"""End-to-end orchestrator for the refusal-direction replication."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from interpret.experiments.directions_common import DirectionModel, build_direction_model
from interpret.experiments.refusal_directions import data
from interpret.experiments.refusal_directions.config import RefusalConfig
from interpret.experiments.refusal_directions.evaluate import evaluate_dataset
from interpret.experiments.refusal_directions.generate_directions import generate_directions
from interpret.experiments.refusal_directions.select_direction import select_direction


class RefusalRunner:
    """Drive the four-phase pipeline.

    Phases:

    1. Load the model (Gemma or Qwen via the adapter), resolve refusal + EOI
       tokens, sample harmful/harmless splits.
    2. Mean-of-difference candidate directions over `cfg.intermediates`.
    3. Three-metric sweep + selection of the best direction.
    4. Greedy completions on the JailbreakBench eval set under
       {baseline, ablation, actadd}, plus harmless test under baseline + actadd(+1).

    Each phase is idempotent: artifacts on disk are reused on rerun.
    """

    def __init__(
        self, config: RefusalConfig, model: DirectionModel | None = None
    ) -> None:
        self.config = config
        self._model = model
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

    def _save_config(self) -> None:
        path = self.config.output_dir / "config.json"
        with path.open("w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def run(self) -> Path:
        cfg = self.config

        model = self._model or build_direction_model(cfg.model_name)

        # Backend-derived dims override the (Gemma-shaped) config defaults so the
        # pipeline is correct on any model. Refusal ids are resolved against the
        # live tokenizer (Gemma verifies; Qwen recomputes).
        cfg.n_layers = model.n_layers
        cfg.d_model = model.d_model
        cfg.refusal_token_ids = tuple(model.refusal_token_ids(cfg.refusal_token_ids))
        self._save_config()

        eoi_ids = model.eoi_token_ids()
        n_eoi = len(eoi_ids)

        with (cfg.output_dir / "tokens.json").open("w") as f:
            json.dump(
                {
                    "eoi_token_ids": eoi_ids,
                    "n_eoi": n_eoi,
                    "refusal_token_ids": list(cfg.refusal_token_ids),
                },
                f,
                indent=2,
            )

        # 1. Sample data
        harmful_train = data.sample(
            data.load_split("harmful", "train", cfg.splits_dir),
            cfg.n_train,
            cfg.seed,
        )
        harmless_train = data.sample(
            data.load_split("harmless", "train", cfg.splits_dir),
            cfg.n_train,
            cfg.seed,
        )
        harmful_val = data.sample(
            data.load_split("harmful", "val", cfg.splits_dir),
            cfg.n_val,
            cfg.seed,
        )
        harmless_val = data.sample(
            data.load_split("harmless", "val", cfg.splits_dir),
            cfg.n_val,
            cfg.seed,
        )

        # 2. Generate candidate directions
        candidates = generate_directions(
            model,
            data.instructions_only(harmful_train),
            data.instructions_only(harmless_train),
            cfg,
            n_eoi=n_eoi,
        )

        # 3. Select the best direction
        pos_labels = [str(tid) for tid in eoi_ids]
        selection = select_direction(
            model,
            candidates,
            data.instructions_only(harmful_val),
            data.instructions_only(harmless_val),
            pos_labels,
            cfg,
        )
        direction = selection["direction"].to(torch.float32)
        selected_layer = int(selection["layer"])

        # 4. Evaluate on JailbreakBench harmful set
        harmful_eval = data.load_eval_dataset(cfg.eval_dataset, cfg.eval_dir)
        if cfg.n_eval is not None:
            harmful_eval = data.sample(harmful_eval, cfg.n_eval, cfg.seed)
        harmful_rates = evaluate_dataset(
            model,
            harmful_eval,
            direction,
            selected_layer,
            cfg,
            dataset_label=cfg.eval_dataset,
            conditions={"baseline": 0.0, "ablation": 0.0, "actadd": -1.0},
        )

        # 5. Evaluate on harmless test set under baseline + actadd(+1)
        harmless_test = data.sample(
            data.load_split("harmless", "test", cfg.splits_dir),
            cfg.n_test,
            cfg.seed,
        )
        harmless_rates = evaluate_dataset(
            model,
            harmless_test,
            direction,
            selected_layer,
            cfg,
            dataset_label="harmless",
            conditions={"baseline": 0.0, "actadd": 1.0},
        )

        summary = {
            "selection": {k: v for k, v in selection.items() if k != "direction"},
            "harmful_eval_refusal_rates": harmful_rates,
            "harmless_test_refusal_rates": harmless_rates,
            "n_eoi": n_eoi,
        }
        with (cfg.output_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        return cfg.output_dir
