"""JailbreakBench substring-ASR evaluation with shared-baseline cache.

The ``baseline`` condition is just plain generation (no hooks) and so is
identical across all experiments that share the same model + eval dataset
+ sample budget. We cache its per-prompt CSV under
``output_dir_root/_baselines/`` so subsequent experiments only need to
generate the per-experiment ``actadd`` condition — saving ~47 minutes per
experiment on the full 100-prompt JailbreakBench × 256-token settings.

The ``actadd`` condition runs through the refusal pipeline's existing
``evaluate_dataset`` evaluator (different direction per experiment, no
caching makes sense). Skips the ``ablation`` condition entirely (broken on
Gemma-3, see refusal report).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from interpret.experiments.poetry_directions.config import PoetryConfig
from interpret.experiments.refusal_directions import data as refusal_data
from interpret.experiments.refusal_directions.evaluate import evaluate_dataset, is_refusal
from interpret.experiments.refusal_directions.tokens import format_chat
from interpret.utils.results_io import append_csv


def _baseline_completions_cache_path(cfg: PoetryConfig) -> Path:
    """Cache path for baseline completions, keyed by params that affect them.

    Shared across all experiments that consume the same model + eval dataset
    + n_eval + max_new_tokens + seed.
    """
    model_alias = cfg.model_name.replace("/", "_")
    n_eval_str = "all" if cfg.n_eval is None else str(cfg.n_eval)
    fname = (
        f"{cfg.eval_dataset}_baseline_{model_alias}"
        f"_n{n_eval_str}_max{cfg.max_new_tokens}_seed{cfg.seed}.csv"
    )
    return cfg.output_dir_root / "_baselines" / fname


def _generate_baseline_csv(
    wrapper,
    dataset: list[dict],
    out_path: Path,
    cfg: PoetryConfig,
) -> float:
    """Generate baseline (no-hook) completions for `dataset`, write CSV, return refusal rate.

    Mirrors the per-condition loop in `interpret.experiments.refusal_directions.evaluate.evaluate_dataset`
    for the ``baseline`` condition only. Uses ``append_csv`` for per-prompt
    checkpointing so a Ctrl-C preserves progress in the cache.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    refusals: list[int] = []
    for item in tqdm(dataset, desc=f"{cfg.eval_dataset}/baseline (cached)"):
        prompt = item["instruction"]
        response = wrapper.generate_from_template(
            format_chat(wrapper, prompt),
            output_len=cfg.max_new_tokens,
            temperature=None,
        )
        refusal = int(is_refusal(response))
        refusals.append(refusal)
        append_csv(
            out_path,
            {
                "category": item.get("category"),
                "prompt": prompt,
                "response": response,
                "is_refusal": refusal,
            },
        )
    return sum(refusals) / max(len(refusals), 1)


def _refusal_rate_from_csv(csv_path: Path) -> float:
    import pandas as pd

    df = pd.read_csv(csv_path)
    return float(df["is_refusal"].mean())


def _ensure_baseline(
    wrapper,
    dataset: list[dict],
    cfg: PoetryConfig,
) -> float:
    """Get-or-compute the JailbreakBench baseline; place CSV in per-experiment dir.

    Looks up the shared cache; if present, copies to ``cfg.completions_dir``.
    Otherwise generates from scratch into the cache, then copies.
    """
    cache_path = _baseline_completions_cache_path(cfg)
    target_path = cfg.completions_dir / f"{cfg.eval_dataset}_baseline.csv"
    cfg.completions_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"[evaluate] reusing cached baseline at {cache_path}")
        shutil.copyfile(cache_path, target_path)
        return _refusal_rate_from_csv(target_path)

    print(f"[evaluate] computing baseline (will cache at {cache_path})")
    rate = _generate_baseline_csv(wrapper, dataset, cache_path, cfg)
    shutil.copyfile(cache_path, target_path)
    return rate


def evaluate_jailbreakbench(
    wrapper,
    direction: torch.Tensor,
    selected_layer: int,
    coefficient: float,
    cfg: PoetryConfig,
) -> dict[str, float]:
    """Run baseline + actadd(`coefficient`) on the JailbreakBench harmful set.

    Returns ``{"baseline": rate, "actadd": rate}`` and writes per-prompt CSVs
    plus a per-dataset summary under ``cfg.completions_dir``. The baseline
    is shared across experiments via a cache under ``_baselines/``.
    """
    dataset = refusal_data.load_eval_dataset(cfg.eval_dataset, cfg.eval_dir)
    if cfg.n_eval is not None:
        dataset = refusal_data.sample(dataset, cfg.n_eval, cfg.seed)

    # 1. Baseline — shared cache.
    baseline_rate = _ensure_baseline(wrapper, dataset, cfg)

    # 2. actadd — per-experiment, no caching.
    summary_path = cfg.completions_dir / f"{cfg.eval_dataset}_summary.json"
    actadd_csv = cfg.completions_dir / f"{cfg.eval_dataset}_actadd.csv"
    if actadd_csv.exists():
        # Reuse existing actadd CSV if present (per-experiment idempotency).
        actadd_rate = _refusal_rate_from_csv(actadd_csv)
    else:
        actadd_rates = evaluate_dataset(
            wrapper,
            dataset,
            direction,
            selected_layer,
            cfg,
            dataset_label=cfg.eval_dataset,
            conditions={"actadd": coefficient},
        )
        actadd_rate = actadd_rates["actadd"]

    refusal_rates = {"baseline": baseline_rate, "actadd": actadd_rate}
    with summary_path.open("w") as f:
        json.dump(refusal_rates, f, indent=2)
    return refusal_rates
