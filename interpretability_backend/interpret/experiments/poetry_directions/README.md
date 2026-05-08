# `interpret/experiments/poetry_directions/`

Test the **adversarial-poetry jailbreak hypothesis** as a steering experiment on Gemma-3-4b-it: extract a "poetry vs prose" mean-of-difference direction from the residual stream, add it to harmful prose prompts, and measure whether refusal collapses. Built on the same `HookManager` + `SteeringOp` infrastructure validated for the refusal-direction experiment ([`interpret/experiments/refusal_directions/`](../refusal_directions/), report at [documentation/references/refusal_direction_report.md](../../../documentation/references/refusal_direction_report.md)).

## Three experiments

| Name | Class A (positive in mean-diff) | Class B (negative) | Source file |
|---|---|---|---|
| `poems_paraphrase` | 1000 poems | 1000 prose paraphrases of the same content | `paraphrased_poems_aligned.csv` |
| `poetry_prose` | 1154 poetry prompts (safe + unsafe) | 1151 prose prompts (safe + unsafe) | `alignment_dataset_long_clean.tsv` (filter `type=poetry` / `type=prose`) |
| `poetry_unsafe_safe` | 540 unsafe-poetry prompts | 614 safe-poetry prompts | `alignment_dataset_long_clean.tsv` (filter `type=poetry`, label `final_safety_status`) |

`poetry_prose` extracts the cleanest poetry-vs-prose direction by using **all** rows on both sides — no safety filter is applied, since safety is an orthogonal axis we don't want to confound with style. The TSV's `sample_id` pairs nearly every poetry row with a prose counterpart (3 poetry-only orphans), so set-level mean-diff and per-pair mean-diff are mathematically equivalent here; the extractor uses the set-level form for simplicity.

`prose_unsafe_safe` is **not** in the catalog. The new TSV has 122 unsafe-prose rows under `final_safety_status` (vs. zero in the legacy `all_prompts.csv`) — non-zero, but too small to extract a clean direction.

## Folder structure

| File | Purpose |
|---|---|
| `config.py` | `PoetryConfig` dataclass + `EXPERIMENTS` catalog + sweep / eval knobs. |
| `data.py` | `load_pairs` (paired CSV) and `load_filtered_subset` (filtered+deduped from a CSV/TSV — separator inferred from the file suffix). |
| `extract.py` | `extract_direction(...)` — per-`(intermediate, position, layer)` mean-diff in fp64, saved as `mean_diffs_<intermediate>.pt`. |
| `sweep.py` | `sweep_layers_coeffs(...)` — bypass score on `harmful_val` and KL on `harmless_val` over a `(position, layer, coefficient)` grid. Selects the best `(layer, coeff)` per the reused refusal-pipeline filter. |
| `evaluate.py` | `evaluate_jailbreakbench(...)` — substring-ASR via `interpret.experiments.refusal_directions.evaluate.evaluate_dataset`. |
| `runner.py` | `PoetryRunner` orchestrating extract → sweep → evaluate. Each phase is idempotent. |
| `run.py` | Driver — instantiates `PoetryConfig` for each of the three experiments and calls `PoetryRunner(cfg).run()`. |

## Quick start

Run all three experiments end-to-end:

```bash
uv run python -m interpret.experiments.poetry_directions.run
```

Or one experiment at a time:

```python
from interpret.experiments.poetry_directions import PoetryConfig, PoetryRunner

PoetryRunner(PoetryConfig(name="poems_paraphrase")).run()
```

Outputs land under `resources/experiments/poetry_directions/<name>/`:

```
resources/experiments/poetry_directions/<name>/
├── config.json
├── extract/
│   ├── mean_diffs_pre_attn.pt        # (n_eoi, 34, 2560)
│   └── metadata.json
├── sweep/
│   ├── direction_evaluations.json
│   ├── direction_evaluations_filtered.json
│   ├── refusal_scores_pre_attn.png
│   ├── kl_scores_pre_attn.png
│   └── bypass_grid_pre_attn.png
├── direction.pt
├── direction_metadata.json
├── completions/
│   ├── jailbreakbench_baseline.csv
│   └── jailbreakbench_actadd.csv
└── summary.json
```

## Smoke test

```bash
uv run python -c "
from interpret.experiments.poetry_directions import PoetryConfig, PoetryRunner
cfg = PoetryConfig(name='poems_paraphrase', max_per_class=4, n_val=2, n_eval=2, max_new_tokens=32)
PoetryRunner(cfg).run()
"
```

## Refusal-isolation guarantee

This pipeline writes only to `resources/experiments/poetry_directions/<name>/`. It **reads** from `resources/refusal_direction/{splits,processed}/` for the same val/test prompts the refusal experiment used (so the bypass numbers are directly comparable). It **never writes** to `resources/experiments/refusal_directions/`. Running this pipeline does not perturb the refusal direction's saved artifacts in any way.

## Interactive testing

[poetry_steer_tester.ipynb](../../../poetry_steer_tester.ipynb) at the repo root mirrors `refusal_steer_tester.ipynb`: load any of the three directions, steer at a chosen `(layer, coeff)`, eyeball generations on harmful/benign prompts. The notebook also has an optional cell for **stacked steering** — applying both `+poetry_direction` and `-refusal_direction` simultaneously to test composition.

## Sweep methodology

Same primitives as `select_direction.py` but with two important changes for this experiment:

- **Bypass intervention is `actadd` (additive), not `ablation`.** Full 102-site projection ablation collapses Gemma-3's residual stream (see refusal report §"Known divergences"). We reuse the working primitive from the refusal experiment.
- **Sign of the coefficient is unknown a priori.** The refusal direction's bypass coefficient is `-1` (subtract refusal → bypass); the poetry direction's bypass sign is empirical, so the sweep grid covers both signs: `(-1.0, -0.5, -0.3, +0.3, +0.5, +1.0)`.

Filter (mirrors refusal's `select_direction._filter`):
- discard if any of `(refusal, kl)` is NaN
- discard if `kl_score > kl_threshold` (default `0.5` — looser than refusal's `0.1` because additive on Gemma-3 perturbs more than ablation does on Gemma 1/2)
- discard if `source_layer >= int(n_layers · (1 − prune_layer_pct))` (top-20% layer cut, same as refusal)

Best surviving candidate (lowest refusal score) is saved as `direction.pt` along with `(intermediate, position, layer, coefficient)` metadata.
