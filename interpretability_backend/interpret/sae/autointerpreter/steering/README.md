# Steering autointerpreter

Interpret SAE features by **intervention** rather than observation. For each
feature we add a scaled decoder direction into Gemma-3's residual stream during
generation, collect the model's answers to a few fixed probe questions across a
strength sweep, and have an LLM judge name and rate the induced behaviour. This
complements the read-the-top-k-activations autointerpreter in the parent package
— and surfaces behaviour the activation view can't (e.g. L9-w16k feature 1055
*reads* as "informal slang for excellent" but *steers* as a "casual response
starter").

Feasibility was measured in `scripts/scratch/bench_steered_generation.py`: ~9
tok/s on MPS, ~14s per 128-token generation, steering-hook overhead +3%. A
100-feature run at 4 strengths × 4 questions × 128 tokens ≈ 6.3h.

## Pipeline

```
generate (model resident) → tear down model → judge (AgentSystem) → aggregate
   │                                              │                    │
   │                                              │                    └─ verdicts.parquet + summary.csv (+ verdicts/)
   │                                              └─ verdict feature_*.json (queue results)
   └─ baseline.json + generations/feature_*.json
```

One YAML drives everything; no CLI flags. Stages are sequential (the 4B model is
released before the headless judge workers spawn) and each is independently
skippable via `stages.skip_*`.

```bash
uv run python -m interpret.sae.autointerpreter.steering.run   # edits: main() -> config path
```

## Files

| File | Class | Purpose |
|---|---|---|
| `config.py` | `SteeringInterpretConfig` + sub-configs, `load_steering_experiments()` | YAML loader (reuses the parent's `_coerce`/`dump_yaml`, `BaseModelSpec`, `SAESpec`); resolves feature indices + activation-labels from a scores parquet. |
| `generate.py` | `SteeringGenerator` | Stage 1: load model + decoder directions once; baseline once; per (feature × strength × question) steer + generate; one JSON per feature. Resume-safe, degeneracy-flagged. |
| `judge.py` | `AgentQueueDriver`, `SteeringJudgeInputWriter`, `SteeringAggregator` | Stages 2-3: drive the judge task through the AgentSystem queue, then merge verdicts with activation-labels. |
| `run.py` | `SteeringInterpretRunner`, `run_from_yaml()` | Orchestrates the three stages. |
| `label_similarity.py` | `LabelAgreementAnalyzer` | Post-hoc: embed the steering vs activation `short_name` (sentence-transformer cosine), add a `label_cosine` column, and report the agree/partial/diverge split. |

Agent + task (outside the package, fixed contracts):
`.claude/agents/sae-steering-judge.md`, `interpret/agent_system/tasks/steering-judge.json`.
Configs: `configs/autointerpret/steering_L9_16k.yaml` (overnight),
`smoke_steering_L9_16k.yaml` (smoke).

## Output layout (`resources/sae_autointerpret_steering/<slug>/`)

```
experiment.yaml      baseline.json
generations/feature_*.json     # baseline + steered texts + embedded activation_label
verdicts/feature_*.json        # judge verdicts (synced from the queue results)
verdicts.parquet  summary.csv  # steering verdict next to activation-label, side by side
                               # (+ label_cosine after running LabelAgreementAnalyzer)
```

Label agreement (after a run):

```python
from interpret.sae.autointerpreter.steering.label_similarity import LabelAgreementAnalyzer
LabelAgreementAnalyzer("resources/sae_autointerpret_steering/<slug>", device="mps").run()
```

## Direction source

`direction.kind: w_dec_parquet` (default) reads decoder rows from
`resources/sae_vectors/` — **no download**. Steering by `vector=` is identical to
`feature_index=` with `normalise: false`. `direction.kind: sae` instead loads the
SAE via `HookManager.add_sae` and steers by `feature_index` (downloads gemma-scope
weights on first use); use it for layer/width combos with no extracted parquet.

## Smoke test (run before the overnight job)

Three incremental passes over `smoke_steering_L9_16k.yaml` (2 features, 1
question, strengths [1000, 1400], 48 tokens), toggling `stages.*`:

1. `skip_judge + skip_aggregate` → model loads, baseline once, 2 generation files
   with the right schema + embedded `activation_label` + degeneracy flags. (~1 min, $0)
2. `skip_generate` (files resume-skipped) + judge on → input copy + manifest reset
   work; 1 worker judges 2 items; verdicts land in the results dir. (cents)
3. `skip_generate + skip_judge`, aggregate on → verdicts sync into `verdicts/`,
   `verdicts.parquet` + `summary.csv` written with `working_steering` and both
   short_names side by side.

## Notes

- **Queue manifest reset** is the #1 silent-failure guard: `init` preserves item
  statuses, so reusing `feature_*.json` names without resetting makes every
  `next` return `done`. `SteeringJudgeInputWriter.write()` resets it.
- `judge.input_dir` / `results_dir` must match the task JSON's `{variant}`→empty
  collapsed paths, or files land where the agents don't look.
- `AgentQueueDriver` is a runner-agnostic copy of `AutoInterpretRunner`'s queue
  helpers; the parent runner could later be migrated onto it to remove the
  duplication (then it would move up to `autointerpreter/`).
