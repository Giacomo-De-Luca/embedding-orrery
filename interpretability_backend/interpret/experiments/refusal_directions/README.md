# `interpret/experiments/refusal_directions/`

Replicates Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (arXiv:2406.11717) on `google/gemma-3-4b-it`, reusing the project's existing Gemma-3 PyTorch wrapper and `HookManager` / `SteeringOp` system.

The reference upstream lives at [`references/refusal_direction/`](../../../references/refusal_direction/) — we adapt only its two scientific scripts (`generate_directions.py`, `select_direction.py`) and reuse the dataset (relocated to [`resources/refusal_direction/`](../../../resources/refusal_direction/)).

## Folder structure

| File | Purpose |
|---|---|
| `config.py` | `RefusalConfig` dataclass — paths, sample counts, intermediates, refusal token IDs. |
| `data.py` | Load `harmful_*` / `harmless_*` splits and processed eval sets from `resources/refusal_direction/`. |
| `tokens.py` | Compute end-of-instruction (EOI) token IDs from the chat template; verify refusal token IDs against the SentencePiece tokenizer. |
| `generate_directions.py` | Mean-of-difference candidate directions over `(intermediate, position, layer)`. |
| `select_direction.py` | Three-metric sweep (bypass / induce / KL on harmless) + filter + best-direction selection. Produces matplotlib plots. |
| `evaluate.py` | Generate completions under {`baseline`, `ablation`, `actadd`} and score with substring-matching judge. |
| `runner.py` | `RefusalRunner` orchestrates the four phases, idempotent per phase. |
| `run.py` | Thin driver: instantiate config + `RefusalRunner(cfg).run()`. |

## Quick start

```python
from interpret.experiments.refusal_directions import RefusalConfig, RefusalRunner

cfg = RefusalConfig()                 # defaults: pre_attn, n_train=128, n_val=32, JailbreakBench
RefusalRunner(cfg).run()
# outputs under resources/experiments/refusal_directions/
```

For a fast smoke test:

```python
RefusalRunner(RefusalConfig(n_train=4, n_val=2, n_test=2)).run()
```

## Phases (mapped to reference)

1. **Generate directions** — `generate_directions.py` adapts [reference generate_directions.py](../../../references/refusal_direction/pipeline/submodules/generate_directions.py). For each `intermediate` in `cfg.intermediates`, runs forward passes on `cfg.n_train` harmful and harmless prompts under `wrapper.cache_activations(...)` and saves `mean_diffs_<intermediate>.pt` of shape `(n_eoi_pos, n_layers, d_model)`.
2. **Select direction** — `select_direction.py` adapts [reference select_direction.py](../../../references/refusal_direction/pipeline/submodules/select_direction.py). For every `(intermediate, pos, layer)` candidate, computes:
   - **Bypass score** (refusal score on harmful_val) under three-site ablation. We register `3 * n_layers` `SteeringOp`s — one per `(layer, hook_type)` for `hook_type ∈ {RESID_POST, ATTN_OUT, MLP_OUT}` — using `mode=ABLATION, strength=0.0`. This matches the reference's [`get_all_direction_ablation_hooks`](../../../references/refusal_direction/pipeline/utils/hook_utils.py#L80-L88).
   - **Induce score** (refusal score on harmless_val) under a single `ADDITIVE` op at the candidate's source layer.
   - **KL on harmless_val** between baseline last-position logits and ablated last-position logits.
   Filters: drop the top 20 % of layers (paper avoids unembedding-direction interference); drop directions with KL > 0.1 or induce score < 0. Saves the chosen `direction.pt` and metadata.
3. **Evaluate** — `evaluate.py` greedy-generates JailbreakBench completions for each of `{baseline, ablation, actadd}` and scores them with the substring judge ported from [reference evaluate_jailbreak.py](../../../references/refusal_direction/pipeline/submodules/evaluate_jailbreak.py#L16-L29). Per-sample CSVs are checkpointed via `interpret.utils.results_io.append_csv`.
4. **Summarise** — aggregate ASR across conditions into `summary.json`.

## Outputs

All artifacts under `resources/experiments/refusal_directions/`:

```
resources/experiments/refusal_directions/
├── config.json
├── generate_directions/
│   ├── mean_diffs_pre_attn.pt        # one per cfg.intermediates entry
│   └── metadata.json
├── select_direction/
│   ├── direction_evaluations.json
│   ├── direction_evaluations_filtered.json
│   ├── ablation_scores.png
│   ├── actadd_scores.png
│   └── kl_div_scores.png
├── direction.pt
├── direction_metadata.json
├── completions/
│   ├── jailbreakbench_baseline.csv
│   ├── jailbreakbench_ablation.csv
│   ├── jailbreakbench_actadd.csv
│   └── harmless_baseline.csv
└── summary.json
```

The directory is reused across runs; sub-phases overwrite their own artifacts so re-running with the same config is idempotent.

## Known divergences from the reference

- **Ablation site offset.** The reference attaches forward-pre-hooks on each decoder layer (operating on the layer *input*); this project attaches forward-hooks on layer *output*, plus the same on `attn_out` and `mlp_out` sub-modules. The two coincide on every inter-layer boundary; they differ only at layer-0 input (reference ablates, we don't) and the residual after the final layer (we ablate, reference doesn't). Net effect on Gemma-3-4b is a 1-of-34 site shift. See the docstring on `_ablation_ops` in [select_direction.py](select_direction.py).
- **Chat-template suffix.** The reference encodes `<end_of_turn>\n<start_of_turn>model\n`. `GemmaPytorchInference.format_prompt` matches this; `GemmaPytorchInference.generate` strips the trailing newline. We always feed the model via `generate_from_template(format_prompt(...))` to keep the suffix consistent with the paper.

## Falling back if a single direction is weak

Gemma-3 is more aggressively safety-tuned than Gemma 1/2. If `intermediates=("pre_attn",)` does not yield a direction with bypass-score improvement of at least ~30 percentage points over baseline, widen the candidate pool:

```python
RefusalConfig(intermediates=("pre_attn", "post_attn", "mlp_out", "post_mlp"))
```

The selection sweep cost grows ×4 in candidate count, but the architecture is unchanged.
