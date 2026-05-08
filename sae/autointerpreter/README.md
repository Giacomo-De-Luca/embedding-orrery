# SAE Autointerpreter

Generate and evaluate per-feature labels for Gemma-scope SAEs using WordNet
as a probing corpus and the `scripts/AgentSystem/` concurrent job queue for
two roles of LLM agents (LabelInterpreter + Evaluator).

## Pipeline

```
collect → extract → label-agents → eval-agents → score
  │           │           │             │           │
  │           │           │             │           └─ scores.parquet (+ push to FeatureLabelStore)
  │           │           │             └─ evaluator/feature_*.json  (0-10 predictions)
  │           │           └─ labels/feature_*.json  (short_name + explanation)
  │           └─ topk/ + linspace/ feature_*.json
  └─ activations.npz (CSR) + index.parquet
```

All stages are driven from a single YAML/JSON config — see
`configs/autointerpret/debug_L29_16k.yaml`. Run with:

```bash
uv run python -m interpret.sae.autointerpreter.run_autointerpret
```

Edit the path in `main()` (or call `run_from_yaml(path)` directly) to pick
a different experiment.

## Files

| File | Class | Purpose |
|---|---|---|
| `config.py` | `AutoInterpretConfig` + sub-configs, `load_experiments()` | YAML/JSON loader. Supports single experiment or `experiments:` sweep list. |
| `sparse_activation_store.py` | `SparseActivationStore` | Incremental `scipy.sparse` CSR builder + parallel `index.parquet`. |
| `collect_activations.py` | `ActivationCollector` | Stage 1: Gemma forward per WordNet entry, SAE hook, aggregate to one vector. |
| `extract_top_k.py` | `TopKFeatureExtractor` | Stage 2: per-feature top-k + `np.linspace`-over-sorted samples. |
| `prepare_agent_inputs.py` | `AgentInputWriter` | Populates AgentSystem input folders. Handles zero-hint A/B split. |
| `score_autointerpret.py` | `AutoInterpretScorer` | Pearson/Spearman per feature; pushes good labels to `FeatureLabelStore`. |
| `run_autointerpret.py` | `AutoInterpretRunner` | Orchestrates the stages, shells out to AgentSystem launchers. |

## Config knobs

All configurable — nothing hardcoded:

- **SAE**: `layer_index`, `hook_type`, `width` (16k/65k/262k), `l0_size`, `variant`.
- **Aggregation**: `last_token` | `mean_prefill` | `max_prefill`.
- **Corpus scope**: `limit`, `pos_filter`, `prompt_template`, `use_chat_template`.
- **Feature selection**: `density_min/max`, `require_min_nonzero`, explicit `feature_indices` list.
- **Agents**: worker count, reps per worker, `show_zero_fraction_to_evaluator` (`on`/`off`/`ab`).
- **Scoring**: `min_pearson`, whether to push back to `FeatureLabelStore`.

## Output layout

```
resources/sae_autointerpret/{model_id}_L{layer}_{hook}_w{width}_{aggregation}/
    experiment.yaml              # driving config (reproducibility)
    activations.npz              # scipy.sparse CSR
    index.parquet                # row_idx → word, synset_id, definition, prompt
    feature_stats.parquet        # density, mean_nonzero, nnz per feature
    topk/feature_*.json          # interpreter input (top-k activating samples)
    linspace/feature_*.json      # evaluator ground truth (_true_activations hidden)
    ab_split.parquet             # which features saw the zero-fraction hint
    scores.parquet               # per-feature Pearson/Spearman + labels
```

## Agents

- `.claude/agents/sae-label-interpreter.md` — reads `topk/feature_*.json`,
  writes `{short_name, explanation, polarity}`.
- `.claude/agents/sae-label-evaluator.md` — reads label + shuffled samples,
  scores each 0–10; scorer computes correlation against true activations.

Task configs: `scripts/AgentSystem/tasks/autointerpret-{label,eval}.json`.

## Notes

- Storage is `scipy.sparse` CSR, not Qdrant — the query pattern is
  "top-k rows per column" which Qdrant doesn't index for. Per-row meta
  lives in the parallel `index.parquet` keyed by `row_idx`.
- The collector calls `generate_from_template(prompt, output_len=1)` so the
  raw `"{word}: {definition}."` prompt is not wrapped in chat markers by
  default (toggle `use_chat_template` to compare).
