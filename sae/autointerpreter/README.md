# SAE Autointerpreter

Generate and evaluate per-feature labels for **Gemma-scope and Qwen-scope
SAEs** using WordNet as a probing corpus and the `interpret/agent_system/`
concurrent job queue for two roles of LLM agents (LabelInterpreter +
Evaluator).

One collect pass can capture **multiple SAEs in a single forward pass** —
all SAEs attached to the same `(layer, hook_type)` site share one forward
hook, and SAEs at different sites compose freely. Cross-family (Gemma +
Qwen) is run as separate passes because the base LMs (and tokenizers)
differ.

This README is the **code reference** (file contract, config knobs, output
layout). For the experimental write-ups — method, protocol corrections, and
the current held-out results — see:

- [documentation/experiments/autointerpreter.md](../../../documentation/experiments/autointerpreter.md) — umbrella: method + all held-out results (SAE + embedding) + SAE-vs-embedding comparison.
- [documentation/experiments/embedding_dimension_autointerpret.md](../../../documentation/experiments/embedding_dimension_autointerpret.md) — the embedding-dimension monosemanticity control study in depth.

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
| `config.py` | `AutoInterpretConfig` + sub-configs, `load_experiments()` | YAML/JSON loader. Supports single experiment or `experiments:` sweep list. `source_kind` selects SAE vs embedding. |
| `wordnet_samples.py` | `WordNetSampleIterator` | Shared WordNet corpus walk (`word: definition.` prompts) used by **both** collectors. |
| `activation_store.py` | `ActivationStore` (ABC) | Source-agnostic store base: buffered append, append-only sharded flush, atomic `index.parquet`, resume. |
| `sparse_activation_store.py` | `SparseActivationStore` | `scipy.sparse` CSR shards (SAE features). |
| `dense_activation_store.py` | `DenseActivationStore` | Dense `.npy` shards (embedding dimensions; signed, no thresholding). |
| `collect_activations.py` | `ActivationCollector` | Stage 1 (SAE): Gemma/Qwen forward per WordNet entry, SAE hook, aggregate to one vector. |
| `collect_embeddings.py` | `EmbeddingCollector` | Stage 1 (embedding): sentence-transformer pooled vector per WordNet entry → dense store. |
| `collect_residuals.py` | `ResidualCollector` | Stage 1 (residual): Gemma forward per WordNet entry, raw hidden state at N residual sites → dense store(s). No SAE. |
| `extract_top_k.py` | `TopKFeatureExtractor` | Stage 2 (SAE): per-feature top-k + `np.linspace`-over-sorted samples. |
| `extract_dense.py` | `DenseFeatureExtractor` | Stage 2 (embedding): per-dimension stats + `signed`/`split` sample extraction. |
| `prepare_agent_inputs.py` | `AgentInputWriter` | Populates AgentSystem input folders. Handles zero-hint A/B split. **Source-agnostic.** |
| `score_autointerpret.py` | `AutoInterpretScorer` | Pearson/Spearman per feature; pushes good labels to `FeatureLabelStore` (SAE only). |
| `run_autointerpret.py` | `AutoInterpretRunner` | Orchestrates the stages, dispatches collect/extract on `source_kind`, shells out to AgentSystem launchers. |

### Embedding-dimension variant

The label / eval / score back-half is **source-agnostic** — it only reads the
`topk/` + `linspace/` JSON contract — so the same harness runs on the dense,
signed dimensions of a **sentence-transformer**, as a monosemanticity control
for SAE features. `collect` is mode-independent (it stores raw signed vectors);
`extract.dim_mode` then chooses how each dimension is presented:

- **`signed`** — one feature per dimension. `topk` shows the `top_k/2` most
  positive and `top_k/2` most negative samples (tagged `pole`); `linspace`
  spans the full signed range with signed `_true_activations`. Scored by the
  signed (−10..+10) agents (`embed-axis-{interpreter,evaluator}`).
- **`split`** — two non-negative half-features per dimension: `pos = max(0,x)`
  (feature index `2·dim`) and `neg = max(0,-x)` (`2·dim+1`). Each behaves like
  an SAE feature, reuses the 0–10 rubric (`embed-dim-{interpreter,evaluator}`),
  and is directly comparable to SAE features. Tests whether a dimension's +dir
  and −dir encode different things.

One collect pass feeds both modes: set `stages.skip_collect: true` and
`extract.activations_run_dir: <prior_run_dir>` to run the second mode without
re-embedding. Run with the embedding configs, e.g.:

```bash
# edit run_autointerpret.main() to point at the config, or:
uv run python -c "from interpret.sae.autointerpreter.run_autointerpret import run_from_yaml; \
  run_from_yaml('configs/autointerpret/smoke_embed_minilm.yaml')"
```

### Raw residual-dimension variant

`source_kind: residual` runs the same dense back-half on the **raw residual
stream of the base LM itself** — no SAE, no sentence-transformer. A
monosemanticity baseline directly comparable to SAE features: `post_mlp` at
layer L is the layer output, i.e. the very `resid_post` site the Gemma-scope
SAEs read. The collector (`ResidualCollector`) captures via the forked
`gemma_pytorch` activation cache, so **Gemma only** for now (the validator
rejects Qwen up front).

- `residual.sites` — list of `{layer_index, intermediate}` capture points;
  `intermediate` ∈ `pre_attn` / `post_attn` / `mlp_out` / `post_mlp` /
  `final_norm` (top-level, `layer_index` ignored). One forward pass feeds
  every site; multi-site runs mirror the multi-SAE subdir layout
  (`L29_post_mlp/`, `final_norm/`, …).
- `residual.activation_dtype` — **keep `float32`**: late-layer Gemma-3
  residual components exceed the float16 max (65504) and would saturate
  to `inf`.
- The base LM and `aggregation` come from the usual collect fields
  (`checkpoint`, `dtype`, `use_chat_template`, `last_token` /
  `mean_prefill` / `max_prefill`); BOS is excluded from mean/max
  aggregates (raw BOS residuals are extreme outliers).
- Both `dim_mode`s and the embed-axis / embed-dim agents apply unchanged.

Configs: `full_resid_gemma3_4b_it_L29_L33.yaml` (collect-only, L29 + L33 in
one pass; `..._pt_...` for the base-model checkpoint),
`label_resid_gemma3_4b_it_L29_200_signed_sonnet.yaml` (200 random dims of
2560: `default_rng(123).choice(2560, 200, replace=False)`, reuses the
collect via `extract.activations_run_dir`), `smoke_resid_gemma_L29.yaml`.

## Config knobs

All configurable — nothing hardcoded:

- **Base model**: `base_model.family` (`gemma` | `qwen`), `checkpoint`, `device`,
  `dtype`, `add_bos`, `use_chat_template`. Selects the inference wrapper.
- **SAEs** (`saes:` list, one entry per attached SAE):
  - **Gemma**: `family: gemma`, `layer_index`, `hook_type`, `width`
    (16k/65k/262k), `l0_size`, `model_size`, `variant`.
  - **Qwen**: `family: qwen`, `layer_index`, `hook_type` (resid_post only),
    `model_size` (e.g. `"1.7B"`), `width` (32k/64k/80k), `k` (50 | 100).
  Multiple SAEs may share the same `(layer, hook_type)` site; the
  HookManager keys them by `(layer, hook_type, identity())` and
  enforces `read_only=True` at shared sites — the autointerpreter sets
  this implicitly. A legacy single-Gemma yaml using the flat
  `layer_index/hook_type/width/...` fields still loads unchanged
  (those fields are honoured when `saes:` is absent).
- **Residual source** (`source_kind: residual` + `residual:` block): `sites`
  (list of `{layer_index, intermediate}`), `activation_dtype` (`float32` —
  see above), optional `aggregation` override (else inherits the collect
  field). Base model comes from the standard fields; SAE fields are ignored.
  A `residual:` block may also ride along on a **`source_kind: sae`** collect
  (Gemma only): the same forward pass then side-captures raw base-model
  residuals into `DenseActivationStore`s. Its `aggregation` may be a single
  mode or a **list** (independent of the SAE aggregation), writing one store
  per (site, mode) — e.g. raw `post_mlp` at L9/L29 + `final_norm`, each at
  both `last_token` and `max_prefill`, alongside a `max_prefill` SAE pass.
  `post_mlp` at layer L is exactly the `resid_post` vector the layer-L SAE
  encodes, so raw-dim-vs-SAE-feature is a same-site paired comparison.
  Subdirs: single mode → bare `residual_subdir` (`final_norm`, `L9_post_mlp`);
  multiple → suffixed `_<aggregation>` (see `residual_unit_layout`). See
  `configs/autointerpret/full_gemma_pt_3saes_dual.yaml` for the full 12-store
  base-model example.
- **Embedding source** (`source_kind: embedding` + `embedding:` block): `provider`
  (`sentence_transformers`), `model_name`, `device`, `normalize`, `prompt`
  (sentence-transformer preset name — e.g. `STS` for EmbeddingGemma — or `null`
  for plain encoders like MiniLM), `activation_dtype`, `embed_batch_size`.
  When set, the SAE fields are ignored and `aggregation` does not apply (the
  model pools internally).
- **Aggregation** (SAE + residual): `last_token` | `mean_prefill` |
  `max_prefill`. On the SAE path this can also be a **list** — one forward
  pass then writes one store per (SAE, aggregation) combination (the
  forward dominates collect cost, so extra conventions are nearly free);
  subdirs are named `L{n}_w{w}_{hook}_{aggregation}` in that case (see
  `sae_unit_layout` in config.py). `mean_prefill`/`max_prefill` exclude
  position 0 when the wrapper prepends a BOS (`wrapper.prepends_bos`:
  Gemma yes; Qwen3/3.5 no — they have no BOS token). BOS activations are
  extreme outliers (~2× content maxima on Gemma L9) that would otherwise
  dominate an unmasked max. The standalone `source_kind: residual` path
  accepts a single mode only; the SAE side-capture `residual:` block accepts
  a list too (see the Residual source bullet above).
- **Corpus scope**: `limit`, `pos_filter`, `prompt_template`, `use_chat_template`,
  `wordnet_xml_path` (relative paths resolve against `PROJECT_ROOT`).
- **Feature selection (SAE)**: `density_min/max`, `require_min_nonzero`, explicit `feature_indices` list.
- **Dimension selection (embedding)**: `dim_mode` (`signed` | `split`), `select`
  (`all` | `top_variance`), `min_variance`, `max_features`, `feature_indices`
  (dimension indices), `activations_run_dir` (reuse a prior collect pass).
- **Agents**: worker count, reps per worker, `show_zero_fraction_to_evaluator`
  (`on`/`off`/`ab`), `fail_on_queue_errors` (raise vs. log when any item ends
  up in `failed` status; default `true`).
- **Scoring**: `min_pearson`, whether to push back to `FeatureLabelStore`.

## Output layout

**Single-SAE (legacy):** flat layout directly under `run_dir/`.

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

**Multi-SAE:** one subdir per SAE under the umbrella `run_dir/`. Each
subdir is a self-contained mini-run; downstream stages run per subdir.
The runner clears the shared AgentSystem queue/results between SAEs and
copies the global agent outputs into the matching `labels/` /
`evaluator/` folder before scoring.

```
resources/sae_autointerpret/full_gemma_L9-16k_L29-16k_L29-65k/
    experiment.yaml
    L9_w16k_resid_post/
        activations.npz  index.parquet  feature_stats.parquet
        topk/  linspace/
        labels/  evaluator/         # per-SAE copies of agent outputs
        scores.parquet
    L29_w16k_resid_post/  …
    L29_w65k_resid_post/  …
```

## Agents

- `.claude/agents/sae-label-interpreter.md` — reads `topk/feature_*.json`,
  writes `{short_name, explanation, polarity}`.
- `.claude/agents/sae-label-evaluator.md` — reads label + shuffled samples,
  scores each 0–10; scorer computes correlation against true activations.
- `.claude/agents/embed-axis-{interpreter,evaluator}.md` — embedding `signed`
  mode: interpreter names both poles of the axis; evaluator predicts a **signed**
  score in −10..+10.
- `.claude/agents/embed-dim-{interpreter,evaluator}.md` — embedding `split`
  mode: same 0–10 non-negative rubric as the SAE agents, neutral wording, input
  carries `{dim, half}`.

Task configs: `interpret/agent_system/tasks/autointerpret-{label,eval}.json` (SAE)
and `autointerpret-embed-{axis,dim}-{label,eval}.json` (embedding). Select via
`agents.label_task` / `agents.eval_task` (and the matching `*_dir` fields) in
the config — see `configs/autointerpret/smoke_embed_minilm*.yaml`.

## Re-scoring / comparing scorers

`AutoInterpretScorer` (Stage 5) scores one SAE in place and writes
`scores.parquet`. To **re-score finished AgentSystem job folders** or
**compare several scorer LLMs / SAEs side by side** without re-running the
pipeline, use `scripts/scratch/score_sonnet_autointerpret.py`:

- `ScoreSweepConfig` — a cartesian-product sweep whose axes are all
  `list[str]`: `models`, `layers`, `sizes`, `sites`, `scorers`. The SAE
  subdir is rebuilt as `L{layer}_w{size}_{site}` (matching `sae_subdir`) and
  the job folder as `{sae_subdir}__{scorer}`. `model` (base-LM family) is
  recorded as metadata only — it isn't in the on-disk folder names, and the
  collect `run_dir` slug is usually custom, so it can't be derived from the
  axes; keep `run_dir` pointed at the matching base-LM run.
- `SweepScorer` — enumerates the product, **skips (and logs) combos absent on
  disk**, and reuses the production `AutoInterpretScorer._align` / `_safe_corr`
  so numbers match Stage 5 exactly. Eval predictions are read from
  `resources/jobs/autointerpret-eval/{sae}__{scorer}/queue/results/` and the
  hidden `_true_activations` from `<run_dir>/{sae}/linspace/`.
- Output → `resources/sae_autointerpret/scores/`: one
  `scores_{combo}.parquet` per combo, a `scores_combined.parquet`, and a
  `sweep_summary.csv` (one row per combo: mean/median Pearson + Spearman,
  valid-feature count, fraction Pearson ≥ 0.5). Combos are reported with the
  *mean of per-feature* correlations — activations are not pooled across
  features (magnitudes differ per feature; the 0–10 score is feature-relative).

## Notes

- Storage is `scipy.sparse` CSR, not Qdrant — the query pattern is
  "top-k rows per column" which Qdrant doesn't index for. Per-row meta
  lives in the parallel `index.parquet` keyed by `row_idx`.
- The collector calls `generate_from_template(prompt, output_len=1)` so the
  raw `"{word}: {definition}."` prompt is not wrapped in chat markers by
  default (toggle `use_chat_template` to compare). Both
  `GemmaPytorchInference` and `Qwen3Inference` expose this method.
- **Read-only enforcement at shared sites**: `HookManager.add_sae`
  rejects co-attaching two SAEs at the same `(layer, hook_type)` unless
  both have `read_only=True`. The collector always sets this; if you
  see the error, you've registered a writing SAE (e.g. for SAE
  reconstruction in place) at a site that already has another SAE.
- **Resume across multi-SAE**: a sample is re-run only when at least
  one store is missing it (intersection of per-store seen-sets). Each
  store skips its own `append` when it already has the row, so a
  multi-SAE pass that crashed mid-flush converges without duplicates.
- `SparseActivationStore.flush()` is **append-only sharded**: each call
  writes a fresh `activations_batch_NNNNNN.npz` containing only the
  buffered rows — no rewrite of prior shards. Per-flush cost is O(batch
  size), independent of total dataset size (the prior load-stack-save
  pattern grew linearly and stalled the collect for ~30 s per 500
  samples once the file passed ~500 MB). `load_matrix()` `vstack`s every
  shard back into one CSR on read. A legacy single `activations.npz`
  (from older runs) is recognised as an immutable "base shard" — loaded
  first by `load_matrix` and counted in `existing_row_keys` so a resumed
  run never duplicates rows. Each shard write is still atomic
  (`<file>.tmp.npz` then `os.replace`); a SIGKILL mid-flush leaves the
  prior shard intact and the partial tmp orphaned (the regex
  `^activations_batch_\d+\.npz$` ignores `.tmp.npz` siblings on load).
