# TREC SAE probing — token-level two-stage pipeline

Config-driven experiment finding task-relevant SAE features for text
classification (TREC first), for both **Gemma-3-4b + gemma-scope-2** (all 34
layers × 3 hook sites) and **Qwen3 + qwen-scope** (per-layer TopK, residual
only, model size configurable). Built on the `interpret/probing/` YAML
engine, extended with a token-level extraction stage.

Experiment configs + run commands: `interpretability_backend/experiments/trec_classification/`.
Engine-level docs: `interpretability_backend/interpret/probing/README.md`
("Token-level two-stage pipeline").

## Design

The pre-existing engine pooled residuals to one vector per sample *before*
SAE encoding, so "max over SAE activations across tokens" (the semantics of
the production `sae_document_activations` path) was inexpressible. The new
pipeline splits extraction in two:

1. **`token_residuals`** — one forward pass per sample captures every token
   position at every requested `(layer, site)`, stored ragged
   (`[total_tokens, hidden]` + an N+1 offset table) in bf16. Runs once per
   model on the GPU box; everything downstream is offline. Family-agnostic:
   Gemma via the fork's `configure_cache` (string intermediates, CPU
   tensors), Qwen via `Qwen3Inference.cache_activations` (HookType keys,
   on-device tensors moved to CPU per sample). Canonical site names
   (`resid_post`, `mlp_out`, `attn_out`) map per family.
2. **`sae_pooled`** — per layer, load the family's SAE
   (JumpReLU gemma-scope / TopK qwen-scope, via the shared `load_sae`
   cache, cleared per layer), encode all tokens chunk-wise, pool per sample
   (`max` with the BOS position masked — BOS is an activation sink that
   otherwise tops every sample — or `last`), dead-filter with
   `min_active_samples`, and emit probeable `[N, d_kept]` per-layer keys
   with the `kept_by_layer` true-index map.

   Plus **`residual_pooled`** (same pooling, no SAE — the raw-residual
   baseline from the *same* forward pass) and **`concat`** (stack pooled
   SAE keys across layers/sites into one wide matrix with per-column
   `feature_names` `L{layer}_{site}_f{true_idx}` for joint probing).

Probes are multiclass logreg + SVC (`class_weight: balanced`, 5-fold
stratified CV); analyses are `top_features` (probe-|coef| ranking, fold-
aware, multiclass-aware with per-feature `top_class`) and
`correlation_map`.

## What was changed (2026-07)

- **gemma_pytorch fork**: `Gemma2DecoderLayer.forward` now caches
  `"attn_out"` — the raw attention-block output (pre `post_attention_layernorm`,
  pre residual add), symmetric with the existing `"mlp_out"` cache; the
  site gemma-scope attn_out SAEs are trained on. Verify empirically on
  first use: FVU of the attn_out SAE's decode(encode(x)) should be ≪ 1;
  if not, the alternate placement is after the post-norm.
- **Probing engine**: four new extraction types (configs + extractors +
  orchestrator branches + topo-sort generalised to multi-source
  dependencies); probe stage skips `skip_probes` datasets and prefers
  dataset-level `feature_names`; models are freed
  (gc + cuda/mps `empty_cache`) after extraction; `_load_qwen` added.
- **sae_analysis**: shared `SAE_INTERMEDIATES` constant
  (`sae_feat`/`sae_max`/`sae_last`); `top_features` handles k-fold
  direction files and multiclass coefs; multiclass logreg
  `feature_importance.csv` aggregates per-feature max-|β|-over-classes.
- **`LabeledTextManifestBuilder`** (`interpret/probing/manifests/labeled_text.py`):
  generic TSV/CSV/parquet text+labels manifest — dedupe (sample IDs are raw
  texts), split filter, smoke-run `limit`, int label passthrough,
  alphabetical categorical encoding, per-target `min_class_count`
  (StratifiedKFold(k) needs ≥ k members per class; TREC `fine_label`'s
  smallest class has 4).

## Memory budget (TREC, ~5.9k deduped questions, ~83k tokens)

| Cache | Size |
|---|---|
| Qwen3-1.7B tokens, 28 layers × resid_post, bf16 | ~9.5 GB |
| Gemma-3-4b tokens, 34 layers × 3 sites, bf16 | ~43 GB |
| One pooled SAE dataset (34 layers × 16k, post-filter) | ~1–3 GB fp32 |
| Concat matrix (resid_post, min_active_samples=10) | ~[5.9k, 200–350k] fp32, 5–8 GB |

The orchestrator holds all resolved datasets in memory for the run — the
full Gemma experiment wants a ≥128 GB box; fallback is splitting
`gemma_tokens` per site into separate extractions/YAMLs (caches make
re-runs cheap).

## Caveats

- No canonical train/test split in v1 — 5-fold CV over all rows. The
  builder's `split_column`/`splits` kwargs exist for a later
  canonical-split extension (needs `indices_override` plumbing in the
  orchestrator).
- `correlation_map` against a nominal class index is a rough screen only;
  `top_features` is the primary ranking.
- Concat's single `(0, "concat")` key is skipped by `consolidate`'s wide
  pivots (same pre-existing constraint as `csv_features`).
- Qwen-scope SAEs are trained on Base checkpoints; anomalous results on
  `Qwen/Qwen3-1.7B` → try `Qwen/Qwen3-1.7B-Base`.
- Qwen features have no labels until the autointerp pass (not on
  Neuronpedia); gemma labels exist for resid_post layers 9/17/22/29.

## Tests

`unit_tests/test_labeled_text_manifest.py`, `test_trec_experiment_config.py`
(parses the real YAMLs), `test_token_residual_extraction.py` (fake
wrappers), `test_sae_pooled_extraction.py` (stub SAE; pins equivalence with
`max_pool_feature_acts`), `test_concat_extraction.py`,
`test_top_features_multiclass.py` (fold-glob + multiclass end-to-end via
`train_sklearn_probe`).
