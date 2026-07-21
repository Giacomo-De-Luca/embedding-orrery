# TREC question-type SAE probing

Find task-relevant SAE features for TREC question classification (6 coarse /
50 fine classes) by probing pooled SAE activations per layer — and jointly
across layers via a concatenated feature matrix — for two model families:

| Config | Model | SAEs | Layers × sites |
|---|---|---|---|
| `trec_gemma.yaml` | Gemma-3-4b-it | gemma-scope-2 JumpReLU 16k | 34 × resid_post (pass 1); mlp_out + attn_out as commented pass 2 |
| `trec_qwen.yaml` | Qwen3-1.7B-Base | qwen-scope TopK 32k (k=50) | 28 × resid_post |

Gemma runs as a **two-pass workflow**: pass 1 (as shipped) extracts only the
residual stream (~14 GB token cache) and runs the primary, label-bearing
analysis; uncommenting the `*_raw` blocks and re-running the same command
adds the mlp_out/attn_out sites as a separate extraction (~29 GB) while
everything from pass 1 cache-hits. Qwen uses the **Base** checkpoint —
qwen-scope SAEs are trained on the Base models.

Both use the token-level two-stage pipeline (see `interpret/probing/README.md`):
one forward pass caches every token's residuals; SAE encoding + pooling
(`max` over tokens with BOS masked, or `last` token) happen offline from the
cache, so sweeping widths/pooling/probes never re-runs the model.

## Run

```bash
cd interpretability_backend

# Local smoke first (200 questions, 5 layers, MPS; ~10-20 min each):
uv run python -m interpret.probing.orchestrator experiments/trec_classification/trec_qwen_smoke.yaml
uv run python -m interpret.probing.orchestrator experiments/trec_classification/trec_gemma_smoke.yaml

# Full runs (A100; qwen first — 5x smaller):
uv run python -m interpret.probing.orchestrator experiments/trec_classification/trec_qwen.yaml
uv run python -m interpret.probing.orchestrator experiments/trec_classification/trec_gemma.yaml
```

Prerequisites: `resources/datasets/SAE/trec.tsv`; `google/gemma-3-4b-it` in
the HF cache (`huggingface-cli download google/gemma-3-4b-it`); SAE weights
auto-download per layer on first use. Devices auto-detect (MPS locally,
CUDA on the A100); set `device:` on a `sae_pooled` block only to override.

Budget (full runs): qwen token cache ≈ 9.5 GB, < 1 h end-to-end. Gemma pass 1
token cache ≈ 14 GB bf16 (pass 2 adds ~29 GB); extraction ~20–40 min per
pass, the 5-fold logreg/SVC probes are the long pole (hours for the full
layer × site grid).

## Results tree

`resources/probing_results/trec_{gemma,qwen}/`:
- `probes/<extraction>/<target>/logreg/probe_results.csv` — accuracy/F1 per
  (layer, fold) + mean/std rows. Majority-class baseline for coarse_label is
  ~22.6%.
- `probes/gemma_sae_concat/<target>/logreg/feature_importance.csv` — per-
  feature standardized |β| across folds, columns named
  `L{layer}_{site}_f{true_feature_idx}`.
- `sae_analysis/<extraction>/<target>/top_features/top_features.json` —
  top-K features by |coef| with Neuronpedia labels where available
  (gemma resid_post L9/17/22/29; qwen has no labels yet) and, for
  multiclass, the class each feature most supports (`top_class`).
- `figures/` — layer curves (val_accuracy), probe×target heatmaps.

## Switching Qwen sizes

Edit `checkpoint`, every `model_size`, the `layers` list, and `width` per
`QWEN_SCOPE_MODELS` (`interpret/sae/sae_config.py`): 1.7B → 28 layers/32k,
2B → 24/32k, 8B → 36/64k, 27B → 64/80k. Rename the extractions (or the
experiment) so caches don't collide. Note qwen-scope SAEs were trained on
the Base checkpoints — if probe accuracy looks anomalous on `Qwen/Qwen3-1.7B`,
try `Qwen/Qwen3-1.7B-Base`.

## Caveats

- The canonical TREC train/test split is not used — probes run 5-fold
  stratified CV over all ~5.9k deduped questions. `fine_label` drops classes
  with < 5 members (`min_class_count`).
- `correlation_map` ranks features by Spearman ρ against the *class index*,
  which is nominal — treat it as a rough screen; `top_features` (probe-based)
  is the primary ranking.
- The concat extraction produces a single (layer, intermediate) key, which
  the cross-experiment `consolidate` pivots skip (same pre-existing
  constraint as csv_features); its per-probe CSVs are unaffected.
