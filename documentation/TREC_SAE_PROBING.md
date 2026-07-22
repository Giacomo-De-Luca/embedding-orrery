# TREC SAE probing — the token-level two-stage pipeline

**Status**: implemented, tested (526 unit tests green), smoke-ready. July 2026.

This document is the complete record of the experiment design and of every
change made to build it: what exists, why each decision was taken, how the
code works, how to run it, and what was found and fixed during review.

- Experiment configs + quick-start: `interpretability_backend/experiments/trec_classification/`
- Engine reference: `interpretability_backend/interpret/probing/README.md` ("Token-level two-stage pipeline")
- Backend map entry: `interpretability_backend/CLAUDE.md` → "Probing experiments"

---

## 1. Goal

Find **task-relevant SAE features** for text classification. For a labeled
text dataset (TREC question classification first), collect language-model
activations at every layer, encode them through pretrained sparse
autoencoders, pool to one vector per prompt, train classification probes,
and rank the SAE features the probes rely on. Three representations are
compared on identical splits:

1. **Raw residual vectors** (baseline): the residual stream pooled per
   prompt, `[N, hidden]` per layer.
2. **Single-SAE vectors**: per-layer SAE feature activations pooled per
   prompt, `[N, d_kept]` per layer.
3. **Concatenation**: all layers' pooled SAE features stacked into one wide
   matrix, `[N, Σ d_kept]`, probed jointly so features compete across the
   whole depth of the model.

Two model families, each paired with the SAE suite trained **on that
checkpoint**:

| | Model | SAE suite | Coverage |
|---|---|---|---|
| Gemma | `google/gemma-3-4b-it` | gemma-scope-2 (`-it` variant), JumpReLU, 16k width, l0 "medium" | all 34 layers × 3 hook sites (`resid_post`, `mlp_out`, `attn_out`) |
| Qwen | `Qwen/Qwen3-1.7B-Base` | qwen-scope, TopK (k=50), 32k width | all 28 layers × `resid_post` only |

Neuronpedia labels exist only for gemma resid_post layers 9/17/22/29;
every other layer/site reports bare feature indices (labels arrive with the
qwen autointerp pass later). Missing labels degrade to empty strings — a
34-layer sweep is safe.

---

## 2. Why a new pipeline — the pooling-order problem

The probing engine (`interpret/probing/`) predates this experiment and was
built for the Glasgow psycholinguistic study, where each sample is a single
word. Its extraction stage pooled each sample to **one residual vector**
(`token_position: last/first/mean/max/word_last`) *before* the SAE stage
encoded it. Two consequences:

- "Max over SAE feature activations across tokens" — the semantics of the
  production document-activation path
  (`InterpretService.run_batch_highlight` → `sae_document_activations`,
  implemented by `interpret.sae.activation_store.max_pool_feature_acts`) —
  was **inexpressible**: max-pooling the residual dimension-wise and then
  encoding it feeds the SAE an off-manifold vector that never occurred in
  the stream.
- Comparing pooling strategies required a full re-extraction per strategy.

The fix is the engine's own long-documented "future extension": emit
token-level activations once, let downstream consumers pool however they
like — **in SAE-feature space** where that is the right thing.

### Why two-stage (cache tokens, encode later) instead of fused

The model forward pass is the only GPU-bound step. Caching token-level
residuals once means every subsequent sweep — SAE widths, l0 variants,
pooling modes, probe hyperparameters — runs offline from the cache in
minutes, with no model in memory. It also gives exact experimental
control: `max` and `last` pooling derive from the *same* forward pass, so
pooling is the only variable. The cost is disk (§8). A fused
encode-during-forward variant remains the right choice for long-text
datasets (alpaca) where token caches would explode; it is not built yet.

---

## 3. The dataset

`interpretability_backend/resources/datasets/SAE/trec.tsv` (gitignored —
copy it to any new machine by hand): 5,952 rows with columns `text`,
`coarse_label` (int 0–5), `fine_label` (int, 50 classes), `split`
(train/test).

Facts that shaped the design:

- **81 duplicate texts.** Sample IDs throughout the engine are the raw
  prompt strings (`ActivationDataset.subset` requires unique IDs), so the
  manifest dedupes, keeping the first occurrence → 5,871 samples.
- **Class imbalance**: coarse class 0 (ABBR) has 95 members → probes use
  `class_weight: balanced`.
- **`fine_label`'s smallest class has 4 members**, but
  `StratifiedKFold(n_splits=5)` raises when any class count < 5 → the
  manifest supports per-target `min_class_count` and the YAMLs set
  `{fine_label: 5}` (drops sub-threshold classes for that target only).
- Mean 10 words / ~14 tokens per question, ~83k tokens total — extraction
  is minutes, caches are tens of GB not TB.
- **No canonical split in v1**: probes run 5-fold stratified CV over all
  rows. The engine has no fixed-split plumbing at the YAML level (the
  `indices_override` hook exists in the trainers but is not exposed);
  `split_column`/`splits` kwargs are in the manifest for a later extension.
- Integer labels reach the probes **unbinned**: `train_sklearn_probe`
  checks `np.issubdtype(targets.dtype, np.integer)` and uses them directly
  as class indices (verified; float targets would be percentile-binned).

---

## 4. Pipeline architecture

```
LabeledTextManifestBuilder  (generic TSV/CSV/parquet text+labels)
        │  manifest.samples = deduped texts
        ▼
token_residuals             (stage 1 — ONE forward pass per sample)
  ragged bf16 cache: per (layer, site) one Tensor[total_tokens, hidden]
  + metadata token_offsets [N+1], prepends_bos; skip_probes=True
        │
        ├── residual_pooled     raw baseline  → (layer, "res_last")   [N, hidden]
        ├── sae_pooled × modes  SAE encode+pool → (layer, "sae_max"/"sae_last") [N, d_kept]
        └── concat              all pooled keys → (0, "concat")       [N, Σ d_kept]
        ▼
probes   logreg + linear_svc (directions saved) + RBF svc (accuracy ceiling)
        ▼
sae_analysis   top_features (|coef| ranking, labels) + correlation_map
        ▼
figures  layer curves (val_accuracy), probe×target heatmaps
```

### 4.1 `token_residuals` (stage 1)

Config: `interpret/probing/configs/token_extraction.py::TokenLevelExtractionConfig`
— `name`, `family: gemma|qwen`, `checkpoint` (gemma `null` → auto-resolve
from HF cache; qwen defaults to `Qwen/Qwen3-1.7B`), `layers` (required
explicit list), `sites` (canonical names), `storage_dtype`
(bfloat16 default), `skip_probes=True`, `device` (qwen only — setting it
for gemma **raises**, because `GemmaPytorchInference` auto-selects its
device and a silently-ignored knob is worse than an error).

Extractor: `interpret/probing/extraction/extract_token_residuals.py`.

**Canonical sites** decouple the config from per-family naming:

| canonical | Gemma fork cache key | Qwen `HookType` | meaning |
|---|---|---|---|
| `resid_post` | `"post_mlp"` | `RESID_POST` | residual stream after the full layer |
| `mlp_out` | `"mlp_out"` | `MLP_OUT` | raw MLP output, pre-residual-add |
| `attn_out` | `"attn_out"` (new, §6.1) | `ATTN_OUT` | raw attention output, pre-residual-add |

**Family dispatch** (mirroring `residual_norm_profiler`'s reconciliation):

- *Gemma*: `configure_cache(layers, intermediates, prefill=True)` once;
  per sample `reset_prefill_cache()` →
  **`generate_from_template(sample, output_len=1)`** →
  `get_cached_activations()["prefill"]` (CPU tensors). The method choice
  is load-bearing: `generate()` wraps the prompt in the chat template
  (`<start_of_turn>user … <start_of_turn>model`), whose constant tokens
  would dominate max pooling for every prompt — the exact failure the
  production BOS/template-masking fix addressed. `generate_from_template`
  passes raw text through; the tokenizer still prepends BOS. The test
  fake wrapper deliberately does **not** implement `generate()`, so a
  regression fails loudly.
- *Qwen*: a fresh `cache_activations(layers, hook_types, prefill_only=True)`
  context per sample around `generate_from_template(sample, output_len=1,
  add_bos=True)`; cache tensors arrive **on-device bf16** and are moved to
  CPU + storage dtype per sample so GPU memory never accumulates. Cache
  keys may be `HookType` members or their string values — both accepted.

**Ragged storage.** Per `(layer, site)` key one
`Tensor[total_tokens, hidden]` (per-sample slices concatenated on dim 0),
with a single shared `metadata["token_offsets"]` list of length N+1
(sample *i* owns rows `offsets[i]:offsets[i+1]`). Chosen over padded
`[N, T_max, hidden]` because TREC lengths (4–60 tokens) would waste ~4× in
padding, and `ActivationDataset` serializes any tensor shape. Per-sample
sequence lengths are asserted identical across keys. Metadata additionally
records `family`, `checkpoint`, `layers`, `sites`, `prepends_bos`
(gemma True / qwen False), `storage_dtype`, `hidden_size`, `n_tokens`,
`token_level: True`.

**Not probeable.** Dim 0 is tokens, not samples, so the orchestrator's
probe loop skips any extraction whose config has `skip_probes=True`.

### 4.2 `sae_pooled` (stage 2)

Config: `configs/sae_pooled_extraction.py::SAEPooledExtractionConfig` —
`source_extraction`, `site`, `pooling: max|last`, `layers` (None = every
source layer for the site), SAE identity (`width` — set explicitly per
family since the label lookup reads it; gemma-only `l0_size`/`variant`;
`model_size`; qwen-only `k`), `exclude_bos=True`, `drop_dead_features=True`,
`min_active_samples`, `device` (None → auto-detect MPS→CUDA→CPU, making
one YAML portable between the Mac and the A100 and the cache sidecar
machine-independent), `batch_size_tokens=8192`.

Extractor: `extraction/extract_sae_pooled.py`. Per layer:

1. Build the family-correct SAE config — family is read from the source
   dataset's metadata, never guessed:
   `GemmaScopeSAEConfig(layer, hook_type=site, model_size, variant, width,
   l0_size, dtype="float32", device)` or
   `QwenScopeSAEConfig(layer, model_size, k, width, dtype="float32",
   device)` — and `load_sae()` it (shared module cache; qwen `site` other
   than resid_post raises).
2. **`pooling: last`** — gather rows `offsets[1:] − 1` (each sample's last
   token) and encode the `[N, hidden]` batch in one call.
3. **`pooling: max`** — encode the full `[total_tokens, hidden]` tensor in
   `batch_size_tokens` chunks (fp32 on the target device) and segment-max
   into `[N, d_sae]` via
   `out.scatter_reduce_(0, sample_idx, feats, reduce="amax",
   include_self=True)` with `out` initialized to −inf. `include_self=True`
   + −inf init is what makes the running max correct **across chunks**
   (`include_self=False` would discard earlier chunks' results); a
   post-loop `isfinite` assertion catches any unwritten sample.
   **BOS masking**: when the source `prepends_bos` and `exclude_bos`, each
   sample's position 0 is dropped from pooling (the BOS activation sink
   otherwise tops every sample identically) — except samples whose *only*
   token is the BOS, which fall back to full-range pooling. These are
   exactly the semantics of the production
   `max_pool_feature_acts`, and a unit test pins the equivalence.
4. `clear_sae_cache()` in a `finally` after each layer — the module-level
   `_SAE_CACHE` never evicts, and 28–34 resident fp32 SAEs would hold
   10+ GB.
5. **Dead-feature filter**: keep feature *j* iff `pooled[:, j] > 0` in at
   least `min_active_samples` samples (matches the production convention
   that non-positive activations are inactive; TopK can keep negative
   pre-activations — they never survive this filter). Surviving columns'
   original indices go into `metadata["kept_by_layer"][layer]`, the map
   every downstream label lookup depends on. Output key:
   `(layer, "sae_max"|"sae_last")`, fp32 CPU `[N, d_kept]`.

### 4.3 `residual_pooled`

`configs/residual_pooled_extraction.py` + `extraction/extract_residual_pooled.py`:
the no-SAE twin — pools the same ragged source per sample
(`last`/`max`/`mean`, same BOS logic) into `(layer, "res_last" …)` fp32
`[N, hidden]`. This is the raw-representation baseline, from the **same
forward pass** that fed the SAEs, for both families (the legacy pooled
`gemma` extraction has no qwen counterpart and would need a second model
pass).

### 4.4 `concat`

`configs/concat_extraction.py` + `extraction/extract_concat_activations.py`:
stacks one or more pooled sources' keys (sorted, optional `layers` filter)
along the feature axis into a single `(0, "concat")` key. Asserts
identical `sample_ids` across sources and per-key row counts. Writes
`metadata["feature_names"]` — `"L{layer}_{site}_f{true_idx}"` with the
true SAE index recovered through each source's `kept_by_layer` — plus
`concat_spans` (source, layer, site, col-start, col-end) for tracing any
column back. The probe stage prefers dataset-level `feature_names` over
the manifest's, so `feature_importance.csv` rows are directly
interpretable. `source_extractions` is a **list** (the multi-source case
is the pass-2 mega-concat across sites), which required generalising the
engine's dependency handling (§6.2).

### 4.5 Manifest: `LabeledTextManifestBuilder`

`interpret/probing/manifests/labeled_text.py` — generic, reusable for the
other files in `resources/datasets/SAE/` (toxicity, safety, alpaca):

- loader by suffix (`.tsv` tab, `.csv` comma, `.parquet`);
- `text_column`, `target_columns` (required), `source_name` (matches the
  YAML `targets:` `source` field);
- `split_column` + `splits` filter (unused in v1);
- `dedupe=True` keep-first (prints the drop count);
- `limit` — deterministic head-N for smoke runs;
- `min_class_count={col: m}` applied per target inside
  `get_rated_samples` (other targets keep the full manifest);
- target encoding: numeric columns pass through as int64 (original TREC
  ids preserved; non-integral floats raise); string columns get a
  deterministic alphabetical map exposed on `target_label_maps`.

---

## 5. Probes and analyses

Three probes per (extraction, target), all `class_weight: balanced`,
5-fold `StratifiedKFold`, shared engine trainer (`train_sklearn_probe`):

| kind | estimator | role | feature ranking? | on concat? |
|---|---|---|---|---|
| `logreg` | `LogisticRegression` (OvR) | linear reference, logistic loss | yes — `coef_ [C, d]` | yes (O(n·d)) |
| `linear_svc` | `LinearSVC` (OvR, hinge) | **independent linear ranking** — the "extract features through SVM" path | yes — `coef_ [C, d]`, same plumbing as logreg | yes (liblinear O(n·d)) |
| `svc` | `SVC` (RBF kernel) | nonlinear accuracy ceiling | no (kernel space) | **no** — `skip_extractions` |

Notes:

- `linear_svc` was added *because* kernel-SVC cannot rank features: the
  decision function lives in kernel space. A linear max-margin classifier
  has one hyperplane per class exactly like logreg; ranking by |w| is the
  classic SVM feature-selection method (SVM-RFE's one-shot form).
  `LinearSVC` (one-vs-rest) was chosen over `SVC(kernel="linear")`
  (one-vs-one → 15 pairwise hyperplanes for 6 classes) so the `[C, d]`
  coef drops straight into the existing multiclass machinery. AUC falls
  back to `decision_function` (no `predict_proba`) automatically.
- Features that rank highly under **both** losses (logistic + hinge) are
  the robust task-feature candidates.
- **`skip_extractions`** (new field on both probe spec types, validated
  against extraction names at config load): RBF SVC skips the concat
  matrix, where the n×n Gram over ~300k dims (O(n²·d) ≈ 10¹³ ops/fold,
  single-threaded libsvm) would take days and yields no ranking anyway.
- Metrics: `val_accuracy` (plotted in layer curves), `val_f1_weighted`
  (CSV only), `val_auc` (binary only — multiclass degrades to None
  gracefully). Coarse-label majority baseline: **22.6 %**.

Analyses (per SAE-typed extraction × target):

- **`top_features`** (`source_probe: logreg` or `linear_svc`): loads
  direction `.npz` files — single-split `L{layer}_{intermediate}_{name}.npz`
  or, for k-fold runs, the signed mean over `…_fold_{i}.npz` (shared
  loader `sae_analysis/directions.py::load_direction_coef`) — collapses
  multiclass `[C, d]` coefs by the strongest class's |coef| (recording
  `top_class` per feature), maps columns to true SAE indices via
  `kept_by_layer`, attaches Neuronpedia labels where the decoder-vector
  parquet exists, writes `top_features.json`.
- **`correlation_map`**: per-feature Spearman ρ against the class index —
  kept as a rough screen but explicitly weak for nominal labels;
  `top_features` is the primary ranking.
- `feature_importance.csv` (written by the trainer itself when a linear
  classifier runs k-fold with `save_directions`): per-feature mean/std of
  standardized |β| across folds; multiclass coefs collapse to per-feature
  max-|β|-over-classes before aggregation (the sign is class-relative, so
  only magnitude aggregates meaningfully).

---

## 6. Every code change, in order

### 6.1 `gemma_pytorch` fork — raw `attn_out` caching

`interpret/forked/gemma_pytorch/gemma/model.py`, `Gemma2DecoderLayer.forward`
(the class Gemma-3 instantiates; Gemma-1's `GemmaDecoderLayer` has no
caching at all): one `self._cache(..., "attn_out", hidden_states)` call
inserted **after `self_attn` returns, before `post_attention_layernorm`**
— Gemma-2/3 use sandwich norms, so the raw pre-norm, pre-residual-add
attention output is the symmetric analogue of the existing `"mlp_out"`
cache and the site gemma-scope attn_out SAEs are trained on. `"attn_out"`
added to `LAYER_INTERMEDIATES`; docstring updated. **One empirical check
remains**: during the first gemma run with attn SAEs, verify the FVU of
`decode(encode(x))` on captured attn_out tensors is ≪ 1; if it is not,
the alternate placement is after the post-norm (a one-line move).

### 6.2 Engine registration & orchestration

- `configs/experiment.py`: the four new config classes joined the
  `ExtractionConfig` union, the `_EXTRACTION_TYPES` tagged-union registry
  (`token_residuals`, `sae_pooled`, `residual_pooled`, `concat`) and
  `_DERIVED_EXTRACTION_TYPES`. A `_dependency_names()` helper generalises
  sibling-name validation and `topo_sorted_extractions()` from the single
  `source_extraction` field to the plural `source_extractions`
  (Kahn-style rounds, cycle detection unchanged). Probe `skip_extractions`
  entries are validated against extraction names (typos raise).
- `orchestrator.py`: dispatch branches for the four types;
  `_load_qwen(checkpoint, device)` (lazy import, bf16);
  `_release_accelerator_memory()` (gc + cuda/mps `empty_cache`, hasattr-
  guarded) called in `finally` after model-bound extractions — **also
  retrofitted to the pre-existing gemma branch**, which never freed the
  model; `_require_source()` helper; the probe loop skips
  `skip_probes` datasets and honours per-probe `skip_extractions`;
  `feature_names` prefers dataset metadata over the manifest; the
  SAE-analysis stage filter widened to include `SAEPooledExtractionConfig`.
- Caching needed **no changes**: any dataclass with `cache_filename()`
  gets sidecar-validated caching automatically. New types mean new stems,
  so existing experiments' caches are untouched. Corollary worth knowing:
  the sidecar compares the **whole** config dict — never mutate a field
  (e.g. `sites`) on an existing extraction name; add a new named
  extraction instead (this is what makes the two-pass gemma workflow
  cache-safe).

### 6.3 Probe trainer (`probes/sklearn_probes.py`)

- `linear_svc` kind (`_build_estimator` → `LinearSVC(C, class_weight,
  max_iter=spec.logreg_max_iter, dual="auto")`), added to
  `_CLASSIFICATION_KINDS` and `_LINEAR_PROBE_KINDS`.
- Multiclass coef aggregation: `[C, d]` collapses to per-feature
  max-|β|-over-classes for `feature_importance.csv` (previously multiclass
  was silently excluded); gate widened to `("logreg", "linear_svc")`.
- **Direction files keyed by probe *name*, not kind** — analyses resolve
  directions via `source_probe`, which is a probe *name*; with the old
  kind-keyed filenames a custom-named probe (`name: logreg_cv`) made
  `top_features` glob nothing and silently skip every layer. Default
  `name == kind`, so default-named probes' filenames are unchanged.

### 6.4 SAE analyses (`sae_analysis/`)

- `constants.py::SAE_INTERMEDIATES = {"sae_feat", "sae_max", "sae_last"}` —
  the four analyses (`top_features`, `correlation_map`, `feature_sweep`,
  `lasso_alpha_sweep`) previously filtered on the literal `"sae_feat"`
  and would have ignored pooled datasets entirely.
- `directions.py::load_direction_coef` — shared fold-aware loader
  (single-file or signed-mean over `_fold_{i}.npz`; fold shapes must
  agree), used by `top_features` and both of `feature_sweep`'s lasso
  paths (which also gained the loop's real `intermediate` in their file
  stems instead of a hardcoded constant).
- `top_features`: multiclass ranking by strongest class + `top_class` in
  the JSON output (previously multiclass coefs failed a shape check and
  were silently skipped).

### 6.5 Post-review fixes (chronological)

An adversarial code review (project `code-quality-reviewer` agent) deep-
verified the pooling numerics, `kept_by_layer` mapping, topo sort, cache
sidecars, fold plumbing and the fork patch, and surfaced:

1. **Critical — Gemma chat-template pollution** (§4.1): fixed to
   `generate_from_template`.
2. Direction name-vs-kind silent skip (§6.3): fixed.
3. Fold-aware loader not shared with `feature_sweep`: fixed (§6.4).
4. Latent CUDA bug: with `drop_dead_features: false` the alive mask was
   built on CPU while the pooled tensor sat on the GPU (device-mismatched
   boolean indexing raises) — mask now created on the tensor's device,
   and the reduction is skipped entirely when the filter is off.
5. `device:` on a gemma `token_residuals` config was silently ignored —
   now raises at config load.
6. Stale metadata (`storage_dtype`, `sites`) no longer carried from the
   bf16 multi-site token source into fp32 single-site pooled datasets.
7. Outdated `"sae_feat"` docstrings in the widened analyses refreshed.

**Deferred follow-ups** (flagged, non-blocking): factor the duplicated
token-source validation in the two pooled extractors into a shared helper
(and use `_require_source` in the pre-existing SAE/Delta branches); evict
`skip_probes` datasets from the orchestrator's in-memory dict after their
last dependent resolves (matters below ~128 GB RAM on the full 3-site
gemma run); exempt execution-only knobs (`batch_size_tokens`) from cache
identity.

### 6.6 Ergonomics added after review

- **Device auto-detect** for `sae_pooled` (`device: None` → MPS→CUDA→CPU,
  resolved at run time): the same YAML runs unmodified on the Mac and the
  A100, and cache sidecars no longer encode the machine.
- **Two-pass gemma workflow** (§7.1).
- **Qwen Base checkpoint** (§7.2).
- **`linear_svc` + `skip_extractions`** (§5).

---

## 7. The experiment configs

`interpretability_backend/experiments/trec_classification/`:
`trec_gemma.yaml`, `trec_qwen.yaml`, `trec_gemma_smoke.yaml`,
`trec_qwen_smoke.yaml`, `README.md`.

Shared: `LabeledTextManifestBuilder` on `trec.tsv` (dedupe,
`min_class_count: {fine_label: 5}`); targets `coarse_label` (6 classes) +
`fine_label` (50); probes as §5; analyses `top_features(source_probe:
logreg)` + `correlation_map`; `automatic_visualisations: true`.

### 7.1 Gemma: two-pass workflow

Pass 1 (as shipped): `gemma_tokens_resid` captures **resid_post only** at
all 34 layers (~14 GB bf16) → `gemma_res_last` baseline,
`gemma_sae_resid_max`, `gemma_sae_resid_last`, `gemma_sae_concat`. This is
the label-bearing primary analysis and keeps the first run light.

Pass 2 (commented blocks in the same file): uncomment `gemma_tokens_raw`
(`sites: [mlp_out, attn_out]`, ~29 GB), `gemma_sae_mlp_max`,
`gemma_sae_attn_max`, optionally the all-sites `gemma_sae_concat_all`
(add it to the RBF probe's `skip_extractions` too), and **re-run the same
command** — pass 1 cache-hits entirely; only the new forward pass and its
probes run. Separate extraction *names* are what make this safe (§6.2).

The gemma *smoke* config still captures all three sites in one small pass
(5 layers, 200 questions) — deliberately, so the attn_out fork patch and
FVU check are exercised before the big pass 2.

### 7.2 Qwen: Base checkpoint, size-configurable

`checkpoint: Qwen/Qwen3-1.7B-Base` — qwen-scope SAEs are trained on the
Base checkpoints (the SAE repo names carry `-Base`), there are no labels
to stay consistent with, and raw-text prefill is the base model's natural
regime. (Gemma stays `it` for the symmetric reason: its SAEs are the
`-it` variant and all ingested Neuronpedia labels are keyed
`gemma-3-4b-it`.) Switching sizes = edit `checkpoint`, every
`model_size`, the `layers` list and `width` per the `QWEN_SCOPE_MODELS`
registry (1.7B → 28/32k, 2B → 24/32k, 8B → 36/64k, 27B → 64/80k) and
rename the extractions for a fresh cache namespace.

---

## 8. Compute & memory budget

TREC ≈ 5,871 samples / ~83k tokens.

| Artifact | Size |
|---|---|
| Qwen token cache (28 × resid_post, bf16) | ~9.5 GB |
| Gemma pass-1 token cache (34 × resid_post, bf16) | ~14 GB |
| Gemma pass-2 token cache (34 × {mlp,attn}, bf16) | ~29 GB |
| One pooled SAE dataset (34 layers, post-filter, fp32) | ~1–3 GB |
| Concat matrix (`min_active_samples: 10`, fp32) | `[5.9k, ~100–300k]`, 2–8 GB |

Runtimes (A100): extraction ~20–40 min per pass (batch-1 prefills; the
loop, not attention, is the cost — hence no flash-attention work: prompts
are ~15 tokens and the HF qwen path already uses fused SDPA); SAE stage
dominated by first-time weight downloads; the 5-fold probe grid is the
long pole (hours). Qwen end-to-end < 1 h. The orchestrator holds all
resolved datasets in memory for the run — pass 1 needs ~30 GB RAM; the
full 3-site gemma configuration wants a big box until the dataset-eviction
follow-up lands.

---

## 9. Running it

### Local smoke (Mac, MPS — run these first)

```bash
cd interpretability_backend
uv run python -m interpret.probing.orchestrator experiments/trec_classification/trec_qwen_smoke.yaml
uv run python -m interpret.probing.orchestrator experiments/trec_classification/trec_gemma_smoke.yaml
```

Pass criteria: no `errors.json`; a second run is all cache hits;
`probes/*/coarse_label/logreg/probe_results.csv` shows `val_accuracy` well
above the 22.6 % majority baseline at mid layers; `top_features.json`
non-empty; the token dataset absent from `probes/`; **attn_out FVU ≪ 1**
(§6.1).

### Remote server (conda + A100)

The repo is not an installable package — only the dependencies matter —
and there is **one trap**: `pyproject.toml` pins torch to the CPU-only
wheel index on Linux (for the GPU-less Docker images). Plain pip from
PyPI sidesteps the pin:

```bash
conda create -n orrery python=3.12 -y && conda activate orrery
pip install uv && uv pip install -r pyproject.toml
python -c "import torch; print(torch.cuda.is_available())"   # MUST be True
# data + gated model (trec.tsv is gitignored — it does not travel with a clone):
scp <mac>:.../resources/datasets/SAE/trec.tsv interpretability_backend/resources/datasets/SAE/
huggingface-cli login && huggingface-cli download google/gemma-3-4b-it
cd interpretability_backend
python -m interpret.probing.orchestrator experiments/trec_classification/trec_qwen.yaml   # < 1 h
python -m interpret.probing.orchestrator experiments/trec_classification/trec_gemma.yaml  # pass 1
```

Devices auto-detect; no YAML edits between machines. Afterwards `rsync`
back `resources/probing_results/` (figures + rankings) and — the payoff of
two-stage — `resources/extracted_activations/` so every further sweep runs
locally without the GPU.

### Results tree

```
resources/probing_results/trec_{gemma,qwen}/
├── experiment.yaml, manifest.csv
├── probes/<extraction>/<target>/<probe>/
│   ├── probe_results.csv          # row per (layer, fold) + mean/std rows
│   ├── summary.json
│   ├── directions/L{L}_{int}_{name}[_fold_i].npz   # logreg + linear_svc
│   └── feature_importance.csv     # k-fold linear probes; concat rows are
│                                  # L{layer}_{site}_f{true_idx}
├── sae_analysis/<extraction>/<target>/
│   ├── top_features/top_features.json   # feature_idx, coef, label, top_class
│   └── correlation_map/...
├── figures/                        # layer curves (val_accuracy), heatmaps
└── errors.json                     # only if a stage failed (per-stage isolation)
```

---

## 10. Known caveats

- **No canonical TREC split** — 5-fold CV over all rows; canonical-split
  evaluation needs `indices_override` exposure in the orchestrator.
- **`correlation_map` on nominal class indices** is a rough screen only.
- **Concat's single key** is dropped from `consolidate`'s wide pivots
  (pre-existing `csv_features` constraint); its own CSVs are unaffected.
- **attn_out placement** is pending the empirical FVU check.
- **Qwen features are label-free** until the autointerp pass; gemma labels
  cover resid_post L9/17/22/29 only.
- **RBF SVC** never runs on concat by design (`skip_extractions`).

---

## 11. Tests

All in `interpretability_backend/unit_tests/`, offline, no model weights:

| File | Pins |
|---|---|
| `test_labeled_text_manifest.py` | dedupe-keep-first, split filter, `limit`, int64 passthrough, alphabetical encoding, per-target `min_class_count`, loader by suffix; real-`trec.tsv` checks (5,871 unique samples, fine classes ≥ 5) |
| `test_trec_experiment_config.py` | all four YAMLs parse + topo-sort, `skip_probes`, multi-source dependency validation, sidecar `_normalise` round-trip, qwen site/default-checkpoint rules, gemma `device` rejection, `skip_extractions` validation + the RBF-skips-concat wiring |
| `test_token_residual_extraction.py` | fake Gemma (no `generate()` — regression trap) + fake Qwen wrappers: ragged offsets, per-key length agreement, bf16/fp32 storage, canonical keys, BOS metadata, HookType-vs-string cache keys, wrong-family/empty errors |
| `test_sae_pooled_extraction.py` | stub SAE: BOS-masked max with single-token fallback, chunk-boundary equality of the scatter max, `last` row selection, `min_active_samples`/`kept_by_layer`, per-layer `clear_sae_cache`, **equivalence with the production `max_pool_feature_acts`**; `residual_pooled` last/max/mean |
| `test_concat_extraction.py` | feature-name/`kept_by_layer` mapping, span order, layers filter, mismatched-IDs and size-mismatch errors |
| `test_top_features_multiclass.py` | fold-glob + signed mean, multiclass `top_class` ranking, binary/unsuffixed path, classic `sae_feat` intermediate; end-to-end k-fold `feature_importance.csv` for **both** `logreg` and `linear_svc` (informative feature ranked first, per-fold `[C, d]` directions on disk) |

Run: `uv run pytest interpretability_backend/unit_tests/` from the repo
root (the torch-free guard's subprocess needs the root CWD).
