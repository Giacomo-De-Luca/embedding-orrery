# Probing engine

YAML-driven experiment runner for activation probing. One experiment defines a
manifest + a list of targets + a list of extractions + a list of probes
(+ optional SAE analyses). The orchestrator runs every probe against every
`(extraction, target)` pair and writes results into a uniform per-experiment
output tree.

The only experiment-specific Python in the engine is the **manifest builder**.
Extractions, probes, and analyses are all generic and configured purely from
YAML.

This package is part of the self-contained `interpret/` toolkit and carries no
parent-repo dependencies. Project-specific bits plug in from outside via two
extension points:

- **Manifest builders** — named from the YAML by dotted path (`"module:Class"`),
  resolved by `ManifestSpec.resolve`. The toolkit ships the abstract
  `manifests.manifest_base.ManifestBuilder` plus a few self-contained concrete
  builders (`feature_csv`, `glasgow`, `xkcd`). Builders that need external image
  assets (colour patches, THINGS-coloured) live with their host project (e.g.
  `scripts.interpretability.probing.manifests`).
- **Distance metrics** — an optional probe-evaluation metric (e.g.
  `val_lab_distance`, mean CIEDE2000 over LAB targets) is selected via a probe
  spec's `distance:` dotted path, resolved to an
  `interpret.utils.distances.ExperimentalDistance`. The colour default is
  `interpret.utils.distances:LabCiede2000Distance`.

## Quick start

```bash
uv run python -m interpret.probing.orchestrator \
    scripts/interpretability/experiments/glasgow_psycholinguistic_norms/experiment.yaml
```

The orchestrator:

1. Loads + validates the YAML (`ExperimentConfig.from_yaml`).
2. Resolves the manifest builder via its Python path string.
3. For each extraction (residual extractions before SAE extractions):
   loads from the **per-experiment cache** at
   `resources/extracted_activations/<experiment_name>/<extraction_name>.pt`
   if the sidecar matches the current config, else re-extracts and caches.
4. For each `(extraction, target, probe)`: trains the probe on the
   subset of activations whose target value is non-null, writes results
   incrementally to a per-probe folder.
5. For each `(sae_extraction, target, sae_analysis)`: runs the analysis.
6. Renders seaborn figures into `<output>/figures/` via
   `ExperimentVisualiser`.
7. Writes `errors.json` if any stage failed.

A second run of the same YAML hits every cache and finishes in seconds for
the extraction stages — the slow work is the probe training itself.

## Pipeline

```
                   manifest.samples (ordered list, sample_to_idx lookup)
                          │
                          ▼
       ┌────────────────────────────────────────────────────┐
       │  Extractions (named, topo-sorted)                  │
       │   • extract_encoder_activations   (HF encoders)    │
       │   • extract_gemma_activations     (text mode)      │
       │   • extract_gemma_activations_from_dataframe       │
       │     (image / multimodal)                           │
       │   • extract_sae_activations       (SAE encode)     │
       │   • extract_delta_activations     (per-pair        │
       │     subtraction; e.g. tinted − per-image baseline) │
       │   • extract_csv_features          (pre-computed    │
       │     vectors loaded from manifest CSV columns;      │
       │     no model inference)                            │
       └────────────────────────────────────────────────────┘
                          │  ActivationDataset per extraction
                          │  (cached at resources/extracted_activations/<exp>/)
                          ▼
       ┌────────────────────────────────────────────────────┐
       │  Probes (per extraction × target × probe)          │
       │   • train_mlp_probes              (ProbeModel MLP) │
       │   • train_sklearn_probe           (ridge/lasso/svr/ │
       │                                    svc/linear_svc/  │
       │                                    logreg/massmean) │
       └────────────────────────────────────────────────────┘
                          │  per-probe CSV + summary.json
                          ▼
       ┌────────────────────────────────────────────────────┐
       │  SAE analysis (per SAE extraction × target)        │
       │   • run_correlation_map           (Spearman ρ)     │
       │   • run_top_features              (|coef| ranking) │
       └────────────────────────────────────────────────────┘
                          │
                          ▼
                ExperimentVisualiser  →  <output>/figures/
```

## Module layout

```
probing/
├── activation_dataset.py      Storage container: tensors, sample_ids,
│                              metadata; save/load to .pt.
├── caching.py                 StageCache: readable filenames + YAML
│                              sidecar validation, atomic writes.
├── orchestrator.py            run_experiment(config) — pipeline driver +
│                              CLI entry point.
├── visualisations.py          ExperimentVisualiser + ConsolidatedVisualiser.
│                              Seaborn figures (layer curves, probe×target
│                              heatmap, best-metric bars, RGB/LAB panels).
│                              Auto-run by orchestrator; also standalone CLI.
├── consolidate.py             Cross-experiment aggregation
│                              (consolidated_long.csv + wide pivots +
│                              cross-experiment figures).
│
├── configs/                   Typed dataclass configs (loaded via OmegaConf).
│   ├── experiment.py            ExperimentConfig (incl. optional
│   │                            group_column), ManifestSpec, TargetSpec,
│   │                            from_yaml/from_dict with tagged-union dispatch.
│   ├── extraction.py            EncoderExtractionConfig, GemmaExtractionConfig.
│   ├── sae_extraction.py        SAEExtractionConfig (source_extraction reference).
│   ├── delta_extraction.py      DeltaExtractionConfig (source_extraction +
│   │                            pairing_column + baseline_filter).
│   ├── csv_features_extraction.py
│   │                            CSVFeaturesExtractionConfig: pull pre-
│   │                            computed feature vectors from manifest
│   │                            DataFrame columns. No model inference.
│   ├── probe.py                 MLPProbeSpec, SklearnProbeSpec
│   │                            (incl. class_weight, kind="svc").
│   ├── sae_analysis.py          CorrelationMapConfig, TopFeaturesConfig.
│   └── cache.py                 (placeholder for future cache helpers)
│
├── extraction/                Each emits an `ActivationDataset`.
│   ├── extract_encoder_activations.py   HF encoder + pooling (cls/mean/last).
│   ├── extract_gemma_activations.py     Gemma3 PyTorch wrapper, text mode.
│   │                                    Plus *_from_dataframe for image mode.
│   ├── extract_sae_activations.py       Encode residuals through Gemma-Scope
│   │                                    SAEs; preserves kept_by_layer.
│   ├── extract_delta_activations.py     Per-pair subtraction: pair each
│   │                                    non-baseline row with its baseline
│   │                                    on `pairing_column` and emit the
│   │                                    difference. Baselines selected via
│   │                                    `baseline_filter` (column equality).
│   └── extract_csv_features.py          Stack pre-computed feature columns
│                                        from the manifest into a single-key
│                                        ActivationDataset. Used when the
│                                        features have already been extracted
│                                        upstream and live in a CSV.
│
├── probes/                    Each writes <output>/probes/<extraction>/
│   ├── mlp_probe.py             <target>/<probe_name>/{probe_results.csv,
│   ├── sklearn_probes.py        summary.json, [directions/, checkpoints/]}.
│   └── mlp_ablation.py          ProbeAblationRunner — leave-one-feature
│                                and leave-one-group-out ablation for MLP +
│                                sklearn classification probes (logreg,
│                                svc). Multi-seed averaging via
│                                ExperimentConfig.ablation_seeds. Group
│                                axes parsed from feature names of the
│                                form "c{ctx}_p{section}_{CATEGORY}".
│                                CLI: `uv run python -m
│                                interpret.probing.probes.
│                                mlp_ablation <experiment.yaml>`.
│
├── sae_analysis/              Each writes <output>/sae_analysis/<extraction>/
│   ├── labels.py                <target>/<analysis>/...
│   ├── correlation_map.py
│   └── top_features.py
│
├── manifests/                 Experiment-specific data sources.
│   ├── manifest_base.py         ABC: prompt_column, target_columns,
│   │                            samples, build_dataframe,
│   │                            get_rated_samples(source, column).
│   ├── glasgow.py               Brysbaert + Glasgow norms; glasgow_only kwarg.
│   ├── xkcd.py                  954 XKCD survey colours.
│   ├── colour_patches.py        Rendered colour patches (RGB + LAB targets).
│   ├── things_coloured.py       THINGS object images tinted + grayscale
│   │                            baseline (paired); exposes source_image,
│   │                            category, is_grayscale for delta + group split.
│   └── feature_csv.py           CSV-backed manifest for already-extracted
│                                feature vectors. Optional row-level
│                                filtering via `kwargs.filters`. Pairs
│                                with the csv_features extraction.
│
└── utils/
    ├── enums.py                 TokenPosition, TaskType.
    └── metrics.py               Shared HIGHER_IS_BETTER + METRIC_PREFERENCE
                                 dicts and `load_probe_results` walker;
                                 imported by report / consolidate /
                                 visualisations to avoid duplication.
```

## YAML schema

```yaml
name: glasgow_psycholinguistic_norms
output_dir: resources/experiments/glasgow_psycholinguistic_norms

# Manifest: Python path string to a ManifestBuilder subclass + kwargs.
manifest:
  path: interpret.probing.manifests.glasgow:GlasgowManifestBuilder
  kwargs:
    glasgow_only: false

# Extractions: each has a unique `name`. SAE references another extraction.
# Cache files live at resources/extracted_activations/<experiment_name>/<name>.pt
# + .yaml. Each experiment gets its own subfolder; sidecar validation catches
# config drift within an experiment.
extractions:
  - type: encoder
    name: minilm
    model_name: sentence-transformers/all-MiniLM-L6-v2
    pooling: cls          # cls | mean | last
    layers: null          # null = auto (first / middle / last)

  - type: gemma
    name: gemma_glasgow
    layers: [0, 9, 17, 22, 29, 33]
    intermediates: [post_mlp]
    token_position: word_last   # last | first | mean | max | word_last
    cache_phase: prefill
    prompt_column: word
    prompt_template: "<start_of_turn>user\n{word}<end_of_turn>\n<start_of_turn>model"

  - type: sae
    name: gemma_glasgow_sae_16k
    source_extraction: gemma_glasgow   # references an extraction by name
    source_intermediate: post_mlp
    token_position: word_last          # MUST match source's
    layers: [9, 17, 22, 29]
    width: 16k                         # 16k | 65k | 262k
    device: cpu
    drop_dead_features: true

  # Delta: subtract a baseline activation per pairing-column group. Used
  # for per-image grayscale subtraction (THINGS-coloured pilot) or per-
  # prompt baseline subtraction. Source can be any prior extraction —
  # delta-on-gemma yields residual deltas, delta-on-sae yields per-feature
  # firing-rate deltas. Both baseline + non-baseline rows must be present
  # in the source extraction.
  - type: delta
    name: delta_gemma_glasgow
    source_extraction: gemma_glasgow
    pairing_column: source_image       # any manifest column
    baseline_filter:                   # column-equality AND filter
      is_grayscale: true

# group_column (optional, top-level). When set, probes use
# GroupShuffleSplit so members of the same group never split across
# train/val. Required when probing paired data (e.g. multiple tinted
# variants of one object) to prevent the probe from exploiting shared
# shape/identity features and inflating val_r2.
group_column: source_image

# Targets: one or more (source, column) pairs.
targets:
  - source: concreteness    # Brysbaert norms (column "Conc.M" etc.)
    column: Conc.M
  - source: glasgow         # Glasgow norms (concreteness, imageability, ...)
    column: imageability

# Probes: every probe runs against every (extraction, target) pair.
probes:
  - type: sklearn
    kind: ridge
    alpha: 1.0
    save_directions: false
  - type: sklearn
    kind: lasso
    alpha: 0.01
    save_directions: true     # required for top_features analysis below
  - type: sklearn
    kind: svr
    C: 1.0
    kernel: rbf
  - type: sklearn
    kind: logreg
    classification_bins: 5
  - type: sklearn
    kind: massmean            # closed-form: mu_high - mu_low direction
  - type: mlp
    hidden_dims: [512]
    dropout: 0.1
    epochs: 100
    patience: 10
    learning_rate: 0.001
    best_metric: val_r2       # or null for default
    distance: null            # dotted path to an ExperimentalDistance, e.g.
                              # interpret.utils.distances:LabCiede2000Distance
                              # → adds val_lab_distance (mean CIEDE2000) for 3-D LAB targets

# SAE analyses: only run for SAE-typed extractions.
sae_analysis:
  - type: correlation_map
    top_k: 30
    max_density: null         # filter features active >X% of samples
    sae_vectors_dir: resources/sae_vectors
  - type: top_features
    source_probe: lasso       # reads <probes>/<source_probe>/directions/
    top_k: 30

cache_enabled: true
```

## Output layout

Each experiment produces a self-contained tree:

```
resources/experiments/<experiment_name>/
├── experiment.yaml                         # the resolved config that ran
├── manifest.csv                            # the manifest as a flat CSV
├── probes/
│   └── <extraction_name>/                  # one folder per extraction
│       └── <target_name>/                  # one folder per target
│           └── <probe_name>/               # one folder per probe spec
│               ├── probe_results.csv       # row per (layer, intermediate)
│               ├── summary.json            # spec + best metric
│               ├── directions/             # only if save_directions=true
│               │   └── L9_post_mlp_lasso.npz
│               └── checkpoints/            # MLP only
│                   └── layer_9_post_mlp.pt
├── sae_analysis/
│   └── <sae_extraction_name>/
│       └── <target_name>/
│           ├── correlation_map/
│           │   ├── correlations_layer9.csv
│           │   ├── all_correlations.csv
│           │   ├── topk_heatmap.png
│           │   ├── correlation_distribution.png
│           │   └── summary.json
│           └── top_features/
│               └── top_features.json
├── figures/                                # seaborn figures
│   ├── layer_curves__<extraction>.png      # one per extraction
│   ├── probe_target_heatmap__<extraction>.png
│   ├── best_metric_bars__<extraction>.png
│   ├── colour_channels_rgb__<extraction>.png   # only if R/G/B in targets
│   ├── colour_channels_lab__<extraction>.png   # only if L/a/b in targets
│   └── extraction_comparison.png           # only when >1 extraction
└── errors.json                             # only if any stage failed
```

To regenerate just the figures without re-running probes:

```bash
uv run python -m interpret.probing.visualisations \
    resources/experiments/<experiment_name>
```

The same command works on `resources/experiments/_consolidated/` to
refresh the cross-experiment figures (auto-detected by the presence of
`consolidated_long.csv`; pass `--consolidated` to force).

The activation cache lives outside experiment folders, namespaced per
experiment:

```
resources/extracted_activations/
├── glasgow_psycholinguistic_norms/
│   ├── minilm.pt + minilm.yaml
│   ├── bge.pt + bge.yaml
│   └── gemma_glasgow.pt + gemma_glasgow.yaml
└── xkcd_colour_directions/
    ├── minilm.pt + minilm.yaml
    └── ...
```

Each `<name>.yaml` sidecar records the full extraction config + torch /
python versions. On cache lookup the orchestrator dict-equals the sidecar's
`config:` block against the current config; mismatch raises
`CacheMismatchError` with a precise diff (no silent overwrite). Caches
are not shared across experiments — extraction config alone does not
capture the manifest sample list, so two experiments with the same
encoder over different manifests would otherwise collide.

## Caching and reproducibility

- **Filenames are readable**, not opaque hashes. Browsing
  `resources/extracted_activations/<experiment_name>/` shows you what's
  cached for that experiment at a glance.
- **Sidecars are the source of truth**: the cache layer compares the current
  config to the sidecar's `config:` dict (after `_normalise` — enums to
  values, Paths to strings). Adding a new field to a config invalidates
  every existing cache; renaming an extraction creates a new cache file.
- **Atomic writes**: the orchestrator writes to `<name>.pt.tmp` and renames
  on success; partial writes can never corrupt a cache entry.
- **Per-experiment isolation**: each experiment writes to its own subfolder
  under `resources/extracted_activations/<experiment_name>/`, so two
  experiments using the same extraction name but different manifests never
  collide. Within one experiment, different configs with the same name →
  mismatch error → user resolves by deleting the stale file or renaming.

## Adding a new experiment

1. Decide on a name describing the *manifest + targets* (not the backbones
   or probes — those are config knobs). Example: `glasgow_imageability_only`.
2. Create `scripts/interpretability/experiments/<name>/experiment.yaml`.
3. Reference an existing manifest builder (`interpret.probing.manifests.glasgow:GlasgowManifestBuilder`)
   or implement a new one in `probing/manifests/`.
4. Pick extraction names. If reusing a previously-extracted activation
   (e.g. `gemma_glasgow`), use the same `name` and matching config — the
   orchestrator will hit the shared cache. To force a fresh extraction,
   pick a new name.
5. Run: `uv run python -m interpret.probing.orchestrator <yaml>`.

## Adding a new manifest

Subclass `ManifestBuilder` (`probing/manifests/manifest_base.py`):

```python
class MyManifestBuilder(ManifestBuilder):
    @property
    def prompt_column(self) -> str: ...
    @property
    def target_columns(self) -> list[str]: ...
    @property
    def samples(self) -> list[str]: ...
    def build_dataframe(self) -> pd.DataFrame: ...
    def get_rated_samples(self, source: str, column: str) -> tuple[list[str], np.ndarray]: ...
```

Reference it from a YAML via the Python path string:

```yaml
manifest:
  path: interpret.probing.manifests.mymodule:MyManifestBuilder
```

## Cross-experiment aggregation

```bash
uv run python -m interpret.probing.consolidate
```

Walks `resources/experiments/`, reads every
`<exp>/probes/<extraction>/<target>/<probe>/probe_results.csv`, and writes
under `resources/experiments/_consolidated/`:

- `consolidated_long.csv` — every row tagged with experiment + extraction +
  target + probe_kind.
- `wide_val_r2.csv` and `wide_val_spearman.csv` — pivots: `(experiment,
  extraction, target, layer)` × `probe_kind` columns.
- `best_per_condition.csv` — best (layer, intermediate) per
  `(experiment, extraction, target, probe_kind)`.
- `summary.md` — markdown coverage table + best-per-condition.

## Token-level two-stage pipeline

The `token_residuals` → `sae_pooled` / `residual_pooled` / `concat` types
implement the token-level extension the classic path lacks: pooling happens
AFTER SAE encoding, so per-feature max over tokens (the production
`sae_document_activations` semantics) is expressible, and one model pass
serves every downstream pooling/width sweep.

```yaml
extractions:
  - name: gemma_tokens          # stage 1: ONE forward pass per sample
    type: token_residuals
    family: gemma               # or qwen (resid_post only)
    checkpoint: null            # gemma: auto from HF cache; qwen: Qwen/Qwen3-1.7B
    layers: [0, 8, 16, 24, 33]
    sites: [resid_post, mlp_out, attn_out]   # canonical, family-agnostic
    storage_dtype: bfloat16
  - name: sae_max               # stage 2: encode every token, pool per sample
    type: sae_pooled
    source_extraction: gemma_tokens
    site: resid_post
    pooling: max                # or last
    width: 16k                  # qwen: 32k etc. — label lookup reads this
    l0_size: medium             # gemma-only; qwen uses model_size + k
    model_size: 4b
    min_active_samples: 10      # dead-feature filter threshold
    sparse: true                # store scipy CSR instead of dense (optional)
  - name: res_last              # raw-residual baseline, same forward pass
    type: residual_pooled
    source_extraction: gemma_tokens
    site: resid_post
    pooling: last
  - name: sae_concat            # all layers jointly, one wide matrix
    type: concat
    source_extractions: [sae_max]
```

Contract notes:

- The token-level dataset is **ragged**: per `(layer, site)` key one
  `Tensor[total_tokens, hidden]`, with `metadata["token_offsets"]`
  (length N+1) delimiting samples. Its dim 0 is tokens, not samples, so
  the config's `skip_probes: true` (default) excludes it from the probe
  stage. `metadata["prepends_bos"]` records whether position 0 of each
  sample is a BOS token.
- `sae_pooled` masks the BOS position for `max` pooling (the BOS
  activation sink otherwise tops every sample identically); a BOS-only
  sample falls back to full-range pooling — the same semantics as
  `interpret.sae.activation_store.max_pool_feature_acts`. Output keys are
  `(layer, "sae_max"|"sae_last")`, dead-filtered with `kept_by_layer`
  recorded like the classic SAE extraction. The SAE family comes from
  the source's metadata; hook site maps to the family's SAE suite
  (gemma-scope: resid_post/mlp_out/attn_out; qwen-scope: resid_post).
- `concat` stacks pooled sources' keys into a single `(0, "concat")` key
  and writes `metadata["feature_names"]` (`L{layer}_{site}_f{true_idx}`),
  which the probe stage prefers over manifest feature names for
  `feature_importance.csv`. Like `csv_features`, its single key is
  skipped by the cross-experiment `consolidate` pivots.
- Analyses recognise all SAE intermediates via
  `sae_analysis/constants.py::SAE_INTERMEDIATES`. `top_features` also
  handles k-fold direction files (`*_fold_{i}.npz`, signed-mean over
  folds) and multiclass logreg coefs (ranked by strongest class,
  `top_class` recorded per feature).
- **Sparse storage** (`sparse: true` on `sae_pooled`): pooled activations
  are stored as scipy CSR matrices instead of dense tensors. Pooled SAE
  rows are typically 1-2% nonzero, so CSR cuts disk/RAM ~25-50× — use it
  for large widths (65k/262k) or safety-scale sample counts. Everything
  downstream accepts CSR: `subset`/save/load, `concat`
  (`scipy.sparse.hstack`, stays CSR), sklearn probes (liblinear/lbfgs/
  libsvm train on CSR natively — standardisation switches to scale-only,
  `StandardScaler(with_mean=False)`, since centering would densify and
  shift the meaningful zero), `correlation_map` (per-column densify),
  and the sweeps (small-data densify). Mass-mean kinds and MLP probes
  reject sparse input with a clear error; `center_only` is likewise
  rejected. Pooling always iterates sample blocks whose dense device
  buffer stays under ~1 GiB (`_POOL_BLOCK_BYTES`), so peak memory never
  scales with `N × d_sae`. Tests: `unit_tests/test_sparse_sae_probing.py`.
- Memory: token caches are big (TREC × gemma-3-4b × 34 layers × 3 sites
  ≈ 43 GB bf16 in one `.pt`). Chunk via layer-subset extractions with
  distinct names when RAM is tight.

Worked example: `experiments/trec_classification/` (TREC question
classification, Gemma + Qwen).

## Known limitations and future extensions

- **The classic (non-token) path pools at extraction time.** Encoders and
  the pooled `gemma` extraction emit one vector per sample (encoder:
  `pooling`; Gemma: `token_position`), and the classic `sae` extraction
  inherits that pooling — its `token_position` is validated against the
  source because SAE encoding can't undo pooling. Use the token-level
  pipeline above when pooling should happen in SAE-feature space.
- **Image-mode Gemma extractions** use a separate code path
  (`extract_gemma_activations_from_dataframe`); the manifest must be
  iterable as a DataFrame with an `image_column`. Not used by current
  experiments but kept for future multimodal probing.
- **Sklearn probes are 1-D-target by default.** Multi-output regression
  (e.g. LAB targets `[N, 3]`) works for Ridge but not for Lasso / SVR
  (sklearn limitation). For multi-dim targets prefer the MLP probe with
  `distance: interpret.utils.distances:LabCiede2000Distance` to get
  `val_lab_distance` (mean CIEDE2000).
- **Per-probe failures are isolated** but not retried automatically.
  `errors.json` records stage + extraction + target + probe + traceback.
  Re-running the same YAML re-attempts only the failed entries (the
  successful ones short-circuit via cache hits and incremental CSVs).
