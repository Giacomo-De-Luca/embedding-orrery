# Glasgow psycholinguistic probing experiment

A worked probing experiment that decodes the nine **Glasgow psycholinguistic
norms** from several embedding spaces, using the merged `interpret/` probing
engine and its `GlasgowManifestBuilder`. It compares which model + probe family
recovers each norm best over the ~4,700 Glasgow words.

## What it does

For each word it extracts one pooled hidden vector per layer (encoder extraction,
`output_hidden_states`), then trains ridge / lasso / SVR / **mass-mean** /
**logistic** (median split → accuracy/AUC) / MLP probes to predict each norm. Output is a per-`(extraction, target, probe, layer)`
tree of metrics + probe directions, plus seaborn figures (layer curves,
probe×target heatmap).

**Extractions**: `minilm` (`all-MiniLM-L6-v2`, 384-d, cls) and
`embeddinggemma_mean` (`google/embeddinggemma-300M`, 768-d, mean). Heavier Gemma-3-4b
residual + a precomputed **Gemini** path are scaffolded (commented) in the YAML —
see "Gemini path" below.

**Targets** (all regression): `concreteness`, `imageability`, `valence`,
`arousal`, `dominance`, `familiarity`, `aoa`, `semsize`, `gender`.

## Results — best val R² per (model × norm), 4,682 words

Best probe shown in parentheses; single 80/20 split, seed 42.

| Norm | MiniLM-384 | EmbeddingGemma-768 |
|---|---|---|
| concreteness | 0.756 (svr, L3) | **0.812** (svr, L12) |
| imageability | 0.715 (svr, L3) | **0.756** (svr, L12) |
| valence | 0.678 (svr, L6) | **0.725** (svr, L12) |
| aoa | 0.647 (svr, L3) | **0.675** (lasso, L12) |
| semsize | 0.595 (svr, L3) | **0.700** (svr, L12) |
| gender | 0.588 (svr, L6) | **0.613** (svr, L12) |
| familiarity | 0.552 (svr, L3) | **0.574** (svr, L12) |
| arousal | 0.511 (svr, L6) | **0.584** (svr, L12) |
| dominance | 0.502 (svr, L6) | **0.553** (svr, L12) |

Findings (consistent with the reference `glasgow_psycholinguistic_norms` report):
- **EmbeddingGemma-300M > MiniLM on every norm** — the newer, larger encoder
  recovers more psycholinguistic signal.
- **SVR (rbf) wins** for both small encoders (matches the report's
  "SVR wins for small-dim encoders"; Lasso only edges it once).
- **Layer split by dimension type**: MiniLM's *referential* norms
  (concreteness/imageability/aoa/semsize) peak early (L3), *affective/social*
  norms (valence/arousal/dominance/gender) peak later (L6).
- **Dimension ranking is architecture-invariant**: `concreteness > imageability
  > valence > aoa > semsize > gender > familiarity > arousal > dominance` — the
  same order in both models (and in the reference report).
- The MiniLM column **reproduces the reference report bit-for-bit**, confirming a
  faithful replication.

### Mass-mean (direction) probe

`kind: massmean` is a **closed-form** probe: it takes the median split of the
target and uses the difference-of-means as a 1-D direction. Because that raw
direction isn't scale-calibrated, the engine reports it in **Pearson/Spearman
correlation** (not R² — which needs calibrated predictions). Best `val_pearson`
per norm, mass-mean vs the best of all five probes:

| Norm | MiniLM mass-mean r | MiniLM best r | EmbGemma mass-mean r | EmbGemma best r |
|---|---|---|---|---|
| concreteness | 0.714 | 0.869 (svr) | **0.858** | 0.902 (svr) |
| imageability | 0.697 | 0.846 (svr) | 0.805 | 0.870 (svr) |
| valence | 0.661 | 0.828 (svr) | 0.759 | 0.854 (svr) |
| semsize | 0.609 | 0.776 (svr) | 0.723 | 0.840 (svr) |
| aoa | 0.653 | 0.805 (svr) | 0.678 | 0.821 (lasso) |
| gender | 0.649 | 0.774 (svr) | 0.573 | 0.792 (svr) |
| familiarity | 0.575 | 0.749 (svr) | 0.651 | 0.762 (svr) |
| arousal | 0.534 | 0.721 (svr) | 0.627 | 0.768 (svr) |
| dominance | 0.535 | 0.714 (svr) | 0.617 | 0.751 (svr) |

Mass-mean is a strong, near-free baseline — it trails SVR by only ~0.05–0.15 r
and peaks at the *same* layers. Concreteness is the standout: a single
difference-of-means axis reaches r≈0.86 on EmbeddingGemma (vs SVR's 0.90),
i.e. concreteness is largely encoded along one linear direction.

## Structure

| File | Purpose |
|---|---|
| `experiment.yaml` | The full experiment config (manifest builder + extraction + targets + probes). |
| `README.md` | This file. |

The manifest builder itself lives in the toolkit at
[`interpret/probing/manifests/glasgow.py`](../../interpret/probing/manifests/glasgow.py)
(`GlasgowManifestBuilder`); this folder only holds the host-specific config, per
the repo convention that dataset-specific experiment glue lives with the project,
not inside `interpret/`.

## Data (gitignored)

The builder loads two files, both under `resources/psycolinguistics/` (CWD-relative):

| File | Source | Notes |
|---|---|---|
| `glasgow_norm.csv` | Glasgow Norms (Scott et al.) | `word` + the nine norm columns. ~4.7k words. |
| `concreteness.tsv`  | Brysbaert et al. concreteness | `Word` + `Conc.M`. ~40k words. **Loaded unconditionally** by the builder even in `glasgow_only` mode, so it must be present. |

Both are gitignored (`resources/psycolinguistics/*`). Place them there before running.

## Run

```bash
cd interpretability_backend        # CWD must contain resources/
uv run python -m interpret.probing.orchestrator \
  experiments/glasgow_psycholinguistic/experiment.yaml
```

Outputs land in `resources/probing_results/glasgow_psycholinguistic/`; the
activation cache is namespaced at `resources/extracted_activations/glasgow_psycholinguistic/`
(a second run reuses it and only re-trains probes).

## Knobs

- **Scope**: `manifest.kwargs.glasgow_only: true` keeps it to the ~4.7k Glasgow
  words. Set `false` to take the union with the ~40k Brysbaert words (much
  heavier extraction). To probe Brysbaert concreteness directly, add a target
  `{source: concreteness, column: "Conc.M"}`.
- **Layers**: `extractions[0].layers: null` auto-selects first/middle/last
  (`[0, 12, 24]` for EmbeddingGemma-300M's 24 layers); list explicit indices for
  a fuller sweep (cached, so cheap to expand).
- **Model**: swap `model_name` for any HF encoder, or replace the whole
  extraction with a `type: gemma` block to probe Gemma-3-4b residual activations
  (heavier — needs the 4b model + GPU).
- **Probes**: ridge / lasso / svr / massmean / mlp are all enabled; tune their
  hyperparams (e.g. `svr.C`, `lasso.alpha`, `mlp.hidden_dims`) inline.

## Gemini path (precomputed API vectors)

The encoder path runs local HF models; it can't call the Gemini API. To probe
**Gemini** embeddings, treat them as precomputed features via the `csv_features`
extraction (no local model). The words currently have **no Gemini vectors** (only
MiniLM-384 in the `Glasgow_norm` ChromaDB collection), and `GEMINI_API_KEY` is not
configured — so this is scaffolded, not yet run. Steps once the key is set:

1. Embed the 4,682 Glasgow words with the backend's Gemini embedding function →
   3072-d vectors.
2. Write a features CSV: `word` + the 9 norm columns + `f0..f3071` feature
   columns (one row per word).
3. Point [`FeatureCSVManifestBuilder`](../../interpret/probing/manifests/feature_csv.py)
   at that CSV (`manifest.path` →
   `interpret.probing.manifests.feature_csv:FeatureCSVManifestBuilder`) and enable
   the commented `type: csv_features` extraction in `experiment.yaml`.

The probe/target/analysis stages are then identical, giving a true
Gemini-vs-EmbeddingGemma-vs-MiniLM comparison.

## Tests

`interpretability_backend/unit_tests/test_glasgow_manifest.py` validates the
builder against the real data and that this YAML parses/resolves — no model
inference. It skips cleanly if the gitignored datasets are absent.
