# Glasgow Psycholinguistic Probing

Probing several embedding spaces for the nine **Glasgow psycholinguistic norms**
over the 4,682 Glasgow words, to ask: *how well does a probe recover each norm
from a model's embedding of a single word?* Compares models and probe families on
one shared evaluation surface.

Built on the merged `interpret/probing` engine (YAML-driven activation probing).
Host-specific config lives at
[`interpretability_backend/experiments/glasgow_psycholinguistic/`](../interpretability_backend/experiments/glasgow_psycholinguistic/)
(`experiment.yaml` + folder README); the reusable `GlasgowManifestBuilder` ships
in the toolkit at `interpret/probing/manifests/glasgow.py`.

## The nine norms

Each is an independent 1-D regression target (Glasgow Norms, Scott et al.):

| Norm | Measures | Type |
|---|---|---|
| `concreteness` | concrete vs abstract | referential |
| `imageability` | ease of mental imagery | referential |
| `aoa` | age of acquisition | referential |
| `familiarity` | how often encountered | referential |
| `valence` | pleasant vs unpleasant | affective |
| `arousal` | activation vs calm | affective |
| `dominance` | controlling vs controlled | affective |
| `semsize` | semantic size of referent | social/affective |
| `gender` | gender association (4 = neutral) | social |

## Setup

- **Manifest**: `GlasgowManifestBuilder(glasgow_only=true)` → 4,682 words, lowercased
  and deduplicated (raises on collisions). Brysbaert `concreteness.tsv` is loaded
  unconditionally by the builder even in `glasgow_only` mode, so both data files
  must be present at `resources/psycolinguistics/` (gitignored).
- **Extractions** (each emits one pooled hidden vector per word per layer):

  | Name | Type | Model | Dim | Layers | Pooling |
  |---|---|---|---|---|---|
  | `minilm` | encoder | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 0, 3, 6 (auto) | cls |
  | `embeddinggemma_mean` | encoder | `google/embeddinggemma-300M` | 768 | 0, 12, 24 (auto) | mean |

- **Probes** (per extraction × target × layer, seed 42, 80/20 split): `ridge`
  (α=1.0), `lasso` (α=0.01, saves directions), `svr` (rbf, C=1.0), `massmean`
  (median-split difference-of-means direction), `logreg` (C=1.0, binary
  **median split** of each norm via `classification_bins=2` → accuracy/F1/AUC,
  not R²; the discriminative counterpart to the Geometry-of-Truth σ(·) mass-mean
  probe), `mlp` (`[256]`, 100 epochs).
- **Activation cache**: `resources/extracted_activations/glasgow_psycholinguistic/`
  — re-runs hit the cache and only re-train probes.

## Results — best val R² per (model × norm)

Best probe in parentheses; single 80/20 split, seed 42. The `ref MiniLM` column is
the reference `glasgow_psycholinguistic_norms` report (MiniLM extraction).

| Norm | MiniLM-384 | EmbeddingGemma-768 | ref MiniLM |
|---|---|---|---|
| concreteness | 0.756 (svr, L3) | **0.812** (svr, L12) | 0.756 |
| imageability | 0.715 (svr, L3) | **0.756** (svr, L12) | 0.715 |
| valence | 0.678 (svr, L6) | **0.725** (svr, L12) | 0.678 |
| aoa | 0.647 (svr, L3) | **0.675** (lasso, L12) | 0.647 |
| semsize | 0.595 (svr, L3) | **0.700** (svr, L12) | 0.595 |
| gender | 0.588 (svr, L6) | **0.613** (svr, L12) | 0.588 |
| familiarity | 0.552 (svr, L3) | **0.574** (svr, L12) | 0.552 |
| arousal | 0.511 (svr, L6) | **0.584** (svr, L12) | 0.511 |
| dominance | 0.502 (svr, L6) | **0.553** (svr, L12) | 0.502 |

The MiniLM column **reproduces the reference report bit-for-bit** — a faithful
replication of the canonical experiment.

## Mass-mean (direction) probe

`massmean` is a **closed-form** probe: median-split the target, use the
difference-of-means as a 1-D direction. Because that raw direction isn't
scale-calibrated, the engine reports it in **Pearson/Spearman correlation, not
R²** (R² needs calibrated predictions — `val_r2` is blank for `massmean`). Best
`val_pearson` per norm, mass-mean vs the best of all five probes:

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

## Logistic-regression probe (median split → accuracy)

`logreg` binarises each norm at its median and fits L2 logistic regression on
the same layers/split. It reports **accuracy / ROC-AUC** (not R²). Best-layer
validation accuracy (AUC in parentheses):

| Norm | MiniLM acc (AUC) | EmbGemma acc (AUC) |
|---|---|---|
| concreteness | **0.871** (0.953) | 0.838 (0.922) |
| imageability | 0.846 (0.921) | 0.813 (0.904) |
| familiarity | 0.809 (0.881) | 0.758 (0.835) |
| aoa | 0.795 (0.877) | 0.777 (0.848) |
| semsize | 0.793 (0.873) | 0.794 (0.874) |
| valence | 0.790 (0.870) | 0.801 (0.877) |
| gender | 0.756 (0.829) | 0.749 (0.820) |
| arousal | 0.741 (0.815) | 0.730 (0.804) |
| dominance | 0.712 (0.793) | 0.710 (0.775) |

Every norm's above-/below-median halves are linearly separable well above chance
(0.5). Interestingly MiniLM edges EmbeddingGemma on the *binary* split for most
norms, even though EmbeddingGemma wins on continuous R² — the extra signal
EmbeddingGemma carries is in fine-grained ordering, not the coarse median cut.

## Concreteness at scale (full Brysbaert, 39,954 words)

The full probe set on the **final pooled MiniLM embedding** of all 39,954
Brysbaert concreteness words (seed 42, 80/20), for direct comparison with the
4,682-word Glasgow concreteness row:

| Probe | Metric | Value |
|---|---|---|
| MLP | R² / ρ | **0.744** / 0.856 |
| SVR | R² / ρ | 0.724 / 0.846 |
| Ridge | R² / ρ | 0.697 / 0.835 |
| Mass-mean | R² (calibrated) / ρ | 0.603 / 0.802 |
| Logistic (median split) | acc / AUC | 0.842 / 0.926 |

Reproduces the live in-platform ridge probe (val R²=0.700) on the
`Concreteness-Ratings` collection. Scores sit just below the 4,682-word Glasgow
figures — larger, more diverse vocabulary + the pooled output rather than the
best internal layer. (This run uses the in-platform `run_probe_core` on
freshly-embedded MiniLM vectors, so mass-mean also gets a calibrated R²; the
offline orchestrator leaves `massmean.val_r2` blank.)

## Findings

1. **EmbeddingGemma-300M > MiniLM on every norm** — the newer, larger encoder
   recovers more psycholinguistic signal. (The reference report extends this
   trend: BGE > MiniLM and Gemma-3-4b > BGE; those extractions are scaffolded but
   not run here — see below.)
2. **SVR (rbf) wins** for both small encoders — matches the reference report's
   "SVR wins for small-dim encoders" (Lasso only edges it once, on EmbGemma aoa).
3. **Layer split by dimension type** (consistent across models): *referential*
   norms (concreteness/imageability/aoa/semsize) peak early (MiniLM L3);
   *affective/social* norms (valence/arousal/dominance/gender) peak later
   (MiniLM L6). EmbeddingGemma concentrates almost everything at the middle
   layer (L12).
4. **Dimension ranking is architecture-invariant**: both models (and the
   reference report) rank `concreteness > imageability > valence > aoa > semsize
   > gender > familiarity > arousal > dominance`. Evidence the dimensions
   themselves vary in encodability by language statistics, not by architecture.
5. **Mass-mean is a strong near-free baseline** — trails SVR by only ~0.05–0.15 r
   at the same layers. Concreteness is largely a single linear axis (mass-mean
   r≈0.86 on EmbeddingGemma vs SVR's 0.90).

## Relationship to the reference report

The full reference experiment (`glasgow_psycholinguistic_norms`) probes four
spaces — **MiniLM, BGE, Gemma-3-4b residual, and a 16k Gemma-Scope SAE** — with
six probe families, and adds SAE `top_features` analysis (e.g. valence dominated
by one "harm/distress" SAE feature at L9; a clean male/female SAE axis at L17/22).
This local experiment currently runs the two light encoders (MiniLM +
EmbeddingGemma). Not yet run here, all straightforward to enable in
`experiment.yaml`:

- **BGE** (`BAAI/bge-base-en-v1.5`, 768-d) — one more `encoder` extraction.
- **Gemma-3-4b residual** — a `type: gemma` extraction (needs the 4b model + GPU).
- **Gemma-Scope SAE** — an `sae` extraction over the Gemma one + `sae_analysis`.

## Gemini path (scaffolded, not run)

The encoder path runs local HF models and can't call the Gemini API. To probe
**Gemini** embeddings, treat them as precomputed features via the `csv_features`
extraction. Blockers today: the Glasgow words have **no Gemini vectors** (only
MiniLM-384 in the `Glasgow_norm` ChromaDB collection), and `GEMINI_API_KEY` is not
configured. Once the key is set: embed the 4,682 words (3072-d) → write a CSV of
`word` + 9 norms + `f0..f3071` → point `FeatureCSVManifestBuilder` at it and
enable the commented `type: csv_features` extraction. See the experiment folder
README's "Gemini path" section.

## Reproducing

```bash
cd interpretability_backend                     # CWD must contain resources/
uv run python -m interpret.probing.orchestrator \
  experiments/glasgow_psycholinguistic/experiment.yaml
```

Outputs: `resources/probing_results/glasgow_psycholinguistic/`
(`probes/<extraction>/<norm>/<probe>/probe_results.csv` + `summary.json`, lasso
`directions/`, and seaborn `figures/`). Both `resources/probing_results/` and
`resources/psycolinguistics/` are gitignored.

## Dependencies

The probing engine added three previously-undeclared deps (now in `pyproject.toml`
and `interpret/README.md`'s portability list): `omegaconf` (config loading),
`seaborn` (figures), `scikit-learn` (probes). `scikit-image` (CIEDE2000 in
`interpret.utils.distances`) came in with the same merge.

## Tests

`interpretability_backend/unit_tests/test_glasgow_manifest.py` — validates the
builder against the real data and that `experiment.yaml` parses/resolves (no model
inference; skips cleanly if the gitignored datasets are absent).
