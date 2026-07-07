# Evaluation

Standalone metrics for scoring embedding analyses, plus config-driven runners.
The metric implementations here are pure (no DB/model). Topic-quality scoring is
orchestrated by `backend/services/topic_quality_service.py::score_topic_quality`,
which persists results on the `topic_extractions.quality_metrics` JSON column
(keyed by level) and is shared by the GraphQL `evaluateTopics` mutation and the
TOML runner below.

Two independent evaluators live here:

1. **Topic quality** — `TopicQualityEvaluator` (this file, below).
2. **Projection fidelity** — `ProjectionFidelityEvaluator`: how faithfully a
   projection (UMAP/PCA) preserves the embedding geometry and, for colour
   datasets, perceptual colour distance, via the Mantel test. See the dedicated
   section at the end of this README and
   [`documentation/PROJECTION_FIDELITY.md`](../../documentation/PROJECTION_FIDELITY.md)
   for methodology + results.

## Structure

| File | Purpose |
|---|---|
| `quality_metrics.py` | `TopicQualityEvaluator` — topic metric implementations (pure, no DB/model). |
| `run_evaluation.py` | Config-driven topic-quality runner: thin caller of `score_topic_quality` (loads current labels + projections, evaluates, persists, prints a report, writes JSON). |
| `eval_config.toml` | Which collections to evaluate and the metric parameters. |
| `evaluation_results.json` | Topic-quality output (generated). |
| `projection_fidelity.py` | `ProjectionFidelityEvaluator` — Mantel-test projection fidelity (pure, no DB/model). |
| `run_projection_fidelity.py` | Config-driven fidelity runner: loads projections + item metadata + embeddings, evaluates, prints a report, writes JSON. |
| `projection_fidelity_config.toml` | Collections, projections, colour field, and Mantel parameters. |
| `projection_fidelity_results.json` | Projection-fidelity output (generated). |

## Run

```bash
uv run python -m interpretability_backend.evaluation.run_evaluation
# custom config:
ORRERY_EVAL_CONFIG=/path/to/config.toml uv run python -m interpretability_backend.evaluation.run_evaluation
```

The runner evaluates the **current active** topic extraction of each collection
listed in `eval_config.toml`.

## Metrics

Metric selection: pass `metrics = [...]` in the config (or `EvaluateTopicsInput.metrics`
over GraphQL) to compute a subset — names `dbcv`, `silhouette`, `diversity`,
`coherence_cv`, `coherence_umass`. C_v is the expensive one on large collections.

| Metric | Measures | Range / direction | Notes |
|---|---|---|---|
| **DBCV** | Density-based cluster validity | [-1, 1], higher better | The HDBSCAN-appropriate metric. **Only available from a live fitted model** — `null` when scoring stored labels (see below). |
| **Silhouette (cluster space)** | Separation in the space the clustering ran in | [-1, 1], higher better | Euclidean; noise excluded; subsampled to `sample_size`. Result key `silhouette_cluster_space`. |
| **Topic diversity** | Redundancy across topics | (0, 1], higher = less overlap | Unique words ÷ total words across topics' top-N keywords. |
| **Coherence C_v** | Keyword interpretability (best human correlation) | typically (0, 1) | gensim `CoherenceModel`. Primary coherence metric. |
| **Coherence U_Mass** | Keyword co-occurrence | ≤ 0, higher (closer to 0) better | gensim `CoherenceModel`. |

### Why there is no raw-embedding silhouette
An earlier revision also computed cosine silhouette on the original high-dimensional
vectors. It was removed after a controlled comparison (emotion, 1k docs): identical
clusterings (ARI = 1.0) and clearly different-quality clusterings all scored
~0.05–0.08 in raw 384-D space — cosine distances concentrate in high dimensions, so
the number cannot discriminate good from bad clusterings and reads as a false
negative. Measured in the clustering's own reduced space the same clusterings score
~0.55, tracking human judgment. Silhouette is therefore reported **only in the
clustering space** (`silhouette_cluster_space`); DBCV is the geometric metric of
record for HDBSCAN. One caveat: the stored-label scorer uses the extraction's stored
projection as the silhouette space, which for `cluster_on="cluster_umap"` extractions
is a *proxy* (the ephemeral 5-D clustering UMAP is not persisted) — the
`cluster_space` field in the result records this provenance.

### Coherence uses no embedding model
C_v / U_Mass are computed against the documents themselves via gensim, so no
embedding model (potentially remote/API) is loaded. Keyword tokenization mirrors the
c-TF-IDF `CountVectorizer` stop-words config.

### DBCV caveat
DBCV is read from a fitted HDBSCAN model's `relative_validity_`, which is not
persisted with an extraction. When re-scoring stored labels here it is therefore
`null`; it is populated only inside a fresh-extraction flow that still holds the
fitted model.

## Projection fidelity (Mantel test)

`ProjectionFidelityEvaluator` scores how well a projection preserves a reference
distance structure, via a Mantel test (Spearman rank correlation between two
pairwise-distance structures + a permutation significance test).

```bash
# with the backend stopped (DuckDB is single-writer)
uv run python -m interpretability_backend.evaluation.run_projection_fidelity
```

| Statistic | Measures | Notes |
|---|---|---|
| **Global ρ** | Whole distance ordering preserved | Spearman over all `N·(N−1)/2` pairs. |
| **kNN-local ρ** | Local neighbourhoods preserved | Neighbours taken in the *reference* space; `k` configurable. |
| **Permutation z / p_emp** | Significance vs a relabelling null | `n_perms` joint row/col permutations of the target. |

References: **embedding** (cosine) and, for colour datasets, **perceptual colour**
(CIEDE2000 via `colour_field`). Targets: the configured projections
(`umap_3d`, `pca_3d`, …). scikit-image is imported lazily and only needed for the
colour reference. Full methodology + the `xkcd_hilbert_gemini` results (UMAP-3D
preserves perceptual colour at ρ = 0.60; PCA-3D preserves embedding *global*
geometry better; UMAP wins *local*) are in
[`documentation/PROJECTION_FIDELITY.md`](../../documentation/PROJECTION_FIDELITY.md).

## Tests
Unit tests (synthetic data, no DB/model):
```bash
uv run pytest interpretability_backend/unit_tests/test_topic_quality_metrics.py -v
uv run pytest interpretability_backend/unit_tests/test_projection_fidelity.py -v
```
