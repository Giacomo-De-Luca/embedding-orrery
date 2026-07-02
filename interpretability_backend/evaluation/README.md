# Evaluation

Standalone metrics for scoring embedding analyses, plus config-driven runners.
This package is intentionally decoupled from the extraction pipeline and from the
GraphQL/DuckDB layers — it reads existing data, computes scores, and reports them.
Persisting metrics to DuckDB or exposing them over GraphQL is deliberately **not**
done here yet (a later decision).

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
| `run_evaluation.py` | Config-driven topic-quality runner: loads current labels + projections + embeddings, evaluates, prints a report, writes JSON. |
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

| Metric | Measures | Range / direction | Notes |
|---|---|---|---|
| **DBCV** | Density-based cluster validity | [-1, 1], higher better | The HDBSCAN-appropriate metric. **Only available from a live fitted model** — `null` when scoring stored labels (see below). |
| **Silhouette (embedding)** | Are the clusters real in the original vector space | [-1, 1], higher better | Cosine; noise excluded; subsampled to `sample_size`. |
| **Silhouette (projection)** | Separation in the 2D/3D space clustering ran on | [-1, 1], higher better | Euclidean. Partly circular when clustering was done on the projection — read alongside the embedding-space number. |
| **Topic diversity** | Redundancy across topics | (0, 1], higher = less overlap | Unique words ÷ total words across topics' top-N keywords. |
| **Coherence C_v** | Keyword interpretability (best human correlation) | typically (0, 1) | gensim `CoherenceModel`. Primary coherence metric. |
| **Coherence U_Mass** | Keyword co-occurrence | ≤ 0, higher (closer to 0) better | gensim `CoherenceModel`. |

### Projection vs embedding space
Clustering currently runs on the 2D/3D projection (UMAP/PCA). A silhouette on those
same coordinates is partly circular (it scores the space clustering optimized in).
The embedding-space silhouette is the more meaningful "are these clusters real?"
number. The extraction pipeline also supports clustering on the original embeddings
directly via `TopicExtractionConfig(cluster_on="embedding")` — BERTopic reports that
UMAP-then-cluster is usually better, so this is opt-in and the default stays
`"projection"`.

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
