# Projection Fidelity via the Mantel Test

How faithfully does a low-dimensional projection (UMAP-3D, PCA-3D, …) preserve
the structure it was built to visualize? This module answers that with a
**Mantel test** — a rank correlation between two pairwise-distance structures —
scoring each projection against one or more *reference* structures:

- the **original embedding geometry** (does the 2D/3D layout keep the high-dim
  distances?), and
- for colour datasets, **perceptual colour distance** (does the layout keep
  colours that look alike close together?).

Code: [`interpretability_backend/evaluation/projection_fidelity.py`](../interpretability_backend/evaluation/projection_fidelity.py)
(pure metrics) + [`run_projection_fidelity.py`](../interpretability_backend/evaluation/run_projection_fidelity.py)
(config-driven runner). Built on the merged `interpret/` toolkit
(`interpret.utils.mantel`, `interpret.utils.distances`).

## Run

```bash
# with the backend stopped (DuckDB is single-writer)
uv run python -m interpretability_backend.evaluation.run_projection_fidelity
# custom config:
ORRERY_PROJECTION_FIDELITY_CONFIG=/path/to/config.toml \
  uv run python -m interpretability_backend.evaluation.run_projection_fidelity
```

Config (`projection_fidelity_config.toml`): `collections`, `projection_types`,
`colour_field` (metadata hex field, `""` to disable colour), `k`, `n_perms`,
`seed`, `sample_size`.

## Methodology

### Distance structures

For a collection of `N` items each structure is a **condensed** pairwise-distance
vector of length `N·(N−1)/2` (scipy convention).

| Structure | Space | Metric | Builder |
|---|---|---|---|
| **Colour** (reference) | perceptual CIELAB | **CIEDE2000** ΔE | `colour_distances(hex_codes)` |
| **Embedding** (reference) | original vectors (e.g. Gemini 3072-d) | **cosine** | `embedding_distances(X)` |
| **Projection** (target) | UMAP-3D / PCA-3D coords | **Euclidean** | `projection_distances(X)` |

Colour: each `#rrggbb` → sRGB → CIELAB (`skimage.color.rgb2lab`) → CIEDE2000
(`interpret.utils.distances.pairwise_lab_ciede2000`). CIEDE2000 is the standard
perceptual colour-difference metric; cosine matches how the embeddings are
indexed (ChromaDB cosine); Euclidean is the natural metric in a projected space.

> scikit-image is imported **lazily inside** `colour_distances`, so importing the
> module and running embedding-space fidelity needs no scikit-image — only the
> colour reference does. If it isn't installed, the runner logs a warning and
> skips colour.

### The three Mantel statistics

For a `(reference, target)` pair (`MantelTest` in `interpret/utils/mantel.py`):

1. **Global ρ** — Spearman rank correlation over *all* `N·(N−1)/2` distance pairs.
   Measures whether the projection preserves the **whole** distance ordering.
2. **kNN-local ρ** (`k`, default 10) — for each item take its `k` nearest
   neighbours *in the reference*, pool the `(reference, target)` distance pairs
   over all items, and correlate. Measures whether **local neighbourhoods**
   survive, even when global structure is weak.
3. **Permutation test** (`n_perms`, default 1000) — jointly permute the target's
   rows/columns `n_perms` times (the standard Mantel relabelling) to build a null
   distribution of ρ, then report the **z-score** and **empirical p** (fraction of
   null ρ ≥ observed). Guards against spuriously "high-looking" correlations.

All metrics degrade to `null` on degenerate input (constant distances, length
mismatch); `evaluate()` never raises.

### Cost guard

Pairwise distances are `O(N²)`. Collections larger than `sample_size` (default
2000) are randomly subsampled (seeded). At `N ≈ 1000` the full run — including a
1000-permutation null for every comparison — takes a couple of minutes.

## Results — `xkcd_hilbert_gemini` (954 XKCD colours, Gemini 3072-d)

Reference baseline — **colour ↔ Gemini embedding: global ρ = 0.29**. The
embedding only moderately reflects perception because it embeds colour *names*
("blue with a hint of purple"), i.e. semantics, not pixels.

| Reference | Projection | Global ρ | kNN@10 ρ | perm z | p_emp |
|---|---|---:|---:|---:|---:|
| **Perceptual colour** (CIEDE2000) | **UMAP-3D** | **0.601** | **0.201** | +120.7 | 0.000 |
| | PCA-3D | 0.475 | 0.123 | +97.0 | 0.000 |
| **Gemini embedding** (cosine) | UMAP-3D | 0.287 | **0.338** | +28.7 | 0.000 |
| | **PCA-3D** | **0.491** | 0.249 | +48.4 | 0.000 |

All correlations are hugely significant (permutation z = 29–121, empirical
p = 0.000).

### Interpretation

- **Global embedding geometry → PCA wins** (0.49 vs 0.29). Expected: PCA is a
  linear variance-preserving projection, so large-scale distances survive.
- **Local neighbourhoods → UMAP wins** (kNN 0.34 vs 0.25). Expected: UMAP
  explicitly optimizes local structure at the expense of global distances.
- **Perceptual colour alignment → UMAP wins, strongly** (0.60 vs 0.47 global).
  Strikingly, the UMAP-3D layout is *more* colour-coherent (0.60) than the raw
  3072-d embedding it was built from (0.29) — the projection concentrates the
  colour signal that is diffuse in the full embedding.

**Bottom line:** UMAP-3D is the better layout for perceptual colour organization
and local neighbourhood structure; PCA-3D is the more honest picture of the
embedding's global geometry.

### Caveats

- **Projections are derived from the embedding**, so embedding→projection fidelity
  is partly a measure of *how much* structure a dimensionality reduction discards,
  not a free-standing ground truth. Colour is the independent reference here.
- **kNN neighbourhoods are defined in the reference space**, so the local probe
  asks "are *reference*-neighbours kept close in the projection?" — reference and
  target are not interchangeable for that statistic.
- Cosine vs Euclidean across spaces is intentional (each space's natural metric);
  Spearman is rank-based, so monotonic metric differences don't bias it.
