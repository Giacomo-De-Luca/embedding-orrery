# Topic-Quality Metrics: Methodology and the BERTopic Comparison

Why the topic-quality evaluator reports what it reports — and the controlled
experiment (2026-07-06, `emotion`, 1,000 docs, `all-MiniLM-L6-v2` 384-d) that
decided it.

Implementation: `interpretability_backend/evaluation/quality_metrics.py`
(metrics) + `backend/services/topic_quality_service.py` (orchestration).
Usage/API reference: `interpretability_backend/evaluation/README.md`.

---

## TL;DR

1. **Silhouette on raw high-dimensional embeddings is useless as a quality
   signal.** It scores ~0.05–0.08 for *any* clustering of this data — good or
   bad — because cosine distances concentrate in 384-D. It was removed.
2. **Measured in the space the clustering actually ran in**, the same clusterings
   score ~0.55, which tracks human judgment of the topics.
3. **Our pipeline and BERTopic are the same algorithm.** Given the same reduced
   input, they produce an *identical* clustering (adjusted Rand index = 1.0) and
   identical metrics. There is no "our topics vs BERTopic topics" gap to close.
4. **DBCV is the geometric metric of record for HDBSCAN** — density-aware,
   noise-aware, and it discriminates where raw silhouette cannot.
5. The real defect surfaced by all this: **the pipeline clustered on the
   visualization projection** (`min_dist=0.1`, 2-D), which is tuned for looking
   good, not for clustering. This motivated `cluster_on="cluster_umap"` (a
   dedicated 5-D, `min_dist=0.0` reduction) becoming the default.

---

## Background: what triggered the investigation

Scoring the stored `emotion` extraction (16 topics, 524/1000 noise) produced
numbers that looked alarmingly bad next to topics whose LLM labels read as
near-perfect:

| Metric | Stored `emotion` extraction |
|---|---|
| Silhouette (raw 384-D embedding, cosine) | **0.061** |
| Silhouette (stored `umap_2d` projection) | 0.310 |
| Topic diversity | 0.875 |
| Coherence C_v | 0.320 |
| Coherence U_Mass | −14.12 |

Two hypotheses: either the metrics were broken, or the topics were worse than
the labels suggested. Both turned out to be partly true.

**Sampling the actual documents** showed the labels *are* flattering: Topic 0
"Fashion and style" contained documents about sauces and spices; "Personal
feelings" appeared as *two separate topics* (3 and 9); and 524 of 1000 documents
were unclustered noise. LLM labeling polishes topics into sounding coherent.

But the metrics were also misleading — see below.

---

## Experiment 1: is noise leaking into the metrics?

HDBSCAN assigns `-1` to noise. Including it as a cluster wrecks any metric.
Verified on a BERTopic fit (523 clustered, 477 noise):

| Computation | Silhouette |
|---|---|
| 384-D, **noise excluded** (what the evaluator does) | **0.054** |
| 384-D, **noise treated as a cluster** (the broken way) | **−0.04** |
| 5-D UMAP (clustering space), noise excluded | **0.522** |
| 2-D UMAP (stored projection), noise excluded | 0.367 |

**Conclusion**: noise is correctly excluded (`labels != -1` for silhouette;
topic `-1` skipped for diversity/coherence). The low raw-space number is *not* a
noise-handling bug.

---

## Experiment 2: our pipeline vs BERTopic (seeded, like-for-like)

Both fitted on the same MiniLM vectors. "Current" = HDBSCAN on the stored
`umap_2d` viz projection (the pipeline's behavior at the time). BERTopic =
its own UMAP (5-D, `min_dist=0.0`) + HDBSCAN. `min_cluster_size=10`, seed 42.

| | Current (HDBSCAN on 2-D viz projection) | BERTopic (UMAP 5-D + HDBSCAN) |
|---|---|---|
| topics / noise | 2 / 38 | 19 / 508 |
| **DBCV** (noise-aware) | 0.010 | **0.173** |
| Silhouette, **raw 384-D** (cosine) | 0.082 | 0.075 |
| Silhouette, **reduced clustering space** | 0.241 (2-D) | **0.552** (5-D) |
| Topic diversity | 0.600 | 0.668 |
| Coherence C_v | 0.349 | 0.280 |
| Coherence U_Mass | −3.23 | −12.52 |

Read the **raw 384-D row**: 0.082 vs 0.075. A degenerate 2-cluster collapse and
a healthy 19-topic clustering are indistinguishable. That row is noise.

Read the **reduced-space row and DBCV**: 0.241 vs 0.552, and DBCV 0.010 vs
0.173. Both correctly prefer the real clustering.

> Caveat on the "Current" column: clustering the *stored viz projection* with
> `min_cluster_size=10` collapsed to 2 blobs. That is not the stored 16-topic
> extraction (which used different parameters) — it is what the projection-based
> path does at these settings, and it is exactly the weakness the experiment was
> built to expose.

---

## Experiment 3: hold dimensionality fixed — do the measures coincide?

Same MiniLM vectors, one UMAP per dimensionality (`n_neighbors=15`,
`min_dist=0.0`, cosine, seed 42), then HDBSCAN (`min_cluster_size=10`).
Silhouette computed *in the same space the clustering ran in*, noise excluded.

| Clustering dim | topics | noise | Silhouette **in that space** | Silhouette **raw 384-D** | DBCV |
|---|---|---|---|---|---|
| 2-D | 25 | 464 | **0.575** | 0.051 | 0.130 |
| 5-D | 19 | 508 | **0.552** | 0.075 | 0.173 |
| 10-D | 20 | 532 | **0.547** | 0.072 | 0.134 |
| 20-D | 21 | 514 | **0.542** | 0.060 | 0.116 |

And the direct head-to-head at the same 5-D input:

```
our UMAP+HDBSCAN vs BERTopic  ->  adjusted Rand index = 1.0
silhouette(ours, 5-D) = 0.5522   silhouette(BERTopic, 5-D) = 0.5522
```

**Conclusions.**
- In-space silhouette is **stable (~0.54–0.58) across every dimensionality** —
  the clusters are genuinely well-formed.
- Raw 384-D silhouette stays pinned at ~0.05–0.08 regardless — a pure
  dimensionality artifact (distance concentration), not a quality signal.
- **BERTopic *is* our pipeline** (UMAP → HDBSCAN → c-TF-IDF). Same reduced
  input ⇒ identical clustering ⇒ identical metrics. ARI = 1.0.

Note that fresh 2-D clustering scores 0.575 — so 2-D per se is not the problem.
The problem was clustering the **viz-tuned** projection.

---

## `min_dist`: visualization vs clustering

UMAP's `min_dist` sets the minimum spacing allowed between neighboring points in
the low-dimensional layout. It does not change *which* points are connected
(that's `n_neighbors`) — only how tightly they may be packed.

| | `min_dist = 0.0` (clustering-tuned) | `min_dist = 0.1+` (viz-tuned) |
|---|---|---|
| Layout | Dense, compact clumps; hard empty gaps | Points spread out; softer gaps |
| Good for | **HDBSCAN** — sharp density boundaries, fewer merges | **Reading the picture** — within-cluster gradient, less overplotting |
| Cost | Overplotted, hard to read | Clusters bleed together; HDBSCAN merges or noises them |

The two goals pull in opposite directions. The pipeline was reusing **one**
projection (`umap_2d`, `min_dist=0.1`) for both the picture and the clustering,
and the clustering paid for it. Hence the dedicated clustering reduction
(`cluster_on="cluster_umap"`, 5-D, `min_dist=0.0`) — which is precisely what
BERTopic does internally, and why its clusters scored better.

---

## What the evaluator reports (and why)

| Metric | Result key | Status |
|---|---|---|
| DBCV | `dbcv` | **Primary geometric metric.** HDBSCAN-native, noise-aware. Needs the live fitted model, so it is `null` when re-scoring stored labels. |
| Silhouette in the clustering space | `silhouette_cluster_space` | Euclidean, noise excluded, subsampled above `sample_size`. |
| Silhouette in raw embedding space | — | **Removed.** Non-discriminative (this document). |
| Topic diversity | `topic_diversity` | Unique ÷ total words over top-N keywords. |
| Coherence C_v | `coherence_cv` | gensim `CoherenceModel`; best human correlation. Expensive. |
| Coherence U_Mass | `coherence_umass` | gensim; relative measure only. |

**Provenance caveat.** When scoring *stored* labels the evaluator uses the
extraction's stored projection as the silhouette space. For
`cluster_on="cluster_umap"` extractions the ephemeral 5-D clustering UMAP is not
persisted, so that projection is a **proxy**. The `cluster_space` field in every
result records what was actually used (e.g. `"cluster_umap/umap_3d"`).

### Interpreting the numbers

- **Silhouette** (−1..1): `>0.5` strong · `0.25–0.5` reasonable · `~0`
  overlapping · `<0` points sit closer to another cluster.
- **DBCV** (−1..1): higher better; density-aware analog of silhouette.
- **Topic diversity** (0..1): higher = topics use distinct vocabulary. `<0.5`
  suggests redundant topics (reduce them).
- **C_v** (~0..1): `>0.65` strong · `0.55–0.65` good · `0.40–0.55` acceptable ·
  `<0.40` weak.
- **U_Mass** (≤0, ~−14..0): closer to 0 better; only comparable **within** one
  corpus.

Absolute thresholds are corpus- and encoder-dependent. These metrics are most
useful for comparing **configurations of the same dataset**, not as universal
pass/fail marks.

---

## Reference results

`emotion` (1,000 docs, 16 stored topics, no reduction; scored via `evaluateTopics`,
cluster space `projection/umap_3d`):

| DBCV | Silhouette (cluster space) | Diversity | C_v | U_Mass |
|---|---|---|---|---|
| — | 0.471 | 0.875 | 0.320 | −14.12 |

`lacan_sentences_gemini_document` (153,772 docs, reduction applied; silhouette
subsampled to 10k, cluster space `projection/umap_2d`):

| Level | clusters | Silhouette | Diversity | C_v | U_Mass |
|---|---|---|---|---|---|
| subtopic (pre-reduction density clusters) | 889 | **−0.249** | 0.814 | 0.358 | −12.31 |
| topic (merged) | 19 | −0.379 | — | 0.340 | — |

The subtopics score better than the merged topics on every geometric measure:
reduction merges topics in **c-TF-IDF space**, which does not preserve geometric
separation in the projection. When reduction has been applied, evaluate at
`level="subtopic"` to judge the clustering itself.

Both collections' negative or near-zero silhouettes on the *stored viz
projection* are the expected consequence of measuring in a `min_dist=0.1` space —
compare against the 0.55 obtainable in a proper clustering space.

---

## Reproducing

BERTopic is **not** a project dependency; install it transiently
(`uv pip install bertopic` — it reuses the existing umap/hdbscan/sklearn and adds
one package). Fix `random_state` on the UMAP: BERTopic's default is unseeded, so
successive `fit_transform` calls give different clusterings (we observed 502 vs
508 vs 523 noise points across runs).

```python
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.metrics import silhouette_score, adjusted_rand_score

u5 = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
          metric="cosine", random_state=42).fit_transform(emb)
hdb = HDBSCAN(min_cluster_size=10, metric="euclidean",
              cluster_selection_method="eom", gen_min_span_tree=True)
labels = hdb.fit_predict(u5)                 # gen_min_span_tree -> relative_validity_ (DBCV)

mask = labels != -1                          # ALWAYS exclude noise
silhouette_score(u5[mask], labels[mask])     # in-space: ~0.55
silhouette_score(emb[mask], labels[mask], metric="cosine")  # raw 384-D: ~0.07 (meaningless)
hdb.relative_validity_                       # DBCV: ~0.17
```

The in-platform equivalent is the `evaluateTopics` GraphQL mutation or the
config-driven runner:

```bash
uv run python -m interpretability_backend.evaluation.run_evaluation
```
