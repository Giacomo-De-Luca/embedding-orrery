# Topic Merge Trees and Topic Reduction

Topic reduction merges an extraction's topics into fewer, broader ones. It is
built on **merge trees**: a dendrogram over the extraction's original clusters
that can be cut at any number of topics. The tree drives a live "Topics" slider
on the Explore page; Apply saves the cut as a reduced extraction.

## Why

The original BERTopic-style reduction did not work on our data (seed snapshot,
`evaluation/run_reduction_comparison.py`):

- **"auto"** ran HDBSCAN (`min_cluster_size=2`) on the topic vectors. c-TF-IDF
  rows are nearly orthogonal (median cosine 0.06 to 0.08), so every topic was
  noise and nothing merged.
- **Fixed N on c-TF-IDF** (the old UI default) chained everything into one
  topic: on emotion at 3 topics, the largest held 95% of the clustered documents.
- **Raw semantic centroids** all look alike (Gemini xkcd: cosine 0.77 to 0.94),
  because averaged embeddings drift toward the dataset mean.
- Reductions were **one-shot and lossy**: a second reduction treated the current
  topics as the new subtopics, so the original clusters were lost and a
  reduction could not be undone or made coarser-then-finer.

## Base topics

A reduction always starts from the extraction's **base topics**: its original
clusters. For an unreduced extraction they are the topics (`topic_id`); for a
reduced one they are the subtopics (`subtopic_id`, the clusters it was reduced
from). So reductions never compound, any topic count can follow any other, and
cutting at the base topic count resets to the original clusters (no subtopics).

A cut that merges nothing and absorbs no noise keeps the base ids (otherwise
groups are numbered by size), so the stored density tree still lines up with
the topics. A reduction counts as applied (`reduction_applied`, subtopics
stored) only when a group has more than one base topic or noise was absorbed;
"before" counts never include noise.

Noise (`-1`) is never a leaf. Noise a density cut absorbed keeps subtopic `-1`,
so it is base noise again for the next reduction.

`BaseTopics` (`backend/services/topic_reduction_service.py`) loads them:
`BaseTopics.load(db, collection, trees)` reads the active extraction's
assignments joined with their documents (`DuckDBClient.get_assignment_documents`,
one join, items of the collection's dataset), labels, keywords and config
snapshot; `BaseTopics.from_clustering(...)` builds them during an extraction.

## Merge trees

`backend/utils/topic_merge_tree.py` (scipy at module level, imported lazily by
schema-reachable code):

- **`TopicMergeTree`**: `leaf_ids` (base topic ids), `leaf_sizes` (documents),
  `children` `(k-1, 2)` and `heights` `(k-1,)` in scipy's linkage convention: an
  index below `k` is a leaf, `k + i` is the node made by row `i`. Rows are in
  merge order (heights non-decreasing, children before parents); the
  constructor validates this. Density trees add `noise_merges`: noise item id
  to the merge row whose branch it fell out of.
- **`TopicTreeCut`**: a cut at N topics applies the first `k - N` rows
  (union-find); a cut at height `h` applies the rows with height `<= h`. Groups
  are named by their root node, so a group keeps its name while it does not
  change. `representatives` gives each group's largest leaf (ties: lowest id),
  which the slider uses for stable colours and preview labels. A noise item is
  absorbed once its merge row has been applied. The frontend mirrors the cut in
  `lib/utils/topicTreeCut.ts`, tested on the same literal examples
  (`unit_tests/test_topic_merge_tree.py`).

## Strategies

Names live in the lean `backend/services/topic_reduction_types.py`
(`STRATEGIES`, `ReductionRequest`, `TopicReductionSettings`).

| Strategy | Tree | Notes |
|---|---|---|
| `average` | average linkage on the cosine distance between **centred** centroid directions | the centred space topic search and the heatmap order use (`TopicCentroidRanker.origin`); reuses `SimilarityOrdering` |
| `ward` | Ward linkage weighted by topic size on the same directions | `SizeWeightedWard`: nearest-neighbour chain plus a Lance-Williams update with initial sizes, exact (checked against brute force); merging into a big topic costs more, so sizes stay balanced |
| `density` | HDBSCAN's condensed tree above the selected clusters | HDBSCAN extractions made after this feature; can **absorb noise** (below) |
| `legacy_ctfidf`, `legacy_semantic` | the old `reduce_to_n_topics` / `auto_reduce_topics` on c-TF-IDF rows or raw centroids | kept only for comparison; to be removed (see the end) |

**Method**: `fixed_n` cuts at `n_topics` (noise not counted, for every strategy;
the legacy methods get N+1 internally when noise exists). `auto` exists for
`average` (cut at `distance_threshold`, centred cosine distance, default
`auto_distance_threshold` = 0.7 from `config/topic_reduction.json`) and for the
legacy strategies (HDBSCAN on the topic vectors). `ward` and `density` need N.

### Density trees

`DensityTreeBuilder.from_condensed_tree(condensed, labels)` reads
`model.condensed_tree_.to_numpy()` (rows of parent, child, lambda, size):

- each cluster label's node is the lowest common ancestor of the nodes its
  points fell out of (no private hdbscan API; on real fits it matches
  `_select_clusters()`);
- the nodes above the selected clusters always split in two, giving exactly
  `k - 1` merges at height `1 / lambda` of the split, deeper splits first
  (disconnected components, lambda 0, go above every finite merge);
- every noise point fell out of one of those nodes before it split: that
  node's merge **absorbs** it. Once a cut applies the merge, the point joins the
  merged topic (it was in that branch until the branch's density dropped below
  the point's). On the emotion seed (fresh clustering, 18 clusters, 50% noise),
  cutting to 6 topics absorbs 161 of 497 noise documents.

`cluster_points` (`topic_extraction/cluster_and_label.py`) builds the tree in
the process that fitted the model (it runs in a CancellableTask child for large
collections) and returns a `ClusteringOutcome(labels, density)`.

## Storage

- **`topic_merge_trees`** (DuckDB, FK to `topic_extractions`, PK
  `(extraction_id, linkage)`): `leaf_ids`, `leaf_sizes`, `left_children`,
  `right_children`, `heights`, `noise_item_ids`, `noise_merges`. Only density
  trees are stored, written with the extraction (and copied to every reduction
  of it, whose base topics are the same). A new table, like `topic_centroids`,
  so no ALTER on an FK table is replayed from the WAL. Deleted with the
  collection (`delete_vector_collection`, `_TOPIC_CHILD_TABLES`) and copied
  into seed snapshots (skipped when the source predates the table).
- **Semantic trees are not stored**: they are rebuilt from `topic_centroids`
  (cached per extraction). Extractions without stored centroids (older ones,
  the current demo seed) get their base centroids computed from the vectors in
  memory, up to `fallback_centroid_max_items` (200 000) items; the
  `scripts/backfill_topic_centroids.py` backfill makes the first load faster.
- **Activation**: topic sync (`_sync_topics_to_duckdb`) writes a new extraction
  inactive (`create_topic_extraction(activate=False)`), inserts its info,
  assignments and merge trees, then activates it in one statement
  (`activate_topic_extraction`: `is_active = (id = ?)`). A failure removes the
  partial rows (`delete_topic_extraction`), keeps the previous extraction
  active and fails the job (it used to log and report success).
- **Config snapshot**: a reduced extraction stores its source's snapshot plus
  `"reduction": {strategy, method, n_topics, distance_threshold, absorb_noise,
  base_extraction_id, absorbed_noise}`; snapshots now include `language`. (The
  standalone reduction used to store no snapshot, so `evaluateTopics` scored
  the wrong space.)
- **`topic_hierarchy`** of new reductions is keyed by topic id
  (`{topic_id: [base topic ids]}`); older rows keep the label-keyed format.
  Readers only test whether it is set.

## Services

- **`TopicMergeTreeService`** (`services/topic_merge_tree_service.py`, modelled
  on `TopicOrderService`): `shared(db)`, `view(collection, linkage)` for the
  query (null when the collection has no topics, fewer than 3 or more than
  `max_tree_topics` base topics, or no centroids and too many items),
  `tree_for_assignments(...)` for reductions, `density_tree(...)`. Base
  centroids (including a failed computation), trees and the stored density
  tree (including "none") are cached per extraction (read by extraction id, so
  a cached entry always belongs to its key); labels are re-read per request.
  Reduction jobs use their own instance (cancellable vector loads); the build
  is deterministic, so the saved cut is still the previewed one.
- **`TopicReductionService`** (`services/topic_reduction_service.py`):
  `plan(base, request)` cuts the base topics and stores nothing (the
  comparison runner uses it too); `assemble(plan, keep_unmerged_labels=...)`
  makes labels, topic infos, assignments, subtopics and the hierarchy;
  `unreduced(base)` assembles an extraction without reduction.
  - Standalone reduction (`keep_unmerged_labels=True`): a topic made of one
    base topic keeps its label (an LLM label survives); merged topics get
    keyword labels, or LLM labels with `regenerate_labels` (merged topics only).
  - Inline reduction during extraction: every final topic gets a fresh keyword
    label, or an LLM label when the extraction uses LLM labels (as before).
- **`topic_extraction_service`** keeps persistence: `extract_topics` (inline
  reduction via `plan`) and `reduce_existing_topics` (a thin wrapper:
  load, plan, assemble, sync, refresh centroids; four integer progress stages)
  call `_sync_topics_to_duckdb`. The reduction services are imported inside
  functions (torch-free / lean schema import boundary).

## GraphQL

- `topicMergeTree(collectionName, linkage = "average", includeNoise = false)`
  returns `TopicMergeTree {extractionId, linkage, baseLevel, leaves {topicId,
  label, size}, merges {left, right, height, size}, noiseCount,
  availableLinkages, noiseItemIds, noiseMerges}` or null. Async, lazily
  imported service; works in the read-only demo.
- `reduceTopics(input)` and the inline `TopicConfigInput.reduction` take
  `strategy` (null: the legacy strategy `useCtfidf` picks), `distanceThreshold`,
  `absorbNoise`; `ReduceTopicsResult.absorbedNoise`. Invalid combinations are
  GraphQL errors (`ReductionRequest` validation, via
  `converters.reduction_request_from_input` / `build_topic_extraction_config`).

## Comparing strategies offline

```bash
uv run python -m interpretability_backend.evaluation.run_reduction_comparison
```

Config `interpretability_backend/evaluation/reduction_comparison.toml` (env
`ORRERY_REDUCTION_COMPARISON_CONFIG`): collections, strategies, topic counts,
distance thresholds, metrics, an optional ground-truth field per collection,
and optional `duckdb_path` / `chroma_path` to run on a copy. It writes JSON and
a Markdown report; nothing is stored. Per case: topic count, largest-topic
share, size entropy, noise share and absorbed noise, keyword diversity, C_v and
U_Mass coherence, silhouette in the stored projection, NMI / ARI against the
ground truth, and ARI between strategies at the same topic count.

On the seed snapshot (no density trees yet: they need a re-extraction):

| Collection, cut | average | ward | legacy_ctfidf | legacy_semantic |
|---|---|---|---|---|
| emotion N=3: largest topic | 0.47 | 0.66 | 0.95 | 0.81 |
| emotion N=3: size entropy | 0.96 | 0.79 | 0.20 | 0.55 |
| xkcd N=3: largest topic | 0.49 | 0.49 | 0.93 | 0.51 |
| xkcd N=5: silhouette | 0.43 | 0.33 | 0.11 | 0.30 |

Legacy c-TF-IDF "auto" merged nothing on either collection (legacy semantic
"auto" kept 9 of 16 and 7 of 11 topics). xkcd `average` at 5 topics gives
reds + yellow + food / greens / blue + muted / purples / ugly.

## Frontend

**Explore: the Topics slider.** `lib/hooks/useTopicMergeTree.ts` reads
`topicMergeTree` (`network-only`; `includeNoise` only for density) and
`app/components/TopicTreeControls.tsx` shows it in the Controls panel's Color
section while colouring by `topic_label`: a Topic tree select (Average, Ward,
and Density when the extraction has a density tree), Absorb noise for
Density, a Topics slider from 2 to the base-topic count, Reset and Apply.
`lib/utils/topicTreeCut.ts` mirrors the backend `TopicTreeCut` on the same
literal examples.

Moving the slider starts a **preview** that rewrites the topic fields in the
browser (`lib/utils/topicCutMetadata.ts`, fed to `useVisualizationPoints`
after any probe fields):

- each item's topic becomes its group's largest base topic (its id, its label
  plus " +m" for m other members);
- its subtopic becomes its own base topic, so "Color by subtopics" works even
  before any reduction;
- noise stays noise unless a density cut absorbs it (then: the group's topic,
  subtopic -1), so noise absorbed by an earlier Apply is noise again in other
  previews.

The rebuild is deferred; above 50k items the slider recuts on release.
**Colours stay put** while sliding: a non-persisted colour pin
(`categoryColorPins` in the store, `buildPinnedCategoryColorMap`) gives each
group the colour its representative base topic has in the saved base field
(`topic_label`, or `subtopic_label` once reduced); the pins cover every tree
node, so slider steps never republish them. Nested colours key subtopics by
(topic, subtopic) (`CategoryKey.subtopicKey`), so absorbed noise never
recolours real noise. While previewing, the topic selection and filters are
cleared, `topic_label` is a plain category (legend and 2D cluster-label clicks
mute preview groups), and the Search panel notes that topic search and
filters use the saved topics. The preview ends on Reset, a collection change
or any colour-field change.

**Apply** calls `reduceTopics` (fixed N, the linkage as strategy, optional
absorb noise and LLM labels for merged topics) with inline, cancellable
progress, and keeps the preview frozen until the item data
(`useEmbeddingData.reload`), topics, tree and similarity order have reloaded
and the topic search is reset (Apply renumbers topic ids). Apply is hidden in
the read-only demo, where the slider still works.

**Collections page.** `TopicConfigForm` (inline reduction) and
`topics/ReduceTopicsSection` (standalone) share `ReductionStrategyFields`: the
strategy picker (Average (centred semantic), the default; Ward
(size-weighted); Density (HDBSCAN tree); Legacy: c-TF-IDF; Legacy: semantic),
Fixed N / Auto (Auto only for Average and the legacy strategies; a distance
threshold for Average auto), and absorb noise for Density. Requests are built
by `lib/utils/topicReduction.ts`, which follows the backend validation rules.
The standalone slider's maximum is the base-topic count from the tree query.

**Cached colour options.** The Explore page caches its colour-by options in
the dataset metadata (`field_analysis`). Every topic sync drops that cache,
so new topic fields (subtopics after a reduction) appear on the next load.
After an Explore Apply, `reload()` does not wait for that: the reload
response carries no cache, so `TopicFieldOptions.refreshCache`
(`lib/utils/fieldAnalysis.ts`) rebuilds one from the analysis already in
memory, with the topic-field options recomputed from the reloaded items. The
reload then takes the cached path (no full analysis of every field on the
main thread) and stores the rebuilt cache. Because a reduction changes
neither the items nor their coordinates, `reload()` also keeps the cached
projections while the item membership signature is unchanged.

Frontend tests: `topicTreeCut.test.ts`, `topicCutMetadata.test.ts`,
`app/collections/lib/__tests__/topicConfigInput.test.ts`,
`stores/__tests__/topicCut.test.ts`, plus pinning and composite-key cases in
`categoryColors.test.ts`, `nestedColorUtils.test.ts`, `categoryKey.test.ts` and
`fieldAnalysis.test.ts`. Details: frontend `claude.md` "Topic merge-tree
slider".

## Tests

`unit_tests/test_topic_merge_tree.py` (cuts, builders, density on hand-built and
real HDBSCAN trees), `test_topic_reducer.py` (`reduce_with_tree`),
`test_topic_merge_tree_service.py` (trees, fallback, cache, GraphQL),
`test_topic_reduction_service.py` (base topics, repeated reductions, labels,
density absorption, preview parity, failed sync, progress),
`test_topic_merge_tree_storage.py`, `test_reduction_comparison.py`,
`test_semantic_reduction_fallback.py` (legacy semantic fallback),
`test_clustering_methods.py` (`cluster_points`), `test_topic_extraction_config.py`
(request validation), `test_cancellable_steps.py` (extraction stores its
density tree; inline reduction; cancel), `test_seed_snapshot.py`.

## Removing the legacy strategies (planned)

Once the comparison has been reviewed: delete `TopicReducer.reduce_to_n_topics`
and `auto_reduce_topics`, the c-TF-IDF matrix plumbing they need,
`legacy_ctfidf` / `legacy_semantic`, `useCtfidf` and its null-strategy default,
their UI options, tests and docs.
