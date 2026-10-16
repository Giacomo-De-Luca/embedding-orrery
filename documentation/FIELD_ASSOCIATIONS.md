# Field Associations (Analytics heatmap)

The **Field associations** section of the Explore page's Analytics panel shows
how two categorical metadata fields relate. The motivating question: in the
`combined` collection, do some `source` datasets talk only about specific
topics, or is each source spread across many?

The section is collapsed by default and costs nothing until opened. Open, it
shows:

- two field pickers, **Group** (rows, e.g. `source`) and **Across** (columns,
  e.g. `topic_label`);
- the overall association strength (Cramér's V, with a verbal reading);
- one bar per Group value with the **effective number of Across values** it
  spreads over ("≈ 3.2 of 64"), most concentrated first;
- an **Open heatmap** button for the full cross-tabulation.

Defaults: Group is the colour field unless it is a topic field (then the
first non-topic categorical field), Across is `topic_label` when the
collection has topics. Picking the field already on the other side swaps them.

## What is counted

`ContingencyTable` (`embedding_visualization/lib/utils/contingencyTable.ts`)
cross-tabulates the two fields over the loaded points in one pass:

- Only points left visible by the **text-search and temporal filters**
  (`combinedMutedIndices`) are counted, so the view follows the filters.
  Category muting (legend isolate) is a view toggle and does not change it.
- Values go through `CategoryKey.of`, so absent, null and empty values form the
  shared **Missing values** row/column, listed last (as in the category list).
- **Unclustered topic noise** (HDBSCAN −1) is left out on topic fields, as in
  the category list (`getUnclusteredValues`).
- Each axis keeps at most **50 values** (`MAX_ASSOCIATION_VALUES`); the smallest
  values beyond that fold into **Other** (ties broken by name). Missing is kept
  apart from the cap. Cramér's V is computed on the folded table.

Statistics, all on `ContingencyTable`:

| Statistic | Definition | Used for |
|---|---|---|
| Expected count | `rowTotal × colTotal / N` | tooltip |
| Pearson residual | `(observed − expected) / √expected` | enrichment colour |
| Ratio | `observed / expected` | "3.4× expected" labels |
| Row / column share | cell ÷ row total, cell ÷ column total | share modes |
| Effective count | `exp(entropy)` of a row's shares: 1 = all in one column, k = even over k | sidebar spread bars |
| Cramér's V | `√(χ² / (N · (min(r, c) − 1)))`, null below two values per axis | summary |

Verbal reading of V: < 0.1 negligible, < 0.3 weak, < 0.5 moderate, otherwise
strong (`AssociationView.strength`).

## The heatmap

`FieldAssociationDialog` + `AssociationHeatmap`
(`embedding_visualization/app/components/charts/`), with all value, colour and
label logic in the pure `AssociationView` (`lib/utils/associationView.ts`).

**Modes** are named by field, so they keep their meaning when the axes swap:

- **Enrichment** (default): Pearson residual on a diverging scale, blue (fewer
  than expected) → gray (as expected) → red (more than expected). The scale is
  symmetric; its limit is the 95th percentile of |residual|, floored at 2 so a
  near-independent table stays pale instead of amplifying noise.
- **Count**: raw counts on a one-hue blue ramp from 0 to the largest cell.
- **% within {Group}** / **% within {Across}**: shares on the same ramp.

Empty cells take the neutral gray in the magnitude modes; in enrichment they
are coloured by their (negative) residual, since absence where points were
expected is information.

**Colours** follow the dataviz reference palette: the blue ramp for magnitude
and the blue/red pair around a gray midpoint for enrichment, with separate
steps for the dark theme (`ASSOCIATION_COLORS`), interpolated in CIELAB.
Cells are separated by a 2px surface gap.

**Layout**: the field with more values goes on the scrolling vertical axis;
**Swap axes** flips it. Row and column labels are sticky. **Values** prints
each cell's compact value (count, share, or ratio) with ink chosen by the
fill's lightness, so the map is readable without hovering.

**Interaction**: one delegated pointer handler serves all cells (they carry
their display position in `data-r`/`data-c`); the hover ring and the tooltip
are overlays, so hovering never re-renders the grid. The grid is focusable and
the arrow keys move a keyboard cursor with the same tooltip. The tooltip leads
with the mode's value ("2.5× expected", "34% of reddit") and lists points,
expected count and both shares.

## PNG export

In **Open heatmap**, click **Export PNG** to download the current view at 2×
resolution. The image includes the title, association summary, selected mode,
legend, labels and full grid, including rows and columns outside the scroll
viewport. It preserves the current axis order, swapped axes, theme and Values
toggle. Label truncation matches the interactive heatmap; hover rings, tooltips
and controls are excluded.

`HeatmapCapture` (`lib/utils/heatmapCapture.ts`) rasterizes the existing heading,
legend and inner grid with the existing `html-to-image` dependency, then stacks
them on an opaque canvas with padding. `downloadCanvasPng` saves the result and
returns whether PNG encoding succeeded. Controls are disabled during export;
failures show a toast and allow retry. Closing the dialog cancels an unfinished
capture without downloading a partial image.

Validation: the association/ordering/contingency Vitest suites remain passing;
`heatmapCapture.test.ts` covers cancellation during lazy import, font embedding
and rasterization. Browser checks compare exported cell pixels against the DOM
in all four modes, both themes, with Values on/off, scrolled rows/columns and
swapped axes. Images are visually checked for title, legend and labels.
Browser checks also cover closing during a delayed first export and retry after
PNG encoding failure. Targeted ESLint with the installed Next flat configs
passes; the repository's default ESLint configuration fails to load, and the
full TypeScript check currently reports unrelated missing `stackglBufferPool`
imports in existing tests.

## Sort orders

`AxisOrdering` (`lib/utils/axisOrdering.ts`), one mode per axis; Other and
Missing always come last.

| Mode | Order |
|---|---|
| **Pattern** (default) | Barycentric seriation: each value sits at the weighted mean position of its cells along the other axis. Weights are each cell's **excess over the independence expectation** (`max(0, observed − expected)`), so a value lands where it is over-represented, not where the biggest columns are; a value with no excess falls back to raw counts. With both axes on Pattern the two sorts alternate (from a size order) until the rows settle, at most 8 rounds. The strongest cells end up on a diagonal. |
| **Similarity** | Topic fields only: the backend order below. Offered (and the default) once available; listed as disabled "(unavailable)" on topic fields whose centroids are not computed. Requesting it without an order falls back to Pattern. |
| **Size** | Largest total first, ties by name. |
| **Name** | The category list's Name order (`comparePeriods`: numbers numerically, otherwise alphabetical). |

## Similarity order (backend)

GraphQL `topicSimilarityOrder(collectionName, field): [String!]` returns a
topic field's values in similarity order, as item metadata carries them, or
null (not a topic field, no active extraction, or centroids not computed).

- **Fields**: `topic_label` / `topic_id` use topic centroids,
  `subtopic_label` / `subtopic_id` subtopic centroids. Ids are returned as the
  strings the collection resolver puts in item metadata; topic labels without
  a label read "Unclustered", as there. The noise topic is excluded.
- **Geometry**: the stored topic centroids (`topic_centroids`, see
  `TOPIC_SEMANTIC_SEARCH.md`) in the same centred cosine space as topic search:
  `TopicCentroidService.ranker.origin(topics)` and
  `CentroidTable.unit_directions(origin)`, so the order follows the
  `CENTER_BY_DEFAULT` choice automatically. Subtopics use the topic-level
  origin, as search does.
- **Ordering** (`backend/utils/similarity_ordering.py`, `SimilarityOrdering`):
  cosine distance, scipy average linkage, then `optimal_leaf_ordering`
  (skipped above 500 rows, where its O(n³) cost bites; WordNet has ~2.9k
  topics). Zero vectors sit at distance 1 from everything. The linkage matrix
  is returned too, for a future dendrogram strip.
- **Subtopics are grouped under their parent topics**: parents in topic
  similarity order, subtopics in similarity order within each parent
  (`SimilarityOrdering.grouped`), matching the nested colour mode.
- **Service** (`backend/services/topic_order_service.py`, `TopicOrderService`):
  caches the order as ids per (extraction id, level), up to 32 entries, behind
  a lock (the resolver runs in a worker thread). Labels are looked up on every
  request (`get_topic_labels` / `get_subtopic_labels`), because topics can be
  renamed. `TopicOrderService.shared(db)` keeps one instance per DuckDB client
  so the cache outlives a request. A new extraction gets a new uuid, so stale
  orders are never served.
- **Import boundary**: `similarity_ordering` imports scipy at module level,
  which `test_torch_free_import` forbids in the schema's import graph. The
  resolver imports `topic_order_service` lazily, like the probing mutation.
- **Cost**: ordering reads only the saved centroids (e.g. 2.9k × 384 floats for
  WordNet, about 4 MB), never the member vectors, so it is safe on the HF Space.
  Centroids must exist: they are written at topic extraction and by
  `scripts/backfill_topic_centroids.py`; the demo seed carries them once it is
  rebuilt after the backfill.

The frontend fetches the order with `useTopicSimilarityOrder` only while the
heatmap dialog is open, for whichever of the two chosen fields is a topic field.

## Shared refactors

- `CategoricalFields` (`lib/utils/fieldAnalysis.ts`): the one rule for which
  fields the Analytics views treat as categorical (a listed field keeps its
  recommended scale; an unlisted one is categorical up to 300 values), used by
  `AnalyticsSidebar`, `CategoryBarList` and this section.
- `computeTemporalCrossTab` (`temporalAnalysis.ts`) is built on
  `ContingencyTable` (periods × categories), so there is one cross-tab
  implementation. `comparePeriods` is the comparator behind `sortPeriods`.
- `isTopicLikeField` (`categoryColors.ts`) names the topic-field check that
  `getUnclusteredValues` already used.

## Tests

- Frontend (vitest): `contingencyTable.test.ts`, `axisOrdering.test.ts`
  (including a shuffled block-diagonal table recovered by Pattern, and the
  excess weighting beating raw counts), `associationView.test.ts` (values,
  labels, colours per theme, enrichment floor, spread rows, field defaults),
  `fieldAnalysis.test.ts` (`CategoricalFields`), `temporalAnalysis.test.ts`
  (cross-tab regressions).
- Backend (pytest): `test_similarity_ordering.py` (cluster contiguity, optimal
  leaf ordering ≥ plain order, scale invariance, tiny and zero inputs, the
  limit, grouping) and `test_topic_order_service.py` (topic pairing on an
  interleaved fixture, ids as strings, subtopic parent blocks, renames without
  recompute, new extraction, unavailable cases, the shared instance, GraphQL
  incl. read-only mode).

## Not done yet

- Clicking a cell to highlight its points on the plot (needs a muting source
  beyond the colour field).
- A field × field overview matrix of Cramér's V across all categorical fields.
- Similarity order for non-topic fields (e.g. `source`), which would need
  on-demand centroids over all member vectors.
- The dendrogram strip beside the heatmap (the linkage is already computed).
