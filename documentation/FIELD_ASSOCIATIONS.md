# Field Associations (Analytics heatmap)

The **Field associations** section of the Explore page's Analytics panel shows
how two metadata fields relate: categorical fields as they are, numeric fields
binned into ordered ranges. Its dialog adds a correlation matrix over numeric
fields. The motivating question: in the `combined` collection, do some
`source` datasets talk only about specific topics, or is each source spread
across many?

The section is collapsed by default and costs nothing until opened. Open, it
shows:

- two field pickers, **Group** (rows, e.g. `source`) and **Across** (columns,
  e.g. `topic_label`);
- the pair's statistic with a verbal reading: Cramér's V, η or Spearman ρ
  depending on the field kinds (see "Pair statistic");
- one bar per Group value with the **effective number of Across values** it
  spreads over ("≈ 3.2 of 64"; Missing and Other count on both sides), most
  concentrated first;
- an **Open heatmap** button for the dialog (Cross-tab and Correlations tabs).

Defaults: Group is the colour field unless it is a topic or numeric field
(then the first non-topic categorical field), Across is `topic_label` when the
collection has topics. Picking the field already on the other side swaps them.
The picks (and the bin settings) live in the section and are shared with the
dialog, whose Cross-tab tab has the same two pickers: changing a field there
changes it in the sidebar too.

Field lists (`AssociationFields`, `lib/utils/fieldAxis.ts`): the cross-tab
offers the categorical fields (`CategoricalFields`), then the continuous
numeric ones (numeric fields recommended a sequential scale), grouped as
"Categorical" and "Numeric (binned)" in the pickers. Discrete numeric fields
(e.g. 0/1 flags) stay categorical there. Topic ids are never numeric.

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
- **Numeric fields are binned** (`NumericBins`, `lib/utils/numericBins.ts`) into
  2–50 ranges (default 10), either **equal count** (edges at quantiles; tied
  values can merge bins) or **equal width** (the range cut into equal steps).
  Edges are computed over all loaded points, so the filters never move them.
  Each bin's key is its range label ("1.90–2.40" for [1.90, 2.40); the last
  bin is closed), printed with the fewest decimals that keep labels distinct
  and within 5% of the narrowest bin of the true edge, never as "-0"; labels
  that still collide (edges closer than 1e-10) get a counter so bins never
  merge. **Whole-number fields** (years, counts) are labelled by the values a
  bin holds ("1993–1996", or "2019") and get one bin per value when they have
  no more distinct values than bins; the toolbar then notes the actual count
  ("Year: 6 bins"). Values that are not numbers go to Missing. A numeric axis
  is always in value order.
- **Bins or values**: each numeric field on the cross-tab has a **Bins | Values**
  switch beside its picker. **Values** shows every distinct value as its own
  category, like a categorical field (Cramér's V, Pattern / Size sorts), but
  ordered by value (−12, −3, 2, 10, not as text). Numeric fields with at most
  50 distinct values start as Values (`AssociationFields.showsValues`: a
  user's choice per field wins), e.g. `century` in the Sanskrit collections
  (30 values); many-valued fields start binned. The choices live in the
  section (`valueOverrides`), shared with the sidebar. Fields that are
  already categorical but hold numbers (0/1 flags, a 19-value `century`)
  also sort by value by default (`FieldAxis.valueOrder`; sorts Value /
  Pattern / Size).
- **Bin labels**: **Ranges** (the keys above) or **Percentiles**, the share of
  points below each bin's edges ("0–10%", "10–20%" for equal-count bins; the
  uneven cumulative shares for equal-width bins, e.g. "0–90%"), with up to two
  decimals when rounded shares would collide. The choice is display only
  (`FieldAxis.labelOf`, `NumericBins.percentileLabel`): it rebuilds no bins,
  applies to the heatmap axes and the sidebar's spread list, and the tooltip
  keeps the value range in brackets while percentiles are shown.
- Each axis keeps at most **50 values** (`AssociationFields.MAX_AXIS_VALUES`); the smallest
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
| Cramér's V | bias-corrected (Bergsma 2013): `φ²` minus its chance level `(r−1)(c−1)/(N−1)`, over the corrected `min(r, c) − 1`; null below two values per axis | summary |

### Pair statistic

`AssociationStatistics` (`lib/utils/fieldAxis.ts`) measures the pair on the
**raw values** (not the bins), over the visible points:

| Fields | Statistic | Range |
|---|---|---|
| categorical × categorical | Cramér's V of the table | 0–1 |
| categorical × numeric | correlation ratio η, bias-corrected (ε²), over the categories as the table folds them (≤ 50 + Other, noise excluded); null with one category | 0–1 |
| numeric × numeric | Spearman ρ (Pearson on average ranks), with Pearson r alongside | −1–+1 |

Why the corrections: plain η² grows by about (groups − 1) / points and plain V
by a similar chance term, so a filter leaving a few thousand points over
thousands of topics made unrelated fields read "strong" (η 0.69 on random
data at 5k points). Corrected, the same data reads 0.00–0.05 at every filter
size. The Spearman statistic reads raw values only, so it is memoised apart
from the bins (changing the bin count never recomputes it).

Verbal reading of the magnitude: < 0.1 negligible, < 0.3 weak, < 0.5
moderate, otherwise strong (`AssociationStatistics.strength`). The coefficient
math is in `FieldCorrelation` (`lib/utils/fieldCorrelation.ts`).

## The heatmap

`FieldAssociationDialog` (title + Cross-tab / Correlations tabs) holds
`CrossTabPanel` and `CorrelationPanel`; both draw with the generic
`HeatmapGrid` over a `HeatmapModel` (`lib/utils/heatmapModel.ts`), share
`HeatmapLegend` and `useHeatmapExport`, and keep their tooltips in
`HeatmapTooltips` (all in `embedding_visualization/app/components/charts/`).
The cross-tab's value, colour and label logic is in the pure `AssociationView`
(`lib/utils/associationView.ts`).

**Cross-tab controls**: Group and Across pickers; when either field is numeric,
**Bins** (a number, 2–50, applied on blur or Enter and clamped; empty or
non-numeric text restores the current count, `NumericBins.parseCount`) and
the method (**Equal count** / **Equal width**); the modes; one sort per axis;
Swap axes; Values; Export PNG (`HeatmapActions`, shared with Correlations).
Both tabs stay mounted (hidden when inactive), so switching tabs keeps each
tab's settings and the last matrix.

**Accessibility**: the grid is a focusable group (`aria-roledescription`
"heatmap") with an always-mounted polite live region announcing the active
cell ("reddit, Politics: 2.5×"); the visual tooltip is `aria-hidden`. Cells
keep a scroll margin so arrow-key navigation never hides them under the
sticky labels; the tooltip opens above the pointer near the bottom edge.

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

Both tabs export (shared `useHeatmapExport` hook). The canvas background is
the first opaque background above the exported heading (the dialog surface).

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
| **Similarity** | `topic_label`, `topic_id`, `subtopic_label`, `subtopic_id` only (`AssociationFields.supportsSimilarity`, the fields the backend can order): the backend order below. Offered (and the default) once available; listed as disabled "(unavailable)" on topic fields whose centroids are not computed. Requesting it without an order falls back to Pattern. |
| **Size** | Largest total first, ties by name. |
| **Name** | The category list's Name order (`comparePeriods`: numbers numerically, otherwise alphabetical). |
| **Value** | Numeric axes: bins low to high (their only order). Categorical axes whose values are numbers (including numeric fields shown as Values): values low to high, the default there. |

Similarity and Value are both external orders (`rowExternal` / `colExternal`
in `AxisOrderRequest`); without one they fall back to Pattern.

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

The frontend fetches the order with `useTopicSimilarityOrder` from the first
time the heatmap dialog opens (and keeps it, so closing never re-sorts the
grid mid-animation), for whichever chosen field supports it.

The service re-reads the active extraction after loading the centroids: if a
new extraction became active in between, it returns null instead of caching
the new centroids' order under the old id.

## Correlations tab

`CorrelationPanel`: a fields × fields matrix of **Spearman ρ** (default) or
**Pearson r** between numeric fields (`AssociationFields.numericOptions`:
every numeric field, 0/1 flags and probe scores included, topic ids excluded).

- **Fields** popover with checkboxes (Select all / Clear); the first 12 are
  selected at first. The matrix keeps the fields' list order.
- **Pairwise complete**: each pair uses the visible points where both fields
  have a value; its count is in the tooltip. With fewer than three shared
  points, or a field that never varies, the pair has no coefficient.
- `CorrelationColumns` (`FieldCorrelation.columns`) extracts and sorts each
  field once; a pair's Spearman ranks over its shared points come from one
  linear pass over that sorted order (no per-pair sort), and pair results are
  cached per method, so ticking another field only computes its new pairs.
  The hook keeps one cache per point set and filter. Measured at 212k points
  × 12 fields (vite-node): Spearman 1.2 s complete / 1.0 s with 5% missing
  (was 8.6 s), Pearson 0.66 s, mostly column extraction and sorting on first
  open. Still synchronous: a Web Worker is the next step if that matters.
- `CorrelationMatrixView` (`lib/utils/correlationView.ts`) colours −1…+1 on
  the enrichment scale (blue negative, gray zero, red positive); the diagonal
  is +1 (red) like any perfect correlation, and only pairs without a
  coefficient are neutral. Values are on by default ("0.62", "−0.05").
- Computed only while the tab is shown (`useCorrelationMatrix`). The tab is
  disabled with fewer than two numeric fields.

## Shared refactors

- **One binning utility**: `NumericBins` (`lib/utils/numericBins.ts`) serves
  both the heatmap axes (`fromValues`: equal-count / equal-width, range and
  percentile labels, whole-number aware) and every histogram
  (`histogramBins(min, max, count, scale)` / `histogramFor(values, …)` /
  `histogram(values, …)`: strictly equal-width bins in linear or log10 space,
  a range reaching zero or below shifted by |min| + 1; `toHistogram` returns
  `HistogramBin` {binStart, binEnd, count, label}). One binary-search
  `indexOf` (−1 outside the range) backs `keyOf`, `counts` and the SAE
  density highlight. It replaced the legend histogram's inline binning in
  `DashboardPanel.tsx` (`LEGEND_HISTOGRAM_BINS = 30`) and
  `app/sae/utils/histogramUtils.ts` (deleted; `ActivationHistogram`,
  `LogitHistogram`, `DensityHistogram` now call `NumericBins`), and the two
  `HistogramBin` types became one (`lib/types/types.ts`). Histogram tick
  labels use one compact format (`NumericBins.tickLabel`), so the density
  histogram's ticks read e.g. "0.05" instead of "0.050".

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

- Frontend (vitest): `contingencyTable.test.ts` (incl. per-axis key
  functions), `axisOrdering.test.ts` (including a shuffled block-diagonal
  table recovered by Pattern, the excess weighting beating raw counts, and the
  Value order), `associationView.test.ts` (values, labels, colours per theme,
  enrichment floor, spread rows), `numericBins.test.ts` (equal count and
  width, merged ties, distinct accurate labels, Missing, clamping),
  `fieldCorrelation.test.ts` (ranks, r, ρ, η, the pairwise matrix with
  missing values and filters), `fieldAxis.test.ts` (field lists and defaults,
  numeric axes, the pair statistic and its summary), `correlationView.test.ts`,
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
- A field × field overview matrix of Cramér's V across all categorical fields
  (the Correlations tab covers numeric fields only).
- Clustering the correlation matrix's fields by similarity.
- A Web Worker for very large correlation matrices (about 1 s at 212k × 12).
- Similarity order for non-topic fields (e.g. `source`), which would need
  on-demand centroids over all member vectors.
- The dendrogram strip beside the heatmap (the linkage is already computed).
