# Missing category values

Frontend handling of points whose colour-by field has no value (2026-10-10).

## Problem

Colour-by fields are often absent on part of a collection. In `combined`,
`category` and `activation_type` are absent on 12,792 of 14,213 items,
`sarcasm` on 9,367 and `polarity_opposition` on 4,853. Before this change:

- the scatter plots drew those points as a grey `'unknown'` group, but
  `useCategoryData` dropped them, so neither the Legend nor the Analytics
  distribution listed them;
- isolating a category muted only the listed values, so the missing points stayed
  visible and the 3D camera refit kept framing them;
- Analytics percentages were computed over the listed categories only (about 10x
  too large on `combined`);
- each view decided on its own what "missing" meant: the 3D refit used `??`
  (an empty string was not missing) and the 2D and 3D cluster-label filters never
  muted missing points;
- `'unknown'` collided with real values spelled `unknown`.

## Rule

A value is missing when it is absent, `null` or `''`. All missing values share
one key, `CategoryKey.MISSING` (`'__missing__'`), which is displayed as
"Missing values" and coloured `MISSING_CATEGORY_COLOR` (`#7f7f7f`). Every view derives
category keys through `CategoryKey`, so plots, Legend, Analytics and the temporal
chart agree on grouping, colour, counts and muting.

## Where it lives

| File | Role |
|---|---|
| `embedding_visualization/lib/utils/categoryKey.ts` | `CategoryKey`: `isMissing`, `of` (raw value to key), `nested` (topic/subtopic keys; a missing subtopic falls back to its topic), `sort` (plain string order, Missing last), `count` (values and counts, optionally skipping point indices) |
| `lib/utils/categoryColors.ts` | `buildCategoryColorMap` gives Missing the fixed colour outside the palette, so real categories keep their colours; `getCategoryLabel` returns "Missing values" for any field |
| `lib/hooks/useCategoryData.ts` | Thin memo over `CategoryKey.count`; the Legend, Analytics list, temporal stacked chart and `CategorySelection` all consume its values |
| `app/components/DashboardPanel.tsx`, `AnalyticsSidebar.tsx` | Filtered counts include Missing |
| `app/components/ScatterPlot2D.tsx`, `ScatterPlot3D.tsx`, `lib/hooks/useDensityOverlay.ts` | Trace grouping, nested grouping, 3D camera refit, cluster-label point filters and density channels use `CategoryKey` |
| `lib/utils/nestedColorUtils.ts` | Points without a topic form a Missing topic with a fixed colour, listed after Unclustered |
| `lib/utils/temporalAnalysis.ts` | `computeTemporalCrossTab` counts Missing, so every period with a temporal value gets a row |
| `lib/utils/hiddenPoints.ts` | `HiddenPoints.filter`: the scatter plots' single filter for the "Hide unclustered points" and "Hide missing values" toggles (it replaced four copies of the unclustered filter in the 2D and 3D plots) |
| `app/components/VisualizationColorControls.tsx` | "Hide missing values" checkbox and "Missing value opacity" slider, shown only when the colour field has missing values |

## Behaviour decisions

- **Legend and Analytics list Missing like any category.** Click isolates it,
  shift+click toggles it, and isolating another category mutes it. Its colour can
  be overridden from the flat Legend's colour picker.
- **Analytics bars stay honest.** Missing shares the bar scale and percentages are
  of all points. Count and Rate sorting rank it like any row; Name sorting pins it
  last because "Missing values" is not a name.
- **Missing is not a cluster.** `groupPointsByCluster` skips it, as it skips
  HDBSCAN noise (no nebula haze, no cluster label).
- **Topic mode.** When colouring by `topic_label` with topics loaded, legend
  clicks select topics. Missing is not a topic, so it is muted while any topic is
  selected. Clicking it clears the topic selection and applies the regular category
  muting (click isolates, shift+click toggles); double-click reset clears both.
- **Nested Legend.** Topic and subtopic names render through `getCategoryLabel`,
  so a Missing topic reads "Missing values". Like Unclustered, it keeps a fixed colour.
- **Hide and fade controls.** When the colour field has missing values, the Color
  section shows "Hide missing values" (store flag `hideMissing`, cleared on colour
  field change like `hideUnclustered`) below a "Missing value opacity" slider
  (`missingPointOpacity`, default 100%, reset per collection; disabled rather than
  removed while missing values are hidden, so the checkbox keeps its place). Each
  view passes `canFadeMissing` to show the slider only where the fade works: the
  scatter plots whenever they draw category traces (no continuous scale with a
  numeric range, the same rule as their `numericData`). `CategoryKey.missingFade` scales every opacity of the
  Missing group, muted or not. Hidden points are removed from traces, the camera refit, cluster
  labels and the density overlay; search highlights stay visible, as with
  unclustered points.

The temporal cross-tab change also resolves the stacked-chart tooltip
misalignment recorded in `AUDIT_2026-07-02.md` (rows were missing for periods
where every point lacked the category, so `chartData` and `allPeriods` diverged).

## Tests

Written before the implementation, in `embedding_visualization/lib/utils/__tests__/`:
`categoryKey.test.ts`, `nestedColorUtils.test.ts`, `clusterGeometry.test.ts`,
`hiddenPoints.test.ts` (new), plus Missing cases added to `categoryColors`,
`categorySelection`, `categoryRowData` and `temporalAnalysis`. The graph-view
cases (`graphPresentation`, `graphAppearance`) live on the `sae-graph` branch.

## Out of scope

Missing-value checks on temporal periods, numeric colour ranges and density
cluster topic titles are separate concerns and were left unchanged.
