# Frontend utilities

This directory holds reusable visualization, color, filtering and data
adaptation code. Focused Vitest regressions live in `__tests__/`; experimental
renderers are in `experimental/`, and the density renderer has its own README.

## Category utilities

- `categoryKey.ts`: `CategoryKey` turns raw metadata values into category keys.
  Absent, null and empty values share `CategoryKey.MISSING` ("Missing values"); its
  `count` builds the sorted values and counts behind the Legend and Analytics.
  See `../../../documentation/MISSING_CATEGORY_VALUES.md`.
- `categorySelection.ts`: `CategorySelection` implements the shared legend
  click (isolate) and shift-click (toggle) semantics.
- `hiddenPoints.ts`: `HiddenPoints.filter` removes the points hidden by the
  "Hide unclustered points" and "Hide missing values" toggles in the scatter plots.
- `contingencyTable.ts`: `ContingencyTable` cross-tabulates two categorical
  fields (Missing kept, values past a cap folded into Other) and derives
  expected counts, Pearson residuals, shares, effective counts and Cramér's V.
  Also backs the temporal cross-tab.
- `axisOrdering.ts`: `AxisOrdering` orders a table's rows and columns by size,
  name, pattern (barycentric seriation) or an external similarity order.
- `associationView.ts`: `AssociationView` (cell values, colours, labels and
  sidebar summaries of the field-association heatmap) and `AssociationFields`
  (its default fields). See `../../../documentation/FIELD_ASSOCIATIONS.md`.
- `heatmapCapture.ts`: `HeatmapCapture` exports the full heatmap grid, heading
  and legend as a 2× canvas for PNG download, cancelling if the dialog closes.

Color and scatter utilities include `plotUtils.ts` (shared highlight palette/size
rules), category palettes and scales, `collisionGrid.ts`, and `overlayLayout.ts`.
Dataset loading, temporal filtering and membership helpers remain reusable by
their corresponding hooks.
