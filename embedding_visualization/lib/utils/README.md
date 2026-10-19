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
  sidebar summaries of the field-association heatmap) and `AssociationScale`
  (its colour scales). See `../../../documentation/FIELD_ASSOCIATIONS.md`.
- `fieldAxis.ts`: `AssociationFields` (comparable fields and defaults),
  `FieldAxis` (a field as a table axis, numeric fields binned) and
  `AssociationStatistics` (Cramér's V, η or Spearman ρ for a pair).
- `numericBins.ts`: `NumericBins`, the one binning utility: axis bins
  (equal-count or equal-width, range or percentile labels) and histogram bins
  (equal-width, linear or log) for the legend and the SAE explorer. `fieldCorrelation.ts`: `FieldCorrelation` (ranks, Pearson,
  Spearman, η, pairwise matrix). `correlationView.ts`: `CorrelationMatrixView`.
  `heatmapModel.ts`: the `HeatmapModel` interface both heatmaps implement.
- `heatmapCapture.ts`: `HeatmapCapture` exports the full heatmap grid, heading
  and legend as a 2× canvas for PNG download, cancelling if the dialog closes.

## Colour utilities

- `categoryPalettes.ts`: the categorical palette registry (`CATEGORY_PALETTES`). The
  three Cosmic Galaxy palettes derive from one `COSMIC_FAMILIES` table and the
  prefix-first `COSMIC_FAMILY_ORDER`; `deepField`, `aurora`, `hubble` and `pulsar`
  are generated data (see below). `categoryColors.ts` builds category → colour maps
  on top of it (presets, Missing/Unclustered colours, cycling past the palette length).
- `colorMath.ts`: `ColorMath`, OKLab/OKLCH conversion and gamut search, WCAG contrast,
  Machado colour-vision-deficiency simulation and OKLab ΔE.
- `paletteDesign.ts`: `PaletteDesigner` (OKLCH candidate grids under hue-arc, lightness,
  chroma and contrast constraints; greedy farthest-first selection; prefix-optimal
  reordering) and `PaletteAudit` (closest pairs, prefix minima, contrast floors).
- `paletteRecipes.ts`: configuration for the generated palettes. Regenerate with
  `npm run palettes:design` (`scripts/designPalettes.ts`); `paletteDesign.test.ts`
  fails when a committed palette no longer matches its recipe.
  See `../../../documentation/CATEGORICAL_PALETTES.md`.

## Plot memory

- `stackglBufferPool.ts`: `StackglBufferPool` releases the large idle buffers from
  the stackgl typed-array pool (`globalThis.__TYPEDARRAY_POOL`), where gl-scatter3d
  parks its vertex arrays after every upload and nothing else ever frees them.
  `ScatterPlot3D` calls `release()` when the point count changes and on unmount.
  See `../../../documentation/RENDER_PERFORMANCE.md`, "Typed-array pool release".

Other colour and scatter utilities include `plotUtils.ts` (shared highlight palette/size
rules), colour scales, `collisionGrid.ts`, and `overlayLayout.ts`.
Dataset loading, temporal filtering and membership helpers remain reusable by
their corresponding hooks.

## Background jobs

- `jobIds.ts`: `JobIds` builds every job id that progress subscriptions, the Jobs
  list and `cancelJob` share (mirror of the backend `services/job_ids.py`, both
  tested against the same examples). Never hand-build `${collection}_…` ids.
- `jobCancellation.ts`: `JOB_CANCELLED_MESSAGE` (the backend's `CANCELLED_MESSAGE`),
  `isJobCancelled` / `jobOutcome` (succeeded, cancelled or failed, for toasts and
  result cards) and `cancelButtonView` (the progress modal's Cancel button
  states). See `../../../documentation/JOB_CANCELLATION.md`.
