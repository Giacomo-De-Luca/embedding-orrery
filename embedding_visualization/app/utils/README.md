# Plot utilities

This folder contains plot-facing algorithms and rendering helpers.

- `densityClusterLabels.ts` defines `DensityCluster`, `LabeledDensityCluster`,
  and `DensityLabelPlacement`, runs density clustering, resolves titles, and
  computes label placements; `DensityLabelColors` colours them. Its internal
  `ClusterMembership` class indexes blob rectangles for efficient membership queries.
- `densityUtils.ts` computes CPU density maps; `density_clustering/` contains
  the Rust/WASM clustering source. The runtime wrapper is in `lib/density-clustering/`.
- `text_summarizer.ts` and `stop_words.ts` support region keyword summaries
  through `TextSummarizer`.
- `labelPlacement.ts` projects 3D points to the screen (`computeMVP`, `projectToScreen`).
  `labelPlacement2D.ts` has `ZoomBandPlacement`, the zoom-band placement of 2D cluster
  labels used by both the centroid and the density labels, and `visibleAt`, the labels
  to draw in a view. `dynamicMapPlacement.ts` and `priorityQueue.ts` are Apple's
  dynamic map labelling, which the zoom bands come from. `dynamicMapPlacement.ts`
  is modified for speed (`LabelGeometry`, `ConflictGraph`, `IndexHeap`) and gives
  the original's placements. Label boxes, fonts, styles
  and drawing passes shared with 3D are in `lib/utils/labels/`; see
  `../../../documentation/PLOT_LABELS.md`.
- `rendeding.ts` and `highlightedText.tsx` support hover text, the data-to-scene
  matrix and highlighted text.
- `__tests__/` contains Vitest tests for density labels, 2D label placement and
  hover rendering. `dynamicMapPlacement.test.ts` checks that the faster placement
  matches the unmodified original, kept as a test oracle in
  `__tests__/fixtures/referenceDynamicMapPlacement.ts` (not used by the app).

Density label field selection is documented in
`../../../documentation/DENSITY_LABELS.md`.
