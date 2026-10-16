# Plot utilities

This folder contains plot-facing algorithms and rendering helpers.

- `densityClusterLabels.ts` defines `DensityCluster`, `LabeledDensityCluster`,
  and `DensityLabelPlacement`, runs density clustering, resolves titles, and
  computes label placements. Its internal `ClusterMembership` class indexes
  blob rectangles for efficient membership queries.
- `densityUtils.ts` computes CPU density maps; `density_clustering/` contains
  the Rust/WASM clustering source. The runtime wrapper is in `lib/density-clustering/`.
- `text_summarizer.ts`, `category_column.ts`, and `stop_words.ts` support region
  keyword summaries through `TextSummarizer` and categorical indexing.
- `labelPlacement.ts`, `labelPlacement2D.ts`, `dynamicMapPlacement.ts`, and
  `priorityQueue.ts` handle label projection, placement, and collision priority.
- `rendeding.ts`, `plotLabels.tsx`, `highlightedText.tsx`, and `colors.ts` support
  hover text, plot labels, highlighted text, and colors.
- `__tests__/` contains Vitest tests for density labels and hover rendering.

Density label field selection is documented in
`../../../documentation/DENSITY_LABELS.md`.
