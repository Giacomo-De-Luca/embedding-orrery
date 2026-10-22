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

## Plot gestures

- `pinchGesture.ts`: `PinchGesture`, the DOM-free two-finger recogniser behind
  `usePinchZoom`. Per touch event it says whether to keep the event from Plotly
  and reports the pair's spread ratio and midpoint.
- `plotPinchZoom.ts`: the `PinchZoomTarget` each plot implements.
  `CartesianPinchZoom` (2D) replays a pinch as wheel events on Plotly's drag
  surface; `CameraPinchZoom` (3D) sets the gl-plot3d camera distance.
  `PinchZoomSession` is the hook's DOM-free event handling (gesture → target,
  one `pinch` per animation frame through an injected scheduler).
  See `../../../documentation/MOBILE_PINCH_ZOOM.md`.
- `coalescedRedraw.ts`: `CoalescedRedraw` merges the redraw requests of one
  synchronous burst (the `plotly_relayouting` events of a pinch frame) into one
  redraw, run in a microtask so it is painted in the same frame as the points.
  Used by the 2D cluster labels (`ScatterPlot2D`) and `useDensityOverlay`.
- `zoomOpacity.ts`: `ZoomOpacity`, the 2D plot's zoom-dependent opacity. Zoomed
  in, each trace's opacity is multiplied by `calculateMarkerStyle`'s opacity for the
  visible share of the points over its opacity for all of them, capped at 1;
  `traceScale` is the factor the Plotly fork's `scatterglOpacityScale`
  config option applies before every draw. Traces marked `FIXED_META` (search glow,
  selected point) keep theirs. See `../../../documentation/ZOOM_OPACITY.md`.

## Screenshot export

- `plotCapture.ts`: `capture2DPlot` / `capture3DPlot` composite each plot with its
  overlay canvases; `drawLegendOverlay` rasterizes the live legend on top.
- `premultipliedPixels.ts`: `PremultipliedPixels.flattenOnto` composites the 3D
  scene's raw `gl.readPixels` output (premultiplied, bottom-up, RGB can exceed
  alpha) onto the screenshot background the way the browser does, so the search
  glow survives the 3D export.

See `../../../documentation/SCREENSHOT_EXPORT.md`.

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

## Model and prompt pickers

- `modelSuggestions.ts`: `ModelSuggestions` over the `embeddingModelCatalog`
  query (`lib/hooks/useEmbeddingModelCatalog.ts`): the model-name field's groups
  ("Suggested" catalog models, then "Used in your collections"), filtered by
  provider and modalities, each model once; hints (modalities, dimension, use);
  the multimodal default (`defaultFor`) and catalog modality lookups.
- `promptPresets.ts`: `PROMPT_PRESETS`, the one prompt preset list, and
  `PromptPresets` groups: a model's own catalog prompt names when known, else the
  presets (embed form), or Auto plus the presets (Explore query prompt).
- `queryPrompt.ts`: `resolveQueryPrompt` ("Auto", empty and blank send no
  prompt: the backend uses the collection's own; typed text is sent as typed,
  spaces included) and `textQueryUnavailableReason` (a collection that records
  no embedding provider and model takes no semantic text query).
- `suggestions.ts`: `SuggestionItem` / `SuggestionGroup` and `matchesSuggestion`,
  shared by the two above and the field
  `lib/ui-primitives/suggestion-combobox.tsx` (free text with grouped
  suggestions, base-ui Autocomplete), which `prompt-combobox.tsx` wraps.

## Topic merge trees

- `topicTreeCut.ts`: `TopicMergeTree` (a validated dendrogram over a topic
  extraction's base topics, scipy's linkage layout), `TopicTreeCut` (groups at N
  topics or at a height, absorbed noise, representatives) and `TopicTreeView`
  (`fromGraphQL` for the `topicMergeTree` query; `baseField` / `baseLabelField`).
  Mirror of the backend's `utils/topic_merge_tree.py`, tested on the same literal
  examples.
- `topicCutMetadata.ts`: `TopicTreeGroups` (every tree node as a preview group:
  representative, label, size, and the colour pin's value map) and
  `TopicCutPreview` (one cut; `apply` rewrites the items' topic fields for the
  Explore slider's preview).
- `topicReduction.ts`: `REDUCTION_STRATEGY_OPTIONS`, `ReductionRules` (which
  options each strategy takes) and `ReductionInputs` (reduction requests the
  backend's `ReductionRequest` accepts), shared by the Collections forms
  (re-exported from `app/collections/lib/topicConfigInput.ts`) and the slider's Apply.
- Related: `categoryColors.ts` `buildPinnedCategoryColorMap` (colour pins),
  `categoryKey.ts` `CategoryKey.subtopicKey` (nested colouring keys),
  `fieldAnalysis.ts` `TopicFieldOptions` (topic-field colour options after a reduction).
  See `../../../documentation/TOPIC_MERGE_TREES.md`.
