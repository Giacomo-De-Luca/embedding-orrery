# Plot labels

Both scatter plots draw their labels on a canvas over the Plotly plot. There are two kinds: cluster (topic) labels, and labels for semantic-search results when "Label search results" is on. Both kinds go through one `CollisionGrid` per redraw, so no two labels overlap. This page covers how they are placed, drawn, redrawn and clicked, and the shared code behind them.

## Order

Labels claim space in this order; a label whose box is taken is skipped.

1. **The selected point's label**, if the selected point is a search result. It always shows.
2. **Cluster labels.**
   - **2D:** the labels whose zoom band holds the current zoom (below), highest priority first. With "Show all cluster labels", every label, still collision-tested.
   - **3D:** the clusters nearest the camera, capped at max(8, ⌊2.5·√n⌋) and fading from 1 to 0.65 with distance. With "Show all", every cluster at full opacity.
3. **The other search labels**, by similarity.

Search labels start to the right of their point:
- **3D:** 8 px.
- **2D:** the radius of the point's inner search glow at the current zoom (`ZoomMarkerStyle.sizeScale`, so labels move out as the markers grow), plus 4 px (`PointLabelPass.offsetBeside`). Points outside the plot area get no label.

Opacity is 0.4 + 0.6 × similarity, or 1 for the selected point. Each label is tested first with a 1 px probe where its text would start, so labels in crowded areas are rejected before their whole box is tested (TensorBoard's projector does the same, there to skip measuring their text). Here the text is already measured: once per change of the labels, not per frame (see "Redraws").

## Shared code

| Piece | File | Role |
|---|---|---|
| `CartesianView` | `lib/utils/cartesianView.ts` | A 2D Plotly view: axis ranges and plot area read from `_fullLayout`. It projects data to CSS pixels, tests the plot area, and gives the zoom relative to the initial view. The 2D labels, the density overlay and the zoom limit use it. |
| `CollisionGrid` | `lib/utils/collisionGrid.ts` | Greedy screen-space packing (TensorBoard's embedding projector). `forViewport` gives 25 × 50 cells; boxes with non-finite coordinates are rejected. |
| `LabelBox` | `lib/utils/labels/labelBox.ts` | Label boxes: `centered` (cluster labels: half the text width + 4 px by 0.6 × font size + 4 px), `rightOf` and `probe` (search labels). |
| `LabelFonts`, `LabelStyle` | `lib/utils/labels/labelStyle.ts` | Label fonts, and per-view, per-theme fills, outlines and opacity. |
| `LabelCanvas` | `lib/utils/labels/labelCanvas.ts` | Sizes the overlay for the pixel ratio and clears it; also used by the 3D camera animation. |
| `PointLabelRanking`, `PointLabelPass` | `lib/utils/labels/pointLabels.ts` | Search labels ranked, and their text measured, once per change of the highlights (not per frame), then drawn through the grid. |
| `ClusterLabelOrder`, `ClusterLabelPainter` | `lib/utils/labels/clusterLabels.ts` | 3D nearest-first and 2D priority order; drawing centred cluster labels and recording the clickable ones. |
| `LabelHitBoxes`, `ClusterLabelClicks` | `lib/utils/labels/labelHitBoxes.ts` | Boxes of the clickable labels drawn last; routing a click to a topic or category; telling clicks from drags. |
| `HighlightedPoints` | `lib/utils/highlightedPoints.ts` | Highlighted points by index, O(k), in highlight order. |
| `ZoomBandPlacement` | `app/utils/labelPlacement2D.ts` | Zoom bands of 2D cluster labels for both the centroid and the density labels, and `visibleAt`, the cluster labels to draw in a view. |
| `useSearchLabels`, `useCoalescedRedraw`, `useLabelMeasurer`, `useClusterLabelClicks` | `lib/hooks/` | The ranked search labels (null for none); one coalesced request of the latest render (also the density overlay's); font-aware text measuring; cluster-label clicks. |

## Redraws

- **2D:**
  - `handlePlotUpdate` is react-plotly's `onUpdate`, which runs after every `Plotly.react`, relayout and relayouting frame, and is read from props when called. It requests one label redraw and one density-map redraw through `CoalescedRedraw`, so a pinch frame's several events cost one redraw, painted in the frame Plotly drew.
  - It also tracks the laid-out plot area. When the margins move (Show axes) or the size changes, the cluster labels are placed again.
  - A change of the labels' inputs (results, selection, placements, theme, "Show all") requests a redraw too.
- **3D:**
  - A camera poll redraws every frame the camera moves.
  - Data, size, font and "Show all" changes request one coalesced redraw. It runs after the commit's `Plotly.react` has applied any new scene ranges, and not while a fly-to hides the labels.
  - The data-to-scene matrix is cached on the scene axis ranges themselves. A redraw between a UMAP ↔ PCA switch and its `Plotly.react` therefore never caches the old ranges' matrix, which would misplace labels and haze.
- **Text widths:** measured once per change of the labels, with the plot's `useLabelMeasurer`, so a camera or pan frame draws without measuring. That covers the search labels in both views (`PointLabelRanking.rank`) and the 3D cluster labels (the `clusterLabels` memo). 2D cluster labels carry the widths their placement measured. With "Show all" on a map of 2,700 topics, 3D used to measure every label on every camera frame.

## Clicks

- **What a click does:** cluster labels are clickable in both views when the plot has a topic or category handler. A click on one toggles its topic (if `topicLabelToIdMap` names it), else its category (a topic-tree preview's groups; shift adds). Density labels from another field or from keyword summaries are drawn without a hit box.
- **Drags:** a press that moves more than 5 px before its release is a drag (pan, rotate), not a click.
- **Not clicks on labels:** a click on the modebar belongs to the modebar. In 3D, labels a fly-to has hidden take no clicks until it lands and redraws them.
- **Points under a label:** Plotly reports a point click before (2D) or during (3D) the press a label takes, so a click on a label selects no point. gl3d's click carries no pointer event, so 3D tests the press position the plot records.
- **Hit-testing:** labels are hit-tested from the boxes drawn last, in the plot container's coordinates.

## Fonts

next/font registers Geist Mono under the family name `GeistMono`, with no space, and publishes the stack as the CSS variable `--font-geist-mono` on `<html>`. Canvas text has to name a registered family. The old strings asked for `Geist Mono`, which isn't registered, so labels silently rendered in the system monospace.

- **Family:** `LabelFonts.family()` reads the variable once and falls back to `GeistMono, ui-monospace, monospace`. The 3D axis labels use the same family.
- **Font strings:** `LabelFonts.cluster(size)` is bold, 13 px by default (density labels use 14 / 12 px per level). `LabelFonts.point()` is 11 px.
- **Font loading:** text measured before the font loads uses fallback widths. `useLabelMeasurer` gives each plot a measurer for label widths and replaces it once `document.fonts` has loaded the label fonts. Placements and measured labels list it as a dependency, so they are measured again with the real font, and the redraw follows.

## Colours

`LabelStyle(isDark, view)` holds the label colours for one view and theme.

| | 2D | 3D |
|---|---|---|
| Cluster text | the cluster colour desaturated by 0.3 (dark) / 0.45 (light) | desaturated by 0.1 (dark) / 0.35 (light) |
| Cluster outline | `rgba(15, 23, 42, 0.85)` 4 px (dark) / `rgba(255, 255, 255, 0.6)` 3 px (light) | same |
| Search-label text | `#e2e8f0` (dark) / `#1e293b` (light), 3 px outline | same |

The two views keep different cluster-text strengths until one pair is chosen. They sit side by side in `CLUSTER_DESATURATION`, so unifying them is a one-line change.

Density labels take their category's colour (`DensityLabelColors.resolve`): the nested topic colour for topic labels, otherwise the colour map, otherwise a neutral grey. They are placed and coloured separately from the clustering, so a new palette, theme, plot area or font re-places them without clustering again.

## Zoom bands (2D cluster labels)

2D cluster labels use Apple's dynamic map labelling (`app/utils/dynamicMapPlacement.ts`; Been, Daiches & Yap 2006). It runs once per set of labels and plot area, not per frame.
- **Order:** `ZoomBandPlacement.place` returns the placements highest priority first, so the per-frame `visibleAt` only iterates.
- **Layout:** each label is laid out as a centred box at the initial view, the data bounds plus a 5 % pad.
- **Zoom band:** each label gets a band `[minScale, maxScale]` in which it doesn't overlap a label of higher priority. Priority is the point count for centroid labels and the density sum for density labels.
- **Drawing:** shows a label only while the current scale is in its band, and only within 20 px of the plot area. The scale is the geometric mean of the span ratios to the initial view (`CartesianView.scaleRelativeTo`; 1 = initial view, below 1 = zoomed in; bands go up to 2).
- **Result:** labels appear and disappear as you zoom but never flicker while you pan. "Show all cluster labels" ignores the bands, and the grid then picks the labels by priority, so labels can change while you pan with it on.

Centroid labels used a hard-coded half-height of 12 px; they now share the density labels' 0.6 × font size + 4 px (11.8 px at 13 px).

### Speed on large maps

Apple's code compared every pair of labels and, at each zoom level, revisited every label, so its time and memory grew with the square of the label count. WordNet's 2,700 topics took about 2.5 s and 590 MB per placement. `dynamicMapPlacement.ts` now gives the same placements with less work:
- **Pairs:** only pairs that can touch below the top zoom level are compared. A grid of cells as wide as the furthest such reach finds them in the 3 × 3 neighbouring cells. Labels whose box doesn't contain its anchor fall back to comparing every pair.
- **Levels:** the sweep down the zoom levels only revisits the labels whose state can change at a level: a conflict ending, the label's own range starting or ending, or a neighbour that has just been hidden. It visits them in index order, as the original loop did, so ties resolve the same way.

| Labels | Before | After |
|---|---|---|
| 1,000 | 303 ms | 32 ms |
| 2,000 | 1,419 ms, 309 MB | 71 ms, 24 MB |
| 2,700 (WordNet) | about 2.5 s, 590 MB | 123 ms, 36 MB |
| 5,000 | not measured | 362 ms |

Node timings, one run each, on random clustered layouts. `app/utils/__tests__/dynamicMapPlacement.test.ts` checks that the placements are identical to the unmodified original's (`__tests__/fixtures/referenceDynamicMapPlacement.ts`). The layouts it checks are random (clustered, uniform, packed), with labels' own zoom ranges, with boxes that don't contain their anchors, and 200 small crowded ones.

### Settling

`ScatterPlot2D` places the labels from inputs that have been still for 150 ms (`useDebounceValue`): the visible points, their clusters, the plot area and the initial ranges. A burst of changes, such as dragging the temporal brush or resizing the panel, then places them once, after the change itself has painted. The density labels are placed from the same settled inputs.

While the inputs settle, the labels stay where they were. The exception is placements made for another data extent (a projection or collection switch): they would land in the wrong places, so they are hidden until the new ones are ready. The initial ranges are keyed on the values of the data bounds, so new points over the same extent, such as a topic-tree preview step, keep their labels.

## Screenshots

`capture2DPlot` and `capture3DPlot` draw the label canvas over the plot (`lib/utils/plotCapture.ts`). Since 2D search labels moved onto the canvas, a 2D screenshot shows them exactly as on screen, collision handling included.
