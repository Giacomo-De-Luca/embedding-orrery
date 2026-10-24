# Plot labels

These are the shared pieces of the canvas labels both scatter plots draw: cluster labels and search-result labels. The design and the placement rules are in `../../../../documentation/PLOT_LABELS.md`.

- `labelBox.ts`: `LabelBox`, the screen-space box of a label.
  - `centered`: cluster labels, also used by the zoom-band layout.
  - `rightOf`: point labels.
  - `probe`: the 1 px strip a point label is tested with before its text is measured.
  - `isFinite`.
- `labelStyle.ts`:
  - `LabelFonts`: the label fonts, from the family next/font registers for `--font-geist-mono`. Also `whenReady`/`isReady` for font loading, and `createMeasurer` for an offscreen measuring context.
  - `LabelStyle`: fills, outlines and opacity for one view and theme. Cluster-label colours differ between 2D and 3D on purpose (`CLUSTER_DESATURATION`).
  - `TextMeasurer`: the measuring part of a 2D context.
- `labelCanvas.ts`:
  - `LabelCanvas.prepare`: sizes the overlay canvas for the pixel ratio, scales it to CSS pixels and clears it.
  - `LabelCanvas.clear`.
  - `LabelTarget`: where a redraw paints (canvas, pixel ratio, hit boxes). Each plot draws with `drawLabelsOn(target)`; its zero-argument `renderLabels()` clears the overlay's hit boxes and draws on `LabelCanvas.overlay` (the overlay at the screen's pixel ratio, null before it mounts).
  - `LabelCanvas.renderOffscreen`: runs `drawLabelsOn` on a canvas of its own at a given pixel ratio with empty hit boxes, for the 4K screenshot.
  - `LabelPaintContext`: the context calls the passes make.
- `pointLabels.ts`:
  - `PointLabelRanking.rank`: the search labels, the selected point's first and the rest by similarity, ranked once per change of the highlights. Given a measurer, it also measures their text, so redraws don't.
  - `PointLabelPass.draw`: draws them right of their projected points through a `CollisionGrid`, probe first.
  - `PointLabelPass.offsetBeside`: the offset that clears a marker.
- `clusterLabels.ts`:
  - `ClusterLabelOrder`: `nearest` for 3D (camera distance, cap, fade) and `byPriority` for 2D.
  - `ClusterLabelPainter.paint`: draws centred cluster labels through the grid and records the clickable ones.
- `labelHitBoxes.ts`:
  - `LabelHitBoxes`: the drawn clickable labels' boxes.
  - `ClusterLabelClicks.route`: a click goes to its topic, else its category.
  - `ClusterLabelClicks.isDrag`: a press that moved more than 5 px is a drag, not a click.
- `__tests__/`: vitest tests, run in Node with no canvas. `recordingContext.ts` is the fake context the pass tests draw on.

Related pieces elsewhere:
- **Collision grid:** `CollisionGrid`, in `../collisionGrid.ts`.
- **2D view geometry:** `CartesianView`, in `../cartesianView.ts`.
- **Highlight lookup:** `HighlightedPoints`, in `../highlightedPoints.ts`.
- **Zoom-band placement of 2D cluster labels:** `app/utils/labelPlacement2D.ts`.
- **Font-aware measurer:** `lib/hooks/useLabelMeasurer.ts`.
- **Label clicks for both plots:** `lib/hooks/useClusterLabelClicks.ts`.
- **Ranked search labels and coalesced redraws:** `lib/hooks/useSearchLabels.ts` and `lib/hooks/useCoalescedRedraw.ts`.
