# Zoom-dependent marker size and opacity (2D)

The 2D scatter plot draws large collections with small, faint points, so dense
regions don't saturate: `calculateMarkerStyle` gives 3.2 px and 0.15 opacity at
150k points. Zoomed in, the same points look washed out and tiny. Points now grow
larger and more opaque as you zoom in, without slowing the zoom.

## Behaviour

- **Rule.** Zooming in shows a share of the data box: per axis, the visible span
  over the data extent (capped at 1), multiplied over both axes.
  `calculateMarkerStyle` already picks size and opacity together from the number of
  points, so each trace's size and opacity are multiplied by its values for the
  visible share of the points over its values for all of them. The view looks like
  a plot of the points you can see. Opacity never goes above 1.
- **Example**, 150k points, dark theme:

  | Zoom (both axes) | Size | Opacity |
  |---|---|---|
  | Full view or zoomed out | 3.2 px | 0.15 |
  | 2× | 4.3 px | 0.22 |
  | 4× | 5.4 px | 0.32 |
  | 10× | 6.8 px | 0.46 |
  | 30× | 8.5 px | 0.62 |
  | ~100× and beyond | 10 px | 0.80 |

  `calculateMarkerStyle`'s maximums are 10 px and 0.8. The light theme multiplies
  the base opacity by 1.6, and the Point opacity slider scales it too; both reach
  fully opaque sooner.
- **Points still separate.** Size grows much more slowly than the zoom: at 4×,
  spacing grows 4× and size 1.7×. Overlap in dense clusters drops to about 18% of
  the full view.
- **Panning** keeps the style: the share depends on the zoom, not on where you look.
- **Opacity** scales on the coloured points, muted points (search, temporal and
  topic filters), missing values and the dimmed context behind search results. The
  search glow layers, the search-result cores and the selected-point layers keep
  their opacity: they carry `meta: ZoomMarkerStyle.FIXED_OPACITY_META`.
- **Size** scales on every trace, glow and selected point included, so highlighted
  results stay larger than their neighbours. Marker outlines keep their width.
- **Box and lasso selections** keep growing with the zoom: selected and unselected
  points both scale in size, and keep Plotly's own selected/unselected opacity.
- **Screenshots** show the same sizes and opacities as the screen.
- **Search-result labels** move out with the grown points: they start right of
  each result's inner glow, sized by `sizeScale` at the current zoom
  (`PLOT_LABELS.md`).
- **Known limits.** The hover label's anchor uses the original marker
  radius; hover hit-testing searches a fixed 20 px around the cursor, so it is
  unaffected. regl-scatter2d stores sizes in steps of 0.39 px, so a smooth zoom
  changes sizes in small steps.

## How it works

The obvious implementation would put the zoom in React state and rebuild the
traces. For scattergl, `marker.size` and `marker.opacity` are `calc` attributes, so
every wheel tick would recalculate every trace (thousands of traces on WordNet).

Instead, the style changes on the GPU side only. scattergl draws each trace as a
regl-scatter2d group. Fork change 6 (`embedding_visualization/forked/plotly_changes.md`)
adds two config options, `scatterglOpacityScale` and `scatterglSizeScale`, each
`(fullTrace, xRange, yRange) → factor`. The scattergl scene calls them for every
marker trace at the start of each draw and multiplies the trace's base values by
the factors.

- **Opacity** is a per-group shader uniform, and a trace with one marker size gets
  that size as a constant attribute read from the group at draw time. Both are
  written to the group directly. `scatter2d.update()` would re-parse every option
  alias per trace, which costs milliseconds per frame with thousands of traces.
- **Per-point sizes** live in a GPU buffer, so they go through
  `scatter2d.update()`, and only when their factor changes; the scene remembers which
  scaled array it uploaded for which base array and factor. The app no longer uses
  them (every search-glow point has the same size, so each glow layer has one size),
  but the fork keeps the path for any trace that does. That buffer stores sizes as
  Uint8 fractions of 100 px, so per-point sizes above 100 px would wrap.
- **Selections**: a trace with a selection keeps Plotly's opacity. Its selected
  points are drawn by Plotly's separate `select2d` instance, so a single size is
  scaled there too (from the selected style's size when it sets one). Per-point
  sizes of a selected trace go back to the base, since `select2d`'s are not scaled.
- Every redraw goes through that draw: each zoom or pan frame, the relayout at the
  end of a gesture, and every `Plotly.react`. The new style therefore lands in the
  same frame as the zoom. There are no extra draws and no position or colour
  uploads; the cost is two function calls per trace per draw (`ZoomMarkerStyle`
  computes the factors once per set of ranges).
- The factors always multiply the base values in `scene.markerOptions`, so draws
  never compound. Plotly resets the groups to those values on every plot call
  outside select mode, and the next draw scales them again either way.
- In single-colour traces the opacity uniform holds `marker.opacity`. In
  colour-scale traces `marker.opacity` is baked into each point's colour alpha and
  the uniform is 1. The factor multiplies either way, and
  `ZoomMarkerStyle.opacityScale` caps it at `1 / marker.opacity`, so a trace never
  goes past fully opaque.
- `ScatterPlot2D`'s config memo must stay stable (a config change makes
  `Plotly.react` replot from scratch), so the options read the current
  `ZoomMarkerStyle` through a ref. A new one is built when the points change and
  stored in a layout effect, which runs before react-plotly's `Plotly.react`
  (called in a promise continuation).
- Screenshots: `Plotly.toImage` plots a clone with the graph's config
  (`gd._context`), so the clone's draws call the same options with the exported
  ranges.

## Files

| File | Role |
|---|---|
| `embedding_visualization/lib/utils/zoomMarkerStyle.ts` | `ZoomMarkerStyle`: the rule (`multipliers`), the per-trace opacity factor (`opacityScale`) and the size factor (`sizeScale`) |
| `embedding_visualization/forked/plotly.js/src/plot_api/plot_config.js` | Declares `scatterglOpacityScale` and `scatterglSizeScale` (fork change 6) |
| `embedding_visualization/forked/plotly.js/src/traces/scattergl/scene_update.js` | `applyMarkerScale`, called at the start of `scene.draw` (fork change 6) |
| `embedding_visualization/app/components/ScatterPlot2D.tsx` | Builds `ZoomMarkerStyle`, passes both options, marks the fixed-opacity traces |

## Tests

- `lib/utils/__tests__/zoomMarkerStyle.test.ts`: the rule on literal values (full
  view, zoomed out, 4× zoom, panning, reversed ranges, an axis without extent, the
  10 px and 0.8 ceilings) and the per-trace factors (opacity cap, fixed traces,
  per-point opacities, the same size factor for every trace).
- `forked/__tests__/scatterglMarkerScale.test.ts`: the fork's real `scene.draw`
  scales each group's opacity and size before drawing, never compounds, passes the
  trace and its ranges, and skips marker-less traces. A trace with a selection
  keeps Plotly's opacity while its unselected and selected (`select2d`) points grow;
  its per-point sizes go back to the base. Per-point sizes
  are uploaded through `update()` only when their factor changes, again after
  Plotly resets them, back to the base array once when the factor returns to 1,
  and not at all at factor 1. It also pins the regl-scatter2d (including its
  100 px Uint8 size encoding) and `toImage` internals the patch relies on.

## Open

- The FPS benchmark (`benchmarks/fps/`) has not been run for this change.
- The curve is fixed. A Controls slider (strength of the zoom effect) could be
  added later; `ZoomMarkerStyle.multipliers` is the place to change it.
