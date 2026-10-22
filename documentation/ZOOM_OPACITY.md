# Zoom-dependent opacity (2D)

The 2D scatter plot draws large collections with faint points, so dense regions
don't saturate: `calculateMarkerStyle` gives 0.15 opacity at 150k points. Zoomed
in, the same faint points look washed out. Points now grow more opaque as you zoom
in, without slowing the zoom.

## Behaviour

- **Rule.** Zooming in shows a share of the data box: per axis, the visible span
  over the data extent (capped at 1), multiplied over both axes. A trace's opacity
  becomes its full-view opacity × `calculateMarkerStyle(N × share).opacity /
  calculateMarkerStyle(N).opacity`, never above 1. In other words, the plot uses
  the opacity it would pick for the number of points you can now see.
- **Example**, 150k points, dark theme:

  | Zoom (both axes) | Opacity |
  |---|---|
  | Full view or zoomed out | 0.15 |
  | 2× | 0.22 |
  | 4× | 0.32 |
  | 10× | 0.46 |
  | 30× | 0.62 |
  | ~100× and beyond | 0.80 (`calculateMarkerStyle`'s maximum) |

  The light theme multiplies the base by 1.6, and the Point opacity slider scales
  it too; both reach fully opaque sooner.
- **Panning** keeps the opacity: the share depends on the zoom, not on where you look.
- **What scales**: the coloured points, muted points (search, temporal and topic
  filters), missing values and the dimmed context behind search results.
- **What stays fixed**: the search glow layers, the search-result cores and the
  selected-point layers. They carry `meta: ZoomOpacity.FIXED_META`.
- **Box and lasso selections switch the scaling off.** Once a selection exists,
  every trace keeps Plotly's own selected/unselected styling at the full-view
  opacity. Zoomed-in points therefore get darker when you select, and zooming has
  no effect until the selection is cleared. The app does nothing with selections;
  if it ever does, scale the `select2d` groups (and the unselected opacity) too.
- **Screenshots** show the same opacity as the screen.
- Point sizes don't change.

## How it works

The obvious implementation would put the zoom in React state and rebuild the
traces. For scattergl, `marker.opacity` is a `calc` attribute, so every wheel tick
would recalculate every trace (thousands of traces on WordNet).

Instead, the opacity changes on the GPU side only. scattergl draws each trace as a
regl-scatter2d group whose opacity is a shader uniform. Fork change 6
(`embedding_visualization/forked/plotly_changes.md`) adds a config option,
`scatterglOpacityScale(fullTrace, xRange, yRange) → factor`. The scattergl scene
calls it for every marker trace at the start of each draw and writes the trace's
base opacity × factor into the group's uniform.

- Every redraw goes through that draw: each zoom or pan frame, the relayout at the
  end of a gesture, and every `Plotly.react`. The new opacity therefore lands in
  the same frame as the zoom. There are no extra draws and no buffer uploads; the
  cost is one function call per trace per draw (`ZoomOpacity` computes the
  multiplier once per set of ranges).
- The uniform is written directly rather than through `scatter2d.update()`, which
  re-parses every option alias per trace: milliseconds per frame with thousands of
  traces.
- The factor always multiplies the base opacity in `scene.markerOptions`, so draws
  never compound. Plotly resets the groups to that base on every plot call
  outside select mode, and the next draw scales them again either way.
- In single-colour traces the uniform holds `marker.opacity`. In colour-scale
  traces `marker.opacity` is baked into each point's colour alpha and the uniform
  is 1. The factor multiplies either way, and `ZoomOpacity.traceScale` caps it at
  `1 / marker.opacity`, so a trace never goes past fully opaque.
- `ScatterPlot2D`'s config memo must stay stable (a config change makes
  `Plotly.react` replot from scratch), so the option reads the current
  `ZoomOpacity` through a ref. A new `ZoomOpacity` is built when the points change
  and stored in a layout effect, which runs before react-plotly's `Plotly.react`
  (called in a promise continuation).
- Screenshots: `Plotly.toImage` plots a clone with the graph's config
  (`gd._context`), so the clone's draws call the same option with the exported
  ranges.

## Files

| File | Role |
|---|---|
| `embedding_visualization/lib/utils/zoomOpacity.ts` | `ZoomOpacity`: the rule (`multiplier`) and the per-trace factor (`traceScale`) |
| `embedding_visualization/forked/plotly.js/src/plot_api/plot_config.js` | Declares `scatterglOpacityScale` (fork change 6) |
| `embedding_visualization/forked/plotly.js/src/traces/scattergl/scene_update.js` | `applyOpacityScale`, called at the start of `scene.draw` (fork change 6) |
| `embedding_visualization/app/components/ScatterPlot2D.tsx` | Builds `ZoomOpacity`, passes the option, marks the fixed traces |

## Tests

- `lib/utils/__tests__/zoomOpacity.test.ts`: the rule on literal values (full view,
  zoomed out, 4× zoom, panning, reversed ranges, an axis without extent, the 0.8
  ceiling) and the per-trace factor (cap, fixed traces, per-point opacities).
- `forked/__tests__/scatterglOpacityScale.test.ts`: the fork's real `scene.draw`
  scales each group before drawing, never compounds, passes the trace and its
  ranges, and leaves selections and marker-less traces alone. It also pins the
  regl-scatter2d and `toImage` internals the patch relies on.

## Open

- The FPS benchmark (`benchmarks/fps/`) has not been run for this change.
- The curve is fixed. A Controls slider (strength of the zoom effect) could be
  added later; `ZoomOpacity.multiplier` is the place to change it.
