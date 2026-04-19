# ScatterPlot3D Performance & Maintainability Analysis

**Date:** 2025-04-15
**File:** `app/components/ScatterPlot3D.tsx`

## Resolved Issues

- **#1** Trace duplication -- refactored to `buildScatter3dTrace`/`buildIndexedScatter3dTrace` helpers
- **#2** Unused `text` arrays -- removed `formatHoverText` from all traces
- **#3** Full Point3D in customdata -- now stores `point.index`, handlers use `pointsRef` lookup
- **#4** Multi-pass array allocations -- single-pass extraction with pre-allocated `Float64Array` for coordinates
- **#6** Duplicated `hideUnclustered` filter -- extracted shared `displayPoints` useMemo

---

## Critical Performance Issues (remaining)

### 1. Massive trace-building duplication in `baseTraces` (lines 416-768)
**Problem:** ~350 lines of deeply nested conditionals with the same trace-building pattern (map x, y, z, text, customdata) copy-pasted **~12 times** across three coloring modes (numeric / categorical / none) x two muting states (combinedMutedIndices or not) x nested-categorical variant. Any bug fix or property change must be applied in all branches.

**Solution:** Extract a `buildTrace(points, markerOpts)` helper that takes a point subset and marker config, returns a `PlotlyData` object. The branching logic should only determine *which points* and *what colors/opacity* -- not rebuild the entire trace shape each time.

---

### 2. `text` arrays built for every point but never displayed (lines 435, 562, 601, 640, 680...)
**Problem:** `formatHoverText` is called on every point to build `text` arrays, but every trace sets `hoverinfo: 'none'`. The custom `FrostedTooltip` reads from `customdata`, not `text`. These arrays are allocated, populated, and passed to Plotly for nothing.

**Solution:** Remove the `text` property from all base traces. Only the highlight core trace (line 841) might need it, and even that uses `hoverinfo: 'none'`.

---

### 3. `customdata` stores full Point3D objects, doubling memory (lines 436, 477, 497, 603, 682...)
**Problem:** Entire `Point3D` objects (including all metadata) are passed as `customdata` to Plotly. Plotly stores a separate internal copy. For a 100k-point dataset with rich metadata, this effectively doubles memory usage for point data.

**Solution:** Store only the index in `customdata` and look up the full `Point3D` from the `points` array in the click/hover handlers. This reduces Plotly's internal data footprint to a single number per point.

---

### 4. Redundant multi-pass array allocations (lines 432-436, 459-476)
**Problem:** Four separate `.map()` calls iterate all `displayPoints` to extract x, y, z, text. Then in the muting branches, these pre-built arrays are indexed *again* with another `.map()` pass (`activeIndices.map(i => allX[i])`), creating yet more intermediate arrays. For categorical mode, each category group does its own `.map(p => p.x)` etc., so total allocations scale with category count.

**Solution:** Build coordinate arrays in a single pass using typed arrays (`Float64Array`) for x/y/z. For the muting split, partition indices in one pass and use typed array views or pre-allocated buffers instead of per-split `.map()` chains.

---

### 5. `numericData` does three passes where one suffices (lines 353-378)
**Problem:** Creates `values` via `.map()`, then `validValues` via `.filter()`, then `cleanValues` via another `.map()`. Three full-array iterations.

**Solution:** Single pass: iterate points once, accumulate min/max inline, write cleaned values directly into the output array.

---

## Moderate Performance Issues

### 6. `hideUnclustered` filter logic duplicated (lines 421-429 vs 917-925)
**Problem:** The exact same filtering logic (check `topic_id === -1` or `topic_label === 'Unclustered'`) runs independently in `baseTraces` and `clusterDataMap`, producing two separate filtered copies.

**Solution:** Extract a single `displayPoints` memo that both consumers share.

---

### 7. Nebula `syncCanvasLayout` forces layout thrashing every GL frame (lines 1386-1405, 1420)
**Problem:** The `onrender` callback calls `getBoundingClientRect()` twice (container + glCanvas) and writes to canvas style properties on every GL render frame. This forces the browser to recalculate layout on each frame.

**Solution:** Only sync layout on resize events (use a ResizeObserver). Cache the last known rect and skip DOM reads/writes when unchanged.

---

### 8. rAF camera polling runs every frame even when camera is stationary (lines 1153-1200)
**Problem:** `requestAnimationFrame(pollCamera)` runs unconditionally at ~60fps as long as any labels are visible. While there's a change-detection check, the rAF callback itself still runs, reads Plotly internals, and does floating-point comparisons every frame.

**Solution:** After N consecutive frames with no camera change (e.g., 30 frames = 0.5s), switch to a lower-frequency `setTimeout` poll (e.g., 200ms). Resume rAF polling on user interaction events (`mousedown`, `wheel`, `touchstart`).

---

### 9. `renderLabels` dependency cascade (line 1149 -> 1200)
**Problem:** `renderLabels` depends on `clusterDataMap`, which changes on mute/unmute. New `renderLabels` reference -> rAF polling effect restarts (line 1200 depends on `renderLabels`). Muting a single category teardowns and recreates the entire polling loop.

**Solution:** Move `clusterDataMap` reads to a ref (like `labelRenderDataRef` already does for point labels) so that `renderLabels` has a stable identity. The polling loop won't restart on data changes -- it'll just pick up the new ref value on the next frame.

---

### 10. Overlay traces use `Plotly.redraw` (full scene rebuild) (line 1244)
**Problem:** When highlight/selected traces change, `gd.data.splice()` + `Plotly.redraw(gd)` redraws the *entire* 3D scene including all base traces. For large datasets, this is expensive.

**Solution:** Use `Plotly.restyle` targeted at only the overlay trace indices to update their properties in-place, or use `Plotly.deleteTraces` + `Plotly.addTraces` which Plotly can optimize better for partial updates.

---

### 11. `tooltipFields` prop likely causes effect churn (line 1360)
**Problem:** The hover event handler effect depends on `[plotReady, tooltipFields]`. If the parent passes `tooltipFields` as a new array literal on each render (e.g., `tooltipFields={['field1', 'field2']}`), this effect re-runs every render -- detaching and reattaching Plotly event listeners each time.

**Solution:** Either memoize `tooltipFields` in the parent, or move it to a ref inside the component since it's only read inside the event handler closure (not used for conditional logic).

---

## Structural / Maintainability Issues

### 12. 1484-line monolithic component
**Problem:** Camera animation, trace building, canvas label rendering, nebula effects, hover/click handling, and zoom limiting are all in one file with shared refs and interleaved effects.

**Solution:** Extract custom hooks: `useCameraAnimation`, `useCanvasLabels`, `useNebulaEffect`, `useHoverTooltip`. Each would own its own refs and effects, reducing coupling and making individual features testable.

---

### 13. `renderedSelectedPoint` eslint-disable hack (line 118-119)
**Problem:** `useMemo(() => selectedPoint, [highlightedIndices])` intentionally lies about dependencies. It works (defers selected-point trace sync until search results arrive), but is fragile -- if `highlightedIndices` changes without `selectedPoint` changing, it still re-evaluates; if `selectedPoint` changes multiple times between `highlightedIndices` changes, intermediate values are silently dropped.

**Solution:** Use an explicit ref + effect pattern: a ref that stores the "rendered" selected point, and an effect that syncs it only when `highlightedIndices` changes. This makes the intent explicit without fighting React's dependency model.

---

### 14. Plotly imported twice via different paths (lines 24-28 and 198-202)
**Problem:** The `Plot` component factory imports `plotly.js-dist-min` via `react-plotly.js/factory`, and then the component imports it *again* into `plotlyLibRef` via a separate `useEffect`. These are two independent dynamic imports with different timing.

**Solution:** Share the Plotly library reference from the factory import. Alternatively, use a module-level promise that both consumers await.

---

### 15. `layout` memo includes constants that could be hoisted (line 1247-1264)
**Problem:** `paperBg` and `sceneBg` are both `'rgba(0,0,0,0)'` regardless of theme, but they're declared inside the component and listed as layout dependencies. The `defaultEye` in the camera config means the layout object is recreated whenever point count changes (via `defaultEye` -> `pointCount`), even though the camera is only set on initial load.

**Solution:** Hoist constant values (`paperBg`, `sceneBg`, axis config objects) to module scope. Use `uirevision` to prevent Plotly from re-applying the camera from layout on every update (already partially done).

---

### 16. `mutedCategories` array reference instability (line 936)
**Problem:** `clusterDataMap` depends on `mutedCategories`. If the parent passes a new `[]` on each render when nothing is muted, the cluster geometry recomputes unnecessarily. Inside the memo, `new Set(mutedCategories)` is created each time.

**Solution:** Accept `mutedCategories` as a `Set` from the parent (avoiding array->Set conversion), or use a ref-based comparison to skip recomputation when the set contents are identical.

---

## Summary by Impact

| Priority | Issue | Est. Impact | Status |
|----------|-------|-------------|--------|
| **High** | #1 Trace duplication | Code quality | Resolved |
| **High** | #2 Unused `text` arrays | Wasted CPU + memory, scales with N | Resolved |
| **High** | #3 Full Point3D in customdata | ~2x memory for point data | Resolved |
| **High** | #4 Multi-pass array allocations | O(N) wasted work per render | Open |
| **Medium** | #6 Duplicated hideUnclustered | Redundant filtering | Resolved |
| **Medium** | #7 Layout thrashing in nebula | Jank during 3D interaction | Open |
| **Medium** | #8 Unconditional rAF polling | Battery/CPU when idle | Open |
| **Medium** | #9 renderLabels cascade | Unnecessary teardown/setup | Open |
| **Medium** | #10 Full redraw for overlay traces | Slow highlight updates on large data | Open |
| **Low** | #12 Monolithic component | Maintainability | Open |
| **Low** | #15-16 Constant/reference instability | Unnecessary memo invalidation | Open |
