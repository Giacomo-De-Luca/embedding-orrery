# Plotly Fork — Change Log

Forked from **plotly.js v3.3.0** and **react-plotly.js v2.6.0**.
These packages will not be updated from upstream. All future changes are documented here.

---

## plotly.js

### 1. GL3D fast-path: skip GPU buffer re-upload for unchanged traces

**File:** `src/plots/gl3d/scene.js` (lines 620–669)

**Problem:** Every `Plotly.react()` call iterates all traces in a 3D scene and calls `trace.update(data)`, which runs `convertPlotlyOptions()` (O(n) per trace — loops all points, applies scaling, rebuilds GPU buffers). When only one trace out of M changes, all M traces still get the full update — O(N\*M) GPU buffer uploads instead of O(1).

**Fix:** Before calling `trace.update()`, check two conditions:
1. The scene-wide `dataScale` hasn't changed (`_sameDataScale`)
2. The trace's input object reference is the same as last frame (`trace._lastInput === data._input`)

If both hold, skip the update and just refresh the data pointer for hover lookups. This relies on callers (ScatterPlot3D) passing memoized, never-mutated trace objects via `useMemo`.

**Debug logging:** A temporary `console.debug('[gl3d patch]')` log on line 668 reports skip/update counts per frame. Remove once validated.

### 2. GL3D camera zoom-out limit

**File:** `src/plots/gl3d/scene.js` (line 186)

**Change:** `zoomMax: 100` → `zoomMax: 2`

**Reason:** Prevents the user from zooming out so far that the 3D scatter plot becomes a tiny dot. The default of 100 allowed excessive zoom-out; 2 keeps the data filling the viewport at maximum zoom-out.

---

## react-plotly.js

### 1. Use plotly.js source build instead of pre-built dist bundle

**File:** `react-plotly.js` (line 10)

**Change:** `require("plotly.js/dist/plotly")` → `require("plotly.js")`

**Reason:** The original entry point hardcodes the pre-built `dist/plotly.js` bundle (11 MB), which is a standalone IIFE that bypasses the source tree entirely — meaning none of the patches above would take effect for the 2D Plot component. By importing `plotly.js` (which resolves to `lib/index.js` → `src/`), both 2D and 3D components use the same patched source build. This also allowed us to drop the entire `dist/` directory (90 MB) from the fork.

---

## Structural changes (not functional)

- **Removed `dist/`** (90 MB) — pre-built bundles, no longer needed since both 2D and 3D use the source build
- **Removed `topojson/`** (4.4 MB) — geographic map data, unused (project only uses scatter/scattergl/scatter3d)
- **Removed `tasks/`**, docs, CI configs — build infrastructure for the upstream project
- **Cleaned `package.json`** — removed `webpack`, `scripts`, `devDependencies`, `browserify` fields
- **Cleaned `stackgl_modules/package.json`** — removed `browserify` transform and `devDependencies` (the module is a pre-built webpack bundle; these were build-time only)
