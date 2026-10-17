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

### 3. GL3D markers: no outline vertices when the line width is 0

**File:** `stackgl_modules/index.js`, gl-scatter3d `proto.update` (module 8418, search for `[fork]`)

**Problem:** every scatter3d marker is a filled glyph (triangles) plus a glyph outline (line segments), packed into the same four vertex buffers at 40 bytes per vertex. The outline is only drawn when `lineWidth > 0` (`drawProject` / `drawFull`), and plotly defaults `marker.line.width` to 0 (1 only for bubble charts with array sizes). `update()` still built and uploaded the outline for every point: about 40% of all marker vertices, never drawn.

**Fix:** after the count pass, `drawOutlines = this.lineWidth > 0`. When false, the outline vertex count is zeroed, so the allocations and uploads cover triangles only, and the outline-writing loop is skipped. The fill pass keeps its original gate (`hasGeometry`, fill + outline count before zeroing), so bounds and `hasAlpha` are computed exactly as before. A translucent `lineColor` still marks the trace transparent even at width 0, so render-pass selection is unchanged. The draw code is untouched.

**Contract:** outline geometry is built only by `update()`. Setting `lineWidth` directly on the object without calling `update()` would draw 0 outline vertices. Plotly never does that: it always passes `lineWidth` through `update(scatterOptions)` (`src/traces/scatter3d/convert.js`).

**Tests:** `forked/__tests__/glScatter3dOutline.test.ts`.

### 4. GL3D picking: no pick pass while the camera moves

**File:** `stackgl_modules/index.js`, gl-plot3d `createScene` (module 4100) and gl-select-static (module 3589); search for `[fork]`.

**Problem:** gl-plot3d resolves the mouse position by drawing every object a second time into offscreen pick buffers (up to 254 pickable objects each, so 11 for WordNet coloured by topic) and copying each buffer to the CPU with a full `readPixels` 1 ms later. `redraw()` did that on every frame the camera moved, although a moving frame's pick image is stale by the next frame and nothing is queried while the button is held. Three smaller faults sat next to it:
- `SelectBuffer.end()` guarded its `clearTimeout` with `if(!this._readTimeout)`, so two pick passes in one task scheduled two readbacks per buffer.
- New pick buffers were allocated at drawing-buffer size, four times the shape `renderPick()` gives them at pixel ratio 2, and the CPU copy never shrinks.
- The mouse handler answered from whatever copy was on the CPU, including one from before the camera moved.

**Fix:**
- `redraw()` renders the pick pass only when the camera did not move this frame (`pickNow = pickDirty && !cameraMoved`). A deferred pass keeps `pickDirty` set; the first still frame renders it, and is forced through the idle early return to do so.
- The mouse handler is a named `handleMouse`. While the pick is stale (`pickStale()`: a deferred pass, or a readback not yet run) a hover is not queried: it is kept as `deferredHover` and replayed by the buffers' new `onread` hook once every buffer has been read. A press instead brings the pick up to date at once (`renderPickNow()`: the deferred pass if there is one, then the pending readbacks), so a click is never lost or answered from a stale image; a press during motion costs one pick pass.
- gl-select-static: the `clearTimeout` guard is fixed, `onread(buffer)` runs after each readback, a `pending` getter reports a scheduled readback, `readNow()` runs it synchronously, and the readback unbinds the pick framebuffer instead of leaving it bound.
- New pick buffers are created at pick resolution (`max(1, drawingBuffer / pixelRatio)`).
- `scene.pickStats` counts `renders` (pick passes), `deferredFrames`, `readbacks` (one per buffer read) and the current `buffers`; the fork tests and the FPS benchmark read it. The counters accumulate for the scene's life.

**Behaviour change:** a mouse move while the camera is moving (wheel zoom, fly-to, or the camera's easing after a drag ends) clears the hover label and is answered on the first still frame. A press during motion still clicks on what is under the cursor at that moment, as before. The main render, the pick shaders and plotly's `scene.js` are unchanged.

**Tests:** `forked/__tests__/glPlot3dPickDeferral.test.ts`.

---

## react-plotly.js

### 1. Use plotly.js source build instead of pre-built dist bundle

**File:** `react-plotly.js` (line 10)

**Change:** `require("plotly.js/dist/plotly")` → `require("plotly.js")`

**Reason:** The original entry point hardcodes the pre-built `dist/plotly.js` bundle (11 MB), which is a standalone IIFE that bypasses the source tree entirely — meaning none of the patches above would take effect for the 2D Plot component. By importing `plotly.js` (which resolves to `lib/index.js` → `src/`), both 2D and 3D components use the same patched source build. This also allowed us to drop the entire `dist/` directory (90 MB) from the fork.

---

## Editing `stackgl_modules/index.js`

The bundle has **mixed line endings**: about 900 lines in a few vendored modules end in CRLF, the rest in LF. Editors and tools that normalise line endings rewrite all of them, which buries a small patch in a ~1,800-line whitespace diff. Edit it with a byte-preserving tool, and check that `git diff --stat` only shows the lines you meant to change. Mark every patched spot with a `[fork]` comment.

---

## Structural changes (not functional)

- **Removed `dist/`** (90 MB) — pre-built bundles, no longer needed since both 2D and 3D use the source build
- **Removed `topojson/`** (4.4 MB) — geographic map data, unused (project only uses scatter/scattergl/scatter3d)
- **Removed `tasks/`**, docs, CI configs — build infrastructure for the upstream project
- **Cleaned `package.json`** — removed `webpack`, `scripts`, `devDependencies`, `browserify` fields
- **Cleaned `stackgl_modules/package.json`** — removed `browserify` transform and `devDependencies` (the module is a pre-built webpack bundle; these were build-time only)
