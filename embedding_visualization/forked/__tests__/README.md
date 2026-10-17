# Fork tests

Unit tests for the patches in `forked/plotly.js` (logged in `../plotly_changes.md`). They run in vitest's node environment with the rest of the frontend suite: `npm run test:run`.

| File | Purpose |
|------|---------|
| `stackglHarness.ts` | Loads the prebuilt `plotly.js/stackgl_modules/index.js` bundle in node. Every fork test loads the bundle through this module. |
| `glScatter3dOutline.test.ts` | gl-scatter3d builds no glyph-outline vertices when the marker line width is 0 (change 3). |
| `glPlot3dPickDeferral.test.ts` | gl-plot3d renders no pick pass while the camera moves, never answers a hover from a stale pick image, renders the pick on demand for a press, allocates pick buffers at pick resolution, and counts its work in `scene.pickStats` (change 4). |

## The harness

The stackgl bundle is a prebuilt webpack bundle that expects a browser: it reads `window` at load time, and vectorize-text grabs a 2D canvas from `document` when its module initializes. `StackglHarness.load()` installs minimal stand-ins for both, once, before requiring the bundle, then returns the bundle's exports (`gl_scatter3d`, `gl_plot3d`). It also gives `window` no-op event-listener methods (gl-plot3d and mouse-change register resize, blur and key listeners on it) and installs a `FrameScheduler` as `requestAnimationFrame`.

- **`FakeCanvasElement` / `FakeCanvas2DContext`:** glyph rasterization. `getImageData` returns a filled disc, so every glyph traces to a circle with both a triangle mesh and an outline. Vertex counts therefore depend on the fake glyph: tests derive reference counts from a build instead of hard-coding them. The canvas also keeps an event-listener registry, so a gl-plot3d test can `dispatch` mouse events (`offsetX`, `offsetY`, `buttons`, which is what mouse-change reads).
- **`RecordingGl`:** a Proxy-based WebGL context (constants in the exported `GL_ENUMS`). Shader compilation and linking always succeed and report no active uniforms or attributes. It records every call, plus `drawArrays` calls, `lineWidth` values, and the latest uploaded bytes per buffer handle. `readPixels` fills its target with `readPixelsFill` when one is set, so a pick query can be made to hit or miss. Constants used in assertions, or validated by the bundle (texture filters and wraps), carry their real WebGL values (`TRIANGLES` = 4, `LINEAR` = 9729, ...).
- **`StackglHarness.frames`** (`FrameScheduler`): the `requestAnimationFrame` queue. `step()` runs one frame; gl-plot3d's render loop re-queues itself, so it runs exactly once per step. The 1 ms pick readbacks run on vitest's fake timers.
- gl-plot3d tests pass a fake camera as `cameraObject` (`tick()` reports whether the view moved, `matrix` is read) and fake pick objects with one pick slot; both live in `glPlot3dPickDeferral.test.ts`.

Draw calls work with `project: [false, false, false]` (no axis-projection pass, which needs reflected uniforms) and `cloud.axes = { bounds }` set before drawing.

The globals stay installed for the whole test file. With vitest's default `isolate: true`, each test file gets a fresh module environment, so they don't leak into other suites. With isolation off, test files in the same worker share Node's require cache, which is why every fork test must load the bundle through the harness. Fork tests need the node environment: under jsdom, the harness throws a clear error because jsdom canvases have no 2D context.
