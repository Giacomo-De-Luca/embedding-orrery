# Fork tests

Unit tests for the patches in `forked/plotly.js` (logged in `../plotly_changes.md`). They run in vitest's node environment with the rest of the frontend suite: `npm run test:run`.

| File | Purpose |
|------|---------|
| `stackglHarness.ts` | Loads the prebuilt `plotly.js/stackgl_modules/index.js` bundle in node. Every fork test loads the bundle through this module. |
| `glScatter3dOutline.test.ts` | gl-scatter3d builds no glyph-outline vertices when the marker line width is 0 (change 3). |

## The harness

The stackgl bundle is a prebuilt webpack bundle that expects a browser: it reads `window` at load time, and vectorize-text grabs a 2D canvas from `document` when its module initializes. `StackglHarness.load()` installs minimal stand-ins for both, once, before requiring the bundle, then returns the bundle's exports. Tests use `gl_scatter3d`; gl-plot3d's `createScene` needs more stand-ins (e.g. `window.addEventListener`), to be added when a test needs it.

- **`FakeCanvasElement` / `FakeCanvas2DContext`:** glyph rasterization. `getImageData` returns a filled disc, so every glyph traces to a circle with both a triangle mesh and an outline. Vertex counts therefore depend on the fake glyph: tests derive reference counts from a build instead of hard-coding them.
- **`RecordingGl`:** a Proxy-based WebGL context (constants in the exported `GL_ENUMS`). Shader compilation and linking always succeed and report no active uniforms or attributes. It records every call, plus `drawArrays` calls, `lineWidth` values, and the latest uploaded bytes per buffer handle. Constants used in assertions carry their real WebGL values (`TRIANGLES` = 4, `LINES` = 1, ...).

Draw calls work with `project: [false, false, false]` (no axis-projection pass, which needs reflected uniforms) and `cloud.axes = { bounds }` set before drawing.

The globals stay installed for the whole test file. With vitest's default `isolate: true`, each test file gets a fresh module environment, so they don't leak into other suites. With isolation off, test files in the same worker share Node's require cache, which is why every fork test must load the bundle through the harness. Fork tests need the node environment: under jsdom, the harness throws a clear error because jsdom canvases have no 2D context.
