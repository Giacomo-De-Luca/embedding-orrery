# Fork tests

Unit tests for the patches in `forked/plotly.js` (logged in `../plotly_changes.md`). They run in vitest's node environment with the rest of the frontend suite: `npm run test:run`.

| File | Purpose |
|------|---------|
| `stackglHarness.ts` | Loads the prebuilt `plotly.js/stackgl_modules/index.js` bundle in node. Every fork test loads the bundle through this module. |
| `glScatter3dOutline.test.ts` | gl-scatter3d builds no glyph-outline vertices when the marker line width is 0 (change 3). |

## The harness

The stackgl bundle is a prebuilt webpack bundle that expects a browser: it reads `window` at load time, and vectorize-text grabs a 2D canvas from `document` when its module initializes. `StackglHarness.load()` installs minimal stand-ins for both, once, before requiring the bundle, then returns its exports (`gl_scatter3d`, `gl_plot3d`, ...).

- **`FakeCanvasElement` / `FakeCanvas2DContext`:** glyph rasterization. `getImageData` returns a filled disc, so every glyph traces to a circle with both a triangle mesh and an outline. Vertex counts therefore depend on the fake glyph: tests derive reference counts from a build instead of hard-coding them.
- **`RecordingGl`:** a Proxy-based WebGL context. Shader compilation and linking always succeed and report no active uniforms or attributes. It records every call, plus `drawArrays` calls, `lineWidth` values, and the latest uploaded bytes per buffer handle. Constants used in assertions carry their real WebGL values (`TRIANGLES` = 4, `LINES` = 1, ...).

Draw calls work with `project: [false, false, false]` (no axis-projection pass, which needs reflected uniforms) and `cloud.axes = { bounds }` set before drawing.

The globals stay installed for the whole test file. Vitest isolates test files, so they don't leak into other suites.
