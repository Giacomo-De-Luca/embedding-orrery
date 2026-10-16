# Rendering performance

A series of targeted fixes to the Explore page's scatter plots, driven by a verified performance review (2026-10-10). The costs show up on large collections, above all `wordnet_senses_full` (212,478 points; 2,694 topic labels plus "Unclustered"). Each step lands on its own, with tests first, a benchmark comparison and explicit sign-off before the next one starts.

## The problems

| # | Problem | Where | Status |
|---|---|---|---|
| 2 | 3D markers upload glyph-outline vertices that are never drawn (~40% of marker vertex memory) | fork: gl-scatter3d | **Implemented, awaiting benchmark** |
| 1 | 3D pick buffers re-render and fully read back on every frame the camera moves (11 buffers, ~63 MB per frame, for the WordNet topic view) | fork: gl-plot3d, gl-select-static | Planned |
| 4 | Plotly's entry file registers ~48 trace types, including two map libraries the app never uses | fork: `lib/index.js` | Planned |
| 7 | One trace per category: ~2,695 traces when WordNet is coloured by topic | `ScatterPlot3D.tsx` | Planned |
| 6 | Plotly's GL pixel ratio is fixed at 2, quadrupling framebuffers on standard screens | both plot configs | Planned |
| 5 | Apollo's lazy query keeps the previous collection's whole payload after a switch | `useEmbeddingData.ts` | Planned |
| 3 | The 2D density overlay allocates four RGBA32F framebuffers (~410 MiB at 1600×950 on Retina) | `densityRenderer/` | Planned (A/B/C comparison first) |

Future work, not scheduled: batching the nebula haze sprites into one draw call; releasing WebGL contexts on unmount; 2D trace grouping (decided after the `2d-topics` measurement); gl-plot3d's pick bookkeeping not re-synced after plotly sorts objects (moot with one pick buffer); the typed-array pool never freeing; the pick framebuffer left bound after a readback.

## How each step is measured

The FPS benchmark (`benchmarks/fps/`, see its README) gained three things for this series:

- **`glStatsPostLoad` / `glStatsPostDrag`** on every run: trace count, marker fill and outline vertices with their buffer size, pick stats (once step 1 exposes them), and the GL canvas backing ratio.
- **`BENCH_DPR`**: the page's `devicePixelRatio`. Historical runs used 1, which does not reflect a Retina screen. Non-1 ratios get their own result label suffix (`-dpr2`).
- **`2d-topics` and `2d-density` passes**, for the 2D trace-count and density-overlay questions.

Benchmarks need the backend on `:8000` and the frontend on `:3000`, and drive a headed Chrome; the user runs them.

## Step 2: no outline vertices at line width 0

**Change.** gl-scatter3d's `update()` packs each marker's fill triangles and outline segments into the same four vertex buffers (40 bytes per vertex). The outline is only drawn when `lineWidth > 0`, and plotly defaults `marker.line.width` to 0. `update()` now skips building and uploading the outline when the width is 0. Bounds, `hasAlpha`, pick ids and everything that is drawn are unchanged. Details: `embedding_visualization/forked/plotly_changes.md`, change 3.

**What still draws outlines.** Only traces with a positive line width: in this app, the search-highlight core trace (`line.width: 1`).

**Tests.** `embedding_visualization/forked/__tests__/glScatter3dOutline.test.ts`, 8 cases on the real bundle through a recording WebGL stub (`forked/__tests__/stackglHarness.ts`):

- At width 0 there are no outline vertices and buffers are sized for triangles only. This was the one test that failed before the patch: 648 outline vertices for 3 points with the test glyph.
- Every draw pass issues only triangle draws.
- Width 1 is unchanged.
- The fill bytes are identical with and without outlines.
- Going 1 → 0 → 1 rebuilds correctly.
- A translucent line colour still marks the trace transparent.
- Pick ids survive invalid positions.

**Benchmark.** Pending; see the commands below. Expected: `glStatsPostLoad.outlineVertices` drops to 0 for the base traces, `vertexMB` falls by the outline share, and GPU-process memory falls with it. FPS should not change, since outlines were never drawn.

```bash
# 1. Baseline with the original bundle (swap it in, restart next dev, run, swap back)
BENCH_RESULT_LABEL=step2-before node bench.mjs 3d-topics wordnet_senses_full
BENCH_RESULT_LABEL=step2-before node bench.mjs 3d wordnet_senses_full
# 2. Patched bundle
BENCH_RESULT_LABEL=step2-after node bench.mjs 3d-topics wordnet_senses_full
BENCH_RESULT_LABEL=step2-after node bench.mjs 3d wordnet_senses_full
```

| Run | Traces | Fill vertices | Outline vertices | Vertex MB | GPU process MB (post-load) | JS heap MB (post-load) | FPS median |
|---|---|---|---|---|---|---|---|
| `3d-topics` before | | | | | | | |
| `3d-topics` after | | | | | | | |
| `3d` before | | | | | | | |
| `3d` after | | | | | | | |
