# Rendering performance

A series of targeted fixes to the Explore page's scatter plots, driven by a verified performance review (2026-10-10). The costs show up on large collections, above all `wordnet_senses_full` (212,478 points; 2,694 topic labels plus "Unclustered"). Each step lands on its own, with tests first, a benchmark comparison and explicit sign-off before the next one starts.

## The problems

| # | Problem | Where | Status |
|---|---|---|---|
| 2 | 3D markers upload glyph-outline vertices that are never drawn (~40% of marker vertex memory) | fork: gl-scatter3d | **Done** (−42% marker vertex memory, −357 MB on WordNet) |
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
- **`BENCH_DPR`**: the page's `devicePixelRatio`. Historical runs used 1, which does not reflect a Retina screen. Non-1 ratios get their own result label suffix (`-dpr2`, `-dpr1p5`).
- **`2d-topics` and `2d-density` passes**, for the 2D trace-count and density-overlay questions.

Benchmarks need the backend on `:8000` and the frontend on `:3000`, and drive a headed Chrome; the user runs them.

## Step 2: no outline vertices at line width 0

**Change.** gl-scatter3d's `update()` packs each marker's fill triangles and outline segments into the same four vertex buffers (40 bytes per vertex). The outline is only drawn when `lineWidth > 0`, and plotly defaults `marker.line.width` to 0 (1 only when `marker.size` is an array). `update()` now skips building and uploading the outline when the width is 0. Bounds, `hasAlpha`, pick ids and everything that is drawn are unchanged. Details: `embedding_visualization/forked/plotly_changes.md`, change 3.

**What still draws outlines.** Only traces with a positive line width: in this app, the search-highlight core trace (`line.width: 1`).

**Tests.** `embedding_visualization/forked/__tests__/glScatter3dOutline.test.ts`, 8 cases on the real bundle through a recording WebGL stub (`forked/__tests__/stackglHarness.ts`):

- At width 0 there are no outline vertices and buffers are sized for triangles only. This was the one test that failed before the patch: 648 outline vertices for 3 points with the test glyph.
- Every draw pass issues only triangle draws.
- Width 1 is unchanged.
- The fill bytes are identical with and without outlines.
- Going 1 → 0 → 1 rebuilds correctly.
- A translucent line colour still marks the trace transparent.
- Pick ids survive invalid positions.

**Benchmark** (2026-10-16, page DPR 1, 8 GB MacBook with under 1 GB of RAM available during the runs). With the `●` marker glyph, each point costs **60 fill + 44 outline vertices** (104 total; 42% outline).

| `wordnet_senses_full` (212,478 pts) | Traces | Outline vertices | Marker vertex buffers | JS heap (post-load, after GC) | FPS median / mean | Load |
|---|---|---|---|---|---|---|
| `3d` before | 1 | 9,349,032 | 843.0 MB | 1,603 MB | 56.8 / 38.5 | 17.6 s |
| `3d` after | 1 | 0 | 486.3 MB | 1,157 MB | 59.9 / 42.3 | 15.4 s |
| `3d-topics` before | 2,695 | 9,349,032 | 843.0 MB | 1,513 MB | 8.6 / 7.9 | 17.3 s |
| `3d-topics` after | 2,695 | 0 | 486.3 MB | 1,253 MB | 8.6 / 8.0 | 16.0 s |

- **Marker vertex buffers: −357 MB (−42%)** in both passes. These figures are exact, from the GL objects' own counts. Before the patch the buffers were 843 MB, matching the review's "~850 MB".
- **JS heap: −446 MB (−28%) uncoloured, −260 MB (−17%) by topic.** The heap falls further than the vertex data in the single-trace view because the typed-array pool also keeps the freed CPU-side arrays, rounded up to powers of two.
- **FPS:** unchanged in the topic view, where per-frame cost is dominated by the 11 pick buffers (step 1). The small gain in the uncoloured view is within single-run noise, possibly helped by less memory pressure.
- The process-RSS figures are omitted: with under 1 GB available, macOS compresses and swaps Chrome's memory, so they don't reflect real use.
- Results: `benchmarks/fps/results/results_{3d,3d-topics}_step2-{before,after}-passthrough.json`. A 50k control (`imdb`, `results_3d_step2-{before,after}.json`) shows the same 42% vertex saving.

**Benchmark harness bug found on the way.** Since mid-July, `bench.mjs` intercepts each real collection's `GetCollectionData` response, buffers it in Node (to record its size) and replays it to the page with `route.fulfill`. Every replayed payload of 77 MB or more crashed the tab (WordNet 77 MB, `gemma_29_65k` 93 MB, `lacan_sentences_gemini_document` 133 MB), while 71 MB (`imdb`) and 53 MB (`synthetic_500k`) worked.

The replayed body travels base64-encoded over the DevTools protocol, which inflates it by a third: 71 MB → ~95 MB, 77 MB → ~103 MB. That's consistent with a ~100 MB message limit. The app itself is fine; WordNet loads normally in a regular browser. The WordNet numbers above came from a scratch copy of the harness that let real collection data pass through (`route.continue()`).

**Fixed (2026-10-16):**
- Real collection data now passes through untouched.
- `GraphqlPayloadTelemetry.recordPassThrough` notes each request, and `measurePending` re-sends it from Node after the browser context closes, recording the decoded size.
- Covered by three new tests in `benchmarkContract.test.mjs`.
- Verified on the three previously crashing collections: WordNet 77.39 MB (same byte count the old interception recorded), `gemma_29_65k` 93.13 MB and `lacan_sentences_gemini_document` 132.57 MB. All load.
- Synthetic payloads still use `route.fulfill`, so `synthetic_1m` (~106 MB) would hit the same limit; see the benchmark README's caveats.
