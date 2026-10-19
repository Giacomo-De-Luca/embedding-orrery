# Rendering performance

A series of targeted fixes to the Explore page's scatter plots, driven by a verified performance review (2026-10-10). The costs show up on large collections, above all `wordnet_senses_full` (212,478 points; 2,694 topic labels plus "Unclustered"). Each step lands on its own, with tests first, a benchmark comparison and explicit sign-off before the next one starts.

## The problems

| # | Problem | Where | Status |
|---|---|---|---|
| 2 | 3D markers upload glyph-outline vertices that are never drawn (~40% of marker vertex memory) | fork: gl-scatter3d | **Done** (−42% marker vertex memory, −357 MB on WordNet) |
| 1 | 3D pick buffers re-render and fully read back on every frame the camera moves (11 buffers, ~63 MB per frame, for the WordNet topic view) | fork: gl-plot3d, gl-select-static | Implemented, tests pass; benchmark pending |
| 4 | Plotly's entry file registers ~48 trace types, including two map libraries the app never uses | fork: `lib/index.js` | Implemented, tests pass (−59% Plotly JS); in-app check pending |
| 7 | One trace per category: ~2,695 traces when WordNet is coloured by topic | `ScatterPlot3D.tsx` | Planned |
| 6 | Plotly's GL pixel ratio is fixed at 2, quadrupling framebuffers on standard screens | both plot configs | Planned |
| 5 | Apollo's lazy query keeps the previous collection's whole payload after a switch | `useEmbeddingData.ts` | Planned |
| 3 | The 2D density overlay allocates four RGBA32F framebuffers (~410 MiB at 1600×950 on Retina) | `densityRenderer/` | Planned (A/B/C comparison first) |
| P | The stackgl typed-array pool keeps every vertex buffer gl-scatter3d hands back to it, so a large collection's 3D buffers (704 MiB for WordNet) stay resident for the life of the tab after switching away | `lib/utils/stackglBufferPool.ts`, `ScatterPlot3D.tsx` | Implemented, tests pass; in-app check pending |

Future work, not scheduled: batching the nebula haze sprites into one draw call; releasing WebGL contexts on unmount; 2D trace grouping (decided after the `2d-topics` measurement); gl-plot3d's pick bookkeeping not re-synced after plotly sorts objects (moot with one pick buffer); size-aware pool eviction for the case where one big collection is viewed both uncoloured and by topic (both buffer sets then stay pooled, see "Typed-array pool release").

## How each step is measured

The FPS benchmark (`benchmarks/fps/`, see its README) gained three things for this series:

- **`glStatsPostLoad` / `glStatsPostDrag`** on every run: trace count, marker fill and outline vertices with their buffer size, pick stats (`scene.pickStats`, from step 1 on), the GL canvas backing ratio, and `poolRetainedMB`, the idle bytes in the stackgl typed-array pool.
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

## Step 1: no pick pass while the camera moves

**Change.** gl-plot3d resolves the mouse against offscreen pick buffers: every object is drawn a second time into them (up to 254 objects per buffer, 11 buffers for WordNet by topic) and each buffer is copied to the CPU with a full `readPixels` 1 ms later. Before this step that happened on every frame the camera moved, which is wasted: the image is stale by the next frame, and the handler never queries while the button is held. Now:

- The pick pass renders only on frames where the camera is still. A pass deferred during motion runs on the first still frame, which is forced to render even if nothing else changed.
- A hover that arrives while the pick is stale is not answered from the old CPU copy. It is remembered and replayed once every buffer has been read back, so the hover label returns about one frame after the camera stops.
- A press while the pick is stale brings it up to date immediately (the deferred pass if there is one, then the pending readbacks), so a click right after a drag or wheel stop, or during a fly-to, lands on what is under the cursor and is never dropped. A press during motion costs one pick pass, instead of one per frame.
- gl-select-static's inverted `clearTimeout` guard is fixed (two passes in one task read each buffer once, not twice), and the readback no longer leaves the pick framebuffer bound.
- New pick buffers are allocated at pick resolution. The CPU copy was four times too large at pixel ratio 2: 2 MiB instead of 8 MiB per buffer for an 800×600 plot, so about 260 MiB less for WordNet's 11 buffers at a typical window size.
- `scene.pickStats` counts pick `renders`, `deferredFrames`, `readbacks` and the current `buffers`; the benchmark records it post-load and post-drag.

Details: `embedding_visualization/forked/plotly_changes.md`, change 4.

**Tests.** `embedding_visualization/forked/__tests__/glPlot3dPickDeferral.test.ts`, 18 cases on the real bundle with fake pick objects, a fake camera, a hand-stepped frame loop and fake timers for the readback. Sixteen failed before the patch (one of them only because `pickStats` did not exist yet):

- no pick pass or readback on moving frames;
- one pass and one readback per buffer on the first still frame (300 objects → 2 buffers), with the pixel ratio restored;
- an update while moving is deferred to the first still frame; a synchronous `scene.redraw()` while moving defers too;
- `pickStats` counters;
- one readback per buffer for two passes in one task; no pick framebuffer left bound; buffers allocated at pick resolution;
- no stale hit for a hover after the camera moved; no hit during a pending readback, then the replayed hit; only the latest deferred hover is replayed, and with two buffers only after both have been read back;
- a press while moving renders the pick on demand and selects; a press while only the readback is pending reads back at once without a second pass; a drag's release is resolved at its own position once the camera is still.

Two guards passed before and after: idle frames draw nothing, and a hover on a fresh pick is answered at once.

**Benchmark.** Pending, run by the user. The Step 2 "after" files are this step's baseline (same bundle), so only the patched runs are needed:

```bash
BENCH_RESULT_LABEL=step1-after node bench.mjs 3d-topics wordnet_senses_full
BENCH_RESULT_LABEL=step1-after node bench.mjs 3d wordnet_senses_full
BENCH_RESULT_LABEL=step1-after node bench.mjs 3d-topics lacan_sentences_gemini_document
```

Expected: `3d-topics` drag FPS up from 8.6 (the 2.7k main-pass draw calls remain until step 7); between post-load and post-drag `pickStats.renders` grows by about one (the first still frame after the release renders the deferred pass) while `deferredFrames` grows by about the number of frames drawn during the drag; process memory down by the smaller CPU readback copies. Camera fly-to animations are not benchmarked, so check them by hand on WordNet and Lacan: click a point or a search result and watch the flight; read the counters from the console with `document.querySelector('.js-plotly-plot')._fullLayout.scene._scene.glplot.pickStats`.

| Run | Traces | Pick buffers | Renders / deferred (drag) | JS heap (post-load) | FPS median / mean |
|---|---|---|---|---|---|
| `3d-topics` before (`step2-after`) | 2,695 | 11 | | 1,253 MB | 8.6 / 8.0 |
| `3d-topics` after | | | | | |
| `3d` before (`step2-after`) | 1 | 1 | | 1,157 MB | 59.9 / 42.3 |
| `3d` after | | | | | |

## Step 4: slim Plotly entry

**Change.** `plotly.js` resolves to the fork's `lib/index.js`, which registered all ~48 upstream trace types and the calendars component. The app only creates `scattergl` (2D) and `scatter3d` (3D) traces. The entry now registers just those two. Plotly's core still registers `scatter`, the modebar, box/lasso selection, hover, legend and colorbar, and `toImage`, `react`, `relayout` and `redraw` are core plot API methods. Details: `embedding_visualization/forked/plotly_changes.md`, change 5.

This step is about load time, not frame rate: it cuts the Plotly code the browser downloads, parses and runs before the first plot. FPS, GPU memory and per-point memory don't change.

**Tests.** `embedding_visualization/forked/__tests__/plotlyEntry.test.ts`, 4 cases. They bundle `plotly.js` the way the app imports it (esbuild, with `next.config.ts`'s Turbopack aliases) and read the module graph:

- Every trace type the app creates is registered. The test scans `app/` and `lib/` for trace names written as string literals in a `type:` value (ternaries and quoted keys included), so a new chart type written that way fails here instead of silently drawing nothing. A type held in a variable is invisible to the scan.
- No trace type the app doesn't create is registered. Failed before (46 unused).
- No map, geo or calendar packages are bundled. Failed before (`maplibre-gl`, `@plotly/mapbox-gl`, `world-calendars`, `d3-geo`, `d3-geo-projection`, `topojson-client`, five `@turf/*`).
- The scattergl and scatter3d traces, the stackgl bundle and `regl-scatter2d` are still in the bundle.

**Measurement** (esbuild 0.27, minified, production `NODE_ENV`, the app's aliases; decimal MB):

| Plotly as the app bundles it | Modules | npm packages | Minified | Gzip | Brotli |
|---|---|---|---|---|---|
| Before (upstream entry) | 1,157 | 118 | 4.93 MB | 1.52 MB | 1.13 MB |
| After (scattergl + scatter3d) | 528 | 82 | 2.00 MB | 0.66 MB | 0.53 MB |
| Change | −629 | −36 | **−2.93 MB (−59%)** | **−0.86 MB (−57%)** | −0.60 MB (−53%) |

More than half of the removed code is the two map libraries, `maplibre-gl` (822 KB minified) and `@plotly/mapbox-gl` (798 KB). Both used to load and run on every page with a plot, because their modules require them at the top level. What remains is dominated by the stackgl bundle (520 KB, the 3D engine), Plotly's cartesian plots and the regl 2D renderer. Next's own chunks will differ in size from esbuild's output, but the removed modules are the same.

**Where it shows:**
- First load of the Explore page, most on phones and slow connections, and on the HF Space demo with its small seed collections, where Plotly is a large share of what loads before the first plot. On WordNet the 77 MB payload dominates, so expect little visible change there.
- `next dev`'s first compile of the Explore page, and the memory-capped Docker build, which has been killed for running out of memory on the forked Plotly's modules before (`next.config.ts`). Both now see half as many Plotly modules. Expected, not measured.

**In-app check** (user):
- Both plots render, and hover and click work in 2D and 3D.
- The modebar buttons work, including the custom ones, plus 2D box and lasso select.
- Screenshot export works in 2D and 3D.
- No new console errors. Plotly only logs "Unrecognized trace type" at `logging: 2` (the default is 1), so a missing registration shows up as an empty plot, not a message. The entry test is the guard.
- Optional: compare `npm run build` chunk sizes before and after. This is a heavy build, so it's the user's call.

## Typed-array pool release

**Problem.** gl-scatter3d borrows its four vertex arrays (positions, colours, glyph offsets, pick ids; 40 bytes per vertex) from `typedarray-pool`, uploads them to the GPU and hands them straight back (`stackgl_modules/index.js`, gl-scatter3d `update`). The pool files every returned ArrayBuffer in a power-of-two bucket on `globalThis.__TYPEDARRAY_POOL.DATA` and nothing in the bundle ever empties it (its `clearCache` is not reachable from the bundle's exports). The buckets are reused only by a later trace of the same size class, so after a large collection has been drawn in 3D its buffers stay resident for the life of the tab, through collection switches, 2D mode and client-side navigation to `/sae`. A full reload is the only thing that cleared them.

For WordNet uncoloured (12,748,680 fill vertices at 60 per point after step 2):

| Array | Bytes needed | Pool bucket |
|---|---|---|
| positions (12 B/vertex) | 153 MB | 256 MiB |
| colours (16 B/vertex) | 204 MB | 256 MiB |
| glyph offsets (8 B/vertex) | 102 MB | 128 MiB |
| pick ids (4 B/vertex) | 51 MB | 64 MiB |
| **Idle in the pool after switching away** | | **704 MiB** |

The step 2 benchmark corroborates the model: its before/after heap drop in the `3d` pass was 446 MB, and the same arithmetic at 104 vertices per point gives 1,152 MiB before, so a predicted drop of 448 MiB. Coloured by topic the 2,695 per-trace arrays add up to a similar order.

**Change.** `lib/utils/stackglBufferPool.ts` (`StackglBufferPool`) reads the pool through its global and drops every idle buffer of 1 MiB or more (`release()`), in place, since the bundle holds references to the bucket arrays; `retainedBytes()` reports what is idle. Buffers in use are never in the buckets, so a release is safe at any moment: the next `update()` allocates afresh. `ScatterPlot3D` calls it from one effect keyed on the point count:

- **Why the point count, not the collection name.** Pooled sizes follow the number of points, and the selected-collection name changes before the new data arrives, while the old points may still be redrawn (and would re-borrow their old sizes).
- **Why the cleanup phase.** React runs effect cleanups before the effects of the same commit, so the release happens before the effect that builds the next collection's traces: the old buffers are evicted, the new ones are allocated fresh and then stay pooled for that collection's rebuilds. On unmount (2D mode, another page) the purge cleanup releases again right after `Plotly.purge`, which is when gl-select-static returns the pick buffer's CPU copy to the pool.
- **Not after every render.** Mute toggles, colour changes and filters rebuild all traces; releasing after each one would make every rebuild pay for ~700 MB of fresh zeroed allocations instead of reusing the pool.

**What it does not cover.** While you stay on a collection its pool buffers are live, so the post-load heap of a WordNet run does not change. Viewing one big collection both uncoloured and by topic leaves both buffer sets pooled, since the point count is the same (future work: size-aware eviction). Apollo's hold on the previous collection's payload is step 5.

**Tests.**
- `lib/utils/__tests__/stackglBufferPool.test.ts`, 8 cases on a pool shaped like the real one: no-op without the global or without buckets, threshold inclusive and configurable, real byte lengths counted (not bucket sizes), in-place emptying, idempotence.
- `forked/__tests__/stackglPoolRelease.test.ts`, 6 cases on the real bundle: the global is the one the bundle writes; an update leaves exactly four idle buffers at power-of-two sizes; a rebuild of the same point count reuses them; a different point count leaves both sets resident (the retention this fixes); a release empties the large buckets and leaves small buffers alone; a point cloud rebuilds byte-identically from an emptied pool and still draws.
- `benchmarks/fps/glSceneStats.test.mjs`: `poolRetainedMB` is read from the pool and shown in the summary line.

**In-app check** (user). The benchmark loads one collection per run, so the effect only shows across a switch, in DevTools:

1. Load WordNet in 3D, then switch to a small collection (or to 2D, or open `/sae`).
2. In the console: `__TYPEDARRAY_POOL.DATA.map((b, i) => b.length ? `${2 ** i / 1048576} MiB x${b.length}` : null).filter(Boolean)`. Before this change: 256 MiB x2, 128 MiB x1, 64 MiB x1 left behind. After: only the small collection's own buffers.
3. Memory panel, collect garbage, then read the heap: expect about 700 MB less than before the change after the same switch.

The benchmark's `glStatsPostLoad.poolRetainedMB` should keep reporting about 704 for WordNet `3d` (idle in the pool between rebuilds, and reused by them) and should be small on a `2d` run started fresh.
