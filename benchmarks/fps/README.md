# FPS + RAM Benchmark

Measures interactive rendering performance (FPS) and memory footprint of the
visualization frontend across collection sizes, from 1k real points up to 1M
synthetic points. Standalone Node harness — not part of the app, has its own
`package.json`.

## Structure

| File | Purpose |
|------|---------|
| `bench.mjs` | The whole harness: Chrome automation, FPS capture, memory telemetry, synthetic data injection |
| `results/` (gitignored) | `results_<pass>.json` per pass + verification screenshots `shot_<pass>_<collection>.png` |

## Prerequisites

- Backend running on `:8000`, frontend on `:3000`
- Google Chrome installed (the harness drives the real browser **headed** so
  WebGL runs on the actual GPU via Metal — headless falls back to software
  rendering and produces meaningless numbers)
- `npm install` in this folder (only dependency: `playwright-core`)

## Usage

```bash
node bench.mjs 3d                        # 3D orbit pass, full ladder
node bench.mjs 3d-nebula                 # 3D with nebula haze on, colored by topic_label
node bench.mjs 2d                        # 2D pan pass (scattergl path)
node bench.mjs 3d emotion synthetic_1m   # specific targets only
npm run bench:all                        # all three passes back to back
```

Rules during a run (~10–15 min per pass over the full ladder):

- **Do not minimize or cover the Chrome window** — Chrome throttles
  `requestAnimationFrame` for occluded windows, corrupting the FPS numbers.
- Leave mouse/keyboard alone; close heavy apps first (results are confounded
  by memory pressure on small-RAM machines).

Results are written incrementally after each collection, so a crashed run
keeps everything completed so far. A failed collection is logged and skipped.

## Ladder

Eight real collections (1k → 212k points, all with stored UMAP projections
and extracted topics) plus three synthetic targets (250k / 500k / 1M).
Any `synthetic_<n>[k|m]` name works as an ad-hoc target.

**Synthetic targets never touch the database.** The harness intercepts the
`GetCollectionData` GraphQL response and substitutes generated Gaussian
clusters (50 clusters, deterministic PRNG, `topic_id`/`topic_label` metadata
so topic coloring and nebula haze work). `GetCollections` is passed through
and the synthetic entry appended (the page only loads URL collections present
in the list); topics/probes/activations queries for the synthetic name return
benign empties. Everything downstream of the network — Apollo parsing, point
transforms, Plotly trace building, WebGL — is the real platform code path.

## Methodology

- Per run: fresh browser context, `viz-preferences` localStorage seeded to a
  fixed baseline (UMAP, target mode, nebula/labels/axes off — nebula pass
  flips nebula on and adds `?colorBy=topic_label`), viewport 1600×950.
- Waits until the plot holds ≥95% of expected points, then a 4s settle.
- FPS: an in-page `requestAnimationFrame` recorder collects frame deltas
  during a continuous 8s mouse drag (orbit in 3D, pan in 2D) driven via CDP.
  Reported: mean / median / 1%-low FPS, p95 + max frame time. The display
  refresh rate (ceiling) is calibrated on a blank page first.
- Memory: JS heap (`performance.memory`), renderer + GPU process RSS and
  whole-Chrome-tree RSS (`ps`), system free/available RAM (`vm_stat`;
  macOS "available" = free + speculative + inactive + purgeable). Sampled
  post-load, mid-drag, and post-drag.

## Caveats

- FPS is capped by the display refresh rate (120 Hz on ProMotion screens).
- Numbers are machine- and load-dependent; report them with the hardware and
  the system-RAM baseline the harness prints at start.
- Synthetic documents are short strings, so heap use per point is lower than
  for real long-document collections at the same count — synthetic runs
  measure rendering scale, not worst-case payload memory.
