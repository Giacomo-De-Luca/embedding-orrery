# Explore first load: why the first collection was slow

Investigation of 2026-10-22 and the change it led to (2026-10-24). The first
collection a visitor opens took much longer than every later switch. The data
path was not the reason.

## What was measured

Local dev, `curl` against the running backend (`localhost:8000`), the first
manifest entry `combined` (14k items, Gemini Embedding 2, 73 topics):

| Request | Time | Payload |
|---|---|---|
| `collections` manifest, first request on a freshly restarted uvicorn worker (includes the lazy `DuckDBClient()` creation and `_ensure_schema`) | 0.41 s | 193 KB |
| `collection(combined, umap_3d)`, first time on that worker | 0.37 s | 2.0 MB gzipped |
| same, warm | 0.21 s | 2.0 MB gzipped |
| `collection(acl_abstracts_emnlp_findings, umap_3d)` | 0.88 s | 13.4 MB gzipped |
| `collectionTopics`, `topicMergeTree`, `collectionProbes` | < 0.05 s each | tiny |

The page cache was warm from earlier requests, so a true cold read of the 25 GB
DuckDB file is slower than these numbers, but not by tens of seconds on an SSD.
The `projections` and `topic_assignments` tables have only their primary keys
(`(collection_name, item_id, projection_type)` and `(extraction_id, item_id)`),
which the per-collection lookups use as a prefix together with DuckDB's zone maps.

## The serial chain in the frontend

Before this change the first load ran strictly in sequence:

1. page shell and base JS chunks;
2. `collections` manifest query, then `resolveInitialCollection`;
3. `collection` data query (topics, probes and the merge tree in parallel);
4. only once `data` exists does `page.tsx` render `DashboardPanel`, whose scatter
   plot ran `import('plotly.js')` in a mount effect, so the forked Plotly bundle
   (traces, stackgl modules, components; about 8 MB unminified in dev, spread
   over hundreds of Turbopack chunk files) was downloaded, parsed and
   GL-initialised only after the data had arrived.

Every later collection switch skips step 4, which is why only the first
collection felt slow, in dev and production alike.

## The change: preload Plotly on page mount

`lib/utils/plotlyLoader.ts` holds `PlotlyLoader`, the one import of the forked
Plotly bundle:

- `plotlyLoader.preload()` is called from an Explore mount effect in
  `app/page.tsx`, so the bundle downloads while the manifest and data queries run;
- `ScatterPlot2D` and `ScatterPlot3D` resolve `plotlyLoader.load()` instead of
  their own `import('plotly.js')`, so they share the one promise (react-plotly.js
  requires the same module, so its dynamic import is a cache hit);
- a failed import is forgotten and retried on the next `load()`, so a transient
  chunk-load failure cannot leave the plots stuck on a rejected promise;
  `preload()` swallows its own failure for the same reason.

The import still targets `plotly.js` (the fork's source build: `package.json`
declares `"plotly.js": "file:./forked/plotly.js"`, symlinked into
`node_modules`), never `plotly.js-dist-min`, so the fork's patches apply.

Tests: `lib/utils/__tests__/plotlyLoader.test.ts` (one import shared by
concurrent callers, preload then load, silent failed preload with retry,
rejected load with retry).

Still sequential after this change: the GL context creation and first
`newPlot` happen when `DashboardPanel` mounts with data. Mounting the panel
with empty points during the loading state would overlap that too; it was left
out of this change.

## Environment factors seen during the investigation

These multiplied the chain above on the development Mac and are worth checking
before measuring anything:

- 8 GB of RAM with 10.7 GB of swap in use; Chrome's renderer processes were
  resident at 16 to 52 MB, so the tab itself had to be paged back in on reload.
- `uvicorn --reload` restarts the backend worker on every `.py` save from any
  parallel session (2.2 s of imports before it serves again). During that gap
  the manifest query fails and `useCollections` retries on a 4 s poll, so a
  reload that coincides with a restart adds 4 to 8 s before the collection query
  can start. Narrowing the watcher (`--reload-dir interpretability_backend/backend`)
  would stop test edits from restarting it.
- The DuckDB file is 25 GB with a 2 GB memory limit (25% of RAM), so its buffer
  pool is cold after each restart.

## `.next` housekeeping

`embedding_visualization/.next` is build output and cache, all regenerable:

- `.next/dev/cache` (2.6 GB): Turbopack's persistent cache. Keep it; it is what
  makes `next dev` compile `/` in seconds instead of minutes.
- `.next/dev/static` and `.next/dev/server` (1.1 GB): dev chunks accumulated over
  months. Safe to delete with the dev server stopped; Turbopack re-emits them.
- `.next/graph-build`, `.next/graph-ux-build`, `.next/graph-smoke` (4.2 GB): old
  build outputs from the `sae-graph` branch experiments, deleted on 2026-10-24.
- `.next/cache`, `.next/server`, `.next/standalone`, `.next/static`: outputs of a
  local `next build`; deleting them costs one slower production build.

Never delete anything under `.next` while `next dev` is running: the running
server references chunk files by name and the page breaks with 404s until a
restart.
