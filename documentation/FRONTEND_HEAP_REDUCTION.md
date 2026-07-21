# Frontend Heap Reduction

## Implemented: active projection loading and typed 3D trace columns

The Explore page now loads and materializes only the projection dimension used
by the active view. This is the first implementation slice of the frontend
heap-reduction roadmap; it does not yet replace the JSON/nested-array core data
model.

### Request lifecycle

`ProjectionLoadPolicy` maps the active method and view mode to exactly one
projection type:

- UMAP + 3D requests `umap_3d`.
- UMAP + 2D requests `umap_2d`.
- PCA + 3D requests `pca_3d`.
- PCA + 2D requests `pca_2d`.
- Manual mode keeps the existing PCA fallback and follows the active dimension.

The first request for a collection sends `includeCore: true`, so the response
contains IDs, documents, item metadata, available fields, collection metadata,
and the active projection. Later method or dimension changes send
`includeCore: false`; those responses contain only the requested projection, a
compact ordered-item signature, and empty core arrays. The GraphQL argument
defaults to `true` for backward compatibility with existing clients.

Core and projection responses each carry a SHA-256 digest of their ordered item
IDs. `ProjectionMembership` compares these before changing the projection
cache, so a partial or differently ordered projection fails explicitly instead
of silently attaching coordinates to the wrong items. Missing signatures are
also rejected; deploy the backend schema before the frontend client.

`LatestRequestGate` invalidates older in-flight loads whenever the collection,
method, or dimension changes. This prevents a slow response for a previous
view from resetting the projection cache after the newer active view loads.

The backend implements projection-only reads through
`DuckDBClient.get_projection_coordinates()`. Shared item columns are not read,
topic assignments are not merged, and core data is not serialized for these
follow-up requests. Projection order follows explicit `row_index` values.
Imports without source indices now receive monotonic indices at insertion.
Before appending to a legacy table, null indices are backfilled in its existing
physical `rowid` order so new rows cannot move ahead of existing points.

### Frontend materialization

`VisualizationPointBuilder` creates wrappers only for the active dimension:
3D mode returns an empty `points2d` array, and 2D mode returns an empty
`points3d` array. This removes the previous requirement that a 2D projection be
present before a 3D view could render.

`PointTraceColumns3D` converts the base/data-marker scatter3d traces' x/y/z
coordinates to `Float32Array` and their global point indices to `Uint32Array`.
It is used by the default, categorical, filtered, and numeric 3D data paths;
small overlay, line, and text traces remain ordinary arrays. Float32 matches
the native DuckDB projection storage type.

### Verification

Tests cover:

- projection policy for PCA, UMAP, manual mode, 2D, and 3D;
- stale-request token invalidation and latest-response selection;
- projection/core membership compatibility and partial-projection rejection;
- active-only 2D/3D wrapper creation;
- typed coordinate/index ordering, indexed subsets, and empty traces;
- the GraphQL `includeCore` contract and backward-compatible default;
- projection-only DuckDB reads, generated insertion indices, and legacy-null
  backfill ordering.

Run the focused checks with:

```bash
cd embedding_visualization
npm run test:run -- \
  lib/utils/__tests__/projectionLoadPolicy.test.ts \
  lib/utils/__tests__/projectionMembership.test.ts \
  lib/utils/__tests__/visualizationPointBuilder.test.ts \
  lib/utils/__tests__/pointTraceColumns3D.test.ts \
  lib/utils/__tests__/latestRequestGate.test.ts

cd ..
uv run pytest \
  interpretability_backend/unit_tests/test_projection_loading.py \
  interpretability_backend/unit_tests/test_duckdb_client.py
```

### Remaining work

- The initial core payload still contains all IDs, documents, and per-item
  metadata objects.
- Projection JSON is still parsed as nested `number[][]` before trace columns
  are materialized.
- Previously visited projections remain cached in `projectionsRef`, although
  only the active dimension has point wrappers.
- Forced-GC and load-peak benchmarks must be rerun before making a quantified
  heap-reduction claim.
- Document deferral, the columnar core, and binary/Arrow transport remain
  separate later phases.

The headed-Chrome harness in `benchmarks/fps/` is compatible with the new
membership-signature contract. New runs use labeled result and screenshot names
so the 2026-07-10 baselines remain intact, and record collection GraphQL bytes
alongside forced-GC heap and process RSS. See its README for the focused 3D and
2D comparison commands.
