# Backend configuration

- `cancellable_tasks.json`: `CancellableTaskSettings` for `utils/cancellable_task.py`
  (row count from which heavy steps run in a killable child process, cancel
  poll interval, array size spilled to a temp `.npy` instead of the pipe,
  terminate grace period). See `../../documentation/JOB_CANCELLATION.md`.
- `http_compression.json`: JSON response gzip settings (level, minimum size,
  worker-thread threshold, media types), loaded by `CompressionSettings` for
  `JsonGzipMiddleware`. See `../../documentation/MULTICLASS_PROBES.md`.
- `response_cache.json`: read-only GraphQL response cache (enabled flag,
  allow-listed root fields, memory budget, stored JSON body-size bounds,
  request-size limit, gzip level, streaming chunk size), loaded by
  `ResponseCacheSettings` for `GraphQLResponseCacheMiddleware`. Only active
  when `ORRERY_READ_ONLY=1`.
  See `../../documentation/GRAPHQL_RESPONSE_CACHE.md`.
- `seed_snapshots/`: seed export/publication manifests and immutable download locks;
  its README describes snapshot generation and deployment.
