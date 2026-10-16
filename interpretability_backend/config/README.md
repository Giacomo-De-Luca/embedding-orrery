# Backend configuration

- `http_compression.json`: JSON response gzip settings (level, minimum size,
  worker-thread threshold, media types), loaded by `CompressionSettings` for
  `JsonGzipMiddleware`. See `../../documentation/MULTICLASS_PROBES.md`.
- `seed_snapshots/`: seed export/publication manifests and immutable download locks;
  its README describes snapshot generation and deployment.
