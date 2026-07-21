# Backend Agent Notes

Follow the root `AGENTS.md` rules. Historical backend notes also live in
`CLAUDE.md`; prefer the root instructions if they conflict.

## Production Docker

- Runtime resource paths are centralized in `backend/utils/resource_paths.py`.
- Local development defaults to `interpretability_backend/resources/`.
- Docker sets `ORRERY_RESOURCE_DIR=/data`,
  `ORRERY_SEED_DIR=/app/interpretability_backend/resources/seed`,
  `ORRERY_DIRECTIONS_DIR=/app/interpretability_backend/resources/directions`, and
  `HF_HOME=/models/huggingface`.
- The optional SAE profile runs `scripts/docker_warmup_sae.py` with
  `uv run python`; it warms volumes only and must not auto-load Gemma into
  memory.

Detailed Docker behavior is documented in `../documentation/DOCKER.md`.
Script structure is documented in `scripts/README.md`.

## Config-Driven Seed Snapshots

- Snapshot manifests live under `config/seed_snapshots/`; collection names are
  resolved to datasets from the live DuckDB store.
- `backend/utils/seed_snapshot.py` owns validation, staged DuckDB/Chroma export,
  integrity manifests, private Hub publication, immutable locks, and download.
- Keep snapshot CLIs limited to selecting one config. Publication must remain a
  separate explicit command because it mutates a remote Dataset repository.
- A failed export or checksum verification must leave the previous snapshot
  untouched. SAE activation examples remain opt-in because of their size.

Full schema, rollout, and commands are documented in
`../documentation/SEED_SNAPSHOTS.md`.

## Projection-Only Collection Reads

- The GraphQL `collection` query accepts `includeCore` (default `true`).
- Follow-up frontend projection requests set it to `false`; the response then
  contains the requested coordinates, their ordered-item signature, and empty
  core arrays.
- `DuckDBClient.get_projection_coordinates()` must not read documents or item
  metadata. It reads item IDs only to compute the membership digest. Ordering
  uses explicit/generated `row_index`; legacy null indices are backfilled in
  physical `rowid` order before later batches are inserted.

See `../documentation/FRONTEND_HEAP_REDUCTION.md` for the cross-stack request
lifecycle and verification coverage.
