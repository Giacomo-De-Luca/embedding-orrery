# Backend Scripts

Utility scripts in this folder support one-off maintenance and Docker demo
setup. Run Python entry points from the repository root with `uv run python`.

- `backfill_topic_centroids.py` computes the embedding-space topic centroids
  that semantic topic search needs, for topic extractions created before they
  were stored (config: `backfill_topic_centroids_config.toml`; backend stopped).
- `compare_embedding_dtypes.py` embeds the same texts with one model in float32
  and bfloat16 (one load at a time) and prints their cosine agreement, top-k
  neighbour agreement and the Metal memory each load held: the check behind
  keeping bf16 checkpoints in bf16 on Apple GPUs (config:
  `compare_embedding_dtypes_config.toml`; loads models, run with the backend
  idle). See `documentation/BACKEND_MEMORY_REPORT.md`.
- `prune_chroma_orphans.py` removes the HNSW index directories chromadb leaves
  on disk for deleted collections (config: `prune_chroma_orphans_config.toml`,
  dry run by default; backend stopped). See
  `documentation/CHROMA_ORPHAN_INDEXES.md`.
- `build_seed_snapshot.py` builds a staged, checksummed DuckDB/Chroma snapshot
  from a JSON manifest under `../config/seed_snapshots/`.
- `publish_seed_snapshot.py` verifies an existing snapshot, uploads it to its
  configured private Hugging Face Dataset path, and writes an immutable lock.
- `download_seed_snapshot.py` downloads a locked Dataset revision and verifies
  it before atomic installation; the demo Dockerfile uses this entry point.
- `docker_warmup_sae.py` warms Docker volumes for the optional SAE demo profile:
  it waits for the backend, downloads the configured HuggingFace checkpoint into
  `HF_HOME`, calls `prepareSaeData`, and exits without loading the model.
- `extract_direction_vectors.py` normalizes steering direction tensors into the
  small runtime `.pt` presets stored under `resources/directions/`.
- `generate_color_strips.py` generates frontend color-map JSON strips from
  backend color data.
- `migrate_chromadb_to_duckdb.py` migrates legacy Chroma-backed collection data
  into the DuckDB-centered schema.
- `poetry_refusal_cosines.py` compares the shipped poetry/refusal steering
  direction vectors.
