# Backend configuration

- `cancellable_tasks.json`: `CancellableTaskSettings` for `utils/cancellable_task.py`
  (row count from which heavy steps run in a killable child process, cancel
  poll interval, array size spilled to a temp `.npy` instead of the pipe,
  terminate grace period). See `../../documentation/JOB_CANCELLATION.md`.
- `duckdb.json`: `DuckDBSettings` for `utils/duckdb_settings.py`, applied to every
  `DuckDBClient` connection: `memory_limit` (DuckDB size such as `"4GB"`, a share
  of this machine's RAM such as `"25%"`, or `null` for DuckDB's default of 80%)
  and `threads` (`null` = one per core). Larger queries spill to `<db>.tmp`.
  See `../../documentation/BACKEND_MEMORY_REPORT.md`.
- `embedding_models.json`: catalog of known embedding models per provider
  (`utils/embedding_model_catalog.py`): form suggestions, prompt names, encoder
  towers, and `max_length`, the input-length limit in tokens for models whose
  own files don't state it reliably (it wins over them in
  `utils/model_max_length.py`; EmbeddingGemma 2's tokenizer declares no limit).
  `task_prefixes` is the task table (`default_task`, then per task `id`,
  `label`, `query_prefix`, optional `document_prefix` (absent = symmetric) and
  the `prompt_names` that alias it) of the models marked
  `"prompt_format": "task_prefix"` (EmbeddingGemma, Gemini Embedding 2). See
  `../../documentation/EMBEDDING_TASK_PREFIXES.md`.
- `http_compression.json`: JSON response gzip settings (level, minimum size,
  worker-thread threshold, media types), loaded by `CompressionSettings` for
  `JsonGzipMiddleware`. See `../../documentation/MULTICLASS_PROBES.md`.
- `response_cache.json`: read-only GraphQL response cache (enabled flag,
  allow-listed root fields, memory budget, stored JSON body-size bounds,
  request-size limit, gzip level, streaming chunk size), loaded by
  `ResponseCacheSettings` for `GraphQLResponseCacheMiddleware`. Only active
  when `ORRERY_READ_ONLY=1`.
  See `../../documentation/GRAPHQL_RESPONSE_CACHE.md`.
- `topic_reduction.json`: `TopicReductionSettings` for
  `services/topic_reduction_types.py`: the default distance threshold of an
  "auto" average-linkage reduction (centred cosine distance), the largest
  number of base topics a merge tree is built for, the merge-tree cache size,
  and the largest collection whose base-topic centroids are computed from the
  vectors when none are stored. See `../../documentation/TOPIC_MERGE_TREES.md`.
- `seed_snapshots/`: seed export/publication manifests and immutable download locks;
  its README describes snapshot generation and deployment.
