# CLAUDE.md - Backend

Instructions for working with the `interpretability_backend` Python backend.

## Running

```bash
# Start server (from project root)
./start_backend.sh
# Or: uv run uvicorn interpretability_backend.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
uv run pytest interpretability_backend/unit_tests/
uv run pytest interpretability_backend/tests/
```

## Code Style

Ruff is the lint + format tool of record. Config lives in the root `pyproject.toml` under `[tool.ruff*]`. Run before committing:

```bash
uv run ruff check interpretability_backend/ --fix    # lint + safe autofixes
uv run ruff format interpretability_backend/         # formatter
```

Active rule set: `E`, `F`, `W`, `I` (isort), `B` (bugbear), `UP` (pyupgrade py312), `ASYNC`. Notebooks (`*.ipynb`) and the vendored `gemma_pytorch/` submodule are excluded. Known residual violations (B905, B904, F841, etc.) are documented in `documentation/RUFF_SETUP.md` as a follow-up cleanup queue — not regressions.

## Architecture Overview

```
GraphQL (Strawberry) + REST (FastAPI)
    ↓
API Layer (queries.py, mutations.py, subscriptions.py, upload.py)
    ↓
Services (topic_extraction_service, progress_emitter, job_state)
    ↓
Clients:
  ├── duckdb_client (orchestrator: docs, metadata, projections, topics)
  ├── chromadb_client (dense vectors only: IDs + embeddings)
  ├── huggingface_client, local_data_client (data source loading)
    ↓
Embedding Functions (create_embedding_function → specific providers)
    ↓
Storage:
  ├── DuckDB (resources/main.duckdb) — documents, metadata, projections, topics
  └── ChromaDB (resources/vector_db/) — dense embedding vectors only
```

### Key Design Patterns

**Dual-database architecture**: DuckDB is the central orchestrator. All reads (collection listing, projection data, text search, item metadata) go through `DuckDBClient`. ChromaDB is used only for dense vector storage and semantic similarity search. Embedding pipelines write documents+metadata to DuckDB and vectors-only to ChromaDB.

**One dataset, many embeddings**: The `vector_collections` table links one dataset to multiple vector stores. Each vector_collection has its own projections and topic extractions. Schema supports future Qdrant sparse vector collections.

**DuckDB bulk insertion**: Uses pandas DataFrames for fast columnar inserts (153k items in ~28s vs 40min with row-at-a-time). FTS extension available for BM25 word-level search.

**Lazy embedding function loading**: `ChromaDBClient.get_collection(load_embedding_function=False)` is the default. Only text-query semantic search sets `True`. This avoids loading 100MB+ models for read-only operations. EF config is currently reconstructed from ChromaDB collection metadata (future: read from DuckDB `vector_collections` table).

**Provider factory**: `create_embedding_function(config, device)` maps `EmbeddingProvider` enum to the correct embedding function. Adding a new provider requires:
1. Add to `EmbeddingProvider` enum in `config.py`
2. Add to `EmbeddingProviderEnum` in `utils/provider_list.py`
3. Create implementation in `embedding_functions/specific_functions/`
4. Add `elif` branch in `create_embedding_function.py`
5. The GraphQL enum mapping in `mutations.py` is auto-generated from the enum.

**Dimension caching**: `utils/known_dimensions.json` stores model → dimension mappings to avoid running test embeddings. The fallback chain is: known_dimension parameter → JSON file → test embedding (saves result).

**Progress emission**: `services/progress_emitter.py` provides an in-memory event bus. Embedding functions call `emit_progress_sync()` (thread-safe). Subscriptions in `API/subscriptions.py` register queues and yield events via WebSocket.

**Job state persistence**: `services/job_state.py` writes to `resources/job_state.json`. On startup, marks "running" jobs as "interrupted". Resume works by loading existing IDs from DuckDB and skipping them.

## Module Reference

### API Layer (`backend/API/`)
- **`types.py`** - All GraphQL type definitions. When adding new fields to queries/mutations, define types here.
- **`queries.py`** - Read operations. `Query` class with `@strawberry.field` methods. Includes `text_search` query (full-text search with field selection, mode, case-sensitivity).
- **`mutations.py`** - Write operations. `Mutation` class with `@strawberry.mutation` methods. Embedding mutations use `asyncio.to_thread()` to run in background threads while the event loop handles WebSocket progress.
- **`subscriptions.py`** - `Subscription.embedding_progress(job_id)` async generator. Registers queue with progress_emitter, yields JobProgress events.
- **`chromadb_instance.py`** - Lazy singleton `get_chromadb_client()`.
- **`duckdb_instance.py`** - Lazy singleton `get_duckdb_client()`.
- **`interpret_instance.py`** - Lazy singleton `get_interpret_service()` for SAE inference.
- **`upload.py`** - REST `POST /upload` endpoint saving files to `resources/uploads/`.

### Clients (`backend/clients/`)
- **`duckdb_client.py`** - Central orchestrator. Key methods:
  - `create_dataset()`, `list_datasets()`, `get_dataset()`, `delete_dataset()` - Dataset CRUD
  - `insert_items_batch()` - Bulk item insert via pandas DataFrame
  - `get_filtered_items(dataset, filters, limit, offset)` - JSON metadata filtering ($eq/$ne/$gt/$gte/$lt/$lte/$in/$nin)
  - `get_items_by_ids()` - Enrich search results with documents + metadata
  - `register_vector_collection()`, `get_vector_collections()` - Vector collection registry
  - `insert_projections_batch()`, `get_projection_data(collection, type)` - Per-type projection storage/retrieval
  - `text_search(dataset, query, fields, mode)` - SQL-based text search (ILIKE for substring, json_extract for metadata)
  - `text_search_bm25(dataset, query)` - FTS extension BM25 search (known issue: needs debugging)
  - `create_topic_extraction()`, `insert_topic_info_batch()`, `insert_topic_assignments_batch()` - Topic storage
  - `get_active_topics()`, `update_topic_label()`, `update_subtopic_label()` - Topic reads/updates
  - `insert_sae_features_batch()`, `get_sae_feature()`, `search_sae_features()`, `list_sae_models()` - SAE feature storage
  - `insert_sae_activations_batch()`, `get_sae_activations()` - SAE activation storage
  - `delete_sae_data()` - Remove SAE data for a model/sae pair
- **`chromadb_client.py`** - Vector-only wrapper (~170 lines, stripped from ~610). Key methods:
  - `get_collection(name, load_embedding_function, for_query, query_prompt)` - Lazy EF loading
  - `semantic_search(...)` - Vector similarity search, returns IDs + distances (no documents/metadata)
- **`huggingface_client.py`** - Dataset info/preview via `datasets` library, portion loading (FIRST_N, RANDOM_SAMPLE, ROW_RANGE, ALL)
- **`local_data_client.py`** - File loading via pandas/pyarrow. Optimized: parquet reads metadata without loading data, CSV reads only headers for info.

### Embedding Functions (`backend/embedding_functions/`)
- **`config.py`** - `DB_PATH`, `EmbeddingProvider`, `EmbeddingModelConfig`, `EmbeddingConfig`, `LocalFileEmbeddingConfig`, `EmbeddingResult`. The `BaseConfig` dataclass uses `kw_only=True`.
- **`create_embedding_function.py`** - Factory pattern. Returns `(EmbeddingFunction, dimension)`. Loads from `.env` via python-dotenv. HuggingFace login happens here if `HUGGINGFACE_API_KEY` is set.
- **`embed_huggingface.py`** - Full HF embedding pipeline: load portion → sort by length → explicit embed → DuckDB (docs+metadata) + ChromaDB (vectors only). Resume via DuckDB ID check.
- **`embed_local_file.py`** - Dispatches to `embed_text_from_local()`, `embed_images()`, or `embed_vectors()` based on `DataType`. Same dual-write pattern.
- **`embed_images.py`** - ViT pipeline (`transformers.pipeline("image-feature-extraction")`). Handles bytes, dicts with "bytes" key, or file paths.
- **`embed_vectors.py`** - Direct vector ingestion (no model needed). Auto-detects vector column.

### SAE Ingestion (`embedding_functions/ingest_sae.py`)
- **`ingest_sae_features(parquet_path, model_id, sae_id, store_vectors, progress_callback)`** — Load SAE feature parquet (index, density, label, top/bottom logits, explanation vectors) into DuckDB `sae_features` table. Optionally stores 2560-d explanation-embedding vectors in ChromaDB for semantic search.
- **`ingest_sae_activations(jsonl_path, model_id, sae_id, batch_size, progress_callback)`** — Stream activation JSONL (~20 samples per feature, 512 tokens each) into DuckDB `sae_activations` table in configurable batches. Auto-detects `model_id`/`sae_id` from JSONL fields.
- Composite join key: `(model_id, sae_id, feature_index)` across both tables.
- GraphQL mutations: `ingestSaeFeatures`, `ingestSaeActivations`.
- GraphQL queries: `saeModels`, `saeFeature`, `saeActivations`, `saeFeatureSearch`.
- Full architecture documentation: `documentation/SAE_ARCHITECTURE.md` (schema, data flow, frontend components, cross-linking, planned enhancements).

### Specific Embedding Functions (`embedding_functions/specific_functions/`)
- **`embed_sentence_transformer.py`** - Fork of ChromaDB's implementation. Adds prompt support (known prompt names like "Retrieval-query" → `prompt_name`, custom strings → `prompt`). Class-level model cache shared across instances.
- **`embed_gemini.py`** - Google Gemini API with `task_type` support and tenacity retry.
- **`embed_qwen.py`** - QWEN3-Embedding with `last_token_pool`, query instruction prepending (`is_query` flag), flash attention on CUDA, fp16.
- **`embed_bge.py`** - BGE-M3 via FlagEmbedding library, dense vector extraction.

### Services (`backend/services/`)
- **`topic_extraction_service.py`** - Three main functions:
  - `extract_topics(config)` - Full pipeline: load projections → HDBSCAN → c-TF-IDF → optional reduction → optional LLM labels → update ChromaDB metadata
  - `reduce_existing_topics(...)` - Standalone: load existing topics → reconstruct c-TF-IDF → reduce → update metadata
  - `generate_llm_labels_for_collection(...)` - Standalone: generate LLM labels for existing topics/subtopics with incremental saves, resume support, and `preserve_ctfidf_labels` option (saves original keyword labels as `ctfidf_label` in topic_summary entries and `ctfidf_subtopic_map` in collection metadata)
- **`progress_emitter.py`** - `emit_progress(job_id, ...)` broadcasts to subscriber queues. Thread-safe (uses `queue.put_nowait`).
- **`job_state.py`** - `JobStateService` singleton with file-based persistence. Methods: `start_job`, `update_progress`, `update_total_expected`, `complete_job`, `fail_job`, `list_jobs`.
- **`interpret_service.py`** - `InterpretService` wrapping the `interpret/` toolkit for live SAE inference. Manages Gemma3-4b-it lifecycle (load/unload, `asyncio.Lock` for GPU serialisation). Four methods: `run_prompt_activations()` (per-token top-k features via PromptExplorer), `generate_steered()` (baseline + steered text via HookManager + SteeringOp), `run_prompt_highlight()` (max-pooled feature activations for scatter plot), `generate_stream()` (streaming chat generation with optional steering and cancel support — emits tokens via `token_emitter.py`). See `documentation/INTERPRET_API.md`.
- **`token_emitter.py`** - Async queue event bus for token streaming (mirrors `progress_emitter.py`). `TokenEvent` dataclass, `register_token_subscriber`/`unregister_token_subscriber` (async), `emit_token` (sync, thread-safe). Queue maxsize 500. Used by `generate_stream` subscription for real-time token delivery over WebSocket.

### Topic Extraction (`backend/topic_extraction/`)
- **`cluster_and_label.py`** - `GenerateTopics` class: HDBSCAN clustering + `ClassTfidfTransformer` (BERTopic-inspired c-TF-IDF). Stores `ctfidf_matrix` and `words` properties for reduction.
- **`topic_reducer.py`** - `TopicReducer` class: `reduce_to_n_topics()` (AgglomerativeClustering on distance matrix), `auto_reduce_topics()` (HDBSCAN with min_cluster_size=2). Supports c-TF-IDF or semantic embeddings for similarity. Returns `topic_hierarchy` mapping new_topic_id → [old_topic_ids].
- **`llm_labeling.py`** - Provider-agnostic LLM labeling. Factory creates `_GeminiLabeler` or `_OpenAILabeler`. Uses tenacity for retries. Prompt includes sample documents + keywords. `generate_llm_labels()` accepts optional `progress_callback(done, total)` for per-topic progress reporting.
- **`extract_topics.py`** and **`_representation_utils.py`** - BERTopic reference implementations (not used directly by the service).

### Utils (`backend/utils/`)
- **`compute_projections.py`** - `compute_projections_for_collection(name, projection_type, job_id)`. Loads embeddings from ChromaDB in 5k batches, computes PCA/UMAP, stores projections as native FLOAT[] arrays in DuckDB. When `job_id` is provided, emits per-projection progress via `emit_progress`.
- **`duckdb_sync.py`** - Helper functions for dual-write during embedding: `sync_dataset_and_collection()`, `sync_items()`. Called from embedding pipelines.
- **`text_processing.py`** - `format_text_for_embedding(row, columns, template)` supports template strings with `{column}` placeholders. `extract_metadata(row, columns)` prepares metadata for storage.
- **`color_preprocessing.py`** - Auto-detects hex color columns (`colour_code`, `color_code`, etc.) and maps each color to a float 0-1 position on a pre-built colorscale strip (Hilbert RGB, hue-sat, XKCD, or rainbow). Adds `mapped_colour` and `mapped_colour_scale` to item metadata. Called from `embed_local_file.py` and `embed_huggingface.py` after `extract_metadata()`. Strips generated by `scripts/generate_color_strips.py`.
- **`batch_utils.py`** - `sort_items_by_length()` sorts by text length descending for efficient transformer batching (reduces padding waste).
- **`id_utils.py`** - `IDDeduplicator` always appends `_N` suffix (1-based) for uniqueness.
- **`provider_list.py`** - Single source of truth for `EmbeddingProviderEnum` (Strawberry enum). Used by both GraphQL types and the internal `EmbeddingProvider` mapping.
- **`logger.py`** - Configures `star_map` logger with file handler (DEBUG) and console handler (ERROR).

## Data Storage

### DuckDB (`resources/main.duckdb`) — Primary data store
- **`datasets`** — one row per dataset (name, description, source info)
- **`items`** — documents + JSON metadata per item, keyed by (dataset_id, id)
- **`vector_collections`** — links datasets to ChromaDB collections (embedding model info, one-to-many)
- **`projections`** — native FLOAT[] coordinate arrays, per (vector_collection, item, projection_type)
- **`projection_metadata`** — PCA variance ratios, timestamps per projection type
- **`topic_extractions`** — extraction config snapshots, reduction metadata, `is_active` flag for history
- **`topic_info`** — per-topic keywords, labels, counts
- **`topic_assignments`** — per-item topic_id/label + subtopic_id/label
- **`sae_features`** — SAE feature metadata: PK (model_id, sae_id, feature_index), density, label, top/bottom logits as JSON
- **`sae_activations`** — SAE activation examples: indexed by (model_id, sae_id, feature_index), stores tokens[512] and activation values[512] as JSON, plus max_value for ordering
- Full schema, API, data flows: `documentation/DATABASE_ARCHITECTURE.md`
- Migration plan: `documentation/DUCKDB_MIGRATION_PLAN.md`
- Migration script: `scripts/migrate_chromadb_to_duckdb.py`

### ChromaDB (`resources/vector_db/`) — Vectors only
- Persistent SQLite-backed HNSW index
- Stores only IDs + dense embedding vectors (no documents, no metadata)
- Used for semantic similarity search and raw embedding reads (projection computation, topic reduction)
- Legacy collections may still contain documents/metadata from before migration

### Job State (`resources/job_state.json`)
- JSON dict of `{collection_name: JobState}` entries
- On server startup, all "running" jobs are marked as "interrupted"
- Completed jobs are removed (not accumulated)

## Common Modifications

### Adding a new embedding provider
1. `embedding_functions/config.py` → add to `EmbeddingProvider` enum
2. `utils/provider_list.py` → add to `EmbeddingProviderEnum` (Strawberry enum)
3. `embedding_functions/specific_functions/` → create `embed_<provider>.py` implementing `EmbeddingFunction[Documents]`
4. `embedding_functions/create_embedding_function.py` → add `elif` for new provider
5. No changes needed in `mutations.py` - the enum mapping is auto-generated

### Adding a new GraphQL query/mutation
1. Define types in `API/types.py`
2. Add resolver method in `API/queries.py` or `API/mutations.py`
3. Types are auto-exported via `API/__init__.py` and `schema.py`

### Adding metadata fields to collections
1. Add field to `CollectionMetadata` in `API/types.py`
2. Add to metadata dict construction in `queries.py:collection()` resolver
3. Store in DuckDB `datasets` or `vector_collections` table (update schema in `duckdb_client.py:_ensure_schema()`)

## Key Gotchas

- **DuckDB is single-writer** — only one process can write at a time (multiple readers OK). Matches the existing single-user deployment model.
- **DuckDB datetime values** — DuckDB returns `datetime` objects that aren't JSON-serializable. Use `_sanitize_for_json()` from `duckdb_client.py` at API boundaries where dicts go into Strawberry `JSON` fields.
- **ChromaDB still stores embedding function config** — `get_collection(load_embedding_function=True)` reconstructs the EF from ChromaDB collection metadata. This is a legacy coupling that should eventually read from DuckDB's `vector_collections` table.
- **Filtered semantic search broken for new collections** — `semantic_search()` passes `where` clauses to ChromaDB, but new collections have no metadata in ChromaDB. Needs pre-filtering via DuckDB IDs.
- **BM25 FTS returns 0 results** — the FTS index + `dataset_id` filtering interaction needs debugging.
- **SentenceTransformer models use a class-level cache** — first load is slow, subsequent calls reuse the cached model
- **The `for_query` flag on `get_collection()`** affects QWEN (adds instruction prefix) and Gemini (maps RETRIEVAL_DOCUMENT → RETRIEVAL_QUERY)
