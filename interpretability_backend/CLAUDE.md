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

## Architecture Overview

```
GraphQL (Strawberry) + REST (FastAPI)
    ↓
API Layer (queries.py, mutations.py, subscriptions.py, upload.py)
    ↓
Services (topic_extraction_service, progress_emitter, job_state)
    ↓
Clients (chromadb_client, huggingface_client, local_data_client)
    ↓
Embedding Functions (create_embedding_function → specific providers)
    ↓
ChromaDB (persistent vector storage)
```

### Key Design Patterns

**Lazy embedding function loading**: `ChromaDBClient.get_collection(load_embedding_function=False)` is the default. Only text-query semantic search sets `True`. This avoids loading 100MB+ models for read-only operations.

**Provider factory**: `create_embedding_function(config, device)` maps `EmbeddingProvider` enum to the correct embedding function. Adding a new provider requires:
1. Add to `EmbeddingProvider` enum in `config.py`
2. Add to `EmbeddingProviderEnum` in `utils/provider_list.py`
3. Create implementation in `embedding_functions/specific_functions/`
4. Add `elif` branch in `create_embedding_function.py`
5. The GraphQL enum mapping in `mutations.py` is auto-generated from the enum.

**Dimension caching**: `utils/known_dimensions.json` stores model → dimension mappings to avoid running test embeddings. The fallback chain is: known_dimension parameter → JSON file → test embedding (saves result).

**Progress emission**: `services/progress_emitter.py` provides an in-memory event bus. Embedding functions call `emit_progress_sync()` (thread-safe). Subscriptions in `API/subscriptions.py` register queues and yield events via WebSocket.

**Job state persistence**: `services/job_state.py` writes to `resources/job_state.json`. On startup, marks "running" jobs as "interrupted". Resume works by loading existing IDs from ChromaDB and skipping them.

## Module Reference

### API Layer (`backend/API/`)
- **`types.py`** - All GraphQL type definitions. When adding new fields to queries/mutations, define types here.
- **`queries.py`** - Read operations. `Query` class with `@strawberry.field` methods.
- **`mutations.py`** - Write operations. `Mutation` class with `@strawberry.mutation` methods. Embedding mutations use `asyncio.to_thread()` to run in background threads while the event loop handles WebSocket progress.
- **`subscriptions.py`** - `Subscription.embedding_progress(job_id)` async generator. Registers queue with progress_emitter, yields JobProgress events.
- **`chromadb_instance.py`** - Lazy singleton `get_chromadb_client()`.
- **`upload.py`** - REST `POST /upload` endpoint saving files to `resources/uploads/`.

### Clients (`backend/clients/`)
- **`chromadb_client.py`** - Core wrapper. Key methods:
  - `get_collection(name, load_embedding_function, for_query, query_prompt)` - Lazy EF loading
  - `semantic_search(...)` - Query with distance → similarity conversion
  - `get_projection_data(name)` - Extracts projections from item metadata (JSON-encoded)
  - `get_all_items(...)` - Filtered item retrieval
  - `update_collection_metadata(...)` - Merge metadata updates
- **`huggingface_client.py`** - Dataset info/preview via `datasets` library, portion loading (FIRST_N, RANDOM_SAMPLE, ROW_RANGE, ALL)
- **`local_data_client.py`** - File loading via pandas/pyarrow. Optimized: parquet reads metadata without loading data, CSV reads only headers for info.

### Embedding Functions (`backend/embedding_functions/`)
- **`config.py`** - `DB_PATH`, `EmbeddingProvider`, `EmbeddingModelConfig`, `EmbeddingConfig`, `LocalFileEmbeddingConfig`, `EmbeddingResult`. The `BaseConfig` dataclass uses `kw_only=True`.
- **`create_embedding_function.py`** - Factory pattern. Returns `(EmbeddingFunction, dimension)`. Loads from `.env` via python-dotenv. HuggingFace login happens here if `HUGGINGFACE_API_KEY` is set.
- **`embed_huggingface.py`** - Full HF embedding pipeline: load portion → sort by length → batch embed → ChromaDB. Supports resume via existing ID check.
- **`embed_local_file.py`** - Dispatches to `embed_text_from_local()`, `embed_images()`, or `embed_vectors()` based on `DataType`.
- **`embed_images.py`** - ViT pipeline (`transformers.pipeline("image-feature-extraction")`). Handles bytes, dicts with "bytes" key, or file paths.
- **`embed_vectors.py`** - Direct vector ingestion (no model needed). Auto-detects vector column.

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

### Topic Extraction (`backend/topic_extraction/`)
- **`cluster_and_label.py`** - `GenerateTopics` class: HDBSCAN clustering + `ClassTfidfTransformer` (BERTopic-inspired c-TF-IDF). Stores `ctfidf_matrix` and `words` properties for reduction.
- **`topic_reducer.py`** - `TopicReducer` class: `reduce_to_n_topics()` (AgglomerativeClustering on distance matrix), `auto_reduce_topics()` (HDBSCAN with min_cluster_size=2). Supports c-TF-IDF or semantic embeddings for similarity. Returns `topic_hierarchy` mapping new_topic_id → [old_topic_ids].
- **`llm_labeling.py`** - Provider-agnostic LLM labeling. Factory creates `_GeminiLabeler` or `_OpenAILabeler`. Uses tenacity for retries. Prompt includes sample documents + keywords. `generate_llm_labels()` accepts optional `progress_callback(done, total)` for per-topic progress reporting.
- **`extract_topics.py`** and **`_representation_utils.py`** - BERTopic reference implementations (not used directly by the service).

### Utils (`backend/utils/`)
- **`compute_projections.py`** - `compute_projections_for_collection(name, projection_type, job_id)`. Loads embeddings in 5k batches, computes PCA/UMAP, stores as JSON strings in item metadata, updates collection metadata with variance ratios. When `job_id` is provided, emits per-projection progress (25% increments) via `emit_progress`.
- **`text_processing.py`** - `format_text_for_embedding(row, columns, template)` supports template strings with `{column}` placeholders. `extract_metadata(row, columns)` converts to ChromaDB-compatible types (str/int/float/bool, lists→JSON).
- **`color_preprocessing.py`** - Auto-detects hex color columns (`colour_code`, `color_code`, etc.) and maps each color to a float 0-1 position on a pre-built colorscale strip (Hilbert RGB, hue-sat, XKCD, or rainbow). Adds `mapped_colour` and `mapped_colour_scale` to item metadata. Called from `embed_local_file.py` and `embed_huggingface.py` after `extract_metadata()`. Strips generated by `scripts/generate_color_strips.py`.
- **`batch_utils.py`** - `sort_items_by_length()` sorts by text length descending for efficient transformer batching (reduces padding waste).
- **`id_utils.py`** - `IDDeduplicator` always appends `_N` suffix (1-based) for uniqueness.
- **`provider_list.py`** - Single source of truth for `EmbeddingProviderEnum` (Strawberry enum). Used by both GraphQL types and the internal `EmbeddingProvider` mapping.
- **`logger.py`** - Configures `star_map` logger with file handler (DEBUG) and console handler (ERROR).

## Data Storage

### ChromaDB (`resources/vector_db/`)
- Persistent SQLite-backed HNSW index
- Projections stored as JSON strings in per-item metadata: `pca_2d`, `pca_3d`, `umap_2d`, `umap_3d`
- Topic assignments in per-item metadata: `topic_id`, `topic_label` (and `subtopic_id`, `subtopic_label` when reduction is applied)
- Collection-level metadata: embedding model info, projection variance, topic summary, `topic_hierarchy` (JSON: reduced label → subtopic labels)

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
2. Add to projection data extraction in `chromadb_client.py:get_projection_data()`
3. Store in collection metadata during embedding in `embed_huggingface.py` / `embed_local_file.py`

## Key Gotchas

- **ChromaDB metadata values cannot be None** - always filter out None values before storing
- **ChromaDB metadata supports only str, int, float, bool** - lists/dicts must be JSON-serialized
- **Projections are stored as JSON strings in per-item metadata**, not as separate fields
- **`embed_dataset.py`** is a facade that re-exports from `embedding_functions/` - don't add logic here
- **SentenceTransformer models use a class-level cache** - first load is slow, subsequent calls reuse the cached model
- **The `for_query` flag on `get_collection()`** affects QWEN (adds instruction prefix) and Gemini (maps RETRIEVAL_DOCUMENT → RETRIEVAL_QUERY)
