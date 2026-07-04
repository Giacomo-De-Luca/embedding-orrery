# CLAUDE.md - Backend

Instructions for working with the `interpretability_backend` Python backend.

## Running

```bash
# Start server (from project root)
./start_backend.sh
# Or: uv run uvicorn interpretability_backend.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests (all pytest tests live in unit_tests/; test/ holds notebooks + debug scripts)
uv run pytest interpretability_backend/unit_tests/
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
  - `insert_document_activations_batch()`, `insert_document_activations_bulk()` - Per-document SAE activation storage
  - `get_document_activation_item_ids()`, `has_document_activations()`, `delete_document_activations()` - Document activation management
  - `search_documents_by_feature_labels()` - Two-hop search: label text → features → ranked documents
- **`chromadb_client.py`** - Vector-only wrapper (~170 lines, stripped from ~610). Key methods:
  - `get_collection(name, load_embedding_function, for_query, query_prompt)` - Lazy EF loading
  - `semantic_search(...)` - Vector similarity search, returns IDs + distances (no documents/metadata)
- **`huggingface_client.py`** - Dataset info/preview via `datasets` library, portion loading (FIRST_N, RANDOM_SAMPLE, ROW_RANGE, ALL)
- **`local_data_client.py`** - File loading via pandas/pyarrow. Optimized: parquet reads metadata without loading data, CSV reads only headers for info.

### Embedding Functions (`backend/embedding_functions/`)
- **`config.py`** - `DB_PATH`, `EmbeddingProvider`, `EmbeddingModelConfig`, `EmbeddingConfig`, `LocalFileEmbeddingConfig`, `EmbeddingResult`. The `BaseConfig` dataclass uses `kw_only=True`.
- **`create_embedding_function.py`** - Factory pattern. Returns `(EmbeddingFunction, dimension)`. Loads from `.env` via python-dotenv. HuggingFace login happens here if `HUGGINGFACE_API_KEY` is set.
- **`embed_huggingface.py`** - Full HF embedding pipeline: load portion → sort by length → explicit embed → DuckDB (docs+metadata) + ChromaDB (vectors only). Resume via DuckDB ID check. **Multi-split**: `EmbeddingConfig.splits` (list) embeds several splits into ONE collection in a single pass — `_load_rows_for_splits()` loads+concatenates them, tagging each row with the reserved `_SPLIT_KEY` (excluded from embedded/metadata columns; survives length-sort), so each item stores its own `source_split` and a single shared `IDDeduplicator` keeps IDs unique across splits. Falls back to `[split]` when unset. (The "All Rows" option in the test-embed HuggingFace tab now sends all split names here in one call instead of looping per-split — the old loop deleted-then-recreated the collection each iteration, leaving only the last split.)
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
  - `extract_topics(config)` - Full pipeline: load projections → HDBSCAN → c-TF-IDF → optional reduction → optional LLM labels → persist to DuckDB (`topic_extractions`/`topic_info`/`topic_assignments` tables via `_sync_topics_to_duckdb`; ChromaDB stores vectors only and is never written here). `TopicExtractionConfig.cluster_on` selects the clustering space (exposed over GraphQL via `TopicConfigInput` → `converters.build_topic_extraction_config` → test-embed `TopicConfigForm`): `"cluster_umap"` (**default**) runs a fresh BERTopic-style UMAP via module helper `_reduce_for_clustering` (n_components=5, min_dist=0, cosine; tunable via `cluster_n_components`/`cluster_min_dist`/`cluster_n_neighbors`) on the raw vectors before HDBSCAN; `"projection"` clusters on the stored UMAP/PCA coords; `"embedding"` clusters on the L2-normalised raw vectors. Both raw-vector modes load via `utils/embedding_loader.load_embeddings_for_ids` and fall back to projection coords if the load fails. `cluster_on` + the UMAP params are recorded in the `topic_extractions.config` JSON snapshot.
  - `reduce_existing_topics(...)` - Standalone: load existing topics → reconstruct c-TF-IDF → reduce → update metadata
  - `generate_llm_labels_for_collection(...)` - Standalone: generate LLM labels for existing topics/subtopics with incremental saves and resume support. Note: the previously documented `preserve_ctfidf_labels` option does not exist in the live code — labels are overwritten in place (keywords survive in `topic_info.keywords`)
- **`progress_emitter.py`** - `emit_progress(job_id, ...)` broadcasts to subscriber queues. Thread-safe (uses `queue.put_nowait`).
- **`job_state.py`** - `JobStateService` singleton with file-based persistence. Methods: `start_job`, `update_progress`, `update_total_expected`, `complete_job`, `fail_job`, `list_jobs`.
- **`interpret_service.py`** - `InterpretService` wrapping the `interpret/` toolkit for live SAE inference. Manages Gemma3-4b-it lifecycle (load/unload, `asyncio.Lock` for GPU serialisation). Four methods: `run_prompt_activations()` (per-token top-k features via PromptExplorer), `generate_steered()` (baseline + steered text via HookManager + SteeringOp), `run_prompt_highlight()` (max-pooled feature activations for scatter plot), `generate_stream()` (streaming chat generation with optional steering and cancel support — emits tokens via `token_emitter.py`; an optional `seed` calls `torch.manual_seed()` right before sampling for both "it"/"pt" paths, so serialised steered+baseline calls sharing one seed get an identical RNG start — backs the frontend chat compare mode). **Direction-vector steering**: `SteeringSpec.direction_name` resolves a pre-extracted 1-D `.pt` vector via the module-level `DIRECTION_REGISTRY` (`refusal`, `poetry` for `gemma-3-4b-it`) — applied at `RESID_POST` of the registry's `layer`, bypassing the `sae.w_dec` lookup. Vectors are loaded from `resources/directions/` lazily and cached on `self._direction_cache` (cleared on unload). **SAE weight cache**: `interpret/sae/loading.py` holds a module-level `_SAE_CACHE` of loaded `SAEBase` objects keyed by the config fields that determine on-disk identity (`layer_index`, `hook_type`, `width`, `model_size`, `variant`, `l0_size`, `dtype`, `device`). `load_sae()` is a thin cache wrapper; `clear_sae_cache()` empties it and is called from `InterpretService.unload_model()` so a model-variant switch frees stale device tensors. Every consumer of `HookManager.add_sae(...)` — chat steering (`_build_steering_session`), `run_prompt_highlight`, `run_batch_highlight`, and `PromptExplorer.run_prompt()` — benefits transparently. (The Gemma cache key uses `variant`/`l0_size`; the Qwen-scope branch keys on `model_size`/`k` instead — see below.) See `documentation/INTERPRET_API.md`.
- **Qwen-scope SAE support (toolkit-level)** — `interpret/sae/sae_config.py` generalises `QwenScopeSAEConfig` across Qwen3/Qwen3.5 sizes via the `QWEN_SCOPE_MODELS` registry: dims are read from the downloaded TopK weights (`loading._load_qwen_scope_sae` validates `config.d_in`), repo naming (family prefix, optional `-Base`) comes from the registry, and only `RESID_POST` is allowed. `Qwen3Inference` (`interpret/inference/qwen3_transformers.py`) gains `generate_stream`/`generate_chat_stream` yielding the shared `interpret/inference/streaming.py::TokenStreamEvent` (mirrors the Gemma streaming contract). The live `InterpretService` is still Gemma-bound — this is wrapper/SAE-layer support only. See `interpret/QWEN_SUPPORT.md`.
- **`token_emitter.py`** - Async queue event bus for token streaming (mirrors `progress_emitter.py`). `TokenEvent` dataclass, `register_token_subscriber`/`unregister_token_subscriber` (async), `emit_token` (sync, thread-safe). Queue maxsize 500. Used by `generate_stream` subscription for real-time token delivery over WebSocket.

### Topic Extraction (`backend/topic_extraction/`)
- **`cluster_and_label.py`** - `GenerateTopics` class: HDBSCAN clustering (`gen_min_span_tree=True` so the fitted model exposes `relative_validity_`/DBCV) + `ClassTfidfTransformer` (BERTopic-inspired c-TF-IDF). Stores `ctfidf_matrix` and `words` properties for reduction.
- **`topic_reducer.py`** - `TopicReducer` class: `reduce_to_n_topics()` (AgglomerativeClustering on distance matrix), `auto_reduce_topics()` (HDBSCAN with min_cluster_size=2). Supports c-TF-IDF or semantic embeddings for similarity. Returns `topic_hierarchy` mapping new_topic_id → [old_topic_ids].
- **`llm_labeling.py`** - Provider-agnostic LLM labeling. Factory creates `_GeminiLabeler` or `_OpenAILabeler`. Uses tenacity for retries. Prompt includes sample documents + keywords. `generate_llm_labels()` accepts optional `progress_callback(done, total)` for per-topic progress reporting.
- **`extract_topics_bertopics.txt`** - Inert BERTopic reference snippet (plain text, not importable). The former `extractRepresentation.py` / `_representation_utils.py` reference implementations were removed as dead code; the live LLM labeling path is `llm_labeling.py`.

### Topic-Quality Evaluation (`interpretability_backend/evaluation/`)
Standalone package (sibling of `backend/`) for scoring extracted topics; decoupled from the extraction pipeline and from DuckDB persistence / GraphQL (those are intentionally deferred). See `evaluation/README.md`.
- **`quality_metrics.py`** - `TopicQualityEvaluator.evaluate(...)` returns a dict of: **DBCV** (from a fitted HDBSCAN model's `relative_validity_`; `None` for stored labels since the model isn't persisted), **silhouette in embedding space** (cosine) and **projection space** (euclidean) with noise excluded + subsampling, **topic diversity**, and **C_v + U_Mass coherence** via gensim's `CoherenceModel` (no embedding model needed). Every metric degrades to `None` on degenerate input; never raises.
- **`run_evaluation.py`** - Config-driven runner (`uv run python -m interpretability_backend.evaluation.run_evaluation`, config `eval_config.toml` or `ORRERY_EVAL_CONFIG`). Loads the active extraction + projections (DuckDB) + embeddings (ChromaDB), reconstructs labels, recomputes per-cluster keywords via c-TF-IDF, evaluates, prints a report, writes JSON. Config `level` selects `"topic"` (active/reduced topics) or `"subtopic"` (pre-reduction HDBSCAN density clusters — the more meaningful geometric evaluation after reduction).
- Tests: `unit_tests/test_topic_quality_metrics.py` (synthetic data, no DB/model). gensim is a dependency (`gensim>=4.4`).
- **`projection_fidelity.py`** - `ProjectionFidelityEvaluator`: Mantel-test fidelity of projections (UMAP/PCA) against reference distance structures — **embedding** (cosine) and, for colour datasets, **perceptual colour** (CIEDE2000). Pure (condensed distance vectors in → results dict out), degrades every metric to `None`, never raises. Three statistics via the merged `interpret.utils.mantel.MantelTest`: global Spearman ρ (whole ordering), kNN-local ρ (neighbourhoods, taken in the reference space), and a permutation z/p_emp significance test. Colour distances (`colour_distances`) **lazily import** scikit-image + `interpret.utils.distances.pairwise_lab_ciede2000`, so importing the module / running embedding-space fidelity needs no scikit-image (only colour does). Runner `run_projection_fidelity.py` (config `projection_fidelity_config.toml` or `ORRERY_PROJECTION_FIDELITY_CONFIG`) loads projections + item metadata (colour) from DuckDB and embeddings from ChromaDB, subsamples to `sample_size` (O(N²) guard), prints a report, writes JSON. Run with the backend stopped. On `xkcd_hilbert_gemini`: UMAP-3D preserves perceptual colour (ρ=0.60) and local neighbourhoods better; PCA-3D preserves the embedding's global geometry better. Tests: `unit_tests/test_projection_fidelity.py`. See `documentation/PROJECTION_FIDELITY.md`.

### Probing experiments (`interpretability_backend/experiments/`)
Host-specific config for the merged `interpret/probing` engine (the generic YAML-driven activation-probing framework lives in the toolkit — see `interpret/probing/README.md`). Per the repo convention, dataset-specific glue (experiment YAMLs) lives here with the project, not inside `interpret/`; the concrete `ManifestBuilder` subclasses ship in the toolkit at `interpret/probing/manifests/{glasgow,xkcd,feature_csv}.py`.
- **`glasgow_psycholinguistic/experiment.yaml`** — probes **EmbeddingGemma-300M** mean-pooled hidden states for six **Glasgow psycholinguistic norms** (concreteness, imageability, valence, arousal, familiarity, aoa) over ~4.7k words, via `GlasgowManifestBuilder` + an `encoder` extraction (`google/embeddinggemma-300M`, layers auto `[0,12,24]`) + ridge & MLP probes. Swap the extraction for a `type: gemma` block to probe Gemma-3-4b residual activations instead (heavier). Data (gitignored) at `resources/psycolinguistics/{glasgow_norm.csv, concreteness.tsv}` — the builder loads Brysbaert concreteness unconditionally even in `glasgow_only` mode. Run from `interpretability_backend/`: `uv run python -m interpret.probing.orchestrator experiments/glasgow_psycholinguistic/experiment.yaml`. Verified result: ridge probes decode concreteness R²=0.77, imageability 0.69, valence 0.68, aoa 0.61, arousal 0.51, familiarity 0.49 — all best at the middle layer, ridge > MLP throughout. Tests: `unit_tests/test_glasgow_manifest.py` (builder + YAML parse, no model). See the folder README.
- **Probing-engine deps** (merged from astrolabe, previously undeclared): `omegaconf` (config loading), `seaborn` (figures), plus `scikit-image` (used by `interpret.utils.distances` CIEDE2000). All added to `pyproject.toml`; also documented in `interpret/README.md`'s dependency list for upstream portability.

### Utils (`backend/utils/`)
- **`compute_projections.py`** - `compute_projections_for_collection(name, projection_type, job_id)`. Loads embeddings from ChromaDB in 5k batches, computes PCA/UMAP, stores projections as native FLOAT[] arrays in DuckDB. When `job_id` is provided, emits per-projection progress via `emit_progress`.
- **`embedding_loader.py`** - `load_embeddings_for_ids(collection_name, ids)`: reads all vectors from ChromaDB (5k batches) and returns a numpy array ordered to match `ids` (or `None` if any id is missing). Shared by embedding-space topic clustering (`cluster_on="embedding"`) and the `evaluation/` package's embedding-space silhouette.
- **`duckdb_sync.py`** - Helper functions for dual-write during embedding: `sync_dataset_and_collection()`, `sync_items()`. Called from embedding pipelines.
- **`text_processing.py`** - `format_text_for_embedding(row, columns, template)` supports template strings with `{column}` placeholders. `extract_metadata(row, columns)` prepares metadata for storage.
- **`color_preprocessing.py`** - Auto-detects hex color columns (`colour_code`, `color_code`, etc.) and maps each color to a float 0-1 position on a pre-built colorscale strip (Hilbert RGB, hue-sat, XKCD, or rainbow). Adds `mapped_colour` and `mapped_colour_scale` to item metadata. Called from `embed_local_file.py` and `embed_huggingface.py` after `extract_metadata()`. Strips generated by `scripts/generate_color_strips.py`.
- **`batch_utils.py`** - `sort_items_by_length()` sorts by text length descending for efficient transformer batching (reduces padding waste).
- **`id_utils.py`** - `IDDeduplicator` keeps IDs unique with **collision-only** suffixing: the first occurrence of a base id is returned verbatim; later collisions get `_1`, `_2`, … bumped until a genuinely free id (so a generated suffix never overwrites a real pre-existing `5_1`). Share one instance across everything written to a collection (e.g. all splits). Used by `embed_huggingface.py`, `embed_local_file.py`, `embed_vectors.py`. **Resume caveat**: this replaced an older always-suffix scheme (`cat` → `cat_1`), so a local-file/vector job with an `id_column` that was *interrupted under the old scheme* should be **restarted, not resumed** (resume replays ids under the new scheme and won't match the stale ones). Fresh runs and the HuggingFace path are unaffected.
- **`provider_list.py`** - Single source of truth for `EmbeddingProviderEnum` (Strawberry enum). Used by both GraphQL types and the internal `EmbeddingProvider` mapping.
- **`logger.py`** - Configures `orrery` logger with file handler (DEBUG) and console handler (ERROR).
- **`seed_bootstrap.py`** - `ensure_seed_loaded()`: on first run copies the committed seed snapshot (`resources/seed/main.duckdb` + `vector_db/`) into the live paths, **only if `main.duckdb` is absent** (never clobbers an existing DB). Called from the FastAPI `lifespan` hook in `main.py`. Build/refresh the seed with `scripts/build_seed_snapshot.py` (run with the backend stopped — DuckDB is single-writer). The seed ships `emotion` + `xkcd_hilbert_gemini` (~23 MB); `.gitignore` un-ignores `resources/seed/`.

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
- **`sae_document_activations`** — Per-document max-pooled SAE activations: PK (collection_name, item_id, feature_index), only nonzero entries stored (~100-150 per doc). Indexed on (collection_name, feature_index) for two-hop label search.
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
