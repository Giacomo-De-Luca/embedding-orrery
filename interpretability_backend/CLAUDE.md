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
- **`upload.py`** - REST `POST /upload` endpoint saving files to `resources/uploads/`. Not mounted when `ORRERY_READ_ONLY=1`.
- **`read_only.py`** - `ReadOnlyExtension` (installed on the schema unconditionally): when `ORRERY_READ_ONLY=1` (read per-operation, not at import) it short-circuits every mutation before execution with a "read-only" GraphQL error by pre-setting `execution_context.result`. Powers the public HF Space demo (`documentation/HF_SPACE_DEMO.md`). Tested in `unit_tests/test_read_only.py`.

### Clients (`backend/clients/`)
- **`duckdb_client.py`** - Central orchestrator. Key methods:
  - `create_dataset()`, `list_datasets()`, `get_dataset()`, `delete_dataset()` - Dataset CRUD
  - `insert_items_batch()` - Bulk item insert via pandas DataFrame
  - `get_filtered_items(dataset, filters, limit, offset)` - JSON metadata filtering ($eq/$ne/$gt/$gte/$lt/$lte/$in/$nin)
  - `get_items_by_ids()` - Enrich search results with documents + metadata
  - `register_vector_collection()`, `get_vector_collections()` - Vector collection registry
  - `insert_projections_batch()`, `get_projection_data(collection, type)` - Per-type projection storage/retrieval
  - `text_search(dataset, query, fields, mode, case_sensitive, filters, limit)` - SQL-based text search (ILIKE for substring, json_extract for metadata; metadata filters applied in SQL; optional `limit` truncates `matches` while `total_matches` stays the full deduplicated count via a single COUNT(*) across all selected fields)
  - `text_search_bm25(dataset, query)` - FTS extension BM25 search (known issue: needs debugging)
  - `create_topic_extraction()`, `insert_topic_info_batch()`, `insert_topic_assignments_batch()` - Topic storage
  - `get_active_topics()`, `update_topic_label()`, `update_subtopic_label()` - Topic reads/updates
  - `insert_sae_features_batch()`, `get_sae_feature()`, `search_sae_features()`, `list_sae_models()` - SAE feature storage
  - `insert_sae_activations_batch()`, `get_sae_activations()` - SAE activation storage
  - `delete_sae_data()` - Remove SAE data for a model/sae pair
  - `insert_document_activations_batch()`, `insert_document_activations_bulk()` - Per-document SAE activation storage
  - `get_document_activation_item_ids()`, `has_document_activations()`, `delete_document_activations()` - Document activation management
  - `search_documents_by_feature_labels()` - Two-hop search: label text → features → ranked documents (MAX ranking)
  - `search_documents_by_feature_indices()` - Hop-2 only search over pre-selected features with ranking modes: `scaled_sum` (default, per-feature max-normalized via inline window function), `max`, `sum`, `matching_features` (count-first, scaled-sum tiebreak). Returns `{"results": [...], "total_matches": int}` — the true pre-limit match count via `COUNT(*) OVER ()` (GraphQL: `DocumentActivationSearchResponse.totalResults`)
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
- **`embed_sentence_transformer.py`** - Fork of ChromaDB's implementation. Adds prompt support (known prompt names like "Retrieval-query" → `prompt_name`, custom strings → `prompt`). Class-level **single-slot** model cache shared across instances — loading a different model evicts the previous one (see Key Gotchas).
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
- **`interpret_service.py`** - `InterpretService` wrapping the `interpret/` toolkit for live SAE inference. Manages Gemma3-4b-it lifecycle (load/unload, `asyncio.Lock` for GPU serialisation). Four methods: `run_prompt_activations()` (per-token top-k features via PromptExplorer), `generate_steered()` (baseline + steered text via HookManager + SteeringOp), `run_prompt_highlight()` (max-pooled feature activations for scatter plot — pooled over the **user-prompt tokens only**: BOS + chat-template positions are masked via `_find_prompt_token_range(..., skip_chat_template=...)`, since the BOS activation sink and the constant template tokens otherwise top the ranking identically for every prompt; `run_batch_highlight()` masks the same way per document; the shared format→range→pool block lives in `_format_for_prefill()`/`_pooled_prompt_features()` (which also warns on a prefill/token_strings length mismatch), and `_max_pool_activations(record, start, end, fallback_to_full)` clamps the range — an empty range degrades to full-sequence pooling interactively but to **no features** in the batch path, so an empty document never persists the constant BOS/template features. Coverage modes' dead-feature floor treats density exactly 0.0 as *unknown*, not dead — qwen-scope features ship with density=0.0. **Data caveat**: `sae_document_activations` rows computed before the masking fix (2026-07) still include BOS/template features — refresh via `deleteDocumentActivations` + rerun `computeDocumentActivations`), `generate_stream()` (streaming chat generation with optional steering and cancel support — emits tokens via `token_emitter.py`; an optional `seed` calls `torch.manual_seed()` right before sampling for both "it"/"pt" paths, so serialised steered+baseline calls sharing one seed get an identical RNG start — backs the frontend chat compare mode). **Direction-vector steering**: `SteeringSpec.direction_name` resolves a pre-extracted 1-D `.pt` vector via the module-level `DIRECTION_REGISTRY` (`refusal`, `poetry` for `gemma-3-4b-it`) — applied at `RESID_POST` of the registry's `layer`, bypassing the `sae.w_dec` lookup. Vectors are loaded from `resources/directions/` lazily and cached on `self._direction_cache` (cleared on unload). **SAE weight cache**: `interpret/sae/loading.py` holds a module-level `_SAE_CACHE` of loaded `SAEBase` objects keyed by the config fields that determine on-disk identity (`layer_index`, `hook_type`, `width`, `model_size`, `variant`, `l0_size`, `dtype`, `device`). `load_sae()` is a thin cache wrapper; `clear_sae_cache()` empties it and is called from `InterpretService.unload_model()` so a model-variant switch frees stale device tensors. Every consumer of `HookManager.add_sae(...)` — chat steering (`_build_steering_session`), `run_prompt_highlight`, `run_batch_highlight`, and `PromptExplorer.run_prompt()` — benefits transparently. (The Gemma cache key uses `variant`/`l0_size`; the Qwen-scope branch keys on `model_size`/`k` instead — see below.) See `documentation/INTERPRET_API.md`.
- **Qwen-scope SAE support** — `interpret/sae/sae_config.py` generalises `QwenScopeSAEConfig` across Qwen3/Qwen3.5 sizes via the `QWEN_SCOPE_MODELS` registry: dims are read from the downloaded TopK weights (`loading._load_qwen_scope_sae` validates `config.d_in`), repo naming (family prefix, optional `-Base`) comes from the registry, and only `RESID_POST` is allowed. `Qwen3Inference` (`interpret/inference/qwen3_transformers.py`) has `generate_stream`/`generate_chat_stream` yielding the shared `interpret/inference/streaming.py::TokenStreamEvent` (mirrors the Gemma streaming contract). **Phase 1 is live in `InterpretService`**: `load_model("Qwen/Qwen3-1.7B")` dispatches on checkpoint family (`_parse_checkpoint` → `(family, model_size, variant)`; qwen checkpoints pass through `_normalize_checkpoint` unchanged), SAE configs are built family-correctly by `_make_sae_config()` (qwen pins `k=50`), streaming chat translates the frontend's Gemma `"model"` role to Qwen's `"assistant"` and threads `GenerateStreamInput.enableThinking` (default False — streams `<think>` reasoning when on, which the frontend routes into the collapsible reasoning panel; ignored for Gemma). **Phase 1.5 is live**: per-token `run_prompt_activations` now works for qwen. `PromptExplorer` is family-agnostic — the `wrapper` supplies the chat template (`format_prompt`) and per-token pieces (a new symmetric `token_strings()` on both `GemmaPytorchInference`/`Qwen3Inference`, aligned to the prefill: Gemma prepends BOS, Qwen doesn't), and `PromptExplorerConfig.sae_config_factory` (injected by the service from `_make_sae_config`) builds the family's SAE config so the explorer no longer constructs Gemma-scope configs itself. `_find_prompt_token_range` gained a qwen ChatML branch (`<|im_start|>`/`<|im_end|>`, forward-scan robust to an injected system turn; base BOS offset from `wrapper.prepends_bos`). Qwen features return index + activation but **blank labels** (qwen-scope isn't on Neuronpedia → Phase 2). The DuckDB ids are `model_id="qwen3-1.7B-base"` (exactly `QwenScopeSAEConfig.neuronpedia_model_id`) / `sae_id="{layer}-qwenscope-1-res-32k"` (`qwen_source_id` in `source_ids.py` — same 5-part positional shape as gemma-scope ids). `services/model_registry.py` (torch-free, literal-only — no `interpret.*` imports) maps stored model ids → HF checkpoints for `compute_document_activations` (which now compares model *ids*, not checkpoints, for its reload check — the old comparison always reloaded). Label-free bootstrap: `scripts/extract_qwen_decoder_vectors.py` extracts decoder vectors (L14/L24, 32k features each) and ingests into `sae_features` + ChromaDB scatter collections; labels/densities/activation examples arrive with the Phase-2 autointerp pass (qwen-scope is not on Neuronpedia). The same extract-and-ingest is also available over GraphQL: `prepareSaeData` with `family: "qwen"` dispatches to `sae_pipeline_service.prepare_qwen_sae_data()` (any `QWEN_SCOPE_MODELS` size/layer, k pinned to 50 since `qwen_source_id` doesn't encode k, `include_activations` rejected, same ids/parquet naming as the script so re-runs are idempotent) — this backs the collections-page SAE tab's Qwen entries. See `interpret/QWEN_SUPPORT.md`.
- **Residual-norm steering hint** (`interpret/inference/residual_norm_profiler.py` + `scripts/profile_residual_norms.py`) — `ResidualNormProfiler` (family-agnostic Gemma/Qwen) measures per-layer residual-stream L2 norm `‖h_L‖` by reusing each wrapper's existing `cache_activations` (no new hooks; reconciles Gemma `{"prefill":{L:{"post_mlp"}}}` vs Qwen `{L:{RESID_POST}}`, masks BOS, pools per-token norms → median/p25/p75). The config-driven runner also records each `DIRECTION_REGISTRY` direction's `‖v‖` and writes a small static JSON straight into the frontend (`embedding_visualization/lib/utils/residualNorms.json`, keyed by model id, merged in place) — no GraphQL, torch-free-boundary-safe. Turns a raw steering coefficient into a "fraction of the residual stream" hint (Gemma-scope decoder rows are unit-norm, so the per-layer table is a complete hint for all SAE features). Pure helpers unit-tested (`unit_tests/test_residual_norm_profiler.py`). Run from `interpretability_backend/`: `uv run python -m scripts.profile_residual_norms`. See `documentation/STEERING_STRENGTH_HINT.md`.
- **`token_emitter.py`** - Async queue event bus for token streaming (mirrors `progress_emitter.py`). `TokenEvent` dataclass, `register_token_subscriber`/`unregister_token_subscriber` (async), `emit_token` (sync, thread-safe). Queue maxsize 500. Used by `generate_stream` subscription for real-time token delivery over WebSocket.
- **`probing_service.py`** - Embedding-space probing: X = collection vectors via `load_embeddings_for_ids`, y = numeric metadata field via `duckdb_client.get_numeric_metadata_field` (quoted JSONPath — dotted fields like `Conc.M` are literal keys). **Binary categorical fallback**: if a field has fewer than `MIN_VALID_SAMPLES` numeric values, the orchestrator reads it as text (`get_text_metadata_field`) and, if it has exactly two distinct non-null values, maps them to 0/1 via `binary_target_mapping` (deterministic: alphabetically-first value → 0, e.g. `{"safe": 0, "unsafe": 1}`) and probes those. The mapping is recorded in the persisted `config` snapshot and surfaced on `ProbeInfo.targetMapping` (GraphQL) for the UI tooltip. Massmean on an imbalanced binary column can degenerate (median split collapses to one class) and fails cleanly with the degenerate-target error — ridge/mlp are unaffected. **Seven kinds**: `ridge`, `lasso` (sparse linear, alpha), `massmean`, `massmean_cov` (Geometry-of-Truth covariance correction: pooled within-class Σ of the standardized train features, Hermitian pseudo-inverse with eigenvalue cutoff `_MASSMEAN_COV_PINV_ATOL=1e-3` — which doubles as the N<D regularizer, null-space directions are dropped — applied to μ⁺−μ⁻, unit-normalized), `svr` (RBF; nonlinear — the toolkit's new opt-in `save_models` flag joblib-dumps the fitted estimator + scaler params to `models/…`, since there's no coef_ to save; training pool hard-capped at `_SVR_MAX_TRAIN=10k` on top of the shared cap because RBF SVR is O(n²)), `logreg` (**binary targets only** — exactly two distinct values, remapped to {0,1} and passed as integer labels so the toolkit doesn't percentile-bin them; scores = P(class 1) via sigmoid of the saved hyperplane; metrics are val_accuracy/f1/auc; no residuals), and `mlp`. Hyperparams (`alpha`, `c`, `kernel`, `class_weight`, `hidden_dims`, `epochs`, `patience`, `dev_split`, `activation` (relu/gelu/tanh/silu, `MLP_ACTIVATIONS` mirror in probing_types — the scoring path MUST rebuild `ProbeModel` with the trained activation, parameterless layers load silently), `seed` — default 7, mirrored by the UI — `train_split`, `max_train_samples`) flow from `TrainProbeInput` through `converters.build_probe_config` (validates kind + class_weight + numeric ranges; tested in `unit_tests/test_probe_converters.py`). `max_train_samples` is an internal guardrail not exposed in the UI. Reuses the `interpret/probing` toolkit trainers (`train_sklearn_probe` for the sklearn kinds, `train_mlp_probes` for mlp) on an in-memory `ActivationDataset`; builds its own seeded split and passes `indices_override` (same split across kinds → comparable metrics; exact n_train/n_val). Parses metrics from `probe_results.csv` (`val_*`/`train_*` columns only — the sklearn writer drops constant layer/intermediate columns), NaN→None sanitized. The mass-mean family additionally gets a **calibrated readout** (`_calibrated_readout`: slope/intercept least squares on train projections → comparable val/train R²; the slope/intercept persist as `calibration` in the probe's config snapshot) and **calibrated residuals** (`slope·score + intercept − y`). Scores ALL items (ridge/lasso: standardized `X @ coef + intercept`; mass-mean kinds: unit-direction projection; mlp: batched checkpoint forward); residuals for predictive kinds (`score − y`) and calibrated mass-mean kinds where y exists; non-finite scores raise (degenerate target). **MLP train/dev/val discipline** (leak fix): `MLPProbeSpec.dev_split` (default 0.2) carves a dev subset out of the train side inside `_train_single_probe` — early stopping AND best-checkpoint selection run on dev only, so the reported `val_*` metrics touch the val split exactly once (previously both selected on the reporting split, inflating val metrics); dev below `_DEV_FLOOR=10` rows or `dev_split=0` degrades to fixed-epoch training with final weights (`unit_tests/test_mlp_probe.py`). Orchestrator `train_probe_for_collection` never raises — error results + terminal `failed` progress on job id `{collection}_probe` (coarse 4-stage progress). Guardrails: `MIN_VALID_SAMPLES=50`, `max_train_samples=50k` subsample (full scoring regardless), float32 downcast. Artifacts under `PROBING_RESULTS_DIR/collections/<collection>/<field>/<kind>/` (offline `consolidate.py`-compatible). Pure core (`run_probe_core`) is DB-free and unit-tested on synthetic data.
- **`probing_types.py`** - Torch-free probe types shared with the API layer: `ProbeConfig`, `PROBE_KINDS`, `sanitize_field_key`, `score_field_names` (derived metadata keys `probe_<field>_<kind>_score` / `_residual`; residuals for `_PREDICTIVE_KINDS` ridge/lasso/mlp/svr plus the `_CALIBRATED_KINDS` mass-mean family — logreg has none), `binary_target_mapping` (two-distinct-value → 0/1 map, else None). Part of the torch-free import boundary (see Key Gotchas).

### Topic Extraction (`backend/topic_extraction/`)
- **`cluster_and_label.py`** - `GenerateTopics` class: HDBSCAN clustering (`gen_min_span_tree=True` so the fitted model exposes `relative_validity_`/DBCV) + `ClassTfidfTransformer` (BERTopic-inspired c-TF-IDF). Stores `ctfidf_matrix` and `words` properties for reduction.
- **`topic_reducer.py`** - `TopicReducer` class: `reduce_to_n_topics()` (AgglomerativeClustering on distance matrix), `auto_reduce_topics()` (HDBSCAN with min_cluster_size=2). Supports c-TF-IDF or semantic embeddings for similarity. Returns `topic_hierarchy` mapping new_topic_id → [old_topic_ids].
- **`llm_labeling.py`** - Provider-agnostic LLM labeling. Factory creates `_GeminiLabeler` or `_OpenAILabeler`. Uses tenacity for retries. Prompt includes sample documents + keywords. `generate_llm_labels()` accepts optional `progress_callback(done, total)` for per-topic progress reporting.
- **`extract_topics_bertopics.txt`** - Inert BERTopic reference snippet (plain text, not importable). The former `extractRepresentation.py` / `_representation_utils.py` reference implementations were removed as dead code; the live LLM labeling path is `llm_labeling.py`.

### Topic-Quality Evaluation (`interpretability_backend/evaluation/` + `backend/services/topic_quality_service.py`)
Metric implementations live in the standalone `evaluation/` package (pure, no DB/model); orchestration + persistence live in `backend/services/topic_quality_service.py::score_topic_quality`, shared by the GraphQL `evaluateTopics` mutation and the TOML runner. See `evaluation/README.md`.
- **`quality_metrics.py`** - `TopicQualityEvaluator.evaluate(...)` returns a dict of: **DBCV** (from a fitted HDBSCAN model's `relative_validity_`; `None` for stored labels since the model isn't persisted), **silhouette in the cluster space** (`silhouette_cluster_space`, euclidean on the coords the clustering ran in; noise excluded + subsampled), **topic diversity**, and **C_v + U_Mass coherence** via gensim's `CoherenceModel` (no embedding model needed). A `metrics` set selects a subset (names: `dbcv`, `silhouette`, `diversity`, `coherence_cv`, `coherence_umass`); unrequested keys are omitted. **Raw-embedding silhouette was removed as non-discriminative** (high-D cosine distances concentrate; identical and clearly-different clusterings scored alike ~0.05–0.08 — see README "Why there is no raw-embedding silhouette"). Every metric degrades to `None` on degenerate input; never raises.
- **`topic_quality_service.py`** - `score_topic_quality(collection_name, level, metrics, sample_size, ...)`: loads the active extraction, reads `projection_type` from the extraction's `config` snapshot (never caller-supplied), rebuilds level-aligned labels, recomputes per-cluster c-TF-IDF keywords, evaluates, and persists to `topic_extractions.quality_metrics` (JSON keyed by level, read-modify-write via `update_topic_quality_metrics`). Emits 4-stage progress on job id `{collection}_evaluate`. Never raises (`{"error": ...}` result). Lazy-imported by the mutation (pulls hdbscan→sklearn + gensim — lean-import boundary).
- **`run_evaluation.py`** - Config-driven runner (`uv run python -m interpretability_backend.evaluation.run_evaluation`, config `eval_config.toml` or `ORRERY_EVAL_CONFIG`); thin caller of the service. Config `level` selects `"topic"` or `"subtopic"` (pre-reduction HDBSCAN density clusters — the more meaningful geometric evaluation after reduction); optional `metrics = [...]` list.
- GraphQL: `evaluateTopics(input: {collectionName, level, metrics, sampleSize})` mutation → `EvaluateTopicsResult{metrics: JSON}`; stored scores surface on `collectionTopics.qualityMetrics` (`{"topic": {...}, "subtopic": {...}}`).
- Tests: `unit_tests/test_topic_quality_metrics.py`, `unit_tests/test_topic_quality_service.py` (fake client + in-memory DuckDB roundtrip). gensim is a dependency (`gensim>=4.4`).
- **Migration gotcha**: the `quality_metrics` column is added by a **guarded** ALTER (pragma_table_info check) followed by an immediate `CHECKPOINT` in `_ensure_schema`. Do NOT use a bare `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` here: replaying an ALTER on this FK-referencing table from the WAL crashes DuckDB 1.4/1.5 with "GetDefaultDatabase with no default database set", wedging the whole DB open (bitten live on 2026-07-06; recovered by truncating the poisoned tail record off the WAL).
- **`projection_fidelity.py`** - `ProjectionFidelityEvaluator`: Mantel-test fidelity of projections (UMAP/PCA) against reference distance structures — **embedding** (cosine) and, for colour datasets, **perceptual colour** (CIEDE2000). Pure (condensed distance vectors in → results dict out), degrades every metric to `None`, never raises. Three statistics via the merged `interpret.utils.mantel.MantelTest`: global Spearman ρ (whole ordering), kNN-local ρ (neighbourhoods, taken in the reference space), and a permutation z/p_emp significance test. Colour distances (`colour_distances`) **lazily import** scikit-image + `interpret.utils.distances.pairwise_lab_ciede2000`, so importing the module / running embedding-space fidelity needs no scikit-image (only colour does). Runner `run_projection_fidelity.py` (config `projection_fidelity_config.toml` or `ORRERY_PROJECTION_FIDELITY_CONFIG`) loads projections + item metadata (colour) from DuckDB and embeddings from ChromaDB, subsamples to `sample_size` (O(N²) guard), prints a report, writes JSON. Run with the backend stopped. On `xkcd_hilbert_gemini`: UMAP-3D preserves perceptual colour (ρ=0.60) and local neighbourhoods better; PCA-3D preserves the embedding's global geometry better. Tests: `unit_tests/test_projection_fidelity.py`. See `documentation/PROJECTION_FIDELITY.md`.

### Probing experiments (`interpretability_backend/experiments/`)
Host-specific config for the merged `interpret/probing` engine (the generic YAML-driven activation-probing framework lives in the toolkit — see `interpret/probing/README.md`). Per the repo convention, dataset-specific glue (experiment YAMLs) lives here with the project, not inside `interpret/`; the concrete `ManifestBuilder` subclasses ship in the toolkit at `interpret/probing/manifests/{glasgow,xkcd,feature_csv}.py`.
- **`glasgow_psycholinguistic/experiment.yaml`** — probes two encoders (`minilm` = `all-MiniLM-L6-v2` 384-d cls; `embeddinggemma_mean` = `google/embeddinggemma-300M` 768-d mean) for all **nine Glasgow psycholinguistic norms** (concreteness, imageability, valence, arousal, dominance, familiarity, aoa, semsize, gender) over ~4.7k words, via `GlasgowManifestBuilder` + `encoder` extractions + five probes (ridge/lasso/svr/**massmean**/mlp). Data (gitignored) at `resources/psycolinguistics/{glasgow_norm.csv, concreteness.tsv}` — the builder loads Brysbaert concreteness unconditionally even in `glasgow_only` mode. Run from `interpretability_backend/`: `uv run python -m interpret.probing.orchestrator experiments/glasgow_psycholinguistic/experiment.yaml`. Verified: MiniLM reproduces the reference `glasgow_psycholinguistic_norms` report bit-for-bit; EmbeddingGemma beats it on all 9 norms; SVR wins throughout; referential norms peak at early layers, affective/social at later. `massmean` reports Pearson/Spearman (uncalibrated direction), not R². Commented scaffolds for Gemma-3-4b (`type: gemma`) and a precomputed **Gemini** path (`csv_features` + `FeatureCSVManifestBuilder`, needs `GEMINI_API_KEY`). Tests: `unit_tests/test_glasgow_manifest.py` (builder + YAML parse, no model). Full writeup: `documentation/GLASGOW_PSYCHOLINGUISTIC_PROBING.md`; see also the folder README.
- **`trec_classification/`** — TREC question-type SAE probing (6 coarse / 50 fine classes) for **both families** via the engine's token-level two-stage pipeline (new 2026-07): `token_residuals` (one forward pass caches every token's residuals, ragged bf16, `skip_probes`) → `sae_pooled` (per-layer family-correct SAE encode + `max`/`last` pooling, BOS-masked like `max_pool_feature_acts`, `kept_by_layer`) / `residual_pooled` (raw baseline) / `concat` (all layers jointly, `feature_names` → `feature_importance.csv`). Gemma-3-4b covers all 34 layers × 3 hook sites (**the fork now caches raw `"attn_out"`** pre-residual-add, symmetric with `"mlp_out"`); Qwen3-1.7B covers 28 layers resid_post (size-configurable via `QWEN_SCOPE_MODELS`). Probes: multiclass logreg/SVC, balanced, 5-fold stratified (`LabeledTextManifestBuilder.min_class_count` drops fine classes < 5); `top_features` is fold- and multiclass-aware (`top_class` per feature). Smoke YAMLs run locally on MPS; full runs sized for an A100 — gemma is a two-pass workflow (pass 1 resid_post ≈ 14 GB token cache; commented pass-2 blocks add mlp_out/attn_out ≈ 29 GB, cache-hitting pass 1). Probes: logreg + linear_svc (both save directions — independent logistic/hinge feature rankings incl. on the concat matrix) + RBF svc accuracy ceiling with skip_extractions keeping it off the concat. **Sparse storage** (2026-07): `sparse: true` on `sae_pooled` stores scipy CSR instead of dense fp32 (~25-50× smaller; pooled SAE rows are 1-2% nonzero) — the enabler for 65k/262k width sweeps and safety-scale datasets. CSR flows through subset/save/load, `concat` (sparse hstack), and the sklearn probes natively; sparse input standardises scale-only (`with_mean=False` — centering would densify and shift the meaningful zero); mass-mean/MLP/center_only reject sparse with clear errors; dispatch helpers in `interpret/probing/utils/matrix_ops.py`. Tests: `test_labeled_text_manifest / test_trec_experiment_config / test_token_residual_extraction / test_sae_pooled_extraction / test_concat_extraction / test_top_features_multiclass / test_sparse_sae_probing`. Docs: `documentation/TREC_SAE_PROBING.md` (§6.7 for sparse) + folder README + probing README "Token-level two-stage pipeline".
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
- **`seed_bootstrap.py`** - `ensure_seed_loaded()`: on first run copies the seed snapshot (`ORRERY_SEED_DIR`, default `resources/seed/`) into the live paths, **only if `main.duckdb` is absent** (never clobbers an existing DB). Called from the FastAPI lifespan hook. Build/refresh seeds with `scripts/build_seed_snapshot.py` and a manifest under `config/seed_snapshots/` while the backend is stopped. `backend/utils/seed_snapshot.py` resolves datasets, copies optional collection/SAE payloads, preserves the committed-vector fallback, writes integrity metadata, and atomically installs only verified output. The large demo snapshot publishes to a private Hugging Face Dataset and is consumed through `demo.lock.json`; see `documentation/SEED_SNAPSHOTS.md` and `documentation/HF_SPACE_DEMO.md`.

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
- **`sae_document_activations`** — Per-document max-pooled SAE activations: PK (collection_name, item_id, feature_index), only nonzero entries stored. Sparsity depends on doc length: ~100-150 features/doc for short sentences, ~2.3k-7.3k for paragraph-length docs (union of JumpReLU-active features across all tokens; ACL collection = 56M rows). Indexed on (collection_name, feature_index) for two-hop label search (~80-130 ms per multi-feature query at 56M rows).
- **`probes`** — one row per trained probe: PK (collection_name, target_field, kind); config/metrics JSON, direction + scaler FLOAT[] (NULL for mlp), artifact_path, n_train/n_val, created_at. Upsert = replace; no FK (cleaned up manually in `delete_dataset`, like sae_document_activations).
- **`probe_scores`** — per-item probe scores: PK (collection_name, target_field, kind, item_id); score FLOAT, residual FLOAT NULL. `insert_probe_scores_bulk` deletes the key's rows first so retrains leave no strays; residual Nones kept as SQL NULL via pandas nullable Float64.
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
- **SentenceTransformer model cache is single-slot** — the class-level cache in `embed_sentence_transformer.py` keeps at most one model resident: requesting a *different* model evicts the cached one first (`_evict_cached_models()`: clear dict → `gc.collect()` → `torch.mps/cuda.empty_cache()`), so alternating between two collections with different models reloads on every switch. Repeat use of the same model still hits the cache. Instances created before an eviction keep their own `self._model` reference and continue to work. Guarded by `unit_tests/test_sentence_transformer_cache.py` (stubbed `sentence_transformers`, no downloads).
- **The `for_query` flag on `get_collection()`** affects QWEN (adds instruction prefix) and Gemini (maps RETRIEVAL_DOCUMENT → RETRIEVAL_QUERY)
- **Torch-free import boundary** — nothing importable from `backend.main` (the FastAPI app + GraphQL schema) may transitively import torch or the `interpret/` toolkit at module level; this keeps startup RSS small and torch resident only once an inference feature is used (linux images now install CPU-only torch via the `[tool.uv.sources]` pin, so the package is on disk but must stay out of the import graph). Heavy imports live behind lazy boundaries: `API/interpret_instance.get_interpret_service()` (torch + interpret toolkit) and the deferred `probing_service` import inside the `train_probe` mutation. Torch-free shared types live in `services/steering_types.py` (`SteeringSpec`) and `services/probing_types.py` (`ProbeConfig`, `PROBE_KINDS`, `score_field_names`) — the API layer imports from those, never from the heavy service modules. The `sys.path` bootstrap for `interpret.*` absolute imports lives in `services/__init__.py` (runs before any service module, so import order no longer matters). Guarded by `unit_tests/test_torch_free_import.py` (subprocess check that importing the schema loads no torch/`interpret.*` — nor hdbscan/sklearn/scipy/umap, see next bullet).
- **Clustering stack is also import-deferred** — `cluster_and_label.py` pulls in hdbscan → sklearn → scipy (~94 MB RSS, measured), so it must stay out of the `backend.main` import graph too (startup RSS ~180 MB vs ~250 MB eager). Two lazy boundaries enforce this: `topic_extraction_service.extract_topics()` imports `GenerateTopics` locally (mirroring the existing `umap` and `ClassTfidfTransformer` local imports), and `topic_extraction/__init__.py` re-exports `GenerateTopics`/`ClassTfidfTransformer` via a PEP 562 `__getattr__` instead of an eager import (the eager re-export was silently re-triggering the whole chain via the package `__init__` whenever `llm_labeling` was imported). Guarded by the same `unit_tests/test_torch_free_import.py` subprocess check.
