# CLAUDE.md

## RULES

- Always update the CLAUDE.md both in the main project and in the frontend or backend folder after significant updates. For minor updates or significant refactors, detail them in .md files in the documentation/ folder, and write in CLAUDE.md which documentation files contains the documentation. 

- Use **uv run python** when you need to launch python. 

- Avoid code duplication whenever possible. Employ a modular approach using classes rather than standalone functions. If you find dysfunctional pattern or duplication in existing code, allert the user directly, before attempting to fix them. 

- Prefer reusable utility functions inside the utils folder rather than stand alone calculation functions.

- Prefer configuration files to command line interfaces. 

- If some of the istructions are unclear or you encounter unexpected roadblocks, alert the user and ask for clarification, rather than writing code that was not agreed upon. 

- If you make a plan, always define and plan tests first, then run the code against those tests after. 

- Never stash changes without being directly asked. 

- For python, never import modules inside function unless strictly necessary. Use imports at the top. 

- For folders with multiple scripts or data files, add a readme explaining both the structure of the folder, the main classes or data structures present there. 

- After finishing a plan, always use the agent: *code-quality-reviewer* to review the quality of the generated code. 


## Project Overview

Embedding analysis platform: embed data from any source (HuggingFace datasets, local files, images, pre-computed vectors), visualize interactively with topic extraction and semantic search. Uses a **dual-database architecture**: DuckDB as the central orchestrator (documents, metadata, projections, topics) and ChromaDB for dense vector storage and similarity search only.

## Directory Structure

- **`interpretability_backend/`** — Python backend (FastAPI + Strawberry GraphQL). Has its own `CLAUDE.md`.
  - `backend/API/` — GraphQL queries, mutations, subscriptions, types
  - `backend/clients/` — DuckDB (orchestrator), ChromaDB (vectors only), HuggingFace, local file clients
  - `backend/embedding_functions/` — Multi-provider embedding infrastructure
  - `backend/services/` — Topic extraction, progress emission, job state
  - `backend/topic_extraction/` — HDBSCAN clustering, c-TF-IDF, LLM labeling, reduction
  - `interpretability_experiments/WordNet/` — WordNet embedding pipeline (153k words)
- **`embedding_visualization/`** — Next.js 15 frontend. Has its own `CLAUDE.md`.
- **`embedding-atlas/`** — Reference: Apple's embedding viz framework
- **`tensorboard/`** — Reference: TensorFlow embedding projector

## Essential Commands

```bash
# Backend
./start_backend.sh
# or: uv run uvicorn interpretability_backend.backend.main:app --host 0.0.0.0 --port 8000 --reload
# GraphQL Playground: http://localhost:8000/graphql

# Frontend
cd embedding_visualization && npm run dev  # http://localhost:3000

# Dependencies
uv sync                                    # Backend (Python)
cd embedding_visualization && npm install  # Frontend

# WordNet (one-time setup, ~8 min)
cd interpretability_backend/interpretability_experiments/WordNet && python embed_wordnet.py
```

## Environment Variables

- `GEMINI_API_KEY` — Gemini embedding & LLM topic labeling
- `CHROMA_OPENAI_API_KEY` — OpenAI embedding & LLM topic labeling
- `CHROMA_COHERE_API_KEY` — Cohere embedding
- `CHROMA_HUGGINGFACE_API_KEY` — HuggingFace API embedding
- `HUGGINGFACE_API_KEY` — HuggingFace model access (gated models)

## Architecture Overview

```
Data Sources → Embedding Providers → DuckDB (docs/metadata) + ChromaDB (vectors) → GraphQL API → Frontend
                                          ↓
                                   Topic Extraction (reads projections from DuckDB, embeddings from ChromaDB)
```

**Dual-database design**: DuckDB (`resources/main.duckdb`) is the central orchestrator storing documents, metadata, projections (native FLOAT[] arrays), and topic data (normalized tables). ChromaDB (`resources/vector_db/`) stores only IDs + dense embedding vectors. One dataset in DuckDB can have multiple `vector_collections` (different embedding models, dense/sparse). Items are stored in **per-dataset tables** (`items_{name}`) for isolation and per-dataset FTS indexes. See `documentation/DATABASE_ARCHITECTURE.md` for full schema, API reference, and data flow diagrams.

**Seed dataset (ships in git)**: The live data stores are gitignored, so a fresh clone would be empty. A small (~23 MB) committed seed snapshot at `interpretability_backend/resources/seed/` (`main.duckdb` + `vector_db/`) holds two demo collections — `emotion` (1000 rows, MiniLM, Gemini topic labels; semantic search works offline) and `xkcd_hilbert_gemini` (954 rows, Gemini-embedded XKCD colors with rainbow `mapped_colour` coloring). On first backend startup, `backend/utils/seed_bootstrap.py::ensure_seed_loaded()` (called from the FastAPI `lifespan` in `main.py`) copies the seed into the live paths **only if `main.duckdb` is absent** — it never clobbers an existing DB. The seed is regenerated from the production stores (backend stopped) via `scripts/build_seed_snapshot.py`. `.gitignore` ignores the live stores but un-ignores `resources/seed/`. Note: live semantic search on the XKCD collection needs `GEMINI_API_KEY` (its vectors are Gemini); viewing/coloring/topics work with no model since projections are precomputed.

**Embedding providers**: SentenceTransformers (default, local), OpenAI, Cohere, Ollama (local), HuggingFace API, Gemini, QWEN (local), BGE (local). Model names are free-form; dimensions auto-detected. Hardware auto-detects MPS → CUDA → CPU.

**SAE (Sparse Autoencoder) feature storage**: Two dedicated DuckDB tables (`sae_features`, `sae_activations`) store Neuronpedia-style feature data with composite key `(model_id, sae_id, feature_index)`. Feature metadata includes density, explanation label, and top/bottom logit arrays (JSON). Activations store 512-token context windows with per-token activation values. Ingestion pipeline (`ingest_sae.py`) loads parquet features and JSONL activations. Explanation-embedding vectors (2560-d) optionally stored in ChromaDB for semantic feature search. GraphQL queries: `saeModels`, `saeFeature`, `saeActivations`, `saeFeatureSearch`; mutations: `ingestSaeFeatures`, `ingestSaeActivations`, `prepareSaeData` (on-demand: downloads from Neuronpedia S3, extracts decoder vectors, and ingests into DuckDB in one call). Frontend feature explorer at `/sae` with token-strip heatmaps, logit bar charts, and right-click cross-linking from the scatter plot. Collection ↔ SAE mapping defined in `lib/utils/saeCollections.ts`. Source string derivation centralized in `interpret/sae/source_ids.py`; all file paths derived from config via `interpret/sae/paths.py`. Standalone pipeline: `interpret/sae/pipeline/prepare_sae_data.py` (`SAEPipelineConfig`/`SAEPipelineRunner`). See `documentation/SAE_ARCHITECTURE.md` for storage schema, `documentation/SAE_PIPELINE.md` for the full download-to-ingestion pipeline.

**SAE document activations** (`sae_document_activations` table): Per-document max-pooled SAE activation vectors stored as sparse rows `(collection_name, item_id, feature_index, activation)` — only nonzero entries (~100-150 per document). Enables **two-hop feature-label search**: user queries a label (e.g., "poetry") → ILIKE matches SAE features → documents ranked by MAX(activation) across matching features. Mutation: `computeDocumentActivations` (batch inference with progress/resume, holds GPU lock for full batch). Query: `searchDocumentsByFeatures`, `hasDocumentActivations`. Service method: `InterpretService.run_batch_highlight()`. Frontend spec in plan file `vast-questing-umbrella.md` (Phases 4-5).

**Steering presets** (`embedding_visualization/lib/utils/steeringPresets.ts` + `backend/services/interpret_service.py::DIRECTION_REGISTRY`): On chat-panel mount the frontend auto-loads a model-specific bundle of steering tools when the config is empty. For `gemma-3-4b-it`: three named SAE features (layer 9, 16k residual: 197 "Religion & spirituality", 3289 "Poetry", 4963 "Sexually explicit") plus one pre-extracted activation direction (`refusal`) stored as a 1-D `.pt` vector in `interpretability_backend/resources/directions/`. (The `poetry` direction vector and its `DIRECTION_REGISTRY` entry still exist server-side but are no longer part of the auto-loaded frontend bundle.) All presets ship at strength 0 (inert); strength-0 specs are filtered before the GraphQL call so no backend SAE loads are wasted. Direction presets resolve server-side via `DIRECTION_REGISTRY` (layer + file baked in) and use raw `vector` on `SteeringOp` instead of an SAE `feature_index`. Direction-row UI uses a ±5 / 0.1-step slider (vs ±2000 / 50 for SAE features). The one-time slice from the multi-candidate refusal tensor to a single 1-D vector is handled by `interpretability_backend/scripts/extract_direction_vectors.py`.

**SAE inference service** (`backend/services/interpret_service.py`): Wraps the `interpret/` toolkit for live inference via GraphQL. `InterpretService` manages Gemma3-4b-it model lifecycle (load on demand, stay resident, explicit unload via `unloadModel` mutation, `asyncio.Lock` for serialised GPU access). Four use cases: (1) `runPromptActivations` — runs a prompt through the model with SAE hooks at specified layers, returns per-token top-k feature activations with Neuronpedia labels via `PromptExplorer`; (2) `generateSteeredResponse` — applies additive steering on `w_dec[feature_index]` at a given strength, returns baseline + steered text; (3) `runPromptHighlight` — runs a prompt, max-pools SAE activations across tokens, returns nonzero `(featureIndex, activation)` pairs for scatter plot highlighting; (4) `generateStream` — **streaming chat generation** via WebSocket subscription, yields tokens one at a time for a chatbot interface with optional SAE steering and model-level abort support. An optional `GenerateStreamInput.seed` calls `torch.manual_seed()` immediately before sampling (covers both "it"/"pt" paths), making generation reproducible; because the GPU lock serialises streams, a steered and a baseline call sharing one seed each get an identical fresh RNG state — this backs the frontend chat **compare mode** (steered vs. baseline threads side-by-side under one seed). Query: `modelStatus` (returns `variant`/"it"|"pt" and `modelSize`); mutations: `loadModel`, `unloadModel`, `runPromptActivations` (supports `skipChatTemplate` flag for raw-token activations), `generateSteeredResponse`, `runPromptHighlight`; subscription: `generateStream` (base "pt" models use non-streaming fallback via `generate_from_template`). Token streaming architecture: `Gemma3ForMultimodalLM.generate_stream()` (sync generator in forked gemma_pytorch) → `GemmaPytorchInference.generate_chat_stream()` (delta-decodes via full-sequence SentencePiece diff) → `InterpretService.generate_stream()` (emits `TokenEvent`s via `token_emitter.py` event bus) → GraphQL subscription reads from `asyncio.Queue`. Abort: `threading.Event` checked per-token in the decode loop; set on client disconnect or timeout. Gemma-only for now; architecture supports future Qwen addition. **SAE weight cache**: `interpret/sae/loading.py` keeps a module-level `_SAE_CACHE` keyed by config identity (`layer_index`, `hook_type`, `width`, `model_size`, `variant`, `l0_size`, `dtype`, `device`) so repeat `load_sae()` calls return the already-resident MPS tensors instead of re-reading the ~320 MB `params.safetensors` from disk. Cleared by `clear_sae_cache()`, which `InterpretService.unload_model()` calls alongside `_direction_cache.clear()`. Matters most for chat TTFT when `HF_HOME` lives on slow storage (e.g. exFAT external drive). See `documentation/INTERPRET_API.md` for full frontend integration guide.

**Topic extraction**: HDBSCAN on projections → c-TF-IDF keywords → optional LLM labels (Gemini/OpenAI). `TopicExtractionConfig.cluster_on` selects the clustering space and is exposed end-to-end (GraphQL `TopicConfigInput` → converter → collections-page `TopicConfigForm`): `"cluster_umap"` (**default**) runs a fresh BERTopic-style UMAP (n_components=5, min_dist=0, cosine; configurable via `cluster_n_components`/`cluster_min_dist`/`cluster_n_neighbors`) on the raw vectors before HDBSCAN; `"projection"` clusters on the stored viz coords (fast); `"embedding"` clusters on the L2-normalised original vectors. The two raw-vector modes share `load_embeddings_for_ids` and degrade to projection coords if vectors can't be loaded; `cluster_on` + the UMAP params are recorded in the `topic_extractions.config` snapshot. Supports reduction via `fixed_n` or `auto` methods. Noise cluster (topic_id=-1) is never merged. When topics are reduced, original topics are preserved as subtopics (`subtopic_id`/`subtopic_label` per item, `topic_hierarchy` at collection level, `subtopics` list on each `TopicInfo`). **Standalone LLM labeling**: `generateLlmLabels` mutation upgrades keyword-pattern labels to LLM-generated ones with incremental saves after each topic, resume support (detects already-labeled via regex), `label_scope` ("both"/"topics_only"/"subtopics_only"), and `preserve_ctfidf_labels` (default true) which saves original keyword labels as `ctfidf_label` in `topic_summary` entries and `ctfidf_subtopic_map` (JSON dict) in collection metadata. Frontend supports **nested color mode**: Tableau-style hierarchical coloring where topics define base hues and subtopics get lightness variations within that hue. **Topic-quality evaluation**: standalone `interpretability_backend/evaluation/` package (`TopicQualityEvaluator` + config-driven `run_evaluation.py`) scores extractions with DBCV, silhouette in both embedding and projection space, topic diversity, and C_v/U_Mass coherence (gensim); supports `level="topic"|"subtopic"`. Not persisted to DuckDB / exposed over GraphQL yet (deferred). See `interpretability_backend/evaluation/README.md`.

**Projection-fidelity evaluation** (`interpretability_backend/evaluation/projection_fidelity.py` + `run_projection_fidelity.py`): standalone Mantel-test evaluator scoring how faithfully a projection (UMAP-3D, PCA-3D, …) preserves a reference distance structure — the original **embedding** geometry (cosine) and, for colour datasets, **perceptual colour** distance (CIEDE2000). Three statistics via the merged `interpret.utils.mantel.MantelTest`: global Spearman ρ, kNN-local ρ (neighbourhoods defined in the reference space), and a permutation z/p_emp significance test. `ProjectionFidelityEvaluator` is pure (condensed distance vectors → results dict, degrades to `None`, never raises); colour distances lazily import scikit-image so only the colour reference needs it. Config-driven runner (`projection_fidelity_config.toml`) reads projections + colour metadata from DuckDB and embeddings from ChromaDB. On `xkcd_hilbert_gemini`: UMAP-3D preserves perceptual colour (ρ=0.60) + local neighbourhoods better, PCA-3D preserves global embedding geometry better. Same deferred-persistence stance as topic-quality. See `documentation/PROJECTION_FIDELITY.md`.

**Real-time progress**: WebSocket subscriptions via progress event bus. Job state persisted to `resources/job_state.json`; interrupted jobs resumable with `resume: true`. Frontend `ProgressModal` (reusable, configurable, with ETA calculation) shows real-time progress for embedding, standalone topic extraction (`{collection}` job ID), topic reduction (`{collection}_reduce` job ID), LLM labeling (`{collection}_llm_labeling` job ID), and per-projection computation (25% increments). `generate_llm_labels()` accepts an optional `progress_callback(done, total)` so callers (`extract_topics`, `reduce_existing_topics`) can emit per-topic LLM labeling progress. `JobsPanel` handles both embedding and `llm_labeling` job types.

**Frontend state management**: Visualization state is managed by a Zustand store (`lib/stores/useVisualizationStore.ts`) with `subscribeWithSelector` middleware. Components read state via selectors (e.g. `useVisualizationStore((s) => s.colorByField)`) for granular re-renders — no prop drilling. The store auto-resets `mutedCategories` + `hideUnclustered` when `colorByField` changes (via subscription, not React effect). Color scale configuration uses a **discriminated union** `ColorScale` (`types.ts`): `{ type: 'categorical' } | { type: 'sequential'; scaleName } | { type: 'diverging'; scaleName } | { type: 'monochrome'; baseColor }`. `categoricalPalette` stays separate from the union because chart components (CategoryBarChart, TemporalChart) need it regardless of the active scale type. `defaultColorScaleForType()` converts a `ColorScaleType` string to a `ColorScale` with sensible defaults. **Model identity store** (`lib/stores/useModelIdentityStore.ts`): Single source of truth for SAE model identity on the features page. Holds `modelId`, `saeId`, cached `parsedSae` (auto-derived from `saeId`), `checkpoint` (auto-derived from `modelId`), backend model status (`backendLoaded`, `backendVariant`, `backendModelSize`), and `steeringConfig`. Bridged from `useSaeSelectors` output via effect in `page.tsx`. Chat components (`ChatPanel`, `ChatInput`, `SteeringControls`, `ModelStatusButton`) read directly from the store — no prop drilling of model identity. `steeringFeatureKey()` utility lives in the store module.

**Nebula cluster effects** (3D only): Boolean toggle (`nebulaMode`) adds translucent haze sprites around topic clusters. Uses `HazeRenderer` (`lib/utils/hazeRenderer.ts`) on a separate overlay canvas with `mix-blend-mode: screen`, hooked into Plotly's GL render loop for camera-synced projection. Cluster geometry computed in `lib/utils/clusterGeometry.ts`. Deprecated renderers (volume isosurface, WebGL billboard sprites, Three.js bloom) archived in `lib/utils/experimental/`.

**Label collision avoidance** (3D only): On each rAF-polled camera change, projects 3D label positions to 2D screen space via MVP matrices from Plotly's WebGL internals, measures text bounding boxes on a canvas overlay, then uses `CollisionGrid` (`lib/utils/collisionGrid.ts`) with a two-pass insertion (thin probe + full box) for greedy set packing. Higher-similarity labels survive; selected point gets infinite priority. **Cluster labels** (`showClusterLabels` on `VisualizationState`): displays topic/subtopic names at cluster centroids with highest collision priority (bold 13px, cluster color). Cluster labels share the same `CollisionGrid` as point labels so they never overlap. Cluster labels are **clickable** (3D only) — hit-test via bounding boxes stored during `renderLabels()`, `onClickCapture` intercepts before Plotly. Utility code in `app/utils/labelPlacement.ts`.

**Resizable Legend** (`lib/hooks/useVerticalResize.ts`): Legend card has a drag handle at its bottom edge controlling `maxHeight` via mouse/touch events. Dragging below the collapse threshold collapses to a Palette pill button; clicking re-expands. Uses callback ref pattern for conditionally rendered elements.

**Synchronized topic selection**: When `colorByField === 'topic_label'`, `selectedTopicIds` (from `useTopicSearch`) is the single source of truth for topic selection state. `DashboardPanel` derives `effectiveMutedCategories` via `useMemo` — no state writes, single render cycle. Legend clicks, TopicSearchSection toggles, and 3D cluster label clicks all feed into `toggleTopic()`. Non-topic color fields preserve the existing `mutedCategories` behavior unchanged.

**Temporal filtering** (Analytics sidebar): Auto-detects year/date fields via name heuristics and numeric range analysis (`lib/utils/temporalAnalysis.ts`). Shows temporal distribution chart in standalone mode (simple area chart) or stacked mode (category breakdown). Custom DOM overlay range picker with draggable handle bars and a slidable selected region filters points by time — points outside the range are muted (gray, low opacity) via trace-splitting in both scatter plots. Double-click resets to full range. Uses `startTransition` + `useDeferredValue` for non-blocking interaction. `TemporalRange` state on `VisualizationState` persists across color field changes but clears on collection change. Shared utilities in `lib/utils/temporalFilters.ts`: `buildTemporalFilterInputs` (backend FilterInput[]) and `isInTemporalRange` (client-side per-point predicate). Both semantic search and text search results are filtered by the active temporal range.

**Analytics category list** (Analytics sidebar): `CategoryBarList` shows category distribution as horizontal background-fill rows (translucent category-color bar behind label + count, normalized to the max visible category) with independent field selection via dropdown (categorical fields from `colorFieldOptions`), defaulting to the active color field. All categories render in a 320px scrollable list (no top-N cap). When a text-search or temporal filter is active, rows show a faded total track + solid matching fill and counts flip to "matches / total". Sort toggle: Count (default; matches-desc while filtered) / Rate (match rate, offered only while filtered) / Name (numeric-aware natural order). Category name filter input appears when there are more than 15 categories. Pure row/sort/summary logic lives in `lib/utils/categoryRowData.ts` (unit-tested). `AnalyticsSidebar` computes filtered counts from `combinedMutedIndices` (text search + temporal) and manages `analysisField` state independently of `colorByField`. The superseded recharts `CategoryBarChart.tsx` is retained temporarily pending verification of the swap.

**Text search as filter**: Text search is **server-side** via the `textSearch` GraphQL query backed by DuckDB's `ILIKE` for document/metadata substring search (vectorized columnar scan). FTS extension available for BM25 word-level search. The `useTextSearch` hook (`lib/hooks/useTextSearch.ts`) fires the query and maps returned IDs to point indices. `SearchSidebar` exposes field selection (which metadata columns to search), match mode (contains/exact), and case-sensitivity controls in the Advanced section. Default: document-only, contains, case-insensitive. Text search muting (`searchMutedIndices`) and temporal muting (`temporallyMutedIndices`) are combined into `combinedMutedIndices` in both scatter plots. Semantic search results retain the glow overlay. Both muting sources stack: a point must match the text query AND be in the temporal range to render normally.

**Filtered point controls**: `hideFilteredPoints` (boolean) removes muted points entirely from the plot instead of graying them out. `mutedPointOpacity` (0–1, default 0.20) is a **multiplier** applied to the current base opacity (not an absolute value) — muted opacity = base opacity × multiplier. This scales properly with point count since base opacity already decreases for large datasets. Both fields live on the Zustand store and are exposed via toggle + slider in VisualizationControls. Reset on collection change.

**Zoom-out limit** (`lib/hooks/useZoomLimit.ts`): Shared hook that intercepts wheel events in the capture phase before Plotly processes them. Each scatter plot provides an `isAtZoomOutLimit` callback: 3D reads live camera distance from `glplot.camera.eye` (max distance 3.0), 2D reads axis ranges from `_fullLayout.xaxis.range` (max 2x data extent). Scroll-zoom-in, pan, and rotation are unaffected.

**Deferred selected point** (3D only): On point click, `selectedPoint` drives the camera fly-to animation via direct glplot manipulation (no Plotly.react). The trace-visible `renderedSelectedPoint` only syncs when `highlightedIndices` (search results) changes, keeping `plotData` stable during animation and avoiding expensive main-thread stalls on large datasets.

## Cross-Cutting Concerns

- **DuckDB path**: `interpretability_backend/resources/main.duckdb` (documents, metadata, projections, topics)
- **ChromaDB path**: `interpretability_backend/resources/vector_db/` (dense vectors only)
- **Similarity metric**: ChromaDB uses cosine distance; similarity = 1 - distance
- **Batch processing**: Embedding pipelines use configurable batch sizes. DuckDB bulk inserts via pandas DataFrames. ChromaDB reads embeddings in 5k batches for projection computation.
- **DuckDB metadata**: JSON column with no type restrictions (native lists, dicts, nulls). Replaces ChromaDB's str/int/float/bool limitation.
- **One dataset, many embeddings**: `vector_collections` table links one dataset to multiple ChromaDB collections (different models). Each has independent projections and topic extractions.
- **WordNet XML** (102MB) downloaded by `embed_wordnet.py`, not in repo
- **Color scheme persistence**: The active colouring (field + `ColorScale` + categorical palette) round-trips through the URL query string, and each collection can store a **default colour scheme** in `datasets.extra_metadata.default_color_scheme` (set via the dashboard "Save as default" button → `updateCollectionMetadata`; merged/spread by the existing DuckDB plumbing, no backend changes). Applied on collection load with precedence **URL > collection default > none**. Frontend helpers in `embedding_visualization/lib/utils/colorScaleUrl.ts`; see frontend CLAUDE.md "Color scheme persistence".
- **Color column preprocessing**: When a dataset has a `colour_code` (or similar) hex column, embedding auto-detects it and adds `mapped_colour` (float 0-1) + `mapped_colour_scale` (strip name) to metadata. The float maps to a pre-built colorscale strip (Hilbert RGB, hue-sat, XKCD, or rainbow). Strips generated by `interpretability_backend/scripts/generate_color_strips.py` as Crameri-format JSONs in `embedding_visualization/lib/colorMaps/colormaps/`. Frontend treats `mapped_colour` as a standard numeric field with the matching colorscale.

## Pages

Cross-page navigation uses a shared pill tab bar (`PageNav` → `PillNav` primitive, see frontend CLAUDE.md). Old routes `/features` and `/test-embed` permanently redirect to `/sae` and `/collections` (next.config.ts).

- `/` — Visualization dashboard (2D/3D scatter plots, semantic search, topic search/filter); nav label "Explore"
- `/sae` — SAE Feature Explorer (token-strip heatmaps, logit charts, feature search, cross-linked from scatter plot via right-click)
- `/collections` — Dataset embedding interface (HuggingFace, local files, collection management, topic extraction)
