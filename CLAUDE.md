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

**Embedding providers**: SentenceTransformers (default, local), OpenAI, Cohere, Ollama (local), HuggingFace API, Gemini, QWEN (local), BGE (local). Model names are free-form; dimensions auto-detected. Hardware auto-detects MPS → CUDA → CPU.

**SAE (Sparse Autoencoder) feature storage**: Two dedicated DuckDB tables (`sae_features`, `sae_activations`) store Neuronpedia-style feature data with composite key `(model_id, sae_id, feature_index)`. Feature metadata includes density, explanation label, and top/bottom logit arrays (JSON). Activations store 512-token context windows with per-token activation values. Ingestion pipeline (`ingest_sae.py`) loads parquet features and JSONL activations. Explanation-embedding vectors (2560-d) optionally stored in ChromaDB for semantic feature search. GraphQL queries: `saeModels`, `saeFeature`, `saeActivations`, `saeFeatureSearch`; mutations: `ingestSaeFeatures`, `ingestSaeActivations`. Frontend feature explorer at `/features` with token-strip heatmaps, logit bar charts, and right-click cross-linking from the scatter plot. Collection ↔ SAE mapping defined in `lib/utils/saeCollections.ts`. See `documentation/SAE_ARCHITECTURE.md` for full schema, data flow, and component reference.

**Topic extraction**: HDBSCAN on projections → c-TF-IDF keywords → optional LLM labels (Gemini/OpenAI). Supports reduction via `fixed_n` or `auto` methods. Noise cluster (topic_id=-1) is never merged. When topics are reduced, original topics are preserved as subtopics (`subtopic_id`/`subtopic_label` per item, `topic_hierarchy` at collection level, `subtopics` list on each `TopicInfo`). **Standalone LLM labeling**: `generateLlmLabels` mutation upgrades keyword-pattern labels to LLM-generated ones with incremental saves after each topic, resume support (detects already-labeled via regex), `label_scope` ("both"/"topics_only"/"subtopics_only"), and `preserve_ctfidf_labels` (default true) which saves original keyword labels as `ctfidf_label` in `topic_summary` entries and `ctfidf_subtopic_map` (JSON dict) in collection metadata. Frontend supports **nested color mode**: Tableau-style hierarchical coloring where topics define base hues and subtopics get lightness variations within that hue.

**Real-time progress**: WebSocket subscriptions via progress event bus. Job state persisted to `resources/job_state.json`; interrupted jobs resumable with `resume: true`. Frontend `ProgressModal` (reusable, configurable, with ETA calculation) shows real-time progress for embedding, standalone topic extraction (`{collection}` job ID), topic reduction (`{collection}_reduce` job ID), LLM labeling (`{collection}_llm_labeling` job ID), and per-projection computation (25% increments). `generate_llm_labels()` accepts an optional `progress_callback(done, total)` so callers (`extract_topics`, `reduce_existing_topics`) can emit per-topic LLM labeling progress. `JobsPanel` handles both embedding and `llm_labeling` job types.

**Frontend state management**: Visualization state is managed by a Zustand store (`lib/stores/useVisualizationStore.ts`) with `subscribeWithSelector` middleware. Components read state via selectors (e.g. `useVisualizationStore((s) => s.colorByField)`) for granular re-renders — no prop drilling. The store auto-resets `mutedCategories` + `hideUnclustered` when `colorByField` changes (via subscription, not React effect). Color scale configuration uses a **discriminated union** `ColorScale` (`types.ts`): `{ type: 'categorical' } | { type: 'sequential'; scaleName } | { type: 'diverging'; scaleName } | { type: 'monochrome'; baseColor }`. `categoricalPalette` stays separate from the union because chart components (CategoryBarChart, TemporalChart) need it regardless of the active scale type. `defaultColorScaleForType()` converts a `ColorScaleType` string to a `ColorScale` with sensible defaults.

**Nebula cluster effects** (3D only): Boolean toggle (`nebulaMode`) adds translucent haze sprites around topic clusters. Uses `HazeRenderer` (`lib/utils/hazeRenderer.ts`) on a separate overlay canvas with `mix-blend-mode: screen`, hooked into Plotly's GL render loop for camera-synced projection. Cluster geometry computed in `lib/utils/clusterGeometry.ts`. Deprecated renderers (volume isosurface, WebGL billboard sprites, Three.js bloom) archived in `lib/utils/experimental/`.

**Label collision avoidance** (3D only): On each rAF-polled camera change, projects 3D label positions to 2D screen space via MVP matrices from Plotly's WebGL internals, measures text bounding boxes on a canvas overlay, then uses `CollisionGrid` (`lib/utils/collisionGrid.ts`) with a two-pass insertion (thin probe + full box) for greedy set packing. Higher-similarity labels survive; selected point gets infinite priority. **Cluster labels** (`showClusterLabels` on `VisualizationState`): displays topic/subtopic names at cluster centroids with highest collision priority (bold 13px, cluster color). Cluster labels share the same `CollisionGrid` as point labels so they never overlap. Cluster labels are **clickable** (3D only) — hit-test via bounding boxes stored during `renderLabels()`, `onClickCapture` intercepts before Plotly. Utility code in `app/utils/labelPlacement.ts`.

**Resizable Legend** (`lib/hooks/useVerticalResize.ts`): Legend card has a drag handle at its bottom edge controlling `maxHeight` via mouse/touch events. Dragging below the collapse threshold collapses to a Palette pill button; clicking re-expands. Uses callback ref pattern for conditionally rendered elements.

**Synchronized topic selection**: When `colorByField === 'topic_label'`, `selectedTopicIds` (from `useTopicSearch`) is the single source of truth for topic selection state. `DashboardPanel` derives `effectiveMutedCategories` via `useMemo` — no state writes, single render cycle. Legend clicks, TopicSearchSection toggles, and 3D cluster label clicks all feed into `toggleTopic()`. Non-topic color fields preserve the existing `mutedCategories` behavior unchanged.

**Temporal filtering** (Analytics sidebar): Auto-detects year/date fields via name heuristics and numeric range analysis (`lib/utils/temporalAnalysis.ts`). Shows temporal distribution chart in standalone mode (simple area chart) or stacked mode (category breakdown). Custom DOM overlay range picker with draggable handle bars and a slidable selected region filters points by time — points outside the range are muted (gray, low opacity) via trace-splitting in both scatter plots. Double-click resets to full range. Uses `startTransition` + `useDeferredValue` for non-blocking interaction. `TemporalRange` state on `VisualizationState` persists across color field changes but clears on collection change. Shared utilities in `lib/utils/temporalFilters.ts`: `buildTemporalFilterInputs` (backend FilterInput[]) and `isInTemporalRange` (client-side per-point predicate). Both semantic search and text search results are filtered by the active temporal range.

**Analytics bar chart** (Analytics sidebar): `CategoryBarChart` shows category distribution with independent field selection via dropdown (categorical fields from `colorFieldOptions`), defaulting to the active color field. When a text search is active, bars show search match counts overlaid on faded total-count background bars, sorted by matches descending. Category name filter input appears when there are more than 15 categories. `AnalyticsSidebar` computes `searchMatchCounts` from `textSearchHighlights` and manages `analysisField` state independently of `colorByField`.

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
- **Color column preprocessing**: When a dataset has a `colour_code` (or similar) hex column, embedding auto-detects it and adds `mapped_colour` (float 0-1) + `mapped_colour_scale` (strip name) to metadata. The float maps to a pre-built colorscale strip (Hilbert RGB, hue-sat, XKCD, or rainbow). Strips generated by `interpretability_backend/scripts/generate_color_strips.py` as Crameri-format JSONs in `embedding_visualization/lib/colorMaps/colormaps/`. Frontend treats `mapped_colour` as a standard numeric field with the matching colorscale.

## Pages

- `/` — Visualization dashboard (2D/3D scatter plots, semantic search, topic search/filter)
- `/features` — SAE Feature Explorer (token-strip heatmaps, logit charts, feature search, cross-linked from scatter plot via right-click)
- `/test-embed` — Dataset embedding interface (HuggingFace, local files, collection management, topic extraction)
