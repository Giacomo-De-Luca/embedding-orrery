# CLAUDE.md

## RULES

- Always update the CLAUDE.md both in the main project and in the frontend or backend folder after significant updates. For minor updates or significant refactors, detail them in .md files in the documentation/ folder, and write in CLAUDE.md which documentation files contains the documentation. 

- Use **uv run python** when you need to launch python. 

- Avoid code duplication whenever possible. Employ a modular approach using classes rather than standalone functions. If you find dysfunctional pattern or duplication in existing code, allert the user directly, before attempting to fix them. 

- Prefer reusable utility functions inside the utils folder rather than stand alone calculation functions.

- Prefer configuration files to command line interfaces. 

- If some of the istructions are unclear or you encounter unexpected roadblocks, alert the user and ask for clarification, rather than writing code that was not agreed upon. 

- If you make a plan, always define and plan tests first, then run the code against those tests after. 

- Never commit or stash changes without being directly asked or explicit approval.

- For python, never import modules inside function unless strictly necessary. Use imports at the top. 

- For folders with multiple scripts or data files, add a readme explaining both the structure of the folder, the main classes or data structures present there. 

- After finishing a plan, always use the agent: *code-quality-reviewer* to review the quality of the generated code. 


## Project Overview

Embedding analysis platform: embed data from any source (HuggingFace datasets, local files, images, pre-computed vectors), visualize interactively with topic extraction and semantic search, and interpret with SAE features, steering, and probing. Uses a **dual-database architecture**: DuckDB as the central orchestrator (documents, metadata, projections, topics) and ChromaDB for dense vector storage and similarity search only.

```
Data Sources → Embedding Providers → DuckDB (docs/metadata) + ChromaDB (vectors) → GraphQL API → Frontend
                                          ↓
                                   Topic Extraction (reads projections from DuckDB, embeddings from ChromaDB)
```

## Directory Structure

- **`interpretability_backend/`** — Python backend (FastAPI + Strawberry GraphQL). **Has its own `CLAUDE.md`** with the module reference, data-storage schema, and gotchas.
  - `backend/` — GraphQL API, DB clients, embedding providers, services, topic extraction
  - `interpret/` — inference/interpretability toolkit (SAE loading, hooks, probing, steering)
  - `evaluation/`, `experiments/`, `interpretability_experiments/` — offline evaluation + experiments (each has a README). WordNet one-time setup (~8 min, downloads the 102 MB XML not in repo): `cd interpretability_backend/interpretability_experiments/WordNet && python embed_wordnet.py`
- **`embedding_visualization/`** — Next.js 16 frontend. **Has its own `claude.md`** with component architecture, state management, and all visualization feature details.
- **`documentation/`** — Canonical feature/architecture docs (referenced throughout this file).
- **`docs/`** — Nextra 4 documentation website (own `package.json`, excluded from Docker). `docs/scripts/sync-content.mjs` copies curated pages from `documentation/` at build time; synced output is gitignored. Gotcha: `zod` pinned to 4.1.12 via npm `overrides` (zod ≥4.2 breaks nextra-theme-docs 4.6). Run: `cd docs && npm install && npm run dev`.
- **`references/`** — Vendored reference code (SAEDashboard, neuronpedia, …) — not part of the app, exclude from reviews/refactors.
- **`benchmarks/fps/`** — Standalone Playwright FPS/RAM benchmark for the scatter plots. See its README.

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
```

## Environment Variables

- `GEMINI_API_KEY` — Gemini embedding & LLM topic labeling
- `CHROMA_OPENAI_API_KEY` — OpenAI embedding & LLM topic labeling
- `CHROMA_COHERE_API_KEY` — Cohere embedding
- `CHROMA_HUGGINGFACE_API_KEY` — HuggingFace API embedding
- `HUGGINGFACE_API_KEY` — HuggingFace model access (gated models)

## Subsystem Map

One-line summaries; follow the pointer for details before working on a subsystem.

| Subsystem | Summary | Details |
|---|---|---|
| Dual-database storage | DuckDB orchestrates documents/metadata/projections/topics (per-dataset `items_{name}` tables, JSON metadata); ChromaDB holds only IDs + dense vectors (cosine). One dataset ↔ many `vector_collections` (different models), each with independent projections/topics. | `documentation/DATABASE_ARCHITECTURE.md`, backend CLAUDE.md "Data Storage" |
| Embedding providers | SentenceTransformers (default), OpenAI, Cohere, Ollama, HuggingFace API, Gemini, QWEN, BGE. Free-form model names, auto-detected dimensions, MPS → CUDA → CPU. | backend CLAUDE.md "Embedding Functions" |
| Seed dataset | ~23 MB committed snapshot (`resources/seed/`) so a fresh clone isn't empty; rebuilt from JSON manifests with checksummed atomic replacement. | `documentation/SEED_SNAPSHOTS.md` |
| HF Space demo | Read-only public demo from the root `Dockerfile` (nginx → uvicorn + Next standalone); backend `ORRERY_READ_ONLY=1`, frontend `NEXT_PUBLIC_DEMO_MODE=1`, seed pinned by `demo.lock.json`, deployed via GitHub Actions. Onboarding: welcome dialog + `?preset=`/`?tour=` URL layer + react-joyride spotlight tour, with parent-page URL sync inside the Space iframe. | `documentation/HF_SPACE_DEMO.md` ("Onboarding"), `documentation/DOCKER.md`, frontend claude.md "Demo onboarding" |
| SAE feature storage | DuckDB tables `sae_features`/`sae_activations` keyed `(model_id, sae_id, feature_index)`; `prepareSaeData` mutation runs download → decoder extraction → ingestion (gemma via Neuronpedia, qwen label-free from HF weights). Explorer UI at `/sae`. | `documentation/SAE_ARCHITECTURE.md` (schema), `documentation/SAE_PIPELINE.md` (pipeline) |
| SAE document activations | Sparse per-document max-pooled activations (`sae_document_activations`) enabling feature-label → document search with selectable ranking modes; computed by `computeDocumentActivations`. | `documentation/DATABASE_ARCHITECTURE.md`, backend CLAUDE.md |
| SAE inference service | `InterpretService` wraps `interpret/` for live GraphQL inference: prompt activations (multi-SAE), steering, prompt highlight, streaming chat (WebSocket, seedable, abortable). Two model families: Gemma (forked gemma_pytorch) and Qwen (transformers). SAE weight cache in `interpret/sae/loading.py`. | `documentation/INTERPRET_API.md`, backend CLAUDE.md |
| Steering presets & strength hint | Frontend auto-loads model-specific steering bundles (SAE features + direction vectors from `DIRECTION_REGISTRY`); an offline residual-norm profiler powers a "% of residual norm" strength hint. | `documentation/STEERING_STRENGTH_HINT.md`, frontend claude.md "Steering presets" / "Steered chat" |
| Topic extraction | HDBSCAN → c-TF-IDF → optional LLM labels; `cluster_on` selects clustering space (`cluster_umap` default, `projection`, `embedding`); reduction preserves originals as subtopics; standalone `generateLlmLabels` with resume. | backend CLAUDE.md "Topic Extraction", `documentation/TOPIC_REDUCTION_FRONTEND_GUIDE.md` |
| Topic-quality evaluation | DBCV / silhouette / diversity / coherence metrics via `evaluateTopics`, persisted on `topic_extractions.quality_metrics`. | `interpretability_backend/evaluation/README.md`, `documentation/TOPIC_QUALITY_METRICS.md` |
| Projection fidelity | Mantel-test evaluator scoring how well projections preserve embedding geometry / perceptual colour distance. | `documentation/PROJECTION_FIDELITY.md` |
| Embedding-space probing | Train probes (ridge/massmean/mlp/…) on stored vectors vs numeric metadata fields server-side; scores merged client-side as extra color fields in the Analytics sidebar. | backend CLAUDE.md "Services", frontend claude.md "Direction probes" |
| Glasgow probing experiment | Offline YAML-driven experiment decoding nine psycholinguistic norms from two encoders. | `documentation/GLASGOW_PSYCHOLINGUISTIC_PROBING.md` |
| Real-time progress | WebSocket progress subscriptions; job state persisted to `resources/job_state.json`, interrupted jobs resumable; frontend `ProgressModal` + page-global `ActiveJobsStrip`. | backend CLAUDE.md "Services", frontend claude.md "Collections page" |
| Frontend visualization | Zustand store (persisted view prefs), 2D/3D Plotly scatter plots (forked plotly.js), nebula haze, label collision avoidance, screenshot export, temporal + text-search filtering, analytics category list, color-scheme persistence (URL > collection default), zoom limits, deferred selected point. | frontend claude.md "State Management" / "Key Patterns" / "Key Files" |

## Cross-Cutting Concerns

- **Data paths**: DuckDB at `interpretability_backend/resources/main.duckdb`; ChromaDB at `interpretability_backend/resources/vector_db/`. Similarity = 1 − cosine distance.
- **Torch-free import boundary**: the GraphQL schema (`backend.main`) must not transitively import torch or `interpret/` at module level — heavy imports are lazy; guarded by `unit_tests/test_torch_free_import.py`. Details in backend CLAUDE.md "Key Gotchas".
- **Color column preprocessing**: hex-colour metadata columns auto-map to `mapped_colour` (float 0-1) + a pre-built colorscale strip. See backend CLAUDE.md and frontend claude.md.

## Pages

Cross-page navigation uses a shared pill tab bar (`PageNav`; see frontend claude.md). Old routes `/features` and `/test-embed` permanently redirect.

- `/` — Visualization dashboard (2D/3D scatter plots, semantic search, topic search/filter); nav label "Explore"
- `/sae` — SAE Feature Explorer (token-strip heatmaps, logit charts, feature search, cross-linked from scatter plot via right-click)
- `/collections` — Dataset embedding interface (HuggingFace, local files, collection management, topic extraction)

## Known Issues

A full multi-agent audit (latent bugs, performance, duplication, UX, docs drift) is in `documentation/AUDIT_2026-07-02.md` — 231 findings, 117 adversarially verified. Highest-severity confirmed items include: filtered semantic search dropping matches (100k pre-filter cap), `deleteCollection` always reporting success while passing a collection name to `delete_dataset`, failed embed jobs marked completed (resume record deleted), semantic topic reduction reading `topic_id` from ChromaDB where it is never written, and sync GraphQL resolvers blocking the event loop.
