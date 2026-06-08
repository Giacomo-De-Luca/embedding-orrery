# Orrery: Interactive Embedding Analysis with Mechanistic Interpretability

Orrery is an open-source platform for embedding visualization, automatic topic extraction, and Sparse Autoencoder (SAE) interpretability. Embed data from any source, visualize up to 500k points at 60 fps, extract topics, and steer a language model live from the scatter plot. *Also perform SAE search on your dataset! Or visualise your data as a galaxy using the nebula mode!*

> **Beta** the platform is functional and under active development, and is going to be sent to EMNLP as demo. If you use the platform or find bugs write me, I'm thankful for any preliminar testing! 

![WordNet 212k](gallery/geometry.png)

## Gallery

| | |
|---|---|
| ![WordNet 212k](gallery/meditation.png) | ![HarmBench](gallery/harmbench.png) |
| WordNet 212k points with nebula cluster effects and semantic search | HarmBench with LLM-generated topic labels |
| ![XKCD Colors](gallery/Gemini_XKCD_PCA.png) | ![Concreteness](gallery/concreteness.jpg) |
| XKCD color words colored by actual hex values -- embedding space mirrors perceptual color space | NRC word norms colored by concreteness score -- psycholinguistic dimensions as spatial gradients |

## Quick Start

### Docker

```bash
docker compose up --build
# Frontend: http://localhost:3000 (ships with demo datasets)
# GraphQL: http://localhost:8000/graphql
```

### Manual

```bash
# Backend
uv sync
./start_backend.sh

# Frontend
cd embedding_visualization && npm install && npm run dev
```

See [`documentation/DOCKER.md`](documentation/DOCKER.md) for SAE cache warmup, volume management, and HuggingFace token options.



## What It Does

**Embedding Visualization** -- Embed from HuggingFace Hub, local files (CSV/JSON/Parquet), images, or pre-computed vectors. Interactive 2D/3D WebGL scatter plots. Eight providers: SentenceTransformers (local, default), Gemini, OpenAI, Cohere, Ollama, QWEN, BGE, HuggingFace API.

**Topic Extraction** -- HDBSCAN clustering on projections with c-TF-IDF keywords and optional LLM labels (Gemini/OpenAI). Hierarchical reduction preserves subtopics with nested coloring.

**Temporal Filtering** -- Auto-detects year/date fields. Draggable range picker filters the scatter plot, semantic search, and text search to a time window. Designed for diachronic analysis of historical corpora and literary collections.

**SAE Feature Analysis** -- Live inference on Gemma 3 with custom JumpReLU/TopK SAE implementations. Per-token activation capture, scatter plot highlighting, additive/ablation/orthogonal steering, and streaming chat. Visualize the SAE feature space as a 3D scatter plot -- right-click any feature to inspect it and steer the model.

**Feature-Grounded Search** -- Connect a dataset to an SAE, compute per-document activations, then search by feature label. Type "poetry" and the system finds features whose descriptions match, then ranks documents by activation strength. Mechanistic search -- not lexical, not semantic, but grounded in the model's internal representations.

**Analytical Coloring** -- Color by any metadata field with 60+ Crameri perceptually-uniform scientific colormaps. Categorical, sequential, diverging, and monochrome scales.

**Search** -- Cosine similarity search with topic and temporal scoping. Server-side text search across documents and metadata fields. Glow-effect highlighting on the scatter plot.

## Architecture

```
Data Sources --> Embedding Providers --> DuckDB (docs, metadata, projections, topics, SAE data)
                                    --> ChromaDB (dense vectors only)
                                         |
              Topic Extraction           v
              SAE Inference        GraphQL API (FastAPI + Strawberry)
                                         |
                                         v
                                    Next.js Frontend
```

**Dual-database design**: DuckDB is the central orchestrator. ChromaDB stores only dense vectors for similarity search. One dataset can have multiple vector collections (different embedding models).

## Pages

- **`/`** -- Visualization dashboard (2D/3D scatter, search, topics, temporal filtering, analytical coloring)
- **`/features`** -- SAE Feature Explorer (activation heatmaps, logit charts, prompt explorer, steering chat)
- **`/test-embed`** -- Dataset management (embed, manage collections, extract topics, configure SAE links)

## Environment Variables

Core features work without API keys (local SentenceTransformers, visualization, topic extraction, SAE analysis).

| Variable | Used For |
|----------|----------|
| `GEMINI_API_KEY` | Gemini embedding + LLM topic labeling |
| `CHROMA_OPENAI_API_KEY` | OpenAI embedding + LLM topic labeling |
| `CHROMA_COHERE_API_KEY` | Cohere embedding |
| `HUGGINGFACE_API_KEY` | HuggingFace gated model access |

## Documentation

- [`documentation/DATABASE_ARCHITECTURE.md`](documentation/DATABASE_ARCHITECTURE.md) -- DuckDB/ChromaDB schema and data flow
- [`documentation/DOCKER.md`](documentation/DOCKER.md) -- Docker setup and SAE cache profile
- [`documentation/SAE_ARCHITECTURE.md`](documentation/SAE_ARCHITECTURE.md) -- SAE storage, ingestion, GraphQL API
- [`documentation/SAE_PIPELINE.md`](documentation/SAE_PIPELINE.md) -- Neuronpedia download-to-ingestion pipeline
- [`documentation/INTERPRET_API.md`](documentation/INTERPRET_API.md) -- SAE inference, steering, streaming
- [`documentation/LABEL_PLACEMENT_GUIDE.md`](documentation/LABEL_PLACEMENT_GUIDE.md) -- 3D label collision avoidance
- [`documentation/NEBULA_CLUSTER_EFFECTS.md`](documentation/NEBULA_CLUSTER_EFFECTS.md) -- Cluster haze rendering

## License

[Apache 2.0](LICENSE)
