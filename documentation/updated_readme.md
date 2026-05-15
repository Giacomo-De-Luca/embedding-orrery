# StarMap: Interactive Embedding Analysis with Mechanistic Interpretability

StarMap is an open-source platform for embedding visualization, automatic topic extraction, and Sparse Autoencoder (SAE) interpretability. It provides an end-to-end workflow from data ingestion through interactive exploration to mechanistic analysis of language model internals.

![Embedding Visualization](gallery/harmbench.png)

## Key Capabilities

**Embedding Visualization** -- Embed datasets from HuggingFace Hub, local files (CSV, JSON, Parquet), images, or pre-computed vectors. Visualize in interactive 2D/3D scatter plots with WebGL rendering, supporting 250k+ points at 60fps. Multiple embedding providers: SentenceTransformers (local), Gemini, OpenAI, Cohere, Ollama, QWEN, BGE.

**Automatic Topic Extraction** -- HDBSCAN clustering on UMAP projections with c-TF-IDF keyword extraction and optional LLM-generated labels (Gemini/OpenAI). Hierarchical topic reduction preserves subtopic structure. Cluster quality metrics (silhouette, Davies-Bouldin) for validation.

**SAE Feature Analysis** -- Load Sparse Autoencoders (GemmaScope, QwenScope) and run live inference. Per-token feature activation visualization, scatter plot highlighting by activated features, and interactive steering. Explore Neuronpedia feature catalogs with semantic search over feature descriptions.

**Analytical Coloring** -- Color any scatter plot by arbitrary metadata fields: categorical labels, numeric scores, temporal dimensions, or mapped color values. Enables replication of embedding geometry analyses (e.g., Geometry of Truth) on any dataset. Sequential, diverging, categorical, and monochrome scales with 60+ Crameri perceptually-uniform colormaps.

**Semantic & Text Search** -- Cosine similarity search with topic and temporal scoping. Server-side text search across documents and metadata fields. Results highlighted on the scatter plot with glow effects.

## Gallery

| | |
|---|---|
| ![WordNet 212k](gallery/geometry.png) | ![HarmBench Topics](gallery/harmbench.png) |
| WordNet Senses (212k points) with nebula cluster effects and semantic search | HarmBench safety benchmark with LLM-generated topic labels |
| ![XKCD Colors](gallery/XKCD_embedded_colourspace.png) | ![Glasgow Norms](gallery/concreteness.jpg) |
| XKCD color words colored by actual hex values -- embedding space mirrors perceptual color space | Glasgow Norms colored by concreteness score -- psycholinguistic dimensions emerge as spatial gradients |

## Quick Start

### Docker (Recommended)

```bash
docker compose up
# Frontend: http://localhost:3000 (pre-loaded with demo datasets)
# GraphQL Playground: http://localhost:8000/graphql
```

### Manual Installation

```bash
# Backend
uv sync
./start_backend.sh
# or: uv run uvicorn interpretability_backend.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
cd embedding_visualization && npm install && npm run dev

# Visit http://localhost:3000
```

### First Steps

1. **Explore pre-loaded data**: Select a collection from the dropdown (top right)
2. **Color by metadata**: Use the "Color By" dropdown to explore different dimensions
3. **Semantic search**: Type a query in the search bar to find similar items
4. **Topic analysis**: Topics appear as categorical color fields with cluster labels
5. **SAE analysis**: Navigate to `/features` for feature exploration and steering

## Architecture

```
Data Sources                  Embedding Providers           Storage
  HuggingFace Hub        -->    SentenceTransformers   -->   DuckDB (documents, metadata,
  Local Files (CSV/JSON) -->    Gemini / OpenAI        -->     projections, topics, SAE data)
  Images                 -->    Cohere / Ollama        -->   ChromaDB (dense vectors only)
  Pre-computed Vectors   -->    QWEN / BGE             -->
                                                              |
                               Topic Extraction               v
                                 HDBSCAN clustering      GraphQL API (FastAPI + Strawberry)
                                 c-TF-IDF keywords            |
                                 LLM labeling                 v
                                                         Next.js Frontend
                               SAE Inference                  2D/3D Scatter Plots
                                 Prompt activations           Semantic Search
                                 Feature steering             Topic Filtering
                                 Streaming chat               SAE Feature Explorer
```

**Dual-database design**: DuckDB is the central orchestrator storing documents, metadata, projections (native FLOAT[] arrays), topic data, and SAE feature metadata. ChromaDB stores only dense embedding vectors for similarity search. One dataset can have multiple vector collections (different embedding models).

## Project Structure

```
starmap/
  interpretability_backend/       # Python backend (FastAPI + Strawberry GraphQL)
    backend/
      API/                        #   GraphQL queries, mutations, subscriptions
      clients/                    #   DuckDB, ChromaDB, HuggingFace, local file clients
      embedding_functions/        #   Multi-provider embedding infrastructure
      services/                   #   Topic extraction, progress tracking, SAE inference
      topic_extraction/           #   HDBSCAN, c-TF-IDF, LLM labeling, topic reduction
    interpret/                    # SAE interpretability toolkit
      inference/                  #   Model wrappers (Gemma, Qwen)
      sae/                        #   SAE models, hooks, steering, feature labels
        exploration/              #   PromptExplorer, NeuronpediaExplorer
        pipeline/                 #   Download-to-ingestion pipeline
      experiments/                #   Refusal directions, poetry directions
    resources/                    #   DuckDB database, ChromaDB vector store
  embedding_visualization/        # Next.js 15 frontend
    app/
      components/                 #   Scatter plots, dashboard, legend, search
      features/                   #   SAE Feature Explorer page
      test-embed/                 #   Dataset embedding interface
    lib/
      hooks/                      #   Data loading, search, visualization hooks
      stores/                     #   Zustand state management
      utils/                      #   Color system, label placement, temporal analysis
      colorMaps/                  #   60+ Crameri scientific colormaps
  documentation/                  # Architecture docs, guides, issue tracking
```

## Pages

- **`/`** -- Visualization dashboard with 2D/3D scatter plots, semantic search, topic filtering, temporal analysis, and analytical coloring
- **`/features`** -- SAE Feature Explorer with token-strip activation heatmaps, logit charts, semantic feature search, and steering chat interface
- **`/test-embed`** -- Dataset management: embed from HuggingFace or local files, manage collections, extract topics, configure SAE links

## Key Technologies

| Layer | Stack |
|-------|-------|
| **Backend** | Python 3.12, FastAPI, Strawberry GraphQL, DuckDB, ChromaDB |
| **Embedding** | SentenceTransformers, Gemini, OpenAI, Cohere, Ollama, QWEN, BGE |
| **ML** | PyTorch, HDBSCAN, scikit-learn, UMAP, HuggingFace Transformers |
| **SAE** | Custom JumpReLU/TopK implementations, GemmaScope/QwenScope weights |
| **Frontend** | Next.js 15, React 19, TypeScript 5, Plotly.js (WebGL), Three.js |
| **UI** | Tailwind CSS 4, Shadcn UI, Zustand, Apollo Client 4 |
| **Colormaps** | D3 scales, 60+ Crameri perceptually-uniform scientific colormaps |

## Forked Dependencies

This project includes two forked libraries with targeted patches:

- **Plotly.js**: Fixes an O(n*m) performance bug in trace updates where Plotly re-reads all existing traces when updating a single trace. Critical for real-time overlay updates (search highlights, selection markers) on large datasets.
- **gemma_pytorch**: Adds internal activation cache for per-layer SAE hook attachment. Standard PyTorch forward hooks only provide module boundary access; the fork exposes mid-layer residual stream states required for SAE analysis.

## Environment Variables

Required only for specific features:

| Variable | Used For |
|----------|----------|
| `GEMINI_API_KEY` | Gemini embedding + LLM topic labeling |
| `CHROMA_OPENAI_API_KEY` | OpenAI embedding + LLM topic labeling |
| `CHROMA_COHERE_API_KEY` | Cohere embedding |
| `HUGGINGFACE_API_KEY` | HuggingFace gated model access |

**Note:** Core features (SentenceTransformers embedding, visualization, topic extraction with keyword labels, SAE analysis) work without any API keys using local models only.

## Documentation

- [`documentation/DATABASE_ARCHITECTURE.md`](documentation/DATABASE_ARCHITECTURE.md) -- DuckDB/ChromaDB schema, data flow
- [`documentation/SAE_ARCHITECTURE.md`](documentation/SAE_ARCHITECTURE.md) -- SAE storage, ingestion, GraphQL API
- [`documentation/SAE_PIPELINE.md`](documentation/SAE_PIPELINE.md) -- Neuronpedia download-to-ingestion pipeline
- [`documentation/INTERPRET_API.md`](documentation/INTERPRET_API.md) -- SAE inference service, steering, streaming
- [`documentation/LABEL_PLACEMENT_GUIDE.md`](documentation/LABEL_PLACEMENT_GUIDE.md) -- 3D label collision avoidance
- [`documentation/NEBULA_CLUSTER_EFFECTS.md`](documentation/NEBULA_CLUSTER_EFFECTS.md) -- Cluster haze rendering

## Running Tests

```bash
# Backend
uv run pytest

# Frontend
cd embedding_visualization && npm test
```

## License

[MIT](LICENSE)
