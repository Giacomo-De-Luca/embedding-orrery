# Embedding Analysis Platform

The project is a platform for generating, analysing and displaying word embeddings. Currently it allows to generate embeddings from either local datasets or hugging face. It uses either sentence-transformers or APIs to generate embeddings, and stores them in a ChromaDB. 

Everything can be done in the frontend, from retriving embeddings, to visualizing them, to analysing them.

Polish is not there, a lot of the functionalities are iffy at best, but the architecture is working, and the visualisation is quite good looking, - so, enjoy! 

(I'm trying to make it work for sparse embeddings as well and qwen embeddings, there were some interpretability experiments in the backend folder, but the code was... raw.)

*Note:* the first time it launches, there are no data to display, probably I should add a test little dataset. Anyways, click on the "Embed" button on the top right, it opens the collection manager page. There one can embed datasets or use the default "emotion one" (which is quite terrible as an example, since it doesn't have labels). Select the columns to embed, and click embed. Remotely the only provider tested that it's working is sentence-transformers. I will add more later. 

## Quick Start

```bash
# 1. Install all dependencies (Python, Rust, Node.js)
./install_requirements.sh

# 2. Start (Optional) Backend API
./start_backend.sh
# Visit http://localhost:8000/graphql

# 3. Start Frontend Visualization
cd embedding_visualization
npm run dev
# Visit http://localhost:3000
```

## Project Structure

### `/interpretability_backend/` - Backend & Core Pipeline

The backend service powered by FastAPI, Strawberry GraphQL, and ChromaDB.

**Key Features:**
- GraphQL API for semantic search and dataset management
- ChromaDB vector database integration
- Embed HuggingFace datasets or local files (CSV, JSON, Parquet)
- Multi-provider support (SentenceTransformers, OpenAI, Cohere, etc.)

**Quick Commands:**
```bash
# Start backend server
./start_backend.sh

# Run tests
uv run pytest
```

See [interpretability_backend/README.md](interpretability_backend/README.md) for detailed documentation.

### `/embedding_visualization/` - Interactive Web Frontend

Modern Next.js web application for 2D/3D visualization, clustering, and analysis.

**Key Features:**
- high-performance 2D/3D scatter plots
- PCA and UMAP dimensionality reduction
- Semantic search and clustering
- Embed new datasets directly from the UI

**Quick Commands:**
```bash
cd embedding_visualization
npm run dev
```

See [embedding_visualization/README.md](embedding_visualization/README.md) for detailed documentation.

## Data Flow

```
Data Sources:
├── HuggingFace Datasets
├── Local Files (parquet, JSON, CSV)
└── Image Datasets
    ↓
Embedding Models:
├── Text: all-MiniLM-L6-v2 (default), OpenAI, Cohere, etc.
├── Image: ViT
└── Vectors: Pre-computed
    ↓
ChromaDB Vector Database (interpretability_backend/resources/vector_db)
    ↓
GraphQL API (FastAPI + Strawberry)
    ↓
Web UI (Next.js)
```

## Key Technologies

**Backend:**
- Python 3.12+, `uv` package manager
- ChromaDB, FastAPI, Strawberry GraphQL
- sentence-transformers, scikit-learn, UMAP

**Frontend:**
- Next.js 15, React, TypeScript
- Plotly.js, WebAssembly (clustering)
- Shadcn UI, Tailwind CSS

## API Reference

### GraphQL API

The backend provides a GraphQL API (`http://localhost:8000/graphql`) for:

```graphql
# Semantic search
query {
  semanticSearch(
    collectionName: "my_collection"
    query: "search query"
    nResults: 10
  ) { ... }
}

# Embed a dataset
mutation {
  embedHuggingfaceDataset(...) { ... }
}
```

## Documentation

- **[interpretability_backend/README.md](interpretability_backend/README.md)** - Backend documentation
- **[embedding_visualization/README.md](embedding_visualization/README.md)** - Frontend documentation
