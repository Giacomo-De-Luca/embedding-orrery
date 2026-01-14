# Embedding Platform Backend

The backend service for the Embedding Analysis Platform, providing a GraphQL API for embedding generation, semantic search, and data management.

## Quick Start

### 1. Start the Server
```bash
# From project root
./start_backend.sh
```
Or manually:
```bash
uv run uvicorn interpretability_backend.backend.main:app --host 0.0.0.0 --port 8000
```
Server runs at: `http://localhost:8000/graphql`

### 2. Run Tests
```bash
uv run pytest interpretability_backend/test
```

## Core Features

- **GraphQL API**: Flexible querying with Strawberry GraphQL.
- **ChromaDB Integration**: Persistent vector storage in `interpretability_backend/resources/vector_db`.
- **Embedding Pipeline**:
    - **HuggingFace Datasets**: Download and embed directly.
    - **Local Files**: Support for `.csv`, `.json`, `.parquet`.
    - **Methods**: SentenceTransformers (local), OpenAI, Cohere, Ollama.
- **Dimensionality Reduction**: Pre-compute PCA and UMAP projections for visualization.

## Directory Structure

```
interpretability_backend/
├── backend/
│   ├── API/            # GraphQL schema, queries, mutations
│   ├── clients/        # ChromaDB, HuggingFace, Local data clients
│   ├── embedding_functions/ # Model implementations
│   └── main.py         # FastAPI entry point
├── resources/
│   └── vector_db/      # ChromaDB storage (Gitignored)
├── test/               # Unit and integration tests
└── utils/              # Shared utilities
```

## GraphQL API Examples

Visit `http://localhost:8000/graphql` for the interactive playground.

**1. Embed a HuggingFace Dataset:**
```graphql
mutation {
  embedHuggingfaceDataset(input: {
    datasetId: "dair-ai/emotion"
    collectionName: "emotion_embeddings"
    columns: ["text"]
    portion: { strategy: FIRST_N, n: 1000 }
    computeProjections: true
  }) {
    totalEmbedded
    error
  }
}
```

**2. Semantic Search:**
```graphql
query {
  semanticSearch(
    collectionName: "emotion_embeddings"
    query: "feeling happy"
    nResults: 5
  ) {
    id
    document
    metadata
    similarity
  }
}
```

**3. List Collections:**
```graphql
query {
  collections {
    name
    count
    metadata
  }
}
```

## Python API

You can also use the backend components directly in Python scripts:

```python
from interpretability_backend.utils.utils import setup_collection

# Connect to the database
collection = setup_collection()

# Query
results = collection.query(
    query_texts=["hello world"],
    n_results=5
)
```
