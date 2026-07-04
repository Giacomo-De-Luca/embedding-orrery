# Embedding Visualization Backend

GraphQL API backend for embedding, searching, and visualizing vector embeddings. Dual-database design: **DuckDB** (`resources/main.duckdb`) orchestrates documents, metadata, projections and topics; **ChromaDB** stores dense vectors only. See `documentation/DATABASE_ARCHITECTURE.md`.

## Features

- **GraphQL API** with Strawberry and FastAPI (queries, mutations, subscriptions)
- **Multi-provider embedding** - SentenceTransformers, OpenAI, Cohere, Ollama, Gemini, BGE, QWEN, HuggingFace API
- **Data sources** - HuggingFace datasets, local files (parquet, JSON, CSV, TSV), pre-computed vectors, images
- **Semantic search** with multiple similarity measures (cosine, L2, inner product)
- **Topic extraction** - HDBSCAN clustering + c-TF-IDF keywords + optional LLM labels (Gemini/OpenAI)
- **Topic reduction** - Merge similar topics via AgglomerativeClustering or auto-HDBSCAN
- **Projection computation** - PCA and UMAP (2D/3D) stored in DuckDB `projections` table (native FLOAT[] arrays)
- **Real-time progress** via WebSocket subscriptions
- **Resumable jobs** with persistent JSON state tracking
- **File upload** endpoint for local file embedding
- **CORS enabled** for frontend development

## Quick Start

```bash
# From project root
./start_backend.sh

# Or manually
uv run uvicorn interpretability_backend.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- API: http://localhost:8000
- GraphQL Playground: http://localhost:8000/graphql
- Health check: http://localhost:8000/health

## Architecture

```
interpretability_backend/
├── backend/
│   ├── main.py                          # FastAPI app, CORS, GraphQL + WS router, upload router
│   ├── schema.py                        # Re-exports from API/ for backward compatibility
│   │
│   ├── API/                             # GraphQL layer
│   │   ├── __init__.py                  # Strawberry schema (Query + Mutation + Subscription)
│   │   ├── types.py                     # All GraphQL types, inputs, enums, scalars
│   │   ├── queries.py                   # Query resolvers (collections, search, datasets, files)
│   │   ├── mutations.py                 # Mutation resolvers (embed, delete, topics, metadata)
│   │   ├── subscriptions.py             # WebSocket subscriptions (embeddingProgress, generateStream)
│   │   ├── converters.py                # Service dataclass → GraphQL type converters
│   │   ├── chromadb_instance.py         # Lazy singleton ChromaDBClient
│   │   ├── duckdb_instance.py           # Lazy singleton DuckDBClient
│   │   ├── interpret_instance.py        # Lazy singleton InterpretService
│   │   └── upload.py                    # REST file upload endpoint (POST /upload)
│   │
│   ├── clients/                         # Data source clients
│   │   ├── duckdb_client.py             # DuckDB orchestrator (items, projections, topics, SAE, chat)
│   │   ├── chromadb_client.py           # ChromaDB wrapper (vectors + semantic search only)
│   │   ├── huggingface_client.py        # HF dataset info, preview, portion loading
│   │   └── local_data_client.py         # Local file info, preview, loading (parquet/JSON/CSV/TSV)
│   │
│   ├── embedding_functions/             # Embedding infrastructure
│   │   ├── config.py                    # Constants, enums (DataType, EmbeddingProvider), dataclasses
│   │   ├── create_embedding_function.py # Factory: provider → embedding function + dimension
│   │   ├── embed_huggingface.py         # HuggingFace dataset → DuckDB+ChromaDB (with resume)
│   │   ├── embed_local_file.py          # Local file → DuckDB+ChromaDB (text/image/vector, with resume)
│   │   ├── embed_existing_dataset.py    # Re-embed an existing dataset with a new model
│   │   ├── embed_images.py              # Image embedding via ViT (google/vit-base-patch16-384)
│   │   ├── embed_vectors.py             # Pre-computed vector ingestion
│   │   ├── ingest_sae.py                # Neuronpedia SAE features/activations → DuckDB
│   │   └── specific_functions/          # Provider-specific embedding implementations
│   │       ├── embed_sentence_transformer.py  # Fork of ChromaDB's ST EF with prompt support
│   │       ├── embed_gemini.py                # Google Gemini embedding with task_type
│   │       ├── embed_qwen.py                  # QWEN3-Embedding with query instruction
│   │       ├── embed_bge.py                   # BGE-M3 with FlagEmbedding
│   │       └── embed_transformers.py          # Template/reference (unused)
│   │
│   ├── services/                        # Business logic services
│   │   ├── topic_extraction_service.py  # Topic extraction + reduction orchestration
│   │   ├── interpret_service.py         # Live SAE inference / steering / streaming chat
│   │   ├── embedding_pipeline.py        # Shared embed→project→topics pipeline wrapper
│   │   ├── sae_pipeline_service.py      # prepareSaeData orchestration (download→ingest)
│   │   ├── progress_emitter.py          # In-memory event bus for WebSocket progress
│   │   ├── token_emitter.py             # Token event bus for streaming chat
│   │   ├── cancel_registry.py           # Per-job cancellation events
│   │   └── job_state.py                 # JSON-based persistent job state (resume capability)
│   │
│   ├── topic_extraction/                # Topic extraction algorithms
│   │   ├── cluster_and_label.py         # GenerateTopics: HDBSCAN + ClassTfidfTransformer
│   │   ├── topic_reducer.py             # TopicReducer: AgglomerativeClustering / auto-HDBSCAN
│   │   └── llm_labeling.py              # LLM label generation (Gemini/OpenAI with tenacity)
│   │
│   └── utils/                           # Shared utilities
│       ├── compute_projections.py       # PCA/UMAP computation → DuckDB projections table
│       ├── duckdb_sync.py               # Dataset/collection sync helpers
│       ├── embedding_loader.py          # Batched embedding reads from ChromaDB
│       ├── seed_bootstrap.py            # First-start seed dataset copy
│       ├── color_preprocessing.py       # Hex colour column → mapped_colour float
│       ├── text_processing.py           # format_text_for_embedding, extract_metadata
│       ├── batch_utils.py               # sort_items_by_length (efficient batching)
│       ├── id_utils.py                  # IDDeduplicator (handles duplicate IDs)
│       ├── provider_list.py             # Single source of truth for EmbeddingProviderEnum
│       ├── logger.py                    # orrery logger setup (file + console)
│       └── known_dimensions.json        # Cached model→dimension mapping
│
├── resources/
│   ├── main.duckdb                      # DuckDB store (documents, metadata, projections, topics)
│   ├── vector_db/                       # ChromaDB persistent storage (vectors only)
│   ├── uploads/                         # Uploaded files for embedding
│   └── job_state.json                   # Persistent job state
│
├── interpretability_experiments/        # Research notebooks and scripts
│   ├── WordNet/                         # WordNet embedding pipeline
│   └── Lacan/                           # Lacan text analysis experiments
│
└── unit_tests/                          # All pytest tests (test/ holds notebooks/debug scripts)
```

## GraphQL API

> **Note**: the tables below cover the core embedding/search/topic surface only. The full schema is roughly 25 queries / 30 mutations / 2 subscriptions — including SAE feature queries (`saeModels`, `saeFeature`, `saeFeatureSearch`, ...), interpret mutations (`loadModel`, `runPromptActivations`, `generateSteeredResponse`, `prepareSaeData`, ...), chat-session persistence, `textSearch`, and the `generateStream` subscription. Explore it in the GraphQL Playground or `backend/API/queries.py` / `mutations.py`.

### Queries

| Query | Description |
|-------|-------------|
| `collections` | List all collections with metadata and counts |
| `collection(name)` | Get full projection data for visualization (ids, documents, metadata, PCA/UMAP) |
| `embeddings(collectionName, ...)` | Get items with optional filtering and embedding vectors |
| `semanticSearch(collectionName, query, ...)` | Text-based semantic search |
| `semanticSearchById(collectionName, itemId, ...)` | Find similar items by existing embedding |
| `embeddingJobs(status?)` | List embedding jobs (running/interrupted/completed) |
| `huggingfaceDatasetInfo(datasetId)` | Get HF dataset configs, splits, features |
| `huggingfaceDatasetPreview(datasetId, ...)` | Preview rows from HF dataset |
| `localFileInfo(filePath)` | Get local file info (columns, rows, size) |
| `localFilePreview(filePath, ...)` | Preview rows from local file |

### Mutations

| Mutation | Description |
|----------|-------------|
| `embedHuggingfaceDataset(input)` | Embed HF dataset → ChromaDB (+ projections + topics) |
| `embedLocalFile(input)` | Embed local file → ChromaDB (text/image/vector) |
| `deleteCollection(collectionName)` | Delete a collection |
| `updateCollectionMetadata(collectionName, metadata)` | Update collection metadata |
| `extractTopics(input)` | HDBSCAN clustering + c-TF-IDF + optional LLM labels |
| `reduceTopics(input)` | Merge similar topics (standalone post-processing) |

### Subscriptions

| Subscription | Description |
|-------------|-------------|
| `embeddingProgress(jobId)` | Real-time WebSocket progress updates |

## Embedding Providers

| Provider | Default Model | Local? | API Key Env Var |
|----------|---------------|--------|-----------------|
| `SENTENCE_TRANSFORMERS` | `all-MiniLM-L6-v2` | Yes | None |
| `OPENAI` | `text-embedding-3-small` | No | `CHROMA_OPENAI_API_KEY` |
| `COHERE` | `embed-english-v3.0` | No | `CHROMA_COHERE_API_KEY` |
| `OLLAMA` | `nomic-embed-text` | Yes | None |
| `HUGGINGFACE_API` | `sentence-transformers/all-MiniLM-L6-v2` | No | `CHROMA_HUGGINGFACE_API_KEY` |
| `GEMINI` | `gemini-embedding-001` | No | `GEMINI_API_KEY` |
| `BGE` | `BAAI/bge-m3` | Yes | None |
| `QWEN` | `Qwen/Qwen3-Embedding-0.6B` | Yes | None |

## Example Queries

### List Collections
```graphql
query {
  collections {
    name
    count
    metadata
  }
}
```

### Semantic Search
```graphql
query {
  semanticSearch(
    collectionName: "my_collection"
    query: "machine learning algorithms"
    nResults: 10
    similarityMeasure: COSINE
  ) {
    id
    document
    metadata
    distance
    similarity
  }
}
```

### Embed HuggingFace Dataset
```graphql
mutation {
  embedHuggingfaceDataset(input: {
    datasetId: "squad"
    collectionName: "squad_questions"
    columns: ["question"]
    portion: { strategy: FIRST_N, n: 1000 }
    computeProjections: true
    extractTopics: true
    topicConfig: {
      minTopicSize: 15
      useLlmLabels: true
      llmProvider: "gemini"
      llmModel: "gemini-3-flash-preview"
    }
  }) {
    totalEmbedded
    embeddingDim
    projectionsComputed
    embeddingProvider
    embeddingModel
  }
}
```

### Extract Topics (Standalone)
```graphql
mutation {
  extractTopics(input: {
    collectionName: "my_collection"
    config: {
      minTopicSize: 10
      nKeywords: 10
      useLlmLabels: true
      reduction: {
        enabled: true
        method: "fixed_n"
        nTopics: 5
      }
    }
  }) {
    numTopics
    numNoisePoints
    topics { topicId label keywords { word score } count }
  }
}
```

### Embed Local File
```graphql
mutation {
  embedLocalFile(input: {
    filePath: "/path/to/data.parquet"
    collectionName: "my_data"
    dataType: TEXT
    columns: ["title", "content"]
    computeProjections: true
  }) {
    totalEmbedded
    embeddingDim
    projectionsComputed
  }
}
```

## Filter Operators

| Operator | Description | ChromaDB |
|----------|-------------|----------|
| `EQ` | Equals | `$eq` |
| `NE` | Not equals | `$ne` |
| `GT` | Greater than | `$gt` |
| `GTE` | Greater than or equal | `$gte` |
| `LT` | Less than | `$lt` |
| `LTE` | Less than or equal | `$lte` |
| `IN` | In array | `$in` |
| `NIN` | Not in array | `$nin` |

## Similarity Measures

| Measure | Description |
|---------|-------------|
| `COSINE` | Cosine similarity (default). Distance [0,2] → similarity [-1,1] |
| `L2` | Euclidean distance → similarity via 1/(1+d) |
| `IP` | Inner product (negated distance) |

## Environment Variables

| Variable | Required For | Description |
|----------|-------------|-------------|
| `CHROMA_OPENAI_API_KEY` | OpenAI embedding + LLM labeling | OpenAI API key |
| `CHROMA_COHERE_API_KEY` | Cohere embedding | Cohere API key |
| `CHROMA_HUGGINGFACE_API_KEY` | HuggingFace API embedding | HF inference API key |
| `GEMINI_API_KEY` | Gemini embedding + LLM labeling | Google Gemini API key |
| `HUGGINGFACE_API_KEY` | Gated model access | HF login for private models |

## CORS Configuration

Allowed origins (edit in `main.py`):
- `http://localhost:3000`
- `http://localhost:3001`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:3001`

## Development

The server runs with `--reload` enabled during development. Changes to Python files will automatically restart the server.

### Running Tests

```bash
uv run pytest interpretability_backend/unit_tests/
```

### Logging

The backend uses the `orrery` logger hierarchy with file output to `orrery.log` and console error output. Key loggers:
- `orrery.topic_extraction` - Topic extraction and reduction
- `orrery.llm_labeling` - LLM label generation
