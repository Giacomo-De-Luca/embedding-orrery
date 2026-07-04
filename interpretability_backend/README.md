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
uv run pytest interpretability_backend/unit_tests/
```
(`test/` contains notebooks and debug scripts, not pytest tests.)

## Core Features

- **GraphQL API**: Flexible querying with Strawberry GraphQL.
- **Dual-database storage**: DuckDB (`resources/main.duckdb`) orchestrates documents, metadata, projections and topics; ChromaDB (`resources/vector_db/`) stores dense vectors only.
- **Embedding Pipeline**:
    - **HuggingFace Datasets**: Download and embed directly.
    - **Local Files**: Support for `.csv`, `.json`, `.parquet`.
    - **Methods**: SentenceTransformers (local), OpenAI, Cohere, Ollama, HuggingFace API, Gemini, QWEN, BGE.
- **Dimensionality Reduction**: Pre-compute PCA and UMAP projections (stored in DuckDB).
- **Topic Extraction**: HDBSCAN clustering with c-TF-IDF keywords and optional LLM labeling.
- **SAE tooling**: Neuronpedia feature ingestion, live Gemma inference with steering (see `documentation/SAE_ARCHITECTURE.md`, `documentation/INTERPRET_API.md`).

## Directory Structure

```
interpretability_backend/
├── backend/
│   ├── API/            # GraphQL schema, queries, mutations, subscriptions
│   ├── clients/        # DuckDB (orchestrator), ChromaDB, HuggingFace, Local data clients
│   ├── embedding_functions/ # Multi-provider embedding + SAE ingestion
│   ├── services/       # Topic extraction, interpret (SAE inference), jobs, progress
│   ├── topic_extraction/ # HDBSCAN, c-TF-IDF, LLM labeling, reduction
│   ├── utils/          # Shared utilities (projections, seed bootstrap, ...)
│   └── main.py         # FastAPI entry point
├── interpret/          # SAE/steering toolkit (see interpret/README.md)
├── evaluation/         # Topic-quality + projection-fidelity evaluators
├── resources/
│   ├── main.duckdb     # DuckDB store (gitignored; seeded on first start)
│   └── vector_db/      # ChromaDB storage (gitignored)
├── unit_tests/         # Pytest tests
└── test/               # Notebooks + debug scripts
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

**3. Extract Topics** (config fields must be nested under `config`):
```graphql
mutation {
  extractTopics(input: {
    collectionName: "emotion_embeddings"
    config: {
      minTopicSize: 10
      nKeywords: 10
      useLlmLabels: false
      projectionType: "umap_2d"
    }
  }) {
    numTopics
    numNoisePoints
    topics {
      topicId
      label
      keywords { word score }
      count
    }
    durationSeconds
  }
}
```

**4. List Collections:**
```graphql
query {
  collections {
    name
    count
    metadata
  }
}
```

## Topic Extraction

Extract semantic topics from your embeddings using HDBSCAN clustering, c-TF-IDF, and optional LLM labeling.

### How It Works

1. **Clustering**: HDBSCAN runs on the space selected by `cluster_on` — default `"cluster_umap"` (fresh 5-D UMAP on the raw vectors); alternatives: `"projection"` (stored viz coords) or `"embedding"` (L2-normalised raw vectors)
2. **Keyword Extraction**: c-TF-IDF identifies representative keywords for each cluster
3. **LLM Labeling** (optional): Gemini (default) or OpenAI generates human-readable topic names from keywords + sample documents
4. **Storage**: Topic assignments and topic info are written to DuckDB (`topic_extractions`/`topic_info`/`topic_assignments`)
5. **Noise Handling**: Points that don't fit any cluster get `topic_id: -1` with label "Unclustered"

### Configuration (key fields)

```python
@dataclass
class TopicExtractionConfig:
    collection_name: str
    min_topic_size: int = 10          # Minimum points per cluster
    n_keywords: int = 10               # Keywords to extract per topic
    use_llm_labels: bool = False       # Generate LLM labels
    llm_provider: str = "gemini"       # "gemini" (GEMINI_API_KEY) or "openai" (CHROMA_OPENAI_API_KEY)
    llm_model: str = "gemini-3-flash-preview"
    projection_type: str = "umap_2d"   # Which stored projection to use
    cluster_on: str = "cluster_umap"   # Clustering space (see above)
    # plus: clustering_method, n_clusters, cluster_n_components/min_dist/n_neighbors,
    # reduction fields — see services/topic_extraction_service.py for the full set
```

### GraphQL Usage

**Standalone extraction:**
```graphql
mutation {
  extractTopics(input: {
    collectionName: "my_collection"
    config: {
      minTopicSize: 15
      nKeywords: 10
      useLlmLabels: true
      projectionType: "umap_2d"
    }
  }) {
    numTopics
    numNoisePoints
    topics {
      topicId
      label
      keywords { word score }
      count
    }
  }
}
```

**Auto-extract during embedding:**
```graphql
mutation {
  embedHuggingfaceDataset(input: {
    datasetId: "squad"
    collectionName: "squad_with_topics"
    columns: ["question"]
    computeProjections: true
    extractTopics: true
    topicConfig: {
      minTopicSize: 20
      useLlmLabels: true
    }
  }) {
    totalEmbedded
    projectionsComputed
  }
}
```

### Environment Variables

- `GEMINI_API_KEY`: Required for LLM labeling with the default Gemini provider
- `CHROMA_OPENAI_API_KEY`: Required when `llm_provider: "openai"` (reuses OpenAI embedding key)

### Code Architecture

**Main Service:**
- `backend/services/topic_extraction_service.py`: Orchestrates the full pipeline
  - `extract_topics(config)`: Main entry point
  - Loads projections, runs clustering, extracts keywords, updates metadata
  - Progress tracking via WebSocket (`progress_emitter.py`)

**Clustering Components:**
- `backend/topic_extraction/cluster_and_label.py`: BERTopic-inspired implementation
  - `GenerateTopics.generate_clusters()`: HDBSCAN clustering
  - `ClassTfidfTransformer`: c-TF-IDF for keyword scoring
  - `GenerateTopics.extract_topics()`: Extract top-N keywords per cluster

**LLM Integration:**
- `backend/topic_extraction/llm_labeling.py`: provider-agnostic labeling (`_GeminiLabeler`/`_OpenAILabeler`)
  - Uses keywords + representative documents as context
  - Rate limiting with exponential backoff
  - Generates concise topic names

### Data Storage

Topic data lives in DuckDB (not ChromaDB): `topic_extractions` (one row per run, with a JSON config snapshot), `topic_info` (per-topic label/keywords/count), and `topic_assignments` (per-item `topic_id`/`topic_label`, plus `subtopic_id`/`subtopic_label` after reduction). See `documentation/DATABASE_ARCHITECTURE.md`.

### Frontend Integration

Topics automatically appear in the visualization:
- `topic_id` and `topic_label` detected as categorical fields
- Available in "Color By" dropdown
- Legend shows topic names and counts
- Click legend to toggle topic visibility

## Python API

You can also use the backend components directly in Python scripts:

```python
from interpretability_backend.backend.clients.duckdb_client import DuckDBClient
from interpretability_backend.backend.clients.chromadb_client import ChromaDBClient

db = DuckDBClient()                    # documents, metadata, projections, topics
items = db.get_items_by_ids("emotion", ["emotion_0", "emotion_1"])

chroma = ChromaDBClient()              # vectors only
results = chroma.semantic_search("emotion", query_texts=["hello world"], n_results=5)
```
