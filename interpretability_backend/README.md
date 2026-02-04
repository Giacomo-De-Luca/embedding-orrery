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
- **Topic Extraction**: HDBSCAN clustering with c-TF-IDF keywords and optional LLM labeling.

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

**3. Extract Topics:**
```graphql
mutation {
  extractTopics(input: {
    collectionName: "emotion_embeddings"
    minTopicSize: 10
    nKeywords: 10
    useLlmLabels: false
    projectionType: "umap_2d"
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

1. **Clustering**: HDBSCAN runs on projection coordinates (UMAP 2D preferred for better cluster separation)
2. **Keyword Extraction**: c-TF-IDF identifies representative keywords for each cluster
3. **LLM Labeling** (optional): OpenAI generates human-readable topic names from keywords + sample documents
4. **Storage**: Each item gets `topic_id` and `topic_label` in metadata
5. **Noise Handling**: Points that don't fit any cluster get `topic_id: -1` with label "Unclustered"

### Configuration

```python
@dataclass
class TopicExtractionConfig:
    collection_name: str
    min_topic_size: int = 10          # Minimum points per cluster
    n_keywords: int = 10               # Keywords to extract per topic
    use_llm_labels: bool = False       # Generate LLM labels (requires CHROMA_OPENAI_API_KEY)
    llm_model: str = "gpt-4o-mini"     # OpenAI model for labeling
    projection_type: str = "umap_2d"   # Which projection to cluster on
```

### GraphQL Usage

**Standalone extraction:**
```graphql
mutation {
  extractTopics(input: {
    collectionName: "my_collection"
    minTopicSize: 15
    nKeywords: 10
    useLlmLabels: true
    projectionType: "umap_2d"
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

- `CHROMA_OPENAI_API_KEY`: Required for LLM labeling (reuses OpenAI embedding key)

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
- `backend/topic_extraction/extractRepresentation.py`: OpenAI labeling
  - Uses keywords + 4 representative documents as context
  - Rate limiting with exponential backoff
  - Generates concise topic names (5 words max)

### Data Storage

**Item Metadata** (per point):
```json
{
  "topic_id": "3",
  "topic_label": "Machine Learning | Neural Networks | Deep Learning"
}
```

**Collection Metadata**:
```json
{
  "has_topics": true,
  "topic_count": 12,
  "topics_extracted_at": "2024-01-15 10:30:00",
  "topic_summary": "[{\"topic_id\": 0, \"label\": \"...\", \"count\": 150, \"keywords\": [...]}, ...]"
}
```

### Frontend Integration

Topics automatically appear in the visualization:
- `topic_id` and `topic_label` detected as categorical fields
- Available in "Color By" dropdown
- Legend shows topic names and counts
- Click legend to toggle topic visibility

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
