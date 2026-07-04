# Database Architecture

The backend uses a dual-database design: **DuckDB** as the central data store and **ChromaDB** for dense vector similarity search only.

## Why Two Databases

DuckDB handles everything that's relational: documents, metadata, projections, topics. ChromaDB handles the one thing it's good at: HNSW-indexed vector similarity search. This separation means:

- No more JSON-serialized projections in ChromaDB metadata
- No more type restrictions (str/int/float/bool only)
- Per-dataset tables with per-dataset FTS indexes
- One dataset can have multiple embedding models (multiple vector collections)
- Topic label updates are O(1) SQL instead of O(N) ChromaDB read-modify-write

## Storage Locations

| Database | Path | Purpose |
|----------|------|---------|
| DuckDB | `resources/main.duckdb` | Documents, metadata, projections, topics, dataset/collection registry |
| ChromaDB | `resources/vector_db/` | Dense embedding vectors only (IDs + vectors, no documents or metadata) |

## DuckDB Schema

> **Note**: beyond the tables documented below, `_ensure_schema()` also creates the SAE tables (`sae_features`, `sae_activations`, `sae_document_activations` — documented in `SAE_ARCHITECTURE.md`) and the chat-history tables `chat_sessions` / `chat_messages` (per-message `steering_snapshot` JSON; CRUD via `create_chat_session` / `list_chat_sessions` / `get_chat_session_with_messages` / `save_chat_message` / `delete_chat_session` in `duckdb_client.py`).

### Global Tables (small, registry-style)

```
datasets
  name (PK)           — unique dataset name, also used to derive items table name
  description, source_type, source_dataset, source_config, source_split, source_file
  embedded_columns (JSON), data_type, total_in_source, item_count (cached)
  created_at, extra_metadata (JSON)

vector_collections
  collection_name (PK) — matches the ChromaDB collection name
  dataset_name (FK)    — which dataset this embedding belongs to
  backend              — "chromadb" (future: "qdrant")
  vector_type          — "dense" (future: "sparse")
  embedding_provider, embedding_model, embedding_dim
  embedding_task, embedding_task_type, embedding_prompt
  item_count, has_projections, has_topics, created_at

projections
  collection_name (FK) — which vector collection
  item_id              — matches item ID in the dataset's items table
  projection_type      — "pca_2d", "pca_3d", "umap_2d", "umap_3d"
  coordinates (FLOAT[])— native array, no JSON parsing
  PK: (collection_name, item_id, projection_type)

projection_metadata
  collection_name (FK), projection_type
  variance (FLOAT[])   — PCA explained variance ratio
  computed_at

topic_extractions
  id (PK, UUID)
  collection_name (FK), dataset_name (FK)
  config (JSON), extracted_at, topic_count
  reduction_applied, reduction_method, reduction_target
  num_topics_before_reduction, topic_hierarchy (JSON)
  is_active            — only one active extraction per collection

topic_info
  extraction_id (FK), topic_id
  label, ctfidf_label, count, keywords (JSON), subtopics (JSON)
  PK: (extraction_id, topic_id)

topic_assignments
  extraction_id (FK), item_id
  topic_id, topic_label, subtopic_id, subtopic_label
  PK: (extraction_id, item_id)
```

### Per-Dataset Tables

Each dataset gets its own items table: `items_{sanitized_name}`

```
items_{dataset_name}
  id (PK)
  document (VARCHAR)   — the embedded text
  metadata (JSON)      — flexible schema, no type restrictions
  row_index (INTEGER)
```

Table names are sanitized: non-alphanumeric characters replaced with underscores. For example, dataset "ag_news" gets table `items_ag_news`.

**Why per-dataset tables:**
- FTS indexes are automatically per-dataset (correct BM25 IDF values)
- No `WHERE dataset_id = ?` on every query — the table IS the scope
- Smaller tables = faster scans
- `DROP TABLE` for instant dataset deletion
- Zero cross-dataset interference

## Data Flow

### Embedding Pipeline
```
Source data → Embedding model → explicit embed()
                                     ↓
                          ┌──────────┴──────────┐
                          ↓                      ↓
                    DuckDB                  ChromaDB
                 items_{name}           collection (IDs + vectors)
              (docs + metadata)
```

### Projection Computation
```
ChromaDB → read embeddings in 5k batches → PCA/UMAP → DuckDB projections table
```

### Topic Extraction
```
DuckDB projections → HDBSCAN clustering → c-TF-IDF keywords → DuckDB topic tables
                                                     ↓
                                              (optional LLM labeling)
```

### Frontend Data Load
```
Frontend → GraphQL collection(name, projectionType) →
  DuckDB: items + projections JOIN (one query, one projection type)
  DuckDB: topic_assignments merged into item_metadata
  → ProjectionData response
```

### Semantic Search
```
Frontend → GraphQL semantic_search(query, filters?) →
  If filters: DuckDB get_filtered_items() → allowed_ids set
  ChromaDB: vector search (over-fetch if filtered) → IDs + distances
  Post-filter by allowed_ids
  DuckDB: get_items_by_ids() → documents + metadata enrichment
  → SemanticSearchResult list
```

### Text Search
```
Frontend → GraphQL text_search(query, fields, mode) →
  DuckDB: ILIKE on per-dataset items table (substring mode)
  DuckDB: json_extract_string for metadata field search
  → TextSearchResponse
```

BM25 word-level search available via `text_search_bm25()` using per-dataset FTS indexes (Porter stemmer, English stopwords).

## DuckDBClient API

Core class: `backend/clients/duckdb_client.py`
Singleton: `backend/API/duckdb_instance.py` → `get_duckdb_client()`

### Datasets
- `create_dataset(name, **kwargs)` → creates dataset row + items table
- `list_datasets()` → `[{name, metadata, count}]`
- `get_dataset(name)` → dataset dict or None
- `update_dataset(name, **kwargs)`
- `delete_dataset(name)` → drops items table, cascades all related data

### Items
- `insert_items_batch(dataset_name, ids, documents, metadatas)` → bulk insert via DataFrame
- `get_item_ids(dataset_name)` → set of IDs
- `get_items_by_ids(dataset_name, ids)` → list of item dicts
- `get_filtered_items(dataset_name, filters, limit, offset)` → filtered items via JSON operators

### Vector Collections
- `register_vector_collection(dataset_name, backend, collection_name, vector_type, **embedding_info)`
- `get_vector_collections(dataset_name)` → list
- `get_vector_collection(collection_name)` → dict or None

### Projections
- `insert_projections_batch(collection_name, item_ids, projection_type, coordinates)`
- `upsert_projection_metadata(collection_name, projection_type, variance=, computed_at=)`
- `get_projection_data(collection_name, projection_type)` → items + coordinates + metadata

### Text Search
- `text_search(dataset_name, query, fields, mode, case_sensitive)` → ILIKE-based
- `text_search_bm25(dataset_name, query, limit)` → FTS BM25-scored results

### Topics
- `create_topic_extraction(collection_name, dataset_name, config)` → extraction UUID
- `insert_topic_info_batch(extraction_id, topics)`
- `insert_topic_assignments_batch(extraction_id, assignments)`
- `get_active_topics(collection_name)` → extraction + topic_info list
- `get_items_for_topic(extraction_id, topic_id)` → item IDs
- `update_topic_label(extraction_id, topic_id, new_label)`
- `update_subtopic_label(extraction_id, subtopic_id, new_label)`

## ChromaDBClient API

Stripped to vector-only operations (~170 lines): `backend/clients/chromadb_client.py`
Singleton: `backend/API/chromadb_instance.py` → `get_chromadb_client()`

- `get_collection(name, load_embedding_function, for_query, query_prompt)` — lazy EF loading
- `semantic_search(collection_name, query_texts, query_embeddings, n_results, distance_metric, query_prompt)` — returns IDs + distances + similarities (no documents/metadata)

## Migration

Script: `scripts/migrate_chromadb_to_duckdb.py`

```bash
# Migrate all collections
uv run python -m interpretability_backend.scripts.migrate_chromadb_to_duckdb

# Single collection
uv run python -m interpretability_backend.scripts.migrate_chromadb_to_duckdb --collection emotion

# Re-migrate (overwrite)
uv run python -m interpretability_backend.scripts.migrate_chromadb_to_duckdb --force

# Verify only
uv run python -m interpretability_backend.scripts.migrate_chromadb_to_duckdb --verify
```

## Known Limitations

- **ChromaDB still stores embedding function config** in collection metadata. `get_collection(load_embedding_function=True)` reads model info from there. Future: read from DuckDB `vector_collections`.
- **DuckDB is single-writer** — one process can write at a time. Matches the single-user deployment model.
- **DuckDB datetime values** need `_sanitize_for_json()` at API boundaries (Strawberry JSON fields).
