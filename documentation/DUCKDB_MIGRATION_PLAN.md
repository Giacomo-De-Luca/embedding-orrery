# DuckDB Migration Plan

## Why This Migration

The current backend stores everything in ChromaDB: dense vectors, documents, per-item metadata, projections (as JSON strings), and topic assignments (as string fields). This has real limitations:

- **Type restrictions**: ChromaDB metadata supports only str/int/float/bool. Lists/dicts must be JSON-serialized, None values forbidden.
- **Projections as JSON strings**: `json.loads("[0.5, -0.3]")` parsed per-item per-load (~600k JSON parses for a 153k collection).
- **Topic updates are O(N)**: Changing a label requires reading all item metadata, modifying, and batch-writing back.
- **One collection = one embedding**: Can't have multiple embeddings (e.g., BGE + Gemini) of the same dataset.
- **No sparse vector support locally**: ChromaDB hybrid search is cloud-only.
- **Text search limited**: Metadata field search falls back to Python iteration over the entire collection.

## Target Architecture

```
DuckDB (in-process, resources/main.duckdb)
  datasets              ── replaces collection-level metadata
  items                 ── documents + flexible JSON metadata
  projections           ── native FLOAT[] arrays, per vector_collection
  projection_metadata   ── variance, timestamps per projection type
  vector_collections    ── references to ChromaDB / Qdrant collections
  topic_extractions     ── extraction configs, supports history
  topic_info            ── per-topic keywords, labels, counts
  topic_assignments     ── per-item topic/subtopic

ChromaDB (stripped to vectors only)
  IDs + dense embedding vectors, no documents, no metadata

Qdrant (future)
  IDs + sparse vectors
```

**Key relationship**: One dataset in DuckDB can have multiple vector_collections (different models, dense/sparse). Each vector_collection has its own projections and can have its own topic extractions.

```
dataset "my_papers"
  ├── vector_collection "my_papers_bge" (ChromaDB, dense, BGE-M3)
  │   ├── projections (pca_2d, pca_3d, umap_2d, umap_3d)
  │   └── topic_extraction (HDBSCAN on umap_2d projections)
  ├── vector_collection "my_papers_gemini" (ChromaDB, dense, Gemini)
  │   └── projections (pca_2d, pca_3d, umap_2d, umap_3d)
  └── vector_collection "my_papers_sparse" (Qdrant, sparse, SPLADE) [future]
```

---

## Schema

```sql
CREATE TABLE datasets (
    id                VARCHAR PRIMARY KEY,       -- UUID
    name              VARCHAR UNIQUE NOT NULL,
    description       VARCHAR,
    source_type       VARCHAR,                   -- 'huggingface'|'local_file'|'image'|'vector'
    source_dataset    VARCHAR,                   -- HF dataset ID
    source_config     VARCHAR,
    source_split      VARCHAR,
    source_file       VARCHAR,                   -- local file path
    embedded_columns  JSON,                      -- native JSON array
    data_type         VARCHAR,                   -- 'text'|'image'|'vector'
    total_in_source   INTEGER,
    created_at        TIMESTAMP DEFAULT current_timestamp,
    extra_metadata    JSON DEFAULT '{}'          -- extensible key-value store
);

CREATE TABLE items (
    id          VARCHAR NOT NULL,
    dataset_id  VARCHAR NOT NULL REFERENCES datasets(id),
    document    VARCHAR,                         -- the embedded text
    metadata    JSON,                            -- flexible schema, no type restrictions
    row_index   INTEGER,
    PRIMARY KEY (dataset_id, id)
);

CREATE TABLE vector_collections (
    id                 VARCHAR PRIMARY KEY,      -- UUID
    dataset_id         VARCHAR NOT NULL REFERENCES datasets(id),
    backend            VARCHAR NOT NULL,         -- 'chromadb'|'qdrant'
    collection_name    VARCHAR NOT NULL,         -- name inside the vector DB
    vector_type        VARCHAR NOT NULL,         -- 'dense'|'sparse'
    embedding_provider VARCHAR,
    embedding_model    VARCHAR,
    embedding_dim      INTEGER,
    embedding_task     VARCHAR,                  -- QWEN instruction
    embedding_task_type VARCHAR,                 -- Gemini task type
    embedding_prompt   VARCHAR,                  -- SentenceTransformers prompt
    item_count         INTEGER DEFAULT 0,
    has_projections    BOOLEAN DEFAULT FALSE,
    has_topics         BOOLEAN DEFAULT FALSE,
    created_at         TIMESTAMP DEFAULT current_timestamp
);

-- Projections are per vector_collection (different embeddings = different projection spaces)
CREATE TABLE projections (
    vector_collection_id VARCHAR NOT NULL REFERENCES vector_collections(id),
    item_id              VARCHAR NOT NULL,
    dataset_id           VARCHAR NOT NULL,
    projection_type      VARCHAR NOT NULL,       -- 'pca_2d'|'pca_3d'|'umap_2d'|'umap_3d'
    coordinates          FLOAT[] NOT NULL,       -- native array, no JSON
    PRIMARY KEY (vector_collection_id, item_id, projection_type),
    FOREIGN KEY (dataset_id, item_id) REFERENCES items(dataset_id, id)
);

-- Per projection-type metadata (variance, timestamps) — one row per (collection, type)
CREATE TABLE projection_metadata (
    vector_collection_id VARCHAR NOT NULL REFERENCES vector_collections(id),
    projection_type      VARCHAR NOT NULL,
    variance             FLOAT[],                -- PCA explained variance ratio
    computed_at          TIMESTAMP,
    PRIMARY KEY (vector_collection_id, projection_type)
);

-- Topic extractions reference a vector_collection (clustering runs on its projections)
CREATE TABLE topic_extractions (
    id                   VARCHAR PRIMARY KEY,    -- UUID
    vector_collection_id VARCHAR NOT NULL REFERENCES vector_collections(id),
    dataset_id           VARCHAR NOT NULL REFERENCES datasets(id),
    config               JSON,                   -- extraction config snapshot
    extracted_at         TIMESTAMP DEFAULT current_timestamp,
    topic_count          INTEGER,
    reduction_applied    BOOLEAN DEFAULT FALSE,
    reduction_method     VARCHAR,
    reduction_target     INTEGER,
    num_topics_before_reduction INTEGER,
    topic_hierarchy      JSON,                   -- {new_label: [old_subtopic_labels]}
    is_active            BOOLEAN DEFAULT TRUE    -- marks the "current" extraction
);

CREATE TABLE topic_info (
    extraction_id VARCHAR NOT NULL REFERENCES topic_extractions(id),
    topic_id      INTEGER NOT NULL,
    label         VARCHAR,
    ctfidf_label  VARCHAR,                       -- original keyword label before LLM
    count         INTEGER DEFAULT 0,
    keywords      JSON,                          -- [{word, score}, ...]
    subtopics     JSON,                          -- [subtopic_label, ...] if reduction applied
    PRIMARY KEY (extraction_id, topic_id)
);

CREATE TABLE topic_assignments (
    extraction_id VARCHAR NOT NULL REFERENCES topic_extractions(id),
    dataset_id    VARCHAR NOT NULL,
    item_id       VARCHAR NOT NULL,
    topic_id      INTEGER NOT NULL,
    topic_label   VARCHAR,
    subtopic_id   INTEGER,
    subtopic_label VARCHAR,
    PRIMARY KEY (extraction_id, dataset_id, item_id),
    FOREIGN KEY (dataset_id, item_id) REFERENCES items(dataset_id, id)
);

-- Indexes
CREATE INDEX idx_items_dataset ON items(dataset_id);
CREATE INDEX idx_projections_vc_type ON projections(vector_collection_id, projection_type);
CREATE INDEX idx_vector_collections_dataset ON vector_collections(dataset_id);
CREATE INDEX idx_topic_assignments_extraction ON topic_assignments(extraction_id);
CREATE INDEX idx_topic_assignments_topic ON topic_assignments(extraction_id, topic_id);
CREATE INDEX idx_topic_extractions_vc ON topic_extractions(vector_collection_id);
```

---

## Phases

Each phase is self-contained and results in a working system. They are designed to be executed as separate implementation tasks.

### Phase 0: Foundation

**Scope**: Add DuckDB dependency, create client class with schema initialization, singleton factory. No existing behavior changes.

**New files**:
- `backend/clients/duckdb_client.py` — `DuckDBClient` class with `_ensure_schema()` and all public methods (see below)
- `backend/API/duckdb_instance.py` — `get_duckdb_client()` singleton

**Modified files**:
- `pyproject.toml` — add `duckdb>=1.2.0`
- `backend/embedding_functions/config.py` — add `DUCKDB_PATH` constant

**DuckDBClient public interface**:
```
# Datasets
create_dataset(name, **kwargs) -> str (UUID)
list_datasets() -> List[Dict]
get_dataset(name) -> Optional[Dict]
update_dataset(name, **kwargs)
delete_dataset(name)  # CASCADE

# Items
insert_items_batch(dataset_id, ids, documents, metadatas)
get_item_ids(dataset_name) -> Set[str]
get_items_by_ids(dataset_name, ids) -> List[Dict]

# Vector collections
register_vector_collection(dataset_id, backend, collection_name, vector_type, **embedding_info) -> str
get_vector_collections(dataset_name) -> List[Dict]
get_vector_collection_by_name(collection_name) -> Optional[Dict]

# Projections
insert_projections_batch(vector_collection_id, dataset_id, item_ids, projection_type, coordinates)
upsert_projection_metadata(vector_collection_id, projection_type, variance, computed_at)
get_projection_data(vector_collection_name, projection_types) -> Dict
    # Returns: {ids, documents, item_metadata, available_fields, projections, metadata}
    # Same structure as current ChromaDBClient.get_projection_data()

# Text search
text_search(dataset_name, query, fields, mode, case_sensitive) -> Dict

# Topics
create_topic_extraction(vector_collection_id, dataset_id, config) -> str
insert_topic_info_batch(extraction_id, topics)
insert_topic_assignments_batch(extraction_id, dataset_id, assignments)
get_active_topics(vector_collection_name) -> Dict
get_items_for_topic(extraction_id, topic_id) -> List[str]
update_topic_label(extraction_id, topic_id, new_label)
update_subtopic_label(extraction_id, subtopic_id, new_label)

# Field analysis (computed via SQL, no cache needed)
compute_field_analysis(dataset_name) -> Dict
```

**Tests**: `unit_tests/test_duckdb_client.py` — schema creation + CRUD for all tables using `:memory:` DuckDB.

---

### Phase 1: Dual-Write

**Scope**: All write paths write to both DuckDB AND ChromaDB. All reads still come from ChromaDB. Validates DuckDB writes with no user-visible change.

**Modified files**:

| File | Change |
|------|--------|
| `backend/embedding_functions/embed_huggingface.py` | After `collection.add()`, also `duckdb.insert_items_batch()`. After collection creation, also `duckdb.create_dataset()` + `duckdb.register_vector_collection()` |
| `backend/embedding_functions/embed_local_file.py` | Same dual-write in `embed_text_from_local()` |
| `backend/embedding_functions/embed_vectors.py` | Same dual-write |
| `backend/embedding_functions/embed_images.py` | Same dual-write |
| `backend/services/compute_projections.py` | After ChromaDB projection metadata write, also `duckdb.insert_projections_batch()` + `duckdb.upsert_projection_metadata()` |
| `backend/services/topic_extraction_service.py` | After `_batch_update_topic_metadata()`, also write to DuckDB topic tables. After collection metadata update, also update DuckDB. Affects `extract_topics()`, `reduce_existing_topics()`, `generate_llm_labels_for_collection()` |

**Tests**: Embed a test dataset end-to-end, verify DuckDB data matches ChromaDB data.

---

### Phase 2a: Read — Collection Listing

**Scope**: `collections()` GraphQL query reads from DuckDB instead of ChromaDB.

**Modified files**:

| File | Change |
|------|--------|
| `backend/API/queries.py` | `collections()` resolver: switch from `chromadb_client.list_collections()` to `duckdb_client.list_datasets()`. Return format must match `[{name, metadata, count}]` |

**Tests**: Compare response from old vs new path.

---

### Phase 2b: Read — Projection Data (main visualization load)

**Scope**: The `collection()` query (the main data load for the scatter plots) reads from DuckDB.

**Modified files**:

| File | Change |
|------|--------|
| `backend/API/queries.py` | `collection()` resolver: switch to `duckdb_client.get_projection_data()`. Topic summary from `duckdb_client.get_active_topics()` |
| `backend/clients/duckdb_client.py` | Implement `get_projection_data()` — single SQL joining items + projections. Must return identical structure to `ChromaDBClient.get_projection_data()` |

**Key SQL**:
```sql
SELECT i.id, i.document, i.metadata,
       p2.coordinates AS pca_2d, p3.coordinates AS pca_3d,
       u2.coordinates AS umap_2d, u3.coordinates AS umap_3d
FROM items i
LEFT JOIN projections p2 ON p2.vector_collection_id = ?
    AND p2.item_id = i.id AND p2.projection_type = 'pca_2d'
LEFT JOIN projections p3 ON p3.vector_collection_id = ?
    AND p3.item_id = i.id AND p3.projection_type = 'pca_3d'
LEFT JOIN projections u2 ON ...
LEFT JOIN projections u3 ON ...
WHERE i.dataset_id = ?
```

**Tests**: Load a collection via both paths, diff the ProjectionData response.

---

### Phase 2c: Read — Text Search

**Scope**: `text_search()` query uses SQL instead of ChromaDB `where_document` + Python filtering.

**Modified files**:

| File | Change |
|------|--------|
| `backend/API/queries.py` | `text_search()` resolver: switch to `duckdb_client.text_search()` |
| `backend/clients/duckdb_client.py` | Implement SQL-based search with `ILIKE` for documents and `json_extract_string()` for metadata fields. Snippet extraction via `POSITION()` + `SUBSTR()` |

**Tests**: Search queries against both paths, compare results.

---

### Phase 2d: Read — Item Retrieval

**Scope**: `embeddings()` query gets documents + metadata from DuckDB, embedding vectors from ChromaDB.

**Modified files**:

| File | Change |
|------|--------|
| `backend/API/queries.py` | `embeddings()` resolver: metadata + documents from DuckDB; vectors (if requested) from ChromaDB; merge by ID |

---

### Phase 2e: Read — Semantic Search Enrichment

**Scope**: Semantic search gets vector similarity from ChromaDB, enriches with documents + metadata from DuckDB.

**Modified files**:

| File | Change |
|------|--------|
| `backend/API/queries.py` | `semantic_search()`: ChromaDB returns IDs + distances; DuckDB enriches via `get_items_by_ids()`. `semantic_search_by_id()`: get embedding from ChromaDB, query ChromaDB, enrich from DuckDB |

---

### Phase 2f: Read — Collection Metadata Mutations

**Scope**: Metadata updates and collection deletion go through DuckDB.

**Modified files**:

| File | Change |
|------|--------|
| `backend/API/mutations.py` | `update_collection_metadata()` → `duckdb_client.update_dataset()`. `delete_collection()` → `duckdb_client.delete_dataset()` (cascades + deletes linked ChromaDB collections) |

---

### Phase 3: Strip ChromaDB Writes

**Scope**: ChromaDB receives only IDs + embedding vectors. No documents, no metadata.

**Key architectural change — explicit embedding**:

Before (ChromaDB auto-embeds):
```python
collection.add(ids=ids, documents=documents, metadatas=metadatas)
```

After (compute embeddings explicitly, dual-target write):
```python
embeddings = embedding_func(documents)
duckdb.insert_items_batch(dataset_id, ids, documents, metadatas)
chroma_collection.add(ids=ids, embeddings=embeddings)
```

Note: `embed_vectors.py` and `embed_images.py` already provide explicit embeddings — they just drop the `documents`/`metadatas` args.

**Modified files**:

| File | Change |
|------|--------|
| `backend/embedding_functions/embed_huggingface.py` | Compute embeddings via EF before `collection.add()`. ChromaDB gets `(ids, embeddings)` only. DuckDB gets `(ids, documents, metadatas)` |
| `backend/embedding_functions/embed_local_file.py` | Same pattern |
| `backend/embedding_functions/embed_vectors.py` | Remove `documents`/`metadatas` from `collection.add()` |
| `backend/embedding_functions/embed_images.py` | Same |
| `backend/services/compute_projections.py` | Remove ChromaDB metadata update loop entirely. Projections only go to DuckDB |
| `backend/services/topic_extraction_service.py` | Remove `_batch_update_topic_metadata()` calls. Remove `_update_collection_topic_metadata()` calls. Read projections from DuckDB. Read/write topic data from DuckDB only. `update_topic_label()` becomes a single SQL UPDATE |
| `backend/topic_extraction/topic_reducer.py` | `_compute_semantic_embeddings()`: get item IDs per topic from DuckDB, fetch embeddings from ChromaDB by ID |
| `backend/clients/chromadb_client.py` | Strip to: `get_collection()`, `semantic_search()`, `add_vectors()`, `get_embeddings()`, `delete_collection()`. Remove: `list_collections()`, `get_collection_info()`, `update_collection_metadata()`, `get_projection_data()`, `text_search()`, `get_all_items()` |

**Tests**: Embed a new dataset, verify ChromaDB only contains IDs + vectors (no documents/metadata). Full end-to-end: embed → projections → topics → search → visualization.

---

### Phase 4: Migration Script + Cleanup

**Scope**: Migrate existing ChromaDB data to DuckDB. Clean up dead code.

**New files**:
- `scripts/migrate_chromadb_to_duckdb.py`

**Migration logic**:
```
For each ChromaDB collection:
  1. Read collection metadata → INSERT into datasets + vector_collections
  2. Read all items in 5k batches:
     a. Separate projection keys (pca_2d/3d, umap_2d/3d) from regular metadata
     b. INSERT items (id, document, clean_metadata)
     c. INSERT projections (vector_collection_id, id, type, parsed coordinates)
  3. If has_topics:
     a. INSERT topic_extraction row (linked to vector_collection)
     b. Parse topic_summary JSON → INSERT into topic_info
     c. For each item with topic_id → INSERT into topic_assignments
  4. Verify: DuckDB item IDs == ChromaDB item IDs
  5. Optional --clean flag: recreate ChromaDB collections with vectors only
```

**Cleanup**:
- `backend/utils/text_processing.py` — `extract_metadata()` no longer needs JSON-serialization workarounds
- Remove remaining dual-write code from Phase 1
- Update `CLAUDE.md` files (root, backend, frontend)

---

### Phase 5 (Future): Qdrant Sparse Vectors

**Scope**: Add sparse vector search alongside dense. `qdrant-client` is already in `pyproject.toml`.

**New files**:
- `backend/clients/qdrant_client.py` — wrapper for Qdrant ops
- `backend/API/qdrant_instance.py` — singleton

**Changes**:
- Embedding pipeline gets optional sparse EF (BM25 or SPLADE)
- `register_vector_collection(dataset_id, 'qdrant', name, 'sparse', ...)`
- Semantic search resolver queries both backends, merges via RRF
- `vector_collections` table already supports this

---

## Files Reference

| File | Phases | Role |
|------|--------|------|
| `backend/clients/duckdb_client.py` | 0 (new) | Central orchestrator |
| `backend/API/duckdb_instance.py` | 0 (new) | Singleton factory |
| `pyproject.toml` | 0 | Add duckdb dependency |
| `backend/embedding_functions/config.py` | 0 | Add DUCKDB_PATH |
| `backend/embedding_functions/embed_huggingface.py` | 1, 3 | Dual-write → DuckDB-only + explicit embedding |
| `backend/embedding_functions/embed_local_file.py` | 1, 3 | Same |
| `backend/embedding_functions/embed_vectors.py` | 1, 3 | Same |
| `backend/embedding_functions/embed_images.py` | 1, 3 | Same |
| `backend/services/compute_projections.py` | 1, 3 | Dual-write → DuckDB-only |
| `backend/services/topic_extraction_service.py` | 1, 3 | Dual-write → DuckDB-only reads/writes |
| `backend/topic_extraction/topic_reducer.py` | 3 | Read topic→item from DuckDB |
| `backend/API/queries.py` | 2a-2e | All resolvers switch to DuckDB |
| `backend/API/mutations.py` | 2f | Metadata/delete switch to DuckDB |
| `backend/clients/chromadb_client.py` | 3 | Strip to vector-only ops |
| `backend/utils/text_processing.py` | 4 | Remove ChromaDB workarounds |
| `scripts/migrate_chromadb_to_duckdb.py` | 4 (new) | Migration script |

## Verification Checklist

- [ ] Phase 0: `uv run pytest unit_tests/test_duckdb_client.py` passes
- [ ] Phase 1: Embed test dataset, DuckDB data matches ChromaDB data
- [ ] Phase 2 (each): GraphQL response identical between old and new paths
- [ ] Phase 3: New embeddings — ChromaDB has only IDs + vectors
- [ ] Phase 4: Migration script runs on all existing collections
- [ ] End-to-end: embed → projections → topics → text search → semantic search → visualization
