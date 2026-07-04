# SAE (Sparse Autoencoder) Feature Explorer — Architecture

This document describes the SAE subsystem: backend storage, ingestion pipeline, GraphQL API, frontend feature explorer, and cross-linking with the main visualization.

## Overview

The SAE subsystem stores and visualizes features extracted from sparse autoencoders (e.g. GemmaScope). Each SAE decomposes a model's activations into interpretable features. The system stores:

- **Feature metadata** — index, density, human-readable label, top/bottom logits
- **Activation examples** — ~20 top-activating text samples per feature, each with 512 tokens and per-token activation values
- **Explanation embeddings** (optional) — 2560-d vectors for semantic feature search, stored in ChromaDB

## Data Model

### Source Data Formats

| File | Format | Content | Example |
|------|--------|---------|---------|
| Features | Parquet | `index`, `vector` (2560-d), `density`, `label`, `top_logits`, `bottom_logits` | `gemma_9_16k.parquet` (16,384 rows) |
| Activations | JSONL | `id`, `modelId`, `layer`, `index`, `tokens[512]`, `values[512]`, `maxValue`, `maxValueTokenIndex` | `gemma-3-4b-it_9-gemmascope-2-res-16k_activations.jsonl` (327,110 rows, ~20 per feature) |

### Composite Key

All SAE data is keyed by `(model_id, sae_id, feature_index)`. This enables multiple SAEs (different layers, widths, models) to coexist in the same database.

| Field | Example | Source |
|-------|---------|--------|
| `model_id` | `gemma-3-4b-it` | Parquet: passed as arg; JSONL: `modelId` field |
| `sae_id` | `9-gemmascope-2-res-16k` | Parquet: passed as arg; JSONL: `layer` field |
| `feature_index` | `0`–`16383` | Both: `index` field |

### DuckDB Tables

```
sae_features
  model_id        VARCHAR NOT NULL  ─┐
  sae_id          VARCHAR NOT NULL   ├─ PK (composite)
  feature_index   INTEGER NOT NULL  ─┘
  density         FLOAT              — fraction of tokens that activate this feature
  label           VARCHAR            — human-readable explanation from Neuronpedia
  top_logits      JSON               — [{token, score}, ...] tokens the feature promotes
  bottom_logits   JSON               — [{token, score}, ...] tokens the feature suppresses
  created_at      TIMESTAMP

sae_activations
  id                   VARCHAR PK     — unique activation sample ID from Neuronpedia
  model_id             VARCHAR NOT NULL  ─┐
  sae_id               VARCHAR NOT NULL   ├─ indexed (non-unique) for joins
  feature_index        INTEGER NOT NULL  ─┘
  tokens               JSON NOT NULL    — string[512], the tokenized context window
  act_values           JSON NOT NULL    — float[512], per-token activation values
  max_value            FLOAT           — highest activation in this sample
  max_value_token_idx  INTEGER         — token index of the max activation
  min_value            FLOAT
  qualifying_token_idx INTEGER
  INDEX idx_sae_act_feature ON (model_id, sae_id, feature_index)
```

**Design decisions:**
- Dedicated tables (not the generic `items_{name}` pattern) because SAE data has structured typed columns (density, logits as JSON arrays) that benefit from native SQL filtering.
- Column named `act_values` not `values` to avoid SQL keyword conflict.
- `INSERT OR REPLACE` semantics for idempotent re-ingestion.
- Activations ordered by `max_value DESC` for the "top activations" query.

### ChromaDB (Optional)

When features are ingested with `store_vectors=True`, explanation-embedding vectors (2560-d) are stored in a ChromaDB collection. This enables:
- Semantic feature search ("find features related to medical terminology")
- Visualization of feature embeddings in the main scatter plot

The ChromaDB collection name follows the pattern `sae_{model_id}_{sae_id}` and is registered in DuckDB's `datasets` + `vector_collections` tables via the standard dual-write pattern.

## Current SAE Data

| Model | SAE Layer/Width | Features | Activations | DuckDB Collections | ChromaDB Collection |
|-------|-----------------|----------|-------------|---------------------|---------------------|
| `gemma-3-4b-it` | `9-gemmascope-2-res-16k` | 16,384 | 327,110 | `gemma_9_16k` (16,384 items), `Gemma_9_16k_embedded` (15,854 items) | `Gemma_9_16k_embedded` |

The two DuckDB collections both map to the same SAE model/layer:
- `gemma_9_16k` — items with metadata `{index, density, top_logits, bottom_logits, topic_id, topic_label}`
- `Gemma_9_16k_embedded` — items with metadata `{index, density, top_logits, bottom_logits, topic_id, topic_label}`, has explanation-embedding vectors in ChromaDB

## Backend Pipeline

### Ingestion (`embedding_functions/ingest_sae.py`)

Two entry points, callable via GraphQL mutations or Python:

```
ingest_sae_features(parquet_path, model_id, sae_id, store_vectors, progress_callback)
  1. Read parquet → validate required columns
  2. Serialize logit columns (numpy arrays of dicts → JSON strings)
  3. Bulk-insert into sae_features via pandas DataFrame
  4. If store_vectors: parse explanation vectors, create ChromaDB collection,
     batch-insert vectors + register in DuckDB datasets/vector_collections

ingest_sae_activations(jsonl_path, model_id, sae_id, batch_size, progress_callback)
  1. Count lines for progress tracking
  2. Stream JSONL, batch into DataFrames (default 5000 records)
  3. INSERT OR REPLACE into sae_activations
  4. Skips malformed lines with warning (logs first 5)
```

### DuckDB Client Methods (`clients/duckdb_client.py`)

| Method | Purpose |
|--------|---------|
| `insert_sae_features_batch(model_id, sae_id, df)` | Bulk insert features |
| `insert_sae_activations_batch(df)` | Bulk insert activations |
| `get_sae_feature(model_id, sae_id, feature_index)` | Single feature lookup |
| `get_sae_activations(model_id, sae_id, feature_index, limit)` | Top activations by max_value |
| `search_sae_features(model_id, sae_id, query, min_density, max_density, limit, offset)` | ILIKE label search + density range filter |
| `list_sae_models()` | Distinct (model_id, sae_id) with feature/activation counts |
| `delete_sae_data(model_id, sae_id)` | Remove all features + activations for a pair |

### GraphQL API

**Queries** (`API/queries.py`):
| Query | Args | Returns |
|-------|------|---------|
| `saeModels` | — | `[SaeModelInfo]` — model_id, sae_id, feature_count, activation_count |
| `saeFeature` | model_id, sae_id, feature_index | `SaeFeature` — density, label, top/bottom logits |
| `saeActivations` | model_id, sae_id, feature_index, limit | `[SaeActivation]` — tokens, values, max_value, max_value_token_index |
| `saeFeatureSearch` | model_id, sae_id, query, min_density, max_density, limit, offset | `[SaeFeatureSearchResult]` — feature + optional activation_count |

**Mutations** (`API/mutations.py`):
| Mutation | Input | Returns |
|----------|-------|---------|
| `ingestSaeFeatures` | `IngestSaeFeaturesInput` (parquet_path, model_id, sae_id, store_vectors) | `IngestSaeResult` |
| `ingestSaeActivations` | `IngestSaeActivationsInput` (jsonl_path, model_id, sae_id) | `IngestSaeResult` |

**Strawberry Types** (`API/types.py`):
`SaeLogitEntry`, `SaeFeature`, `SaeActivation`, `SaeModelInfo`, `SaeFeatureSearchResult`, `IngestSaeFeaturesInput`, `IngestSaeActivationsInput`, `IngestSaeResult`

## Frontend

### Feature Explorer Page (`/sae`)

```
app/sae/
  page.tsx                       — orchestration: URL state, GraphQL queries, layout
  components/
    FeatureHeader.tsx             — model/SAE selector, feature index nav (prev/next),
                                    text search input, "View in scatter plot" link
    FeatureDetailCard.tsx         — feature metadata: index badge, density, label,
                                    top/bottom logit bar charts
    LogitBarChart.tsx             — horizontal bar chart for logit entries
                                    (blue = positive, orange = negative)
    ActivationExamples.tsx        — list of top-activating token strip heatmaps
    TokenStrip.tsx                — inline token sequence with activation heatmap coloring
                                    (orange-red intensity, hover shows value)
    FeatureSearchResults.tsx      — tabular results with index, label, density columns
```

### TypeScript Types (`lib/types/types.ts`)

```typescript
SaeLogitEntry    { token, score }
SaeFeature       { modelId, saeId, featureIndex, density, label, topLogits, bottomLogits }
SaeActivation    { id, tokens[], values[], maxValue, maxValueTokenIndex }
SaeModelInfo     { modelId, saeId, featureCount, activationCount }
SaeFeatureSearchResult { feature, activationCount }
```

### GraphQL Operations (`lib/graphql/queries.ts`, `mutations.ts`)

| Constant | Operation |
|----------|-----------|
| `GET_SAE_MODELS` | Query all model/SAE pairs |
| `GET_SAE_FEATURE` | Query single feature by index |
| `GET_SAE_ACTIVATIONS` | Query top activations for a feature |
| `SEARCH_SAE_FEATURES` | Search features by label/density |
| `INGEST_SAE_FEATURES` | Mutation: ingest parquet |
| `INGEST_SAE_ACTIVATIONS` | Mutation: ingest JSONL |

### Collection ↔ SAE Mapping (`lib/utils/saeCollections.ts`)

Mapping between visualization collections and SAE identifiers. **Primary lookup is dynamic**: `getSaeInfoFromMetadata()` reads `sae_model_id`/`sae_id` from the collection's DuckDB metadata (written by `prepareSaeData`). The hardcoded `SAE_ENTRIES` table below is the **fallback** for legacy collections only (`getSaeInfo()`), used as `getSaeInfoFromMetadata(metadata) ?? getSaeInfo(name)`.

```typescript
SAE_ENTRIES[]           — [{collectionName, modelId, saeId}, ...]
COLLECTION_TO_SAE       — collection name → {modelId, saeId}
SAE_TO_COLLECTION       — "modelId::saeId" → collection name
getSaeInfo(name)        — helper returning SaeIdentifier | null
SAE_FEATURE_INDEX_FIELD — metadata field name ("index") for cross-linking
```

### Cross-Linking

**Visualization → Features** (right-click):
- `ScatterPlot2D`/`ScatterPlot3D` track the hovered point in a ref
- On `contextmenu`, if the point has SAE metadata (`index` field), a context menu appears with "View Feature #N"
- Clicking navigates to `/sae?modelId=...&saeId=...&featureIndex=N`
- The `saeInfo` prop is threaded: `page.tsx` → `DashboardPanel` → scatter plots

**Visualization → Features** (header button):
- `AppHeader` has a "Features" button that links to `/sae`
- When the selected collection is an SAE collection, the link includes `modelId`/`saeId` params

**Features → Visualization**:
- `FeatureHeader` shows a "View in scatter plot" link when the current SAE maps to a collection

### URL State

The features page reads and writes URL search params for deep-linking:
```
/sae?modelId=gemma-3-4b-it&saeId=9-gemmascope-2-res-16k&featureIndex=42
```

## Planned Enhancements

### Semantic Feature Search
Use the existing `SEMANTIC_SEARCH` GraphQL query on the SAE collection (`Gemma_9_16k_embedded`) which has 2560-d explanation-embedding vectors. The current text search (ILIKE) would be complemented by vector similarity search. Returned IDs map directly to feature indices.

### Histograms
- Per-feature activation value distribution (data already on client from `GET_SAE_ACTIVATIONS`)
- Density distribution across all features (requires aggregation query or fetching all densities)
- Logit score distribution (data already on client)

### Most Similar Features
- **By label/explanation embedding**: Use `SEMANTIC_SEARCH` on the SAE collection with the current feature's label as query. Results are features with similar explanations.
- **By activation pattern**: Would require computing similarity metrics over activation profiles — a more complex backend capability.
