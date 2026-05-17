# SAE Data Pipeline

Download, prepare, and ingest Neuronpedia SAE feature data into DuckDB. The pipeline lives in the `interpret/` module (standalone, no backend dependencies) and is bridged to the backend via a single GraphQL mutation.

Related docs:
- `SAE_ARCHITECTURE.md` — DuckDB schema, GraphQL queries, frontend feature explorer
- `INTERPRET_API.md` — live inference API (prompt activations, steering, highlight)

## Architecture

```
                         interpret/ module (standalone)
    ┌───────────────────────────────────────────────────────────────────┐
    │                                                                   │
    │   Neuronpedia S3 ──download──> JSONL (features + explanations)    │
    │                                    │                              │
    │   Neuronpedia S3 ──download──> batch-*.jsonl.gz (activations)     │
    │                                    │                              │
    │                              merge batches                        │
    │                                    │                              │
    │                                    ▼                              │
    │   HuggingFace ──load SAE──> decoder vectors (w_dec matrix)        │
    │                                    │                              │
    │                        merge vectors + labels                     │
    │                                    │                              │
    │                                    ▼                              │
    │                             Parquet + JSONL                       │
    │                                                                   │
    └──────────────────────────────┬────────────────────────────────────┘
                                   │
                      backend/ bridge (sae_pipeline_service.py)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
             DuckDB sae_features            DuckDB sae_activations
             + ChromaDB vectors             (per-feature examples)
```

## Concepts

### Neuronpedia Source ID

A string that identifies one SAE within a model on Neuronpedia's S3 bucket:

```
{layer}-gemmascope-2-{hook_abbrev}-{width}
```

| Component | Values | Example |
|-----------|--------|---------|
| `layer` | 0-33 (Gemma3-4b has 34 layers) | `9` |
| `hook_abbrev` | `res` (residual), `mlp`, `att` (attention) | `res` |
| `width` | `16k`, `65k`, `262k` | `65k` |

Example: `9-gemmascope-2-res-65k` (layer 9, residual stream, 65k features)

This string is always derived from `GemmaScopeSAEConfig` via `neuronpedia_source_id()` in `interpret/sae/source_ids.py`. Never construct it manually.

### Path Derivation

All file paths are derived from `GemmaScopeSAEConfig` via `interpret/sae/paths.py`:

| Function | Example path |
|----------|-------------|
| `labels_dir(config)` | `resources/sae_labels/neuronpedia_gemma-3-4b-it/` |
| `features_jsonl_path(config)` | `...neuronpedia_gemma-3-4b-it/gemma-3-4b-it_9-gemmascope-2-res-65k_features.jsonl` |
| `activations_jsonl_path(config)` | `...neuronpedia_gemma-3-4b-it/gemma-3-4b-it_9-gemmascope-2-res-65k_activations.jsonl` |
| `activation_batches_dir(config)` | `...neuronpedia_gemma-3-4b-it/activations/9-gemmascope-2-res-65k/` |
| `vectors_parquet_path(config)` | `resources/sae_vectors/w_dec_gemma-3-4b-it_layer9_resid_post_w65k.parquet` |

## Pipeline Stages

### Stage 1: Download from Neuronpedia S3

Downloads three data types per source from `neuronpedia-datasets.s3.us-east-1.amazonaws.com`:

| Data type | Size (per source) | Output |
|-----------|-------------------|--------|
| Features (density, logits) | ~17 MB gz | Merged into `{model}_{source}_features.jsonl` |
| Explanations (labels, embeddings) | ~20 MB gz | Merged into same JSONL |
| Activations (token examples) | ~336 MB gz | Raw `batch-*.jsonl.gz` in `activations/{source}/` |

Activations are opt-in (`skip_activations=True` by default) due to their size.

The download has resume support: already-merged feature indices are skipped, already-downloaded batch files are skipped.

**Code**: `interpret/download/download_neuronpedia_s3.py` — `download_source()`

### Stage 2: Merge Activations

Decompresses raw `batch-*.jsonl.gz` files and concatenates them into a single JSONL sorted by feature index. Only runs when activations were downloaded.

**Code**: `interpret/download/merge_activations.py` — `merge_source()`

### Stage 3: Extract Decoder Vectors

1. Downloads SAE weights from HuggingFace (`google/gemma-scope-2-4b-it`)
2. Extracts the decoder weight matrix `w_dec` (shape: `d_sae x d_in`)
3. Loads Neuronpedia labels from the features JSONL (stage 1 output)
4. Merges vectors + labels into a parquet file

Output parquet schema:

| Column | Type | Description |
|--------|------|-------------|
| `index` | int32 | Feature index (0 to d_sae-1) |
| `vector` | list\<float32\> | 2560-dim decoder direction |
| `density` | float32 | Activation frequency |
| `label` | string | Autointerpreter description |
| `top_logits` | list\<{token, score}\> | Tokens the feature promotes |
| `bottom_logits` | list\<{token, score}\> | Tokens the feature suppresses |

**Code**: `interpret/sae/extract_decoder_vectors.py` — `extract_and_merge()`

## Usage

### CLI (standalone, no backend needed)

```bash
# Full pipeline: download + extract for layer 9, 16k width
uv run python -m interpret.sae.pipeline.prepare_sae_data --layer 9 --width 16k

# Skip download (labels already present locally)
uv run python -m interpret.sae.pipeline.prepare_sae_data --layer 9 --width 16k --skip-download

# Include activation examples (~336 MB download)
uv run python -m interpret.sae.pipeline.prepare_sae_data --layer 9 --width 65k --with-activations

# Non-default hook type
uv run python -m interpret.sae.pipeline.prepare_sae_data --layer 22 --width 16k --hook mlp_out
```

### Python (programmatic)

```python
from interpret.sae.sae_config import GemmaScopeSAEConfig
from interpret.sae.pipeline import SAEPipelineConfig, SAEPipelineRunner

config = SAEPipelineConfig(
    sae=GemmaScopeSAEConfig(layer_index=9, width="65k", device="cpu"),
    skip_activations=True,     # default: skip large activation download
    skip_download=False,       # set True if labels already downloaded
)
result = SAEPipelineRunner(config).run()

# result.features_parquet  -> Path to output parquet
# result.features_jsonl    -> Path to downloaded features JSONL
# result.activations_jsonl -> Path to merged activations JSONL (if downloaded)
# result.model_id          -> "gemma-3-4b-it"
# result.sae_id            -> "9-gemmascope-2-res-65k"
```

### GraphQL (via backend)

The `prepareSaeData` mutation runs the full pipeline and ingests the result into DuckDB in one call:

```graphql
mutation {
  prepareSaeData(input: {
    layer: 9
    width: "16k"
    hookType: "resid_post"
    storeVectors: true
    includeActivations: false
  }) {
    modelId
    saeId
    featuresInserted
    activationsInserted
    durationSeconds
    status        # "completed", "already_ingested", or "failed"
    error
  }
}
```

Returns `status: "already_ingested"` if the `(model_id, sae_id)` pair already exists in DuckDB with a matching parquet file on disk.

Progress is emitted via the existing WebSocket subscription bus with job ID `sae_prepare_{layer}_{hook}_{width}`.

## Module Map

```
interpret/
├── sae/
│   ├── source_ids.py              Canonical source string derivation
│   ├── paths.py                   All file path derivation from SAEConfig
│   ├── sae_config.py              GemmaScopeSAEConfig, QwenScopeSAEConfig, HookType
│   ├── loading.py                 Download + load SAE weights from HuggingFace
│   ├── extract_decoder_vectors.py Extract w_dec matrix, merge with labels -> parquet
│   ├── feature_labels.py          SQLite-backed label lookup (for live inference)
│   └── pipeline/
│       ├── __init__.py            Exports SAEPipelineConfig/Runner/Result
│       └── prepare_sae_data.py    Unified 3-stage orchestrator + CLI
│
├── download/
│   ├── download_neuronpedia_s3.py Bulk S3 download (features, explanations, activations)
│   ├── download_neuronpedia_gemma3_features.py  Alternative REST API download
│   └── merge_activations.py       Decompress + sort activation batches
│
backend/
├── services/
│   └── sae_pipeline_service.py    Bridge: pipeline.run() -> DuckDB ingestion
├── embedding_functions/
│   └── ingest_sae.py              DuckDB insert for features + activations
└── API/
    ├── mutations.py               prepareSaeData mutation
    └── types.py                   PrepareSaeInput, PrepareSaeResult
```

## Data Sizes (typical per source)

| Asset | 16k width | 65k width |
|-------|-----------|-----------|
| Features JSONL | ~15 MB | ~60 MB |
| Activation batches | ~336 MB | ~1.3 GB |
| SAE weights (HF) | ~160 MB | ~650 MB |
| Output parquet (with vectors) | ~160 MB | ~650 MB |
| DuckDB sae_features rows | 16,384 | 65,536 |
| DuckDB sae_activations rows | ~327k | ~1.3M |

SAE weights are cached by `huggingface_hub` after first download.
