# Embedding models, prompts and load options

How the backend knows its embedding models, turns a collection's prompt into what the model
receives, loads sentence-transformers models, and keeps secrets out of ChromaDB. The task-prefix
system of EmbeddingGemma and Gemini Embedding 2 has its own page: `EMBEDDING_TASK_PREFIXES.md`.
Media inputs: `MULTIMODAL_EMBEDDING.md`.

## Model catalog

`interpretability_backend/config/embedding_models.json`, read by `EmbeddingModelCatalog`
(`backend/utils/embedding_model_catalog.py`, torch-free, validated: unknown keys and wrong types
raise). It is the single source for:

- **the known models per provider**, keyed by `EmbeddingProviderEnum` value
  (`"sentence_transformers"`, `"openai"`, …). Each entry:

  | Key | Meaning |
  |---|---|
  | `model` | model name (required) |
  | `description` | shown next to the suggestion |
  | `modalities` | default `["text"]`; EG2 lists `text`, `image`, `audio`, `video` |
  | `prompts` | the model's own prompt names (from its sentence-transformers config) |
  | `default_for` | data types whose default model this is (`["multimodal"]`) |
  | `encoders` | media modality → model-config attribute of its tower (`"image": "vision_config"`) |
  | `max_length` | input limit in tokens, when the model files don't state it reliably |
  | `prompt_format` | `"task_prefix"` for EmbeddingGemma and Gemini Embedding 2 |

- **the prompt-name vocabulary:** the union of every entry's `prompts` plus `extra_prompt_names`
  (`s2p`, `s2s`). It tells a prompt *name* the loaded model lacks (an error) from literal prompt
  text;
- **the document → query maps** used when a query carries no explicit prompt:
  `query_prompt_map` (`Retrieval-document` → `Retrieval-query`, `Document` → `SearchQuery`,
  `document` → `query`) and `query_task_type_map` for Gemini (`RETRIEVAL_DOCUMENT` →
  `RETRIEVAL_QUERY`);
- **the task-prefix table** (`task_prefixes`, see `EMBEDDING_TASK_PREFIXES.md`).

The catalog lists only models the repo already knew, plus `google/embeddinggemma-2`. Dimensions
come from `backend/utils/known_dimensions.json` (`KnownDimensions`, shared by the factory and the
catalog query): the factory records each newly probed model there.

**GraphQL `embeddingModelCatalog`** (synchronous, torch-free) returns the catalog entries with
their dimensions (`suggested: true`), then the models existing collections use
(`DuckDBClient.embedding_models_in_use()`) with their `collectionCount`; models that are only used,
such as local checkpoint paths, come with `suggested: false`. The read-only demo returns the
catalog only, so local paths never reach visitors. The embed forms use it for the model-name and
prompt pickers ("Suggested" / "Used in your collections").

## Prompt resolution

`backend/utils/embedding_prompts.py`, used by the sentence-transformers wrapper
(`embed_sentence_transformer.py`) once, at construction:

**Document side, `DocumentPromptResolver`:**

| The collection's prompt is… | Sent as |
|---|---|
| a name in the loaded model's `prompts` | `prompt_name=` |
| a known name (catalog vocabulary) the model lacks | **error** listing the model's prompts |
| anything else | literal `prompt=` text |

Before, a static name list sent unknown names (EmbeddingGemma's `query`, `document`,
`SearchQuery`…) as literal text, silently prefixing every input with e.g. "SearchQuery".

**Query side, `QueryPromptResolver`:** an explicit query prompt wins. Otherwise the candidates
are the mapped name (`query_prompt_map`), then the stored name, and the first one the model
defines is used. So fine-tuned checkpoints that kept only their document prompt still work, and
symmetric prompts (`Clustering`, `STS`) reuse themselves. Gemini task types map the same way.

The frontend sends `null` for "Auto" and the backend resolves it; it no longer maps prompts
itself. Text queries are embedded explicitly by `ChromaDBClient.embed_query` /
`semantic_search` with the collection's own model; a model that fails to load **raises**. It used
to fall back silently to ChromaDB's default embedder.

## Load options

`backend/utils/model_load_options.py`:

- **`EmbeddingDtypePolicy`** (lazy torch import): bfloat16 on a bf16-capable CUDA device. On MPS
  (macOS 14+), bfloat16 only for checkpoints saved in bfloat16 (EG2, read from `config.json`),
  float32 otherwise, so EmbeddingGemma 1 collections see no drift. CPU is float32. **Never
  float16:** it overflows EmbeddingGemma's activations into NaN or silently degraded vectors.
- **`SentenceTransformerLoadOptions`**: the dtype, the encoder towers to skip (`config_kwargs`,
  from the requested modalities and the catalog's `encoders`), a `cache_key`, and `covers()`. The
  wrapper's single-slot cache reuses a cached load that covers a request; anything else evicts it
  and reloads. Table: `MULTIMODAL_EMBEDDING.md` "Encoder towers".
- **`create_embedding_function(config, device, modalities=("text",))`**: text datasets and every
  text query load text-only. Non-sentence-transformers providers reject media modalities.

## No embedding function in ChromaDB

Collections store vectors only; every write and every query passes explicit vectors. So
`embedding_job.open_embedding_collection` creates every new collection with
`embedding_function=None` passed **explicitly**. Omitting the argument would make Chroma 1.x
record its default 384-d ONNX embedder. A recorded embedding function also stores its config
(`get_config()`), which is how the HF token leaked (below), and a stored config whose name matches
a Chroma built-in (`sentence_transformer`) can be rebuilt by Chroma into its own embedder when the
config is read. The seed exporter keeps its own `{"type": "legacy"}` marker
(`seed_snapshot._LegacyMarkerEmbeddingFunction`), which serves the same purpose for seeds.

## HF token scrub

The sentence-transformers wrapper used to return its load kwargs, the Hugging Face `token` among
them, from `get_config()`, and ChromaDB persisted that config for every collection in
`chroma.sqlite3` (`collections.config_json_str`, and twice in the Rust sysdb's `schema_str`).
`get_config()` now never returns secrets, and new collections store no embedding function at all.
Existing collections are cleaned by `scripts/scrub_chroma_ef_secrets.py`
(`ChromaConfigScrubber`, `backend/utils/chroma_config_scrubber.py`): it removes only keys inside
`embedding_function` objects, never prints values, and is idempotent.

**Runbook** (config `scripts/scrub_chroma_ef_secrets_config.toml`: `dry_run = true`,
`backup = true`, `secret_keys = ["token", "use_auth_token", "hf_token", "api_key"]`):

1. Stop the backend (the script refuses to run while it holds the store).
2. Dry run: `uv run python -m interpretability_backend.scripts.scrub_chroma_ef_secrets`. It
   lists the collections whose configs hold secrets and how many keys it would remove (never the
   values).
3. Set `dry_run = false` and run again: it writes a timestamped `chroma.sqlite3.<time>.bak` (through
   sqlite's backup API), applies everything in one transaction and re-reads to verify.
4. Restart the backend and spot-check a semantic search.

## Dependencies

sentence-transformers ≥ 6.1 (ordered multimodal dict inputs), transformers ≥ 5.19 (the first
release with `embedding_gemma2`; it deprecates `torch_dtype=` in favour of `dtype=`), flagembedding ≥ 1.4.2
(1.3.5 failed to import on transformers 5: `is_torch_fx_available`), torchvision (EG2's video
processor), PyAV (audio and video decoding; scipy resamples), and librosa with soundfile
(sentence-transformers' audio extra; no repo code imports them). On linux, torchvision comes from the same CPU wheel index as torch, so the two stay
matched (macOS: torch 2.9.0 / torchvision 0.24.0; linux: 2.13.0+cpu / 0.28.0+cpu).
`unit_tests/test_dependency_surface.py` checks all of these by import only, and
`test_torch_free_import.py` keeps every one of them out of the GraphQL schema's import graph.
