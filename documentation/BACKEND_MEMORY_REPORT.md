# Backend memory: idle retention and peak usage

Report and plan, 2026-10-19. Analysis only: no code was changed. Scope is the
backend worker process. Out of scope: the frontend dev server (2.6 GB, mostly
Metal, in the same measurement) and the library-version work another session
is handling.

## Summary

On the 16 GB machine the idle backend had a 14 GB footprint (peak 18 GB).
About 13 GB of it was Metal memory (`IOAccelerator`) owned by torch; the
Python heap was about 0.4 GB. Two problems add up.

**Why the memory never comes back (idle retention)**

1. **Models are never unloaded.** The SentenceTransformer cache is emptied
   only when a *different* model is requested. The interpretability model
   (gemma-3-4b-it, ≈8.6 GB) and its SAE cache are freed only by an explicit
   `unloadModel`. Nothing unloads at idle.
2. **torch's MPS allocator keeps every freed block.** It starts trimming only
   above its *low watermark*, 1.4 × the GPU working-set size: ≈14.9 GiB on a
   16 GB Mac, more than the machine can spare. Its hard cap (1.7×,
   ≈18.1 GiB) matches the observed 18 GB peak. No embedding path calls
   `empty_cache()` after a job.
3. **`empty_cache()` returns only part of it** (about half in the other
   agent's test), for mechanical reasons (section 3.3). Only process exit is
   guaranteed to return everything.

**Why so much was allocated (peak), for EmbeddingGemma 2**
(`google/embeddinggemma-2`, 744M parameters, 1.49 GB of bf16 weights)

1. **Truncation is effectively off.** The tokenizer declares
   `model_max_length = 1e30` (the "infinity" sentinel), and
   sentence-transformers caps it only from a top-level
   `max_position_embeddings`, which this composite config does not have.
   Documents go in at full length; the model supports 8,192 tokens.
2. **The batch size is an implicit 32.** Our embedding function never passes
   `batch_size`, so `encode()` uses its default whatever the document length.
3. **Attention memory is quadratic on MPS.** For this model's head sizes,
   torch runs a generic kernel that materialises the full
   batch × heads × L × L score matrix in every layer: 4.3 GB per tensor for
   32 documents of 4,096 tokens, 17 GB at 8,192.
4. **The unused vision and audio towers** are loaded with the text tower:
   ≈0.94 GB.

A length cap alone is not enough: at the model's own limit (8,192 tokens),
32 documents still need ≈17 GB per score tensor. The cap and a token budget
per batch have to go together.

**Secondary, on the CPU side.** ChromaDB keeps the HNSW index of every
collection it has touched for the life of the process (up to 1.9 GB each on
this checkout), DuckDB's buffer pool has no limit (default 80% of RAM), and
several loaders hold whole datasets or float64 copies during jobs
(section 3.7). Another session's uncommitted work already moves projections,
clustering and SAE preparation into child processes.

**Recommended order:** measure (phase 0); bound the peak with a length cap,
token-budget batching and text-only loading (phase 1); release after every
GPU job and cap the allocator (phase 2); unload idle models (phase 3); run
embedding jobs in a child process (phase 4, the only complete fix). Pros and
cons per option are in sections 4 and 5, the plan in section 6.

**Side finding (security):** the Hugging Face token is stored in plain text in
ChromaDB for 16 collections (section 9). The ChromaDB distance-space plan
already removes it; rotate the token regardless.

## Implementation status (2026-10-19)

Sections 1 to 9 are the analysis as written; their line numbers refer to the
code before the changes below.

**Done by commit e6d28a4 (another session):** text-only loading of multimodal
towers (P3, `utils/model_load_options.py`) and the Hugging Face token kept out
of ChromaDB's stored config (section 9). Projections, clustering and SAE
preparation now run in `CancellableTask` child processes.

**Implemented here:**

- **Length cap and real batch size** (P1; P2 without the token budget yet).
  - `max_length` per model in `config/embedding_models.json` (EmbeddingGemma 2:
    8,192; EmbeddingGemma 1: 2,048). `ModelMaxLengthResolver`
    (`utils/model_max_length.py`) looks it up in order: catalog,
    `sentence_bert_config.json`, the tokenizer (its 1e30 "no limit" sentinel is
    ignored), `config.json`'s `max_position_embeddings`. Model files are read
    without loading the model (`utils/model_files.py`).
  - `EmbeddingModelConfig.max_length` / `batch_size`; GraphQL
    `EmbeddingModelInput.maxLength` and the `embeddingModelMaxLength(provider,
    modelName)` query, which the embed form uses to show the model's limit. An
    empty field means the model's limit; a value above it is rejected by the
    form and by the backend.
  - SentenceTransformers: the cap goes to each `encode()` call
    (`processing_kwargs`), never onto the shared cached model, and the batch
    size is passed through. QWEN embeds in batches of `batch_size` (it used to
    run the whole 100-document chunk in one pass). BGE now runs on the job's
    device (it was always MPS).
  - Every local embedder counts the documents it truncates
    (`utils/truncation.py`, prompt tokens included). The job keeps running;
    the count reaches the progress modal as a `warning` and the result as
    `EmbedDatasetResult.warnings`, shown as a toast.
  - The form's batch size is now texts per forward pass for local models;
    defaults are 32 (BGE 12) for local providers and 100 for APIs.
- **bf16 on Apple GPUs** (P4): `EmbeddingDtypePolicy` keeps checkpoints saved
  in bf16 (EmbeddingGemma 2) in bf16 on MPS from macOS 14; float32 checkpoints
  stay float32. `scripts/compare_embedding_dtypes.py` measures the agreement
  and memory of both dtypes on a machine. EmbeddingGemma 2 collections embedded
  in float32 since e6d28a4 now get bf16 query vectors (expected drift: cosine
  around 0.999; run the script to confirm).
- **DuckDB memory limit** (P7): `config/duckdb.json`, default 25% of RAM,
  applied to every `DuckDBClient` connection (`utils/duckdb_settings.py`).
- **Small fixes** (P6): `modelStatus` no longer creates the interpretability
  service (a status poll imported torch); semantic search on Gemini/OpenAI
  collections no longer probes the device (`device_for`); `unload_model`
  clears the SAE cache even without a model; steering SAEs record the prefill
  only; `run_batch_highlight` no longer keeps the results it hands to its
  callback.

**After code review:**

- BGE passes `devices=` (FlagEmbedding's actual parameter; `device=` was
  silently ignored and the library picked every CUDA GPU, else MPS).
- A tokenizer limit is capped by the model's position table, as
  sentence-transformers does. Qwen3-Embedding's tokenizer claims 131,072 and
  its config 32,768, so QWEN's default cap is now 32,768 instead of the old
  hard-coded 8,192. Add `"max_length": 8192` to its catalog entry to keep the
  old cap.
- Model files: only definitive answers are cached (a parsed file, or a file
  the model doesn't have). Offline, timeout and gated failures are retried, so
  a transient failure no longer pins a checkpoint to float32 until a restart.
  The read-only demo resolves limits from the Hub cache only and never
  downloads.
- The "%" DuckDB limit resolves against the container's cgroup memory limit,
  not the host's RAM, and never below 64 MiB. Zero sizes are rejected.
- Truncation counting goes through `TruncationTracker.count`, which disables
  itself instead of failing the job when a model's tokenizer can't count.
- The embed form commits the length cap on blur only, so an invalid entry
  ("9000" against 8,192) no longer leaves a prefix ("900") committed. Provider
  and model name are debounced together for the limit lookup.

**Still open:** R1 (release after every GPU job), R2 (allocator watermarks),
R3 (idle unload, SAE cache budget, QWEN/BGE caching, ChromaDB reset), R4
(embedding jobs in a child process), the P2 token budget, P5, the remaining
P7 items, and a length cap for document activations.

## 1. What was measured

From the other agent's session on the 16 GB machine (`footprint` on the
backend worker):

| Category | Dirty | Note |
|---|---|---|
| IOAccelerator (Metal) | 13 GB | torch MPS buffers, unified memory |
| untagged VM_ALLOCATE | 414 MB | native allocations |
| MALLOC zones | ~390 MB | the Python heap |
| **Footprint** | **14 GB (peak 18 GB)** | `ps` showed 192 MB: misleading |

A freshly started backend sits at 412 MB. `modelStatus` reported no
interpretability model loaded while the 13 GB stayed resident. In a standalone
test on torch 2.9.0 / MPS, the agent allocated 2.1 GB of tensors and deleted
them all: the process still held 2.1 GB. After `torch.mps.empty_cache()` it
still held 1.08 GB, while `torch.mps.current_allocated_memory()` reported 0.

How to measure: `ps`/RSS does not show Metal memory on macOS. Use the physical
footprint (Activity Monitor's "Memory" column, `footprint <pid>`, or
`proc_pid_rusage` from code) plus `torch.mps.driver_allocated_memory()`, which
counts everything Metal holds for the process. `current_allocated_memory()`
counts live tensors only.

## 2. The model

| | EmbeddingGemma 2 | EmbeddingGemma 1 (for comparison) |
|---|---|---|
| Repo | `google/embeddinggemma-2` (not gated) | `google/embeddinggemma-300m` |
| Parameters | 744,371,512: text ≈270M, vision ≈170M, audio ≈300M | 308M |
| Checkpoint dtype / size | bf16 / 1.49 GB | fp32 / 1.21 GB |
| Text tower | 24 layers, hidden 512, 4 query heads, head_dim 256 (512 on the 4 full-attention layers), sliding window 512 | 24 layers, hidden 768, 3 heads, head_dim 256, sliding window 512 |
| Supported context | 8,192 tokens | 2,048 tokens |
| Effective truncation in our code | none (tokenizer sentinel, see 3.4) | 2,048 (set in its `sentence_bert_config.json`) |
| Precision guidance | bf16 or fp32; fp16 produces NaN | bf16 or fp32; no fp16 |
| Library needs | sentence-transformers ≥ 6.1 (declared by the model) and a transformers release that knows `embedding_gemma2` (checkpoint saved with 5.18.0.dev0) | locked versions are fine |

The model was described as "720M": it has 744M parameters in total, but only
the ≈270M text tower is used for text embedding. Sources: the model's
`config.json`, `sentence_bert_config.json`, `tokenizer_config.json` and Hub
metadata;
[Google's EmbeddingGemma page](https://ai.google.dev/gemma/docs/embeddinggemma)
and
[Unsloth's EmbeddingGemma 2 guide](https://unsloth.ai/docs/models/embeddinggemma-2),
which gives the text-only loading recipe and the fp16 warning.

Version caveat: the torch internals below were read from the locked torch
2.9.0 (headers and binary in `.venv`). EmbeddingGemma 2 needs newer
transformers and sentence-transformers and may run with a newer torch. The
allocator defaults have been stable across releases, but phase 0 should
re-check the attention-kernel claim in 3.4 on that machine.

## 3. Root causes

### 3.1 Models are never unloaded

Nothing in the backend unloads a model on a timer, at idle or after a job.
Paths are relative to `interpretability_backend/`; files that other sessions
are editing right now are cited by function name rather than line.

| Holder | What stays resident | When it is released |
|---|---|---|
| SentenceTransformer cache, `backend/embedding_functions/specific_functions/embed_sentence_transformer.py:43` (class-level dict) | The last embedding model: 1.49 GB for EmbeddingGemma 2 as loaded today | Only when a *different* model is requested (`:86-90`, eviction at `:94-118`). Each embedding-function instance also keeps `self._model` (`:91`), so an evicted model survives while any instance does |
| Interpretability service, `backend/services/interpret_service.py:267-294` (process-wide singleton, `backend/API/interpret_instance.py:16-26`) | gemma-3-4b-it ≈ 8.6 GB bf16, including a ≈0.8 GB vision tower that text use never touches (Qwen3-1.7B ≈ 3.4 GB) | Only through the `unloadModel` mutation, or a model switch inside `_compute_document_activations` (`backend/API/mutations.py`), which then leaves its own model loaded |
| SAE weight cache, `interpret/sae/loading.py:121` (plain dict, no size limit) | Every SAE used since the model was loaded: ≈168 MB (16k) to ≈2.7 GB (262k) each for gemma-3-4b. One default `runPromptActivations` call loads 4 | `clear_sae_cache()` inside `unload_model`, which is skipped when no model is loaded (early return at `interpret_service.py:314-315`) |
| QWEN and BGE embedding functions (`embed_qwen.py`, `embed_bge.py`) | Not cached at all: every semantic or topic search on such a collection builds a new embedding function and reloads the model (`ChromaDBClient._embedding_function_for` in `backend/clients/chromadb_client.py`) | Freed after the query, but the blocks stay in the allocator pool, and every query pays a full model load |

Two more findings from the same review:

- **Probe training does not use the GPU.** `ProbeModel` is never moved to a
  device and the sklearn kinds run on numpy, so probing adds only CPU memory
  during a job. This corrects the earlier report, which counted probe training
  among the GPU jobs.
- **Only two backend code paths call `empty_cache()`:** `unload_model`
  (`interpret_service.py:327-331`) and the model-switch eviction
  (`embed_sentence_transformer.py:107-114`). Both are hand-written
  `gc.collect()` + `empty_cache()` sequences, and the offline probing engine
  has a third (`interpret/probing/orchestrator.py:404-408`). Nothing calls
  `torch.mps.synchronize()`, sets an allocator watermark or records memory
  use.

### 3.2 The allocator does not give memory back by itself

torch's MPS allocator (`torch/include/ATen/mps/MPSAllocator.h`, torch 2.9.0)
caches freed buffers in pools and reuses them. Two limits are relative to
Metal's `recommendedMaxWorkingSetSize`, which is about 10.7 GiB on a 16 GB
Mac (`torch.mps.recommended_max_memory()` prints it):

| Limit | Default | 16 GB Mac | Effect |
|---|---|---|---|
| Low watermark (`PYTORCH_MPS_LOW_WATERMARK_RATIO`) | 1.4 on unified memory (line 357) | ≈14.9 GiB | Above it the allocator garbage-collects cached buffers and commits command buffers more often |
| High watermark (`PYTORCH_MPS_HIGH_WATERMARK_RATIO`) | 1.7, upper bound 2.0 (lines 352-354) | ≈18.1 GiB | Hard limit: allocations beyond it raise "MPS backend out of memory" |

So on this machine the allocator never trims its cache until the system is
already deep in swap, and it lets one process reach 18 GiB, the observed
peak. Nothing on our side bounds it: no watermark is configured, and the
embedding path never calls `empty_cache()` after a job.

### 3.3 Why `empty_cache()` returns only part of it

`empty_cache()` releases cached buffers and any Metal heap left empty. There
are three things it cannot release, all visible in the allocator header:

1. **Buffers still in flight.** A freed buffer still referenced by an
   uncompleted command buffer is parked in `buffers_pending_free` until the
   command buffer's completion handler runs. Without `torch.mps.synchronize()`
   first, `empty_cache()` skips it. Probably minor in our code, since most
   operations end with a CPU read-back that synchronises, but cheap to rule
   out.
2. **Heaps pinned by a survivor.** Allocations of 10 to 512 MiB are packed
   into shared 1 GiB heaps (`kXLargeHeapU`, lines 29-30 and 150-156), and a
   heap is freed only when it holds no buffer at all
   (`TORCH_INTERNAL_ASSERT(!n_buffers)` in `releaseMTLHeap`). One surviving
   tensor in the same heap as the activations, such as a model weight or a
   cached rotary table, keeps the whole gigabyte resident. EmbeddingGemma 2's
   embedding matrix (262,144 × 512 in bf16 ≈ 268 MB) lands in exactly such a
   heap.
3. **Memory outside torch's allocator.** MPSGraph allocates the intermediates
   of the graphs it runs itself (the header notes "implicit allocations from
   MPS backend"). They show in `driver_allocated_memory()` but not in
   `current_allocated_memory()`, and `empty_cache()` cannot free them. The
   attention kernel in 3.4 is an MPSGraph graph, so its score matrices are
   candidates. torch also caches one compiled graph per op and input shape,
   and length-sorted batches produce a new shape almost every batch.

These mechanisms are consistent with the other agent's measurement, but which
one dominates has not been measured. Phase 0 separates them by comparing
`driver_allocated_memory()` after `empty_cache()` with and without
`synchronize()`, and with the model unloaded versus loaded.

### 3.4 Peak memory per batch

**No effective truncation.** The factory passes the constructor nothing but a
Hub token (`create_embedding_function.py:137`, used at
`embed_sentence_transformer.py:89`), so sequence length, dtype and towers all
come from the model's own files. sentence-transformers (5.5.1 locally, the
same logic in 6.1.0) caps the tokenizer at `config.max_position_embeddings`
only when the top-level config has that attribute
(`base/modules/transformer.py:667-677`; lines 874-884 in 6.1.0).
EmbeddingGemma 2's config is composite, with the value inside `text_config`
(262,144), and the tokenizer declares `model_max_length = 1e30`. Either way the
effective limit is far above the 8,192 tokens the model was trained on, so
documents are embedded at full length. That is a quality problem as well as a
memory one. One-line check on the other machine: print `model.max_seq_length`
after loading.

**Implicit batch size.** `encode()` defaults to `batch_size=32`, and the
embedding function never overrides it (`embed_sentence_transformer.py:129-141`).
Each call receives a pipeline chunk of 100 documents (`batch_size: int = 100`
in `backend/embedding_functions/config.py`). Inputs are length-sorted, by
the pipeline and again inside `encode()`, so the first batches hold the
longest documents and set the peak. The allocator then keeps that peak for the
rest of the process's life (3.2).

**Quadratic attention on MPS.** The torch 2.9.0 binary ships fused MPS
attention kernels (`sdpa_vector*`) only for head dimensions 64, 96 and 128.
EmbeddingGemma 2 (head_dim 256 and 512) therefore falls back to
`sdpa_general_mps`, an MPSGraph graph that materialises the whole score
matrix. With bidirectional attention the sliding window is applied as a mask,
so all 24 layers compute the full L × L matrix, not only the 4 full-attention
layers. Gemma-family models in transformers also build a full and a sliding
boolean mask per forward pass, each B × L × L bytes.

One attention score tensor is B × 4 heads × L² × 2 bytes (bf16). At least two
are alive per layer (scores and probabilities), twice that if MPSGraph keeps
fp32 intermediates:

| Longest document in batch (L) | B = 32 | B = 8 | B = 1 |
|---|---|---|---|
| 512 tokens | 67 MB | 17 MB | 2 MB |
| 1,024 | 268 MB | 67 MB | 8 MB |
| 2,048 | 1.07 GB | 268 MB | 34 MB |
| 4,096 | 4.3 GB | 1.07 GB | 134 MB |
| 8,192 | 17.2 GB | 4.3 GB | 537 MB |

Example: 32 documents of about 4,096 tokens need roughly 2 × 4.3 GB for scores
and probabilities, 1.1 GB of masks, 1.5 GB of weights and about 1 GB of other
activations. That is around 12 GB at the first batch, which the allocator then
keeps: the order of magnitude observed. For EmbeddingGemma 1 (fp32, 3 heads)
the formula is B × 3 × L² × 4 bytes: 1.6 GB per tensor at B = 32,
L = 2,048.

### 3.5 Unused vision and audio towers

sentence-transformers loads every tower in the checkpoint unless told
otherwise. For text embedding the vision (≈170M) and audio (≈300M) encoders
are dead weight: about 0.94 GB in bf16. The model's text-only recipe passes
`config_kwargs={"vision_config": None, "audio_config": None}`; our embedding
function has no way to pass `config_kwargs` today.

### 3.6 The same pattern elsewhere

Other local models share the "no length cap, batch not bounded by tokens"
shape, and the allocator keeps their peak too:

- **QWEN embeddings** (`embed_qwen.py`): the `batch_size=12` parameter
  (`:13`) is never used. Each call embeds the whole pipeline chunk of 100
  documents in a single forward pass at up to `max_length=8192` (`:15`,
  `:64-70`). With Qwen3-Embedding-0.6B's 16 heads, if it takes the same
  generic attention path, one score tensor for 100 documents of 2,048 tokens
  is ≈13 GB.
- **BGE embeddings** (`embed_bge.py:25-31`): batch 12 at up to 8,192 tokens in
  fp16, always on MPS.
- **Document activations** (`_compute_document_activations`): one document per
  forward pass, with no truncation. Attention runs in fp32 after the mask add
  (≈1.4 GB transient at 4k tokens), and the dense SAE activations stay on the
  device until the next document. The job also keeps every document's results
  in a Python list, although the rows are already written every 50 documents.
- **Steered generation** records SAE activations at every decode step
  (`_make_sae_config` defaults to `prefill_only=False`,
  `interpret_service.py:497`, and `_build_steering_session` uses that
  default), although steering only needs the decoder row, resolved once at
  attach.

### 3.7 CPU-side memory

**Stays resident after a job or request:**

| Holder | Size | Notes |
|---|---|---|
| ChromaDB HNSW index cache (Rust bindings) | About the collection's `data_level0.bin` for every collection touched. On this checkout: 136 segment folders, ≈8 GiB in total, largest 1.9 GB | Any write, embedding read or query loads the collection's index, and it stays loaded. The cache is limited by count (open-file limit ÷ 5: 51 under launchd's default of 256), not by bytes. `chroma_segment_cache_policy` and `chroma_memory_limit_bytes` are not passed to the Rust bindings in chromadb 1.3.4 (`chromadb/api/rust.py:83-92` and `120-125`). `SharedSystemClient.clear_system_cache()` plus `gc.collect()` releases it (verified by the ChromaDB distance-space session) |
| DuckDB buffer pool | Up to 80% of RAM (default `memory_limit`: `DuckDBClient.__init__` calls `duckdb.connect(db_path)` with no config, `backend/clients/duckdb_client.py`) | Cached blocks and primary-key indexes are evicted only under that limit. `sae_document_activations` (≈56M rows) has a primary key and a secondary index |
| torch, transformers and the interpretability toolkit | Import cost, to measure in phase 0 | Imported for good by the first `modelStatus` poll (`backend/API/queries.py:986` creates the interpretability service) or the first text search (`get_device()` imports torch even for Gemini or OpenAI collections, `create_embedding_function.py:234-243`). The torch-free import boundary holds at startup only |
| Numeric stack (sklearn, scipy, umap, pynndescent, numba, hdbscan) | The clustering part alone measured ≈94 MB of RSS, plus numba's compiled code | In committed code these steps run in the server process. The uncommitted work moves them into child processes above 2,000 rows |
| Allocator high-water mark | Unmeasured | Millions of small row dicts and tuples are freed after a job; CPython and macOS malloc rarely hand all of it back |

**Peaks during jobs:**

| Where | Peak |
|---|---|
| Hugging Face "all rows" loads the dataset as Python dicts (`backend/clients/huggingface_client.py:354`), alive for the whole job | The whole dataset, every column |
| Local files: full DataFrame, then `iterrows()` with `.tolist()` on every array cell (`backend/clients/local_data_client.py:284-300`) | ≈8× the array data (Python floats take ≈32 bytes, not 4), with the DataFrame alive at the same time |
| `load_embeddings_for_ids` / `iter_embedding_batches` (`backend/utils/embedding_loader.py`): float64 pages plus a full float64 copy | ≈16·N·D bytes, four times the float32 size (projections moved to float32 in the uncommitted work; this loader still uses float64) |
| Topic extraction: `get_projection_data` parses every item's metadata although only ids, documents and coordinates are used, and the float64 vectors stay alive through LLM labelling | Metadata of the whole collection plus N × D float64 |
| Re-embed: `embed_existing_dataset` calls `get_filtered_items(..., limit=1_000_000)` | Every item with parsed metadata, although only the document is needed |
| Gemma `prepareSaeData`: decoder vectors built with `.tolist()` (`interpret/sae/extract_decoder_vectors.py:169`), and the CPU SAE left in `_SAE_CACHE` | ≈1.3 GB (16k features) to ≈5.4 GB (65k) transient, plus 168 MB to 2.7 GB resident; both move into a child process in the uncommitted work |
| `collection` resolver (one Explore page load) | ≈0.5 GB transient for WordNet (81 MB of JSON, measured in `response_cache.py`); concurrent loads stack |

## 4. Options: getting memory back when idle

### R1. Release after every GPU job

One shared helper in `backend/utils/` (for example a `DeviceMemory` class)
runs `gc.collect()`, then `torch.mps.synchronize()` + `empty_cache()` (or the
CUDA equivalents), and logs `driver_allocated_memory()` before and after. It
must look torch up in `sys.modules` instead of importing it, so a process that
never used torch never loads it (the torch-free import boundary). Call it in a
`finally` at the end of every GPU job: embedding (right after `_do_embed` in
`EmbeddingPipeline.run`, before projections, which are CPU-only), re-embed,
document activations, and model unload. It also replaces the three
hand-written cleanup sequences listed in 3.1.

| Pros | Cons |
|---|---|
| Small, low-risk change | Partial: about half came back in the other agent's test (3.3); `synchronize()` may recover more, unmeasured |
| Also lowers the peak of the projection and topic steps that follow the embedding | The model itself stays loaded |
| One place to log what was freed | The next job re-grows the cache (first batches slightly slower) |
| | Every GPU path must call it; a forgotten path leaks again |

### R2. Cap the allocator

Set `PYTORCH_MPS_LOW_WATERMARK_RATIO` (soft limit: the allocator trims its
cache above it) and `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (hard limit) from a
config file, exported to `os.environ` at startup before torch first touches
MPS. Startup is a safe place because the API never imports torch at module
level; spawned children inherit the values. Example for a 16 GB Mac: low 0.5
(≈5.3 GiB), high 0.9 (≈9.6 GiB). The ratios are relative to each machine's
working-set size, so one setting scales across machines.
`torch.mps.set_per_process_memory_fraction()` can change the high limit at
runtime; there is no Python setter for the low limit.

| Pros | Cons |
|---|---|
| A few lines; bounds the worst case on any machine | A job that needs more than the hard cap fails with "MPS backend out of memory" instead of swapping: pair it with P2's out-of-memory retry |
| The low watermark makes the allocator trim during jobs | Trimming costs some throughput (more allocation churn) |
| Protects the rest of the machine (frontend, browser) | Does nothing at idle by itself: trimming only happens when something allocates |
| | A ratio of 0.0 means "no limit", an easy misconfiguration to guard against |

### R3. Unload idle models

A registry (for example `ResidentModels` in `backend/utils/`) hands out models
through a lease (`with registry.lease(key, loader) as model:`), records the
last use, unloads models idle longer than a configured TTL, then calls R1. An
asyncio task started in `main.lifespan` checks every 30 seconds or so. Config,
for example `config/model_residency.json`:
`{"embedding_idle_seconds": 300, "interpret_idle_seconds": 900, "check_interval_seconds": 30}`.

Two details decide whether it works:

- **Embedding functions must not keep their own model reference.** Today each
  instance stores `self._model` (`embed_sentence_transformer.py:91`), which
  keeps an evicted model alive (the documented gotcha). They should fetch the
  model from the registry on each call.
- **Never unload during use.** A lease covers a whole job, stream or request;
  the interpretability model is also guarded by its existing `asyncio.Lock`.

The same registry covers the other holders:

- **Interpretability model:** the reaper calls `unload_model()` under the
  service's lock, so it never runs during a stream or a batch job, and
  `computeDocumentActivations` stops leaving its model loaded indefinitely.
- **SAE cache:** a byte budget with least-recently-used eviction instead of an
  unbounded dict.
- **QWEN and BGE:** cached through the registry, which ends the full model
  reload on every query.
- **ChromaDB:** the reaper can also drop the client and call
  `SharedSystemClient.clear_system_cache()` (see P7).

| Pros | Cons |
|---|---|
| Frees the weights automatically: 1.5 GB for EmbeddingGemma 2 as loaded today, 8.6 GB for gemma-3-4b-it | The first request after an unload pays a reload (seconds, depending on disk and model) |
| Config-driven, per model family | Concurrency logic (leases, reaper against job threads) needs careful tests |
| Fixes the stale-reference gotcha as a side effect | Still subject to 3.3: what MPSGraph or pinned heaps hold stays until process exit |
| Covers the interactive features, where a child process does not fit | |

### R4. Run GPU batch jobs in a child process

Embedding and re-embedding run in a spawned child that exits when the job
ends; the OS then takes back everything, Metal included. The infrastructure
exists: `backend/utils/cancellable_task.py` (`CancellableTask`) runs a
module-level target in a spawned process, streams items back over a pipe to an
`on_item` callback in the parent, spills large numpy inputs to memory-mapped
`.npy` files, kills the child (and its children) on cancel, and keeps every
DuckDB write in the parent, as DuckDB's single-writer rule requires. A child
embedding target would load the model, embed in token-budget batches (P2) and
`channel.send()` each chunk's ids and vectors; the parent writes DuckDB and
ChromaDB and emits progress as today, so resume and cancel semantics stay the
same.

Status: in committed code nothing calls `CancellableTask`. In the working tree,
another session is wiring projections, topic clustering, topic-quality metrics
and SAE preparation through it (uncommitted). The embedding step itself is not
part of that work: `_do_embed` still runs in a thread of the server process.

Notes: `_ArraySpill` handles numeric arrays only, so large text inputs need a
text spill (for example a temporary Arrow file), or the child reads the source
file or Hugging Face dataset itself (never DuckDB). Query-time embedding
(semantic and topic search) stays in-process with R3: it embeds one short
text, so its activation memory is negligible. Document activations could move
too, but they share the interactive model, so the parent would have to unload
its copy first.

| Pros | Cons |
|---|---|
| The only option that returns 100%, whatever torch or Metal retain | Startup cost per job: process spawn, torch/transformers import (a few seconds) and model load |
| A crash or out-of-memory in the child cannot take down the API | Results cross a pipe (≈3 MB per 1,000 vectors of 768 floats: fine) |
| The API process stays torch-free (~0.4 GB) during embedding jobs | If the API process also holds the same model for queries, both copies coexist during a job |
| Reuses existing, tested code (`CancellableTask`) | Debugging and logging inside a child are less convenient |
| Small jobs can keep running inline (`isolate=SETTINGS.should_isolate(n_rows)`) | Text inputs need the spill extension above |

## 5. Options: lowering peak usage

### P1. Explicit sequence-length cap (correctness and memory)

A per-model `max_seq_length` from config, passed to the constructor (or as
`processor_kwargs={"model_max_length": N}`, the form the sentence-transformers
docs prefer). Default to the model's supported context (8,192 for
EmbeddingGemma 2), so nothing is silently truncated below what the model
handles; lower it (for example to 2,048) where speed matters more than the
tail of long documents. For long-document corpora, an alternative is to chunk
each document into windows and average the chunk embeddings: no truncation,
more compute, slightly different semantics.

| Pros | Cons |
|---|---|
| Removes the unbounded case (a single 32k-token document needs ≈8.6 GB for one score tensor) | Text past the cap is ignored |
| Stops embedding past the trained context, which also degrades quality | Collections embedded with different caps are not strictly comparable: record the cap with the collection |
| | On its own it still allows 17 GB per score tensor at batch 32: needs P2 |

### P2. Token-budget batching with out-of-memory retry

One shared batcher in `backend/utils/` for all local providers
(SentenceTransformers, QWEN, BGE) replaces the implicit 32, QWEN's unused 12
and BGE's 12 with a budget on padded tokens per batch (B × L ≤
`max_batch_tokens`, plus a `max_batch_size`). text-embeddings-inference uses
the same scheme (its default budget is 16,384 tokens). Inputs are already
length-sorted, so the batcher only has to cut the sorted list: long documents
get small batches, short ones large batches. On an out-of-memory error it
releases (R1), halves the batch, retries, and keeps the smaller size for the
rest of the job.

With 16,384 tokens, EmbeddingGemma 2's score tensor stays at or below about
1.1 GB: B = 2 at L = 8,192, B = 8 at 2,048 (268 MB), B = 32 at 512 (67 MB).

| Pros | Cons |
|---|---|
| Bounded peak whatever the document lengths | Our embedding function batches itself (tokenise or estimate lengths first) instead of making one `encode()` call |
| Keeps throughput for short texts | One more tunable per model |
| The retry turns R2's hard cap into a slowdown instead of a failure | Batch sizes vary, so per-batch progress is less uniform |

### P3. Load only the text tower of multimodal models

A per-model `config_kwargs={"vision_config": None, "audio_config": None}`
(EmbeddingGemma 2's own recipe). The embedding function needs a way to pass
`config_kwargs` and `model_kwargs`: today it forwards `**kwargs` to the
constructor and accepts only primitive values
(`embed_sentence_transformer.py:78-80`), so nested dicts work but `torch.dtype`
objects do not (use dtype strings such as `"bfloat16"`).

Do not rely on ChromaDB's stored embedding-function config to carry these
settings: that stored config is how the Hugging Face token leaks (section 9),
and the ChromaDB distance-space plan stops persisting it. Read the profile from
the per-model config at both index and query time, and record the profile used
in the collection's DuckDB `vector_collections` row.

| Pros | Cons |
|---|---|
| ≈0.94 GB less for EmbeddingGemma 2 | Image or audio collections would need a separate profile |
| No numerical effect on text embeddings | Relies on the model's config accepting `None` towers (true for this model; check per model) |

### P4. Dtype policy

Never upcast bf16 checkpoints. transformers ≥ 5 loads the checkpoint's own
dtype by default, so EmbeddingGemma 2 is already bf16; keep that explicit.
Never use fp16 for Gemma models. Optionally load fp32 checkpoints such as
EmbeddingGemma 1 in bf16.

| Pros | Cons |
|---|---|
| bf16 halves weights and activations (EmbeddingGemma 1: 1.21 → 0.6 GB) | Small numeric drift against existing fp32 vectors in the same collection: record the dtype and keep queries consistent, or re-embed |
| | bf16 on MPS needs macOS 14 or later; check speed on older Apple GPUs |

### P5. Shape bucketing (optional, measure first)

Pad batch lengths to a multiple of 64. Fewer distinct shapes mean fewer cached
MPSGraph graphs and better reuse of cached blocks (3.3).

| Pros | Cons |
|---|---|
| May reduce fragmentation and recompilation | Up to 63 padding tokens of extra compute per batch |
| | Benefit unproven; only worth it if phase 0 shows graph-cache or fragmentation growth |

### P6. Small independent fixes

Each is a few lines and can ship on its own:

| Fix | Effect | Risk |
|---|---|---|
| `modelStatus` answers "not loaded" without creating the interpretability service | A status poll no longer imports torch and transformers into the API process | None in practice |
| Call `get_device()` only for local providers | A search on a Gemini or OpenAI collection no longer imports torch | None |
| `unload_model` runs `clear_sae_cache()` and the release even when no model is loaded | Frees SAEs and cached blocks left by a failed or partial load | None |
| Steering sessions build SAEs with `prefill_only=True` (or a no-record mode) | No per-decode-step activation storage on the device during steered chat | Check that nothing reads those recorded activations |
| `computeDocumentActivations` stops collecting results it has already written, and caps document length from config | Bounded CPU growth and a bounded per-document peak | Truncated documents lose their tail features |

### P7. CPU-side options

| Option | Pros | Cons |
|---|---|---|
| DuckDB `memory_limit` and `threads` from config, passed to `duckdb.connect(db_path, config=...)` | One line; a predictable ceiling (for example 1-2 GB) | Large queries spill to `main.duckdb.tmp` (slower) or fail where an operator cannot spill; too low a value slows page loads |
| ChromaDB reset at idle: drop the client singleton and call `SharedSystemClient.clear_system_cache()` from the R3 reaper (the modules that open their own `PersistentClient` must go through the singleton first) | The only way to release HNSW indexes without a restart | Semi-private API (used by Chroma's tests) that may change; the next query reloads the index (seconds for GB-size collections); must not run while a request uses the client |
| float32 vector loading: preallocate a float32 array and fill it page by page in `embedding_loader.py` | Four times lower peak for topic extraction, centroids and evaluation; no API change | Last-digit differences in float64-sensitive steps; exact-value tests need tolerances |
| Stream source rows instead of whole-dataset lists: sort (index, length) pairs and fetch rows by index; read local files in pyarrow batches without `iterrows()` and `.tolist()` | CPU peak bounded by the batch, not the dataset | The largest refactor here; touches the loaders another session is reworking |
| Topic extraction reads only ids, documents and coordinates, and drops the vectors after clustering | Avoids parsing all metadata and holding N × D float64 through labelling | Small; centroids must be computed before the vectors are dropped |

## 6. Plan

Tests are written before each phase's code, per the project rules. Every phase
can ship on its own and is measured with the phase 0 tool on the 16 GB
machine. Phases 1 and 2 are small; 3 and 4 are larger.

**Coordination.** Several files these phases touch are being edited right now
by other sessions: the embedding pipeline refactor (`embedding_job.py`,
`config.py`, the `embed_*.py` files, `embedding_pipeline.py`), the
`CancellableTask` wiring (`cancellable_task.py`, `compute_projections.py`,
`topic_extraction_service.py`, `sae_pipeline_service.py`, `mutations.py`),
the ChromaDB distance-space plan (`chromadb_client.py`, `get_collection`,
`semantic_search`) and `duckdb_client.py`. Land or coordinate with those
before phases 1, 3 and 4; phase 0 and most of phase 2 are independent.

### Phase 0: measure (no behaviour change)

- `ProcessMemoryProbe` in `backend/utils/`: physical footprint via
  `proc_pid_rusage` on macOS (psutil RSS elsewhere), plus
  `current_allocated_memory()`, `driver_allocated_memory()` and
  `recommended_max_memory()` when torch is already imported; DuckDB's
  `duckdb_memory()` when the client exists.
- `scripts/profile_embedding_memory.py`, driven by a TOML config: model,
  sample file, and a list of variants (dtype × length cap × batching ×
  towers). Each variant runs in a fresh spawned process and records the
  footprint after load, at peak, after deleting outputs, after
  `empty_cache()`, after `synchronize()` + `empty_cache()`, and after
  unloading the model. Output is a markdown table.
- Tests first: the probe returns positive numbers and never imports torch;
  the config loader rejects unknown keys (reuse `JsonFileSettings` or the TOML
  pattern of the other runners).
- It loads models, so the user runs it. It answers how much each fix saves
  and which mechanism in 3.3 holds what `empty_cache()` leaves.

### Phase 1: bound the peak (P1 to P4)

- Config `config/embedding_models.json`: defaults plus per-model entries
  (`max_seq_length`, `max_batch_tokens`, `max_batch_size`, `dtype`,
  `text_only`), loaded by a settings class like the existing ones.
- The local embedding functions build `model_kwargs`, `config_kwargs` and
  `processor_kwargs` from the profile, never persist secrets, and batch through
  the shared token-budget batcher with the out-of-memory retry.
- Tests first (stubbed `sentence_transformers`, as in
  `test_sentence_transformer_cache.py`): the profile merge; the constructor
  receives the expected kwargs; no token appears in the stored config; the
  batcher keeps B × L within budget, restores input order, puts an
  over-budget document in a batch of one, and halves after a simulated
  out-of-memory error; QWEN and BGE go through the same batcher.
- Acceptance: EmbeddingGemma 2 job peak under about 4 GB on the 16 GB machine.

### Phase 2: release, caps and quick fixes (R1, R2, P6, part of P7)

- `DeviceMemory.release()` in `finally` blocks on every GPU path, replacing
  the existing cleanup sequences; watermark ratios from a config file,
  exported at startup unless already set; the P6 fixes; DuckDB `memory_limit`
  from config; float32 vector loading.
- Tests first: with a fake `torch` in `sys.modules`, the call order is
  gc → synchronize → empty_cache; without torch imported it is a no-op and
  does not import it; the pipeline calls it after the embed step on success,
  error and cancel, before projections; the settings reject ratios outside
  0 < low ≤ high ≤ 2 and never override an existing environment value;
  `modelStatus` without a service imports no torch (extend
  `test_torch_free_import.py`); the DuckDB client applies the configured
  limit; `load_embeddings_for_ids` returns float32 with unchanged ordering.
- Acceptance: idle footprint a minute after a job within about 1.5 GB of the
  fresh-start baseline (model still loaded).

### Phase 3: unload idle models (R3, ChromaDB reset)

- `ResidentModels` registry with leases and TTL; all local embedding
  functions fetch the model per call; a lifespan task reaps idle models,
  including the interpretability model under its lock, and resets the
  ChromaDB client; `_SAE_CACHE` gets a byte budget.
- Tests first, with a fake clock and stub loaders: eviction after the TTL;
  never while leased; re-acquiring after eviction reloads; eviction calls
  `DeviceMemory.release()`; a thread race between acquire and evict is safe;
  the interpretability model is not unloaded during a stream; the ChromaDB
  reset waits for in-flight requests; the lifespan task starts and stops
  cleanly.
- Acceptance: within the TTL plus one check interval, the footprint is back
  to baseline plus whatever 3.3 leaves (phase 0 says how much).

### Phase 4: embedding jobs in a child process (R4)

- A module-level embedding target run through `CancellableTask` for jobs
  above `isolate_min_rows`; the parent keeps all DuckDB and ChromaDB writes;
  a text spill for large inputs. Optionally afterwards: streaming source rows
  (P7).
- Tests first, with a fake encoder in the child: chunks arrive in order and
  are written per chunk; cancel kills the child and keeps written chunks
  (resumable); child errors surface as `TaskFailed`; inline and isolated runs
  give identical results; the parent never imports torch for an isolated job.
- Acceptance: footprint back to the fresh-start baseline after every
  embedding job.

## 7. Expected results (estimates, to confirm in phase 0)

| State | Idle after an EmbeddingGemma 2 job | Peak during the job |
|---|---|---|
| Today | 12-14 GB | Up to the 18 GB hard cap |
| After phase 1 | ≈2-3 GB (model plus a bounded cache) | ≈3-4 GB |
| After phase 2 | ≈1-1.5 GB (model plus what `empty_cache()` cannot free) | ≈3-4 GB, never above the cap |
| After phase 3 | ≈0.4 GB plus the 3.3 leftover, after the TTL | Unchanged |
| After phase 4 | ≈0.4 GB right after the job | Unchanged, in the child |

On top of these: the HNSW indexes of the collections touched (until the
phase 3 reset) and DuckDB's buffer pool (bounded after phase 2).

## 8. Decisions needed

1. **Truncation:** cap at the model's context (8,192) and rely on token-budget
   batching, cap lower (for example 2,048) for speed, or chunk and average
   long documents?
2. **Idle TTLs:** how long to keep the embedding model and the
   interpretability model, given that the first request after an unload
   waits for a reload.
3. **bf16 for fp32 checkpoints** (EmbeddingGemma 1): new collections only, or
   existing ones too (small drift against their stored vectors)?
4. **DuckDB memory limit** value, and whether the semi-private ChromaDB reset
   is acceptable.
5. **Child process for embedding jobs:** worth the per-job startup cost? It
   builds on `cancellable_task.py`, which another session is wiring up now.
6. **Duplication to clean up** (flagged per the project rules, not fixed):
   three hand-written `gc.collect()` + `empty_cache()` sequences
   (`embed_sentence_transformer.py:107-114`, `interpret_service.py:327-331`,
   `interpret/probing/orchestrator.py:404-408`), and three local providers
   that each batch their own way (SentenceTransformers' implicit 32, QWEN's
   unused `batch_size`, BGE's 12). Phases 1 and 2 replace both with one shared
   utility each.

## 9. Side findings (not memory)

- **Hugging Face token stored in plain text.** The factory passes the token
  as a constructor kwarg (`create_embedding_function.py:137`), the fork's
  `get_config()` returns the kwargs (`embed_sentence_transformer.py:174-180`),
  and ChromaDB persists them in `collections.schema_str`: 16 collections in
  the live store on this checkout. The committed seed has none. The ChromaDB
  distance-space plan already stops persisting embedding-function config and
  migrates those 16 collections. Rotate the token in any case, especially if
  this `vector_db` folder has been copied to another machine.
- **Progress events cross threads unsafely.** `emit_progress` and
  `emit_token` call `asyncio.Queue.put_nowait` from worker threads
  (`progress_emitter.py:97`, `token_emitter.py:74`). asyncio queues are not
  thread-safe, so delivery can stall until the event loop wakes for another
  reason. The fix is `loop.call_soon_threadsafe(queue.put_nowait, event)`.
