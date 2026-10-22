# Job cancellation and resume

Cancel buttons on the Collections page progress modals, and resuming
interrupted jobs (2026-10-17, extended 2026-10-19).

## What can be cancelled

Every progress modal on the Collections page has a Cancel button:

| Job | Job id (`JobIds`) | How it stops | Work kept | Afterwards |
|---|---|---|---|---|
| Embed: HF, local text / image / vector file, re-embed | `{collection}` | between batches; projections and the embed pipeline's topic step stop mid-step | items embedded so far (DuckDB + ChromaDB), projections already stored | Jobs strip → Resume. A cancel after projections only skipped topic extraction: the collection is usable, so the job is closed |
| Topic extraction | `{collection}` | the clustering UMAP / HDBSCAN fit stops mid-step; LLM labels between topics | nothing (the previous extraction stays active) | run it again |
| Topic reduction | `{collection}_reduce` | between steps; centroid loads (merge trees without stored centroids, legacy semantic) between batches; LLM relabels between topics | nothing (the previous extraction stays active) | run it again |
| Topic evaluation | `{collection}_evaluate` | c-TF-IDF and the metrics (gensim C_v) stop mid-step | nothing (stored scores unchanged) | run it again |
| LLM topic labeling | `{collection}_llm_labeling` | before each topic / subtopic LLM call | labels saved after every call | Jobs strip → Resume |
| SAE preparation (SAE tab) | `sae_prepare_{size}_{variant}_{layer}_{hook}_{width}` | the download / decoder extraction stops mid-step; DuckDB ingestion between steps; the optional collection creation like an embed | downloaded batch files, an extracted parquet | run it again (downloads resume per file) |
| SAE document activations | `{collection}_sae_activations` | before each document; a cancel while items load or another inference holds the GPU skips the model load | every processed document | run Compute again (skips processed items) |

A cancelled job reports the error `Cancelled by user` (`CANCELLED_MESSAGE`) in
its failed progress event and its mutation result. The frontend matches it
(`isJobCancelled` / `jobOutcome`) to show an info toast and a neutral
"Cancelled" result instead of a failure, and the error stops follow-up steps
(the post-embed SAE compute). A cancelled extraction, reduction or evaluation
leaves the panels showing the existing topics and scores.

## How a running step is stopped

Two mechanisms, chosen per step:

- **Cooperative checks.** A job's resolver registers a `threading.Event` under
  its job id (`cancellable_job`); `cancelJob(jobId)` sets it. Loops check it
  between batches, documents, topics and steps (`is_cancelled`,
  `raise_if_cancelled` → `JobCancelled`). `JobCancelled` is a `BaseException`,
  like `asyncio.CancelledError`, so the pipelines' broad `except Exception`
  fallbacks cannot swallow a cancel and carry on; each job entry point catches
  it explicitly.
- **Killable child process (`CancellableTask`).** UMAP, HDBSCAN, PCA, c-TF-IDF,
  gensim coherence and the SAE weight download spend their time in compiled code
  or blocking I/O that a thread cannot interrupt. These run as a
  `CancellableTask` target in a spawned child process: the calling thread polls
  the cancel event and kills the child's process group (the child and anything
  it started) when it is set.
  - The target is a module-level `target(channel, *args, **kwargs)`. It streams
    items (progress, or each finished projection) with `channel.send`, which
    reach `on_item` in the parent, and returns its result. DuckDB allows one
    writing process, so the child never opens the database; every write
    happens in the parent.
  - Numeric arrays of at least `spill_min_bytes` reach the child through a
    temporary `.npy` mapped copy-on-write, not through the pipe.
  - Inputs under `isolate_min_rows` run in-process: a process start costs one
    or two seconds of imports. Inline runs keep the same API and honour a cancel
    at each `channel.send` and at the end. Tunables are in
    `interpretability_backend/config/cancellable_tasks.json`.
  - Stopping: SIGTERM to the child's process group (the child turns it into
    `SystemExit`, so pure-Python steps unwind and flush), SIGKILL after
    `terminate_grace_ms` for a step still inside compiled code, then a final
    SIGKILL to the group once the child is reaped, so worker pools it started
    (joblib, gensim) die with it. The child watches its parent and kills its
    group if the server dies.
  - Files a killed step may leave are safe to reuse. Neuronpedia batch files
    are written to `.partial` and renamed (`write_atomic`), the merged features
    JSONL drops a half-written last line before appending (`drop_torn_tail`),
    and `extract_and_merge` writes its parquet through a `.partial` file too,
    so resume never trusts a truncated file.

## Resume

Embedding jobs store the GraphQL input they were started with in their job-state
config (`config["input"]`, camelCase, built by `graphql_input_payload`).
`resumePlanFor(job)` replays it with `resume: true`, so a resumed job keeps its
columns and template, topic settings and SAE link. Jobs recorded before inputs
were stored fall back to unpacking the snake_case config summary.

The SAE link is part of the embed input (`saeLink`). The pipeline links the
collection to the SAE as soon as every item is embedded
(`link_collection_to_sae`), before the optional projection and topic steps,
so a cancel there cannot drop it. With `computeActivations` (set by the embed
tabs' post-embed SAE step) the frontend then computes document activations,
after a fresh run and after a resumed one (`runPostEmbedSaeStep`). The SAE
tab's feature collections are link-only: `prepareSaeData` builds their embed
input through the same converter, so a resumed SAE collection keeps its link
and topic settings without computing activations. That collection phase is
also registered under the collection's own id, so the Jobs list can cancel it.

A resumed job is marked running again (`JobStateService.resume_job`), so the
Jobs list does not offer Resume for a job that is running, and a failed embed
keeps its interrupted record instead of being marked completed.

Image and vector file embeds now register job state, report progress and
resume (rows already in the collection are skipped) like text embeds.

Fixed along the way: resuming an HF or local-text embed of an existing
collection crashed with `UnboundLocalError`, because the collection metadata was
only built when a new collection was created. Re-embed resume also dropped the
original columns and template.

## Where it lives

| File | Role |
|---|---|
| `backend/services/job_ids.py` | `JobIds`: every job's progress / job-state / cancel id (mirrored by `embedding_visualization/lib/utils/jobIds.ts`; both tested against the same literal examples) |
| `backend/services/cancel_registry.py` | Event registry (`cancellable_job` with identity-checked unregister, `request_cancel`), `is_cancelled` / `raise_if_cancelled` / `JobCancelled`, `report_cancelled` (job state interrupted, or closed with `resumable=False`, + failed event), `emit_cancelled` (event only), `CANCELLED_MESSAGE` |
| `backend/utils/cancellable_task.py` | `CancellableTask` + `CancellableTaskSettings` (see above) |
| `backend/embedding_functions/embedding_job.py` | Shared by all five embedders: `open_embedding_collection` (overwrite, or resume returning the ids already embedded), `EmbeddingBatchJob` (job state, per-batch progress, cancel between batches, result), `report_embedding_failure` |
| `backend/embedding_functions/config.py` | `to_job_config()` on the embed configs: Jobs-list summary fields + the replayable `input` |
| `backend/services/embedding_pipeline.py` | Embed → projections (cancellable) → topics (cancellable) → SAE link → complete |
| `backend/utils/compute_projections.py` | `ProjectionSpec` table + `fit_projections` task target; each finished projection is stored as it arrives |
| `backend/services/topic_extraction_service.py` | `cluster_documents` task target (cluster space + UMAP + clustering), checkpoints, `JobCancelled` handlers for extraction, reduction and LLM labeling |
| `backend/topic_extraction/cluster_and_label.py` | `cluster_labels()` (module-level so a child can run it), `GenerateTopics.documents_frame()` |
| `backend/topic_extraction/llm_labeling.py` | `generate_llm_labels(..., cancel_event)` |
| `backend/services/topic_quality_service.py` | `score_metrics` task target; nothing is persisted on cancel |
| `backend/services/sae_pipeline_service.py` | `run_gemma_pipeline` / `extract_qwen_vectors` task targets; ingestion checkpoints |
| `backend/utils/embedding_loader.py` | `cancellable_embedding_batches`, `load_all_embeddings` (projections), `load_embeddings_for_ids(..., cancel_event)` |
| `backend/services/job_state.py` | `resume_job` (a resumed job runs again), `JobState.source` (incl. re-embed source dataset) |
| `interpret/download/download_neuronpedia_s3.py` | `write_atomic`, `drop_torn_tail` (interrupted downloads stay resumable) |
| `backend/API/mutations.py` | `cancelJob`; every long-running resolver registers its `JobIds` id; one `_run_embedding_pipeline` for the three embed mutations |
| `backend/API/converters.py` | `graphql_input_payload`, `build_sae_link`, `build_reembed_config` |
| `backend/clients/duckdb_client.py` | `update_collection_extra_metadata` (SAE link and other collection settings, on the dataset row a re-embedded collection shares) |
| `embedding_visualization/lib/utils/jobCancellation.ts` | `JOB_CANCELLED_MESSAGE` (guarded by a backend test), `isJobCancelled`, `jobOutcome`, `announceJobOutcome` (the one outcome → toast switch), `cancelButtonView` |
| `lib/hooks/useCancelJob.ts`, `app/collections/components/JobProgressBody.tsx` | The Cancel button (`<ProgressModal cancellable />`) |
| `lib/hooks/useEmbedDataset.ts` | One `runEmbed` runner for the three embed mutations; cancel-aware outcomes for embed, extraction, reduction, labels and activations |
| `app/collections/lib/embeddingFormUtils.ts` | `saeLinkFromParams`, `runPostEmbedSaeStep`, `resumePlanFor`, `resumeJob` |

To make another job cancellable: build its id in `JobIds` (both sides), register
it with `cancellable_job` in the resolver, check the event at the job's
checkpoints or run its heavy step through `CancellableTask`, catch `JobCancelled`
at the entry point and report `CANCELLED_MESSAGE`, and pass `cancellable` to its
`ProgressModal`. Probe training (`{collection}_probe`) is the remaining job
without a Cancel button.

## Tests

- `unit_tests/test_cancellable_task.py` (+ `cancellable_task_workers.py`): results
  and items in both modes, array spilling, error propagation, a busy child and
  its grandchild killed on cancel, settings validation.
- `unit_tests/test_neuronpedia_download_resume.py`: atomic batch writes, torn
  JSONL tails dropped.
- `unit_tests/test_embedding_job.py`: collection opening (create / overwrite /
  resume), batch-job bookkeeping, cancel between batches, failure reporting, the
  local-text resume regression, vector cancel-then-resume, the extra-metadata
  merge for re-embedded collections.
- `unit_tests/test_cancellable_steps.py`: projections (inline cancel between
  projections and while loading; a real child computing PCA), topic extraction
  (cancel before clustering; HDBSCAN in a real child), LLM labels, reduction,
  evaluation, SAE preparation, the pipeline's topic step and SAE link, and each
  mutation registering its `JobIds` id.
- `unit_tests/test_job_cancellation.py`: registry, `cancelJob`, pipeline cancel
  semantics, re-embed, LLM labeling, document activations.
- `unit_tests/test_graphql_input_payload.py`: stored inputs are valid mutation
  variables (checked with graphql-core coercion against the schema).
- `unit_tests/test_job_ids.py` / `lib/utils/__tests__/jobIds.test.ts`: the same
  id examples on both sides.
- `lib/utils/__tests__/jobCancellation.test.ts`, `app/collections/lib/__tests__/resumeJob.test.ts`,
  `saePostEmbed.test.ts`: outcome classification, button states, resume plans
  (stored input and legacy configs), the post-embed SAE step.
