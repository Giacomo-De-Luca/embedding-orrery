# Job cancellation

Cancel button on the Collections page progress modal (2026-10-17).

## Problem

Embedding, LLM topic labeling and SAE document activations run for minutes to
hours, and the only way to stop one was to kill the backend from the terminal.
Before this change:

- the embed modal already had a Cancel button, but only HuggingFace and
  local-file embeds registered a cancel event: on a re-embed (manage tab) the
  button called `cancelEmbeddingJob`, which found nothing and did nothing;
- a cancelled embed returned no error, so the frontend showed a success toast
  and the optional post-embed SAE step (`runPostEmbedSaeStep`) still ran;
- a cancel that arrived after the last batch returned early without updating
  job state, so the job stayed `running` in `job_state.json` until restart;
- LLM labeling and SAE document activations had no cancel at all (the Jobs strip
  hid Cancel for `llm_labeling` jobs).

## Behaviour

Cancellation is cooperative. A job stops at its next checkpoint and keeps the
work it has done:

| Job | Modal / job id | Stops | Work kept | Afterwards |
|---|---|---|---|---|
| Embed (HF, local text file, re-embed) | `{collection}` | between batches; also between embed → projections → topic extraction | items embedded so far (DuckDB + ChromaDB) | Jobs strip → Resume (skips existing ids, recomputes projections). A cancel after projections only skipped topic extraction: the collection is usable, so the job is closed instead |
| LLM topic labeling | `{collection}_llm_labeling` | before each topic / subtopic LLM call | labels saved after every call; the topic list refreshes | Jobs strip → Resume (skips topics with LLM labels) |
| SAE document activations | `{collection}_sae_activations` | before each document. The cancel is registered as the resolver starts, so a cancel while items load or another inference holds the GPU stops before the model load | every processed document (the pending batch is flushed) | run Compute again (skips processed items) |

A step that is already running finishes first: one embedding batch, one LLM
call (including its retries), one document's forward pass, or a whole PCA/UMAP
projection or topic extraction, which have no checkpoints inside them. Until the
job stops, the button reads "Cancelling…" and stays disabled. A cancel that lands
during the embed pipeline's final topic extraction is accepted but has nothing
left to stop, so the job completes normally.

A cancelled job reports the error `Cancelled by user` (`CANCELLED_MESSAGE`) in
its `failed` progress event and its mutation result. The frontend matches the
string (`isJobCancelled`) to show an info toast and a neutral "Cancelled" result
card instead of a failure, and the non-null error is what stops the post-embed
SAE step.

Image and vector local-file embeds have no per-batch check (nor job state to
resume), so their progress modal does not offer Cancel.

## Where it lives

| File | Role |
|---|---|
| `interpretability_backend/backend/services/cancel_registry.py` | Thread-safe registry of `threading.Event`s keyed by job id. `cancellable_job(job_id)` registers one for a job's lifetime (identity-checked unregister, so a late-finishing job never removes a newer job's event); `request_cancel`, `is_cancelled`, `raise_if_cancelled` / `JobCancelled`, `report_cancelled` (failed progress event + job state: interrupted, or closed with `resumable=False`), `CANCELLED_MESSAGE` |
| `backend/API/mutations.py` | `cancelJob(jobId)` (replaces `cancelEmbeddingJob(collectionName)`). The embed, re-embed and `generateLlmLabels` resolvers wrap their work in `cancellable_job` and pass the event down; `computeDocumentActivations` registers first and runs its body in `_compute_document_activations` |
| `backend/services/embedding_pipeline.py` | `EmbeddingPipeline.run()` checks after the embed step and after projections; `_cancelled()` reports a cancel the batch loop didn't (resumable only while projections are missing) and sets the error |
| `backend/embedding_functions/embed_huggingface.py`, `embed_local_file.py`, `embed_existing_dataset.py` | Per-batch check: `report_cancelled` + return `error=CANCELLED_MESSAGE` with the count embedded so far |
| `backend/services/topic_extraction_service.py` | `llm_labeling_job_id()` (shared with the mutation); `raise_if_cancelled` before each LLM call; an `except JobCancelled` branch reports progress and returns the real label counts |
| `backend/services/interpret_service.py` | `run_batch_highlight(..., cancel_event)` breaks before the next document |
| `embedding_visualization/lib/utils/jobCancellation.ts` | `JOB_CANCELLED_MESSAGE` (mirror of the backend constant, guarded by a backend test), `isJobCancelled`, `CancelState` + `cancelButtonView` (button visibility / label / disabled) |
| `lib/hooks/useCancelJob.ts` | Wraps the `CANCEL_JOB` mutation; resolves whether the backend found the job |
| `app/collections/components/JobProgressBody.tsx` | `cancellable` prop: renders the button, owns the cancel state, toasts when the job was no longer running. `ProgressModal` keys it by `jobId` |
| `app/collections/page.tsx`, `TopicExtractionCard.tsx` | Embed (`activeJobCancellable` from `useEmbedDataset`: false for image/vector files), LLM labeling (incl. resumed runs) and SAE activations modals pass `cancellable`; the Jobs strip cancels via `useCancelJob` (now also `llm_labeling` jobs) |
| `lib/hooks/useEmbedDataset.ts`, `EmbedResultCard.tsx`, `topics/LlmLabelingSection.tsx`, `manage/CollectionDetailPane.tsx` | Cancel-aware toasts, no error banner for a cancel, "Embedding Cancelled" result card, and refreshes of saved labels / the activations badge after a cancel |

To make another job cancellable: register its progress job id with
`cancellable_job` in the resolver, check the event at the job's checkpoints, call
`report_cancelled` (or raise `JobCancelled` into a handler that does), return
`CANCELLED_MESSAGE` as the error, and pass `cancellable` to its `ProgressModal`.
Topic extraction, topic reduction, topic evaluation, image/vector file embeds and
the SAE tab's `prepareSaeData` download are not cancellable yet.

## Tests

- `interpretability_backend/unit_tests/test_job_cancellation.py` (job state routed to
  a temp file for every test): registry semantics (incl. the stale-unregister case
  and non-resumable reports), the frontend constant mirror, `cancelJob` through the
  real schema, the pipeline cancelling after embed (resumable) and after projections
  (closed) without double reporting, re-embed passing the event and stopping
  between batches, LLM labeling stopping between topics with saved labels and real
  counts kept, and `computeDocumentActivations` flushing partial rows, plus skipping
  the model load when cancelled while items load.
- `unit_tests/test_interpret_service.py::TestBatchHighlightCancel`: no forward pass
  after the cancel.
- `embedding_visualization/lib/utils/__tests__/jobCancellation.test.ts`:
  `isJobCancelled` and the button states.
- `app/collections/lib/__tests__/saePostEmbed.test.ts`: a cancelled embed skips the
  SAE step.

Not covered by unit tests: the HF/local per-batch blocks (same shape as the tested
re-embed one) and the subtopic labeling loop.
