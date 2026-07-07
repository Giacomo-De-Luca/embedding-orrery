# Collections Page UX Refactor (2026-07)

Phased refactor of `/collections` addressing the traps found in the UX review:
tab switches destroyed in-progress form state, nothing was URL-addressable,
running jobs became invisible after a page reload, a blocking modal locked the
page during long embeds, the Manage tab's only browser was a Select dropdown,
validation used `alert()`, and the HF/Local tabs duplicated ~80% of their flow.

All paths below are relative to `embedding_visualization/`.

## Phase A — State & URL fixes

- Tabs stay mounted once visited (`TabsContent forceMount` +
  `data-[state=inactive]:hidden`, visited-set in `app/collections/page.tsx`),
  so column/template/portion/name state survives tab switches. The shared
  `lastEmbedResult` is gated per-tab by a new `lastEmbedSource`
  (`'hf' | 'local' | 'reembed'`) on `useEmbedDataset`.
- `?tab=` and `?collection=` round-trip through the URL
  (`app/collections/lib/urlState.ts`; pure helpers unit-tested). The page is
  wrapped in `<Suspense>` (Next 15 `useSearchParams` requirement).
  `?collection=` without `?tab=` implies the manage tab. Manage-tab selection
  was lifted from `CollectionManagerTab` to the page (controlled props).

## Phase B — Jobs visibility & progress

- Progress logic layered: pure math in `app/collections/lib/jobProgress.ts`
  (`computePercent`, `formatElapsed`, `nextEtaState` — unit-tested) →
  `useJobProgress(jobId)` (WebSocket subscription + elapsed + ETA; resets on
  jobId change) → presentational `JobProgressBody` → wrapped by
  `ProgressModal` (centered blocking overlay, used by all client-initiated
  jobs). A non-blocking bottom-right `JobProgressDock` was tried for embeds
  and **reverted on user feedback** (off-center, cramped at 420px — judged a
  downgrade vs. the centered modal); the strip below covers the
  visibility-after-reload goal on its own.
- Page-global `ActiveJobsStrip` (polls `GET_EMBEDDING_JOBS`, all statuses,
  5s) lists running + interrupted jobs on every tab — driven purely by server
  state, so jobs stay visible after a reload. Resume dispatches on
  `job.jobType` via the shared `resumeJob()` in
  `app/collections/lib/embeddingFormUtils.ts` (this also fixes the old
  cross-tab resume bug where each tab assumed its own job type). Cancel is
  hidden for `llm_labeling` jobs (backend cancel registration unverified).
  `JobsPanel.tsx` and `EmbedProgressSection.tsx` were deleted (absorbed).
- `<Toaster />` is now mounted in `app/providers.tsx` (it never was —
  every existing `toast()` call app-wide was invisible). `useEmbedDataset`
  fires success/error toasts for embed, re-embed, topic extraction,
  reduction, and LLM labeling.

## Phase C — Manage tab master–detail

- `CollectionManagerTab` is now a thin `grid md:grid-cols-[280px_1fr]` shell
  over `manage/CollectionListPane` (search input + scrollable list with
  item-count / provider / has-topics badges; pure `filterCollections` in
  `lib/collectionFilter.ts`) and `manage/CollectionDetailPane` (header card
  with Quick Action links — `next/link` + `buttonVariants`, no more
  hand-copied class strings — plus `manage/DeleteCollectionDialog`, an
  AlertDialog replacing the inline confirm). Detail sections (preview,
  metadata editor, SAE link/activations, TopicExtractionCard) moved over
  verbatim. New `lib/ui-primitives/alert-dialog.tsx` (shadcn, imports fixed
  to house paths).
- Preview-table cells truncated at 100 chars open a popover with the full
  value (`PreviewCell`).

## Phase D — Embed-flow ergonomics

- Numbered section cards ("1 · Data Source" … "5 · Embedding Model") +
  scroll-into-view when the info card appears after fetch.
- Sticky `EmbedFooterBar` (config recap + primary CTA + first blocking
  issue) replaces the buried embed buttons; hidden while an embed runs (the
  dock takes over). Validation is pure (`getEmbedValidationIssues` /
  `buildEmbedSummary` in `lib/embedValidation.ts`, unit-tested); all
  `alert()` calls removed.
- `PortionSelector` gained `allowedStrategies`; LocalFileTab passes
  `['ALL','FIRST_N','RANDOM_SAMPLE']`, fixing the shown-but-inert Row Range
  option (`EmbedLocalFileInput` has no range fields).

## Phase E — HF/Local logic dedup (zero UI change, per user decision)

- `lib/useEmbedFormState.ts`: the form state previously duplicated verbatim
  in both tabs (columns, template, id column, portion, collection name,
  reset-on-source-change), composed with `useEmbeddingModelState`. Each tab
  instantiates its own copy — state stays per-tab, JSX stays per-tab.
- Pure input builders in `embeddingFormUtils.ts`: `buildHFEmbedInput`,
  `buildLocalEmbedInput`, `buildReEmbedInput`, `mergeMetadataColumns`,
  `buildPortionInput` — unit-tested (portion mapping, metadata merge rule,
  TEXT/IMAGE/VECTOR routing, `__document__` special case). Tab handlers
  shrank to validate → build → mutate.

## Phase F — Polish

- Hover-only action icons (metadata delete, topic rename/regenerate) now
  visible at rest (`opacity-40`).
- Page identity: `TestEmbedPage` → `CollectionsPage`, h1 "Collections",
  subtitle no longer mentions ChromaDB.
- Batch-size input is `type="number" min=1` with a parse guard.
- `TopicExtractionCard` split into a shell over `topics/TopicListSection`,
  `topics/ReduceTopicsSection`, `topics/LlmLabelingSection`.

## Tests

New colocated vitest suites under `app/collections/lib/__tests__/`:
`urlState`, `jobProgress`, `collectionFilter`, `embedValidation`,
`embedInputBuilders`. Full suite: 225 tests green.

## Repo-level issues surfaced (pre-existing, NOT fixed here)

1. **Clean Turbopack builds are broken**: `npm run build` is
   `next build --turbopack`, but the glslify fix lives only in the webpack
   config (`next.config.ts` alias). From a clean `.next`, Turbopack fails on
   glslify's dynamic requires; warm caches mask it. Either port the alias to
   Turbopack config or drop `--turbopack`.
2. **`app/components/charts/TemporalFilterChart.tsx:146`** has a type error
   at HEAD (`string | null | undefined` passed to `getCategoryLabel`) that
   fails `next build`'s typecheck (webpack path). Left alone — it belongs to
   the in-flight probe work.
