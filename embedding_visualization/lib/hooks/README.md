# Frontend hooks

Hooks in this folder connect React components to application state, queries,
and browser lifecycles. Shared types live in `lib/types/types.ts`; reusable
calculations live in `lib/utils/` or `app/utils/`.

- Collection and embedding hooks load datasets, active projections, topics,
  and visualization points. Category/color hooks derive plotting and legend data.
- Search hooks manage semantic, text, topic, prompt, and document-feature search.
- `useTopicMergeTree.ts` drives the Explore topic slider: the `topicMergeTree`
  query, the store's `topicCut`, the deferred preview metadata, the colour pins
  and Apply (`reduceTopics`). `useEmbeddingData.reload()`,
  `useCollectionTopics().refetch` and `refreshTopicSimilarityOrders` reload what
  an Apply replaces (`../../../documentation/TOPIC_MERGE_TREES.md`).
- Embed-form data: `useEmbeddingModelCatalog.ts` (model suggestions, as
  `ModelSuggestions`) and `useMediaEmbeddingSettings.ts` (media options' defaults
  and limits), both fetched once per page, cache-first.
- Chat and steering hooks manage chat sessions and streamed model responses.
- Browser/layout hooks manage resizing, scrolling, theme, zoom, and URL syncing.
  `useZoomLimit.ts` caps 2D wheel zoom-out; `usePinchZoom.ts` adds two-finger
  pinch zoom to both scatter plots (`../../../documentation/MOBILE_PINCH_ZOOM.md`).
- `useLatestRef.ts` keeps a ref on the latest value (written in a layout effect)
  for identity-stable callbacks that run outside React's commit, such as a queued
  `CoalescedRedraw` request or a Plotly config function.
- `useDensityOverlay.ts` owns the 2D density renderer lifecycle.
  `useFloatRenderTargets.ts` reads the cached `FloatRenderTargets` probe
  (SSR-safe) that gates both density renderers' controls.
  `useDensityClusterLabels.ts` produces `DensityLabelPlacement[]` asynchronously,
  or `null` while disabled, rebuilding, or using centroid fallback.
- `__tests__/` covers steering helpers and semantic-search abort races (the
  store's topic-cut lifecycle is tested in `../stores/__tests__/topicCut.test.ts`).

Density label field selection and regression tests are documented in
`../../../documentation/DENSITY_LABELS.md`.
