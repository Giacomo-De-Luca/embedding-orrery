# Frontend hooks

Hooks in this folder connect React components to application state, queries,
and browser lifecycles. Shared types live in `lib/types/types.ts`; reusable
calculations live in `lib/utils/` or `app/utils/`.

- Collection and embedding hooks load datasets, active projections, topics,
  and visualization points. Category/color hooks derive plotting and legend data.
- Search hooks manage semantic, text, topic, prompt, and document-feature search.
- Chat and steering hooks manage chat sessions and streamed model responses.
- Browser/layout hooks manage resizing, scrolling, theme, zoom, and URL syncing.
- `useDensityOverlay.ts` owns the 2D density renderer lifecycle.
  `useDensityClusterLabels.ts` produces `DensityLabelPlacement[]` asynchronously,
  or `null` while disabled, rebuilding, or using centroid fallback.
- `__tests__/` covers steering helpers and semantic-search abort races.

Density label field selection and regression tests are documented in
`../../../documentation/DENSITY_LABELS.md`.
