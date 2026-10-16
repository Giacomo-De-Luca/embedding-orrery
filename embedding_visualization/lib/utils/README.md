# Frontend utilities

This directory holds reusable visualization, color, filtering and data
adaptation code. Focused Vitest regressions live in `__tests__/`; experimental
renderers are in `experimental/`, and the density renderer has its own README.

## Category utilities

- `categoryKey.ts`: `CategoryKey` turns raw metadata values into category keys.
  Absent, null and empty values share `CategoryKey.MISSING` ("Missing values"); its
  `count` builds the sorted values and counts behind the Legend and Analytics.
  See `../../../documentation/MISSING_CATEGORY_VALUES.md`.
- `categorySelection.ts`: `CategorySelection` implements the shared legend
  click (isolate) and shift-click (toggle) semantics.
- `hiddenPoints.ts`: `HiddenPoints.filter` removes the points hidden by the
  "Hide unclustered points" and "Hide missing values" toggles in the scatter plots.

Color and scatter utilities include `plotUtils.ts` (shared highlight palette/size
rules), category palettes and scales, `collisionGrid.ts`, and `overlayLayout.ts`.
Dataset loading, temporal filtering and membership helpers remain reusable by
their corresponding hooks.
