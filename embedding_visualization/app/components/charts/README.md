# Analytics charts

This folder holds the Analytics sidebar's categorical and temporal charts,
category lists and field-association views.

- `FieldAssociationSection` holds the Group/Across picks (categorical or
  binned numeric fields) and bin settings, and shows the pair statistic and
  spread summary; `AssociationFieldSelect` is its field picker (also in the
  dialog).
- `FieldAssociationDialog` is the dialog shell: title plus Cross-tab and
  Correlations tabs.
- `CrossTabPanel` (field pickers, bins, modes, axis sorts, swap, values,
  export) and `CorrelationPanel` (numeric field picker, Spearman/Pearson,
  values, export) are the two tabs.
- `HeatmapGrid` renders any `HeatmapModel` as a scrollable cell grid with a
  pointer/keyboard tooltip (`HeatmapTooltips`); its export ref exposes the
  complete inner grid. `HeatmapLegend` is the shared scale strip and
  `HeatmapActions` the shared Values toggle + Export PNG button.
- Shared statistics and display models live in `lib/utils/contingencyTable.ts`,
  `associationView.ts`, `axisOrdering.ts`, `numericBins.ts`, `fieldAxis.ts`,
  `fieldCorrelation.ts` and `correlationView.ts`; `heatmapCapture.ts` (via
  `lib/hooks/useHeatmapExport.ts`) captures the existing DOM for export.

See `../../../../documentation/FIELD_ASSOCIATIONS.md` for behavior and validation.
