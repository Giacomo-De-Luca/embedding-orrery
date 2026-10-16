# Analytics charts

This folder holds the Analytics sidebar's categorical and temporal charts,
category lists and field-association views.

- `FieldAssociationSection` selects two categorical fields and shows their
  association strength and spread summary.
- `FieldAssociationDialog` manages heatmap modes, axis order and PNG export.
- `AssociationHeatmap` renders the scrollable cell grid and pointer/keyboard
  tooltip; its export ref exposes the complete inner grid.
- Shared statistics and display models live in `lib/utils/contingencyTable.ts`,
  `associationView.ts` and `axisOrdering.ts`; `heatmapCapture.ts` captures the
  existing DOM for export.

See `../../../../documentation/FIELD_ASSOCIATIONS.md` for behavior and validation.
