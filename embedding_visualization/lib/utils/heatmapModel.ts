/**
 * What `HeatmapGrid` draws: labelled rows and columns of coloured cells.
 * Implemented by the field-association cross-tab (`AssociationView`) and the
 * correlation matrix (`CorrelationMatrixView`).
 */
export interface HeatmapModel {
  readonly rowLabels: readonly string[];
  readonly colLabels: readonly string[];
  color(i: number, j: number): string;
  /** Short text printed in the cell when values are shown. */
  cellLabel(i: number, j: number): string;
  /** Whether that text needs light ink on the cell's fill. */
  usesLightInk(i: number, j: number): boolean;
}
