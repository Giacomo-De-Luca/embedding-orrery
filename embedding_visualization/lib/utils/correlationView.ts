/**
 * View model for the correlation-matrix heatmap: coefficients from -1 to +1
 * on the association view's diverging scale (blue negative, gray zero, red
 * positive; the diagonal is +1), pairs without a coefficient in the neutral colour.
 */

import { rgb } from 'd3-color';
import { ASSOCIATION_COLORS, AssociationScale, type AssociationLegend } from './associationView';
import { FieldCorrelation, type CorrelationMatrix } from './fieldCorrelation';
import type { HeatmapModel } from './heatmapModel';

export class CorrelationMatrixView implements HeatmapModel {
  /** Fill of pairs without a coefficient. */
  readonly emptyColor: string;
  readonly rowLabels: string[];
  readonly colLabels: string[];

  private constructor(
    readonly matrix: CorrelationMatrix,
    readonly scale: AssociationScale,
    neutral: string,
    labelOf: (field: string) => string,
  ) {
    this.emptyColor = rgb(neutral).toString();
    this.rowLabels = matrix.fields.map(labelOf);
    this.colLabels = this.rowLabels;
  }

  static of(
    matrix: CorrelationMatrix,
    isDark: boolean,
    labelOf: (field: string) => string = field => field,
  ): CorrelationMatrixView {
    const colors = isDark ? ASSOCIATION_COLORS.dark : ASSOCIATION_COLORS.light;
    return new CorrelationMatrixView(matrix, new AssociationScale('diverging', 1, colors), colors.neutral, labelOf);
  }

  /** The coefficient of a pair (1 on the diagonal); null when undefined. */
  value(i: number, j: number): number | null {
    return this.matrix.values[i][j];
  }

  color(i: number, j: number): string {
    const value = this.value(i, j);
    return value === null ? this.emptyColor : this.scale.color(value);
  }

  cellLabel(i: number, j: number): string {
    if (i === j) return '1';
    return FieldCorrelation.format(this.value(i, j));
  }

  usesLightInk(i: number, j: number): boolean {
    return AssociationScale.needsLightInk(this.color(i, j));
  }

  legend(): AssociationLegend {
    return { gradient: this.scale.gradient(), start: '−1', middle: '0', end: '+1' };
  }
}
