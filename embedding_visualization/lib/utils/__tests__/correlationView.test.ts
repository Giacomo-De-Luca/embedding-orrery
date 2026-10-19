import { describe, it, expect } from 'vitest';
import { CorrelationMatrixView } from '../correlationView';
import { ASSOCIATION_COLORS } from '../associationView';
import type { CorrelationMatrix } from '../fieldCorrelation';

const MATRIX: CorrelationMatrix = {
  fields: ['a', 'b', 'c'],
  values: [
    [1, 1, -1],
    [1, 1, null],
    [-1, null, 1],
  ],
  counts: [
    [5, 5, 5],
    [5, 5, 1],
    [5, 1, 5],
  ],
};

/** Normalises a hex to the rgb() string the scales return. */
function hex(value: string): string {
  const n = parseInt(value.slice(1), 16);
  return `rgb(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255})`;
}

describe('CorrelationMatrixView', () => {
  const view = CorrelationMatrixView.of(MATRIX, false, field => field.toUpperCase());

  it('labels both axes with the field names', () => {
    expect(view.rowLabels).toEqual(['A', 'B', 'C']);
    expect(view.colLabels).toEqual(['A', 'B', 'C']);
  });

  it('colours +1 red (the diagonal too), -1 blue, and pairs without a coefficient neutral', () => {
    const { over, under, neutral } = ASSOCIATION_COLORS.light;
    expect(view.color(0, 1)).toBe(hex(over));
    expect(view.color(0, 2)).toBe(hex(under));
    expect(view.color(0, 0)).toBe(hex(over));
    expect(view.color(1, 2)).toBe(hex(neutral));
  });

  it('prints coefficients with two decimals and a true minus sign', () => {
    const small = CorrelationMatrixView.of(
      { fields: ['x', 'y'], values: [[1, -0.0512], [-0.0512, 1]], counts: [[3, 3], [3, 3]] },
      false,
    );
    expect(small.cellLabel(0, 1)).toBe('−0.05');
    expect(view.cellLabel(0, 1)).toBe('1.00');
    expect(view.cellLabel(0, 0)).toBe('1');
    expect(view.cellLabel(1, 2)).toBe('–');
  });

  it('spans the legend from -1 to +1', () => {
    const legend = view.legend();
    expect([legend.start, legend.middle, legend.end]).toEqual(['−1', '0', '+1']);
  });

  it('uses the dark-mode steps in dark mode', () => {
    const dark = CorrelationMatrixView.of(MATRIX, true);
    expect(dark.color(0, 1)).toBe(hex(ASSOCIATION_COLORS.dark.over));
  });
});
