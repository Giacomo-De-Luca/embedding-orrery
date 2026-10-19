import { describe, it, expect } from 'vitest';
import { AssociationView, ASSOCIATION_COLORS } from '../associationView';
import { ContingencyTable } from '../contingencyTable';
import { CategoryKey } from '../categoryKey';

/** Points reproducing a count matrix with row keys r0.. and column keys c0.. */
function tableOf(matrix: number[][]) {
  const points: { index: number; metadata: Record<string, unknown> }[] = [];
  matrix.forEach((row, i) =>
    row.forEach((count, j) => {
      for (let k = 0; k < count; k++) points.push({ index: points.length, metadata: { src: `r${i}`, topic: `c${j}` } });
    }),
  );
  return ContingencyTable.fromPoints(points, 'src', 'topic');
}

describe('AssociationView values', () => {
  const table = tableOf([[30, 10], [0, 60]]);

  it('encodes the value of the selected mode', () => {
    expect(AssociationView.of(table, 'count', false).value(0, 1)).toBe(10);
    expect(AssociationView.of(table, 'rowShare', false).value(0, 0)).toBe(0.75);
    expect(AssociationView.of(table, 'colShare', false).value(0, 1)).toBeCloseTo(10 / 70);
    expect(AssociationView.of(table, 'enrichment', false).value(0, 0)).toBeCloseTo(table.residual(0, 0));
  });

  it('labels cells compactly per mode', () => {
    expect(AssociationView.of(table, 'count', false).cellLabel(0, 0)).toBe('30');
    expect(AssociationView.of(table, 'rowShare', false).cellLabel(0, 0)).toBe('75%');
    // r0/c0: 30 observed vs 40 * 30 / 100 = 12 expected.
    expect(AssociationView.of(table, 'enrichment', false).cellLabel(0, 0)).toBe('2.5×');
    expect(AssociationView.of(table, 'enrichment', false).cellLabel(1, 0)).toBe('0×');
  });

  it('shows a share under half a percent as <1%', () => {
    const tiny = tableOf([[1, 999]]);
    expect(AssociationView.of(tiny, 'rowShare', false).cellLabel(0, 0)).toBe('<1%');
  });

  it('flags cells with no points as empty', () => {
    const view = AssociationView.of(table, 'count', false);
    expect(view.isEmpty(1, 0)).toBe(true);
    expect(view.isEmpty(0, 0)).toBe(false);
  });
});

describe('AssociationView colours', () => {
  it('maps enrichment onto blue / neutral / red with the neutral at zero', () => {
    const view = AssociationView.of(tableOf([[30, 10], [0, 60]]), 'enrichment', false);
    const { neutral, under, over } = ASSOCIATION_COLORS.light;
    expect(view.scale.color(0)).toBe(view.scale.color(-0));
    expect(view.scale.color(0)).toBe(AssociationView.of(tableOf([[1]]), 'enrichment', false).scale.color(0));
    expect(view.scale.color(view.scale.limit * 10)).toBe(view.scale.color(view.scale.limit));
    expect(view.scale.color(view.scale.limit)).toBe(hex(over));
    expect(view.scale.color(-view.scale.limit)).toBe(hex(under));
    expect(view.scale.color(0)).toBe(hex(neutral));
  });

  it('floors the enrichment limit so a near-independent table stays pale', () => {
    const view = AssociationView.of(tableOf([[20, 21], [20, 19]]), 'enrichment', false);
    expect(view.scale.limit).toBe(AssociationView.ENRICHMENT_FLOOR);
  });

  it('runs the sequential ramp from 0 to the largest value, darkest last in light mode', () => {
    const view = AssociationView.of(tableOf([[30, 10], [0, 60]]), 'count', false);
    const ramp = ASSOCIATION_COLORS.light.sequential;
    expect(view.scale.limit).toBe(60);
    expect(view.scale.color(60)).toBe(hex(ramp[ramp.length - 1]));
    expect(view.scale.color(0)).toBe(hex(ramp[0]));
  });

  it('uses the dark-mode steps in dark mode', () => {
    const view = AssociationView.of(tableOf([[30, 10], [0, 60]]), 'count', true);
    const ramp = ASSOCIATION_COLORS.dark.sequential;
    expect(view.scale.color(60)).toBe(hex(ramp[ramp.length - 1]));
  });

  it('paints empty cells in the neutral colour in sequential modes', () => {
    const view = AssociationView.of(tableOf([[30, 10], [0, 60]]), 'count', false);
    expect(view.color(1, 0)).toBe(hex(ASSOCIATION_COLORS.light.neutral));
  });

  it('picks light text on dark fills and dark text on light fills', () => {
    const view = AssociationView.of(tableOf([[30, 10], [0, 60]]), 'count', false);
    expect(view.usesLightInk(1, 1)).toBe(true);
    expect(view.usesLightInk(1, 0)).toBe(false);
  });
});

describe('AssociationView summaries', () => {
  it('labels Other and Missing, and resolves other values through the field preset', () => {
    expect(AssociationView.axisLabel('src', ContingencyTable.OTHER)).toBe(ContingencyTable.OTHER_LABEL);
    expect(AssociationView.axisLabel('src', CategoryKey.MISSING)).toBe(CategoryKey.MISSING_LABEL);
    expect(AssociationView.axisLabel('src', 'reddit')).toBe('reddit');
  });

  it('lists the most concentrated rows first, pinned rows last', () => {
    const points = [
      ...[0, 1, 2, 3].map(i => ({ index: i, metadata: { src: 'spread', topic: `t${i}` } })),
      ...[4, 5, 6, 7].map(i => ({ index: i, metadata: { src: 'focused', topic: 't0' } })),
      { index: 8, metadata: { topic: 't1' } },
    ];
    const rows = AssociationView.spreadRows(ContingencyTable.fromPoints(points, 'src', 'topic'));
    expect(rows.map(r => r.key)).toEqual(['focused', 'spread', CategoryKey.MISSING]);
    expect(rows[0].effective).toBeCloseTo(1);
    expect(rows[1].effective).toBeCloseTo(4);
    expect(rows[0].count).toBe(4);
  });

  it('takes custom labels per axis, e.g. percentile labels of numeric bins', () => {
    const table = tableOf([[1, 0], [0, 1]]);
    const view = AssociationView.of(table, 'count', false, { row: key => `R:${key}`, col: key => `C:${key}` });
    expect(view.rowLabels).toEqual(['R:r0', 'R:r1']);
    expect(view.colLabels).toEqual(['C:c0', 'C:c1']);
    expect(AssociationView.spreadRows(table, key => key.toUpperCase()).map(r => r.label)).toEqual(['R0', 'R1']);
  });

  it('resolves the display labels of both axes', () => {
    const view = AssociationView.of(tableOf([[1, 0], [0, 1]]), 'count', false);
    expect(view.rowLabels).toEqual(['r0', 'r1']);
    expect(view.colLabels).toEqual(['c0', 'c1']);
  });
});

/** Normalises a hex to the rgb() string d3 interpolators return. */
function hex(value: string): string {
  const n = parseInt(value.slice(1), 16);
  return `rgb(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255})`;
}
