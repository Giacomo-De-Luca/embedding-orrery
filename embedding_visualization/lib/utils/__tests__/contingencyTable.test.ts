import { describe, it, expect } from 'vitest';
import { ContingencyTable } from '../contingencyTable';
import { CategoryKey } from '../categoryKey';

const { MISSING } = CategoryKey;
const { OTHER } = ContingencyTable;

/** One point per `[source, topic]` pair; `undefined` leaves the field absent. */
function pointsOf(pairs: Array<[string | undefined, string | undefined]>) {
  return pairs.map(([source, topic], index) => ({
    index,
    metadata: {
      ...(source !== undefined && { source }),
      ...(topic !== undefined && { topic }),
    },
  }));
}

/** Points reproducing a `rows x cols` count matrix with row keys r0.. and column keys c0.. */
function pointsFromMatrix(matrix: number[][]) {
  const pairs: Array<[string, string]> = [];
  matrix.forEach((row, i) =>
    row.forEach((count, j) => {
      for (let k = 0; k < count; k++) pairs.push([`r${i}`, `c${j}`]);
    }),
  );
  return pointsOf(pairs);
}

describe('ContingencyTable.fromPoints', () => {
  it('counts each pair of values with row, column and grand totals', () => {
    const table = ContingencyTable.fromPoints(
      pointsOf([['a', 'x'], ['a', 'x'], ['a', 'y'], ['b', 'y']]),
      'source',
      'topic',
    );
    expect(table.rows).toEqual(['a', 'b']);
    expect(table.cols).toEqual(['x', 'y']);
    expect(table.counts).toEqual([[2, 1], [0, 1]]);
    expect(table.rowTotals).toEqual([3, 1]);
    expect(table.colTotals).toEqual([2, 2]);
    expect(table.total).toBe(4);
    expect(table.countOf('a', 'x')).toBe(2);
    expect(table.countOf('b', 'x')).toBe(0);
    expect(table.countOf('unknown', 'x')).toBe(0);
  });

  it('groups absent, null and empty values under the Missing key, listed last', () => {
    const points = [
      { index: 0, metadata: { source: 'a', topic: 'x' } },
      { index: 1, metadata: { source: 'a' } },
      { index: 2, metadata: { source: 'a', topic: null } },
      { index: 3, metadata: { source: '', topic: 'x' } },
    ];
    const table = ContingencyTable.fromPoints(points, 'source', 'topic');
    expect(table.rows).toEqual(['a', MISSING]);
    expect(table.cols).toEqual(['x', MISSING]);
    expect(table.countOf('a', MISSING)).toBe(2);
    expect(table.countOf(MISSING, 'x')).toBe(1);
  });

  it('skips the excluded point indices (active filters)', () => {
    const table = ContingencyTable.fromPoints(
      pointsOf([['a', 'x'], ['a', 'y'], ['b', 'y']]),
      'source',
      'topic',
      { exclude: new Set([1]) },
    );
    expect(table.counts).toEqual([[1, 0], [0, 1]]);
    expect(table.total).toBe(2);
  });

  it('drops the excluded values of each axis (unclustered noise)', () => {
    const table = ContingencyTable.fromPoints(
      pointsOf([['a', 'x'], ['a', '-1'], ['b', 'y']]),
      'source',
      'topic',
      { colExclude: new Set(['-1']) },
    );
    expect(table.cols).toEqual(['x', 'y']);
    expect(table.total).toBe(2);
  });

  it('folds the smallest values past maxValues into Other without changing totals', () => {
    const table = ContingencyTable.fromPoints(
      pointsOf([
        ['a', 'x'], ['a', 'x'], ['a', 'x'],
        ['a', 'y'], ['a', 'y'],
        ['b', 'z'],
        ['b', 'w'],
      ]),
      'source',
      'topic',
      { maxValues: 3 },
    );
    expect(table.cols).toEqual(['x', 'y', OTHER]);
    expect(table.countOf('b', OTHER)).toBe(2);
    expect(table.total).toBe(7);
    expect(table.colTotals.reduce((sum, n) => sum + n, 0)).toBe(7);
  });

  it('tells which key a raw value is counted under', () => {
    const table = ContingencyTable.fromPoints(
      pointsOf([['a', 'x'], ['a', 'x'], ['a', 'y'], ['b', 'z']]),
      'source',
      'topic',
      { maxValues: 2, colExclude: new Set(['w']) },
    );
    expect(table.colKeyFor('x')).toBe('x');
    expect(table.colKeyFor('z')).toBe(OTHER);
    expect(table.colKeyFor('never-seen')).toBeUndefined();
    expect(table.rowKeyFor('a')).toBe('a');
  });

  it('keeps Missing explicit and outside the maxValues cap', () => {
    const table = ContingencyTable.fromPoints(
      pointsOf([['a', 'x'], ['a', 'y'], ['a', 'z'], ['a', undefined]]),
      'source',
      'topic',
      { maxValues: 2 },
    );
    expect(table.cols).toContain(MISSING);
    expect(table.cols.at(-1)).toBe(MISSING);
    expect(table.cols.filter(c => c !== MISSING)).toHaveLength(2);
  });

  it('adds no Other value when the axis fits under the cap', () => {
    const table = ContingencyTable.fromPoints(
      pointsOf([['a', 'x'], ['b', 'y']]),
      'source',
      'topic',
      { maxValues: 2 },
    );
    expect(table.cols).toEqual(['x', 'y']);
  });

  it('reads each axis through its own key function when given one', () => {
    const points = [
      { index: 0, metadata: { source: 'a', score: 0.2 } },
      { index: 1, metadata: { source: 'a', score: 0.9 } },
      { index: 2, metadata: { source: 'b', score: 0.7 } },
    ];
    const band = (raw: unknown) => ((raw as number) < 0.5 ? 'low' : 'high');
    const table = ContingencyTable.fromPoints(points, 'source', 'score', { colKey: band });
    expect(table.cols).toEqual(['high', 'low']);
    expect(table.countOf('a', 'low')).toBe(1);
    expect(table.countOf('b', 'high')).toBe(1);
  });

  it('builds an empty table from no points', () => {
    const table = ContingencyTable.fromPoints([], 'source', 'topic');
    expect(table.rows).toEqual([]);
    expect(table.cols).toEqual([]);
    expect(table.total).toBe(0);
  });
});

describe('ContingencyTable statistics', () => {
  it('computes expected counts, residuals and ratios on a perfectly associated table', () => {
    const table = ContingencyTable.fromPoints(pointsFromMatrix([[10, 0], [0, 10]]), 'source', 'topic');
    expect(table.expected(0, 0)).toBe(5);
    expect(table.residual(0, 0)).toBeCloseTo(Math.sqrt(5));
    expect(table.residual(0, 1)).toBeCloseTo(-Math.sqrt(5));
    expect(table.ratio(0, 0)).toBe(2);
    expect(table.ratio(0, 1)).toBe(0);
    expect(table.cramersV()).toBeCloseTo(1);
  });

  it("corrects Cramér's V for small samples (Bergsma)", () => {
    // Uncorrected V would be 0.5 on this 8-point table.
    const table = ContingencyTable.fromPoints(pointsFromMatrix([[3, 1], [1, 3]]), 'source', 'topic');
    expect(table.cramersV()).toBeCloseTo(Math.sqrt(0.125), 3);
  });

  it('gives zero residuals and V = 0 on an independent table', () => {
    const table = ContingencyTable.fromPoints(pointsFromMatrix([[2, 4], [3, 6]]), 'source', 'topic');
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) expect(table.residual(i, j)).toBeCloseTo(0);
    }
    expect(table.cramersV()).toBeCloseTo(0);
  });

  it('computes the shares within each row and within each column', () => {
    const table = ContingencyTable.fromPoints(pointsFromMatrix([[3, 1], [0, 4]]), 'source', 'topic');
    expect(table.rowShare(0, 0)).toBe(0.75);
    expect(table.colShare(0, 1)).toBe(0.2);
  });

  it('counts how many columns a row effectively spreads over', () => {
    const table = ContingencyTable.fromPoints(
      pointsFromMatrix([[2, 2, 2, 2], [5, 0, 0, 0]]),
      'source',
      'topic',
    );
    expect(table.effectiveCount(0)).toBeCloseTo(4);
    expect(table.effectiveCount(1)).toBeCloseTo(1);
  });

  it('returns null instead of NaN for V when an axis has fewer than two values', () => {
    expect(ContingencyTable.fromPoints([], 'source', 'topic').cramersV()).toBeNull();
    const oneRow = ContingencyTable.fromPoints(pointsOf([['a', 'x'], ['a', 'y']]), 'source', 'topic');
    expect(oneRow.cramersV()).toBeNull();
  });

  it('swaps axes: rows become columns and each share becomes the other', () => {
    const table = ContingencyTable.fromPoints(pointsFromMatrix([[3, 1], [0, 4]]), 'source', 'topic');
    const swapped = table.transpose();
    expect(swapped.rowField).toBe('topic');
    expect(swapped.colField).toBe('source');
    expect(swapped.rows).toEqual(table.cols);
    expect(swapped.counts).toEqual([[3, 0], [1, 4]]);
    expect(swapped.rowShare(1, 0)).toBe(table.colShare(0, 1));
    expect(swapped.cramersV()).toBeCloseTo(table.cramersV()!);
  });
});
