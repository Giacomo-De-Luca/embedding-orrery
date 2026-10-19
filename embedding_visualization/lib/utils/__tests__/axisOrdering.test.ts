import { describe, it, expect } from 'vitest';
import { AxisOrdering } from '../axisOrdering';
import { ContingencyTable } from '../contingencyTable';
import { CategoryKey } from '../categoryKey';

const { MISSING } = CategoryKey;
const { OTHER } = ContingencyTable;

function tableOf(pairs: Array<[string, string, number]>) {
  const points: { index: number; metadata: Record<string, unknown> }[] = [];
  for (const [row, col, count] of pairs) {
    for (let k = 0; k < count; k++) {
      points.push({ index: points.length, metadata: { row, col } });
    }
  }
  return ContingencyTable.fromPoints(points, 'row', 'col');
}

/** Values of `keys` in the order given by `order`. */
const ordered = (keys: string[], order: number[]) => order.map(i => keys[i]);

describe('AxisOrdering.bySize', () => {
  it('orders values by descending total, ties by name', () => {
    expect(ordered(['b', 'a', 'c'], AxisOrdering.bySize(['b', 'a', 'c'], [2, 2, 5]))).toEqual(['c', 'a', 'b']);
  });

  it('pins Other and then Missing last, whatever their totals', () => {
    const values = ['a', OTHER, MISSING, 'b'];
    expect(ordered(values, AxisOrdering.bySize(values, [1, 50, 99, 2]))).toEqual(['b', 'a', OTHER, MISSING]);
  });
});

describe('AxisOrdering.byName', () => {
  it('orders numbers numerically and other labels alphabetically', () => {
    expect(ordered(['10', '2', '1'], AxisOrdering.byName(['10', '2', '1']))).toEqual(['1', '2', '10']);
    expect(ordered(['pear', 'apple'], AxisOrdering.byName(['pear', 'apple']))).toEqual(['apple', 'pear']);
  });

  it('sorts by the display label when one is given', () => {
    const labels: Record<string, string> = { a: 'Zebra', b: 'Ant' };
    expect(ordered(['a', 'b'], AxisOrdering.byName(['a', 'b'], v => labels[v]))).toEqual(['b', 'a']);
  });
});

describe('AxisOrdering.byExternal', () => {
  it('follows the given order, then appends unlisted values by size, with Missing last', () => {
    const values = ['a', 'b', 'c', 'd', MISSING];
    const order = AxisOrdering.byExternal(values, [5, 1, 3, 9, 7], ['c', 'not-present', 'a']);
    expect(ordered(values, order)).toEqual(['c', 'a', 'd', 'b', MISSING]);
  });
});

describe('AxisOrdering.resolve', () => {
  // Two blocks, shuffled: rows r0/r2 and columns c1/c3 belong together, as do r1/r3 and c0/c2.
  const blockOf: Record<string, 'A' | 'B'> = { r0: 'A', r2: 'A', c1: 'A', c3: 'A', r1: 'B', r3: 'B', c0: 'B', c2: 'B' };
  const shuffledBlocks = tableOf([
    ['r0', 'c1', 6], ['r0', 'c3', 5],
    ['r1', 'c0', 5], ['r1', 'c2', 4],
    ['r2', 'c1', 3], ['r2', 'c3', 4],
    ['r3', 'c0', 3], ['r3', 'c2', 5],
  ]);

  it('pattern on both axes recovers the block-diagonal structure', () => {
    const { rowOrder, colOrder } = AxisOrdering.resolve(shuffledBlocks, { rowMode: 'pattern', colMode: 'pattern' });
    const rowBlocks = ordered(shuffledBlocks.rows, rowOrder).map(v => blockOf[v]);
    const colBlocks = ordered(shuffledBlocks.cols, colOrder).map(v => blockOf[v]);
    // Each block is contiguous, and the first row block meets the first column block.
    expect(rowBlocks.join('')).toMatch(/^(AABB|BBAA)$/);
    expect(colBlocks.join('')).toMatch(/^(AABB|BBAA)$/);
    expect(rowBlocks[0]).toBe(colBlocks[0]);
  });

  it('pattern on one axis follows the order of the other axis', () => {
    const { rowOrder, colOrder } = AxisOrdering.resolve(shuffledBlocks, { rowMode: 'pattern', colMode: 'name' });
    expect(ordered(shuffledBlocks.cols, colOrder)).toEqual(['c0', 'c1', 'c2', 'c3']);
    // Block B owns c0 and c2, block A c1 and c3: B sits slightly earlier on average.
    expect(ordered(shuffledBlocks.rows, rowOrder).map(v => blockOf[v]).join('')).toBe('BBAA');
  });

  it('places a row where it is over-represented, not where most of its points are', () => {
    // `s` has most points in the huge last column but is over-represented only in the first.
    // Weighting by raw counts would put it after `t`; weighting by excess puts it first.
    const table = tableOf([
      ['s', 'a_left', 10], ['s', 'c_huge', 90],
      ['t', 'b_mid', 5],
      ['u', 'c_huge', 100],
    ]);
    const { rowOrder } = AxisOrdering.resolve(table, { rowMode: 'pattern', colMode: 'name' });
    expect(ordered(table.rows, rowOrder)).toEqual(['s', 't', 'u']);
  });

  it('uses the semantic order when one is given', () => {
    const { colOrder } = AxisOrdering.resolve(shuffledBlocks, {
      rowMode: 'size',
      colMode: 'semantic',
      colExternal: ['c3', 'c2', 'c1', 'c0'],
    });
    expect(ordered(shuffledBlocks.cols, colOrder)).toEqual(['c3', 'c2', 'c1', 'c0']);
  });

  it('falls back to pattern order when no semantic order is available', () => {
    const semantic = AxisOrdering.resolve(shuffledBlocks, { rowMode: 'semantic', colMode: 'name', rowExternal: null });
    const pattern = AxisOrdering.resolve(shuffledBlocks, { rowMode: 'pattern', colMode: 'name' });
    expect(semantic.rowOrder).toEqual(pattern.rowOrder);
  });

  it('orders a binned numeric axis by its value order', () => {
    const table = tableOf([['a', '5–10', 1], ['a', '0–5', 3], ['b', '10–15', 2]]);
    const { colOrder } = AxisOrdering.resolve(table, {
      rowMode: 'size',
      colMode: 'value',
      colExternal: ['0–5', '5–10', '10–15'],
    });
    expect(ordered(table.cols, colOrder)).toEqual(['0–5', '5–10', '10–15']);
  });

  it('keeps Missing and Other last in pattern order', () => {
    const table = ContingencyTable.fromPoints(
      [
        ...Array.from({ length: 5 }, (_, i) => ({ index: i, metadata: { row: 'a', col: 'x' } })),
        { index: 5, metadata: { row: 'b', col: 'y' } },
        { index: 6, metadata: { col: 'x' } },
      ],
      'row',
      'col',
    );
    const { rowOrder } = AxisOrdering.resolve(table, { rowMode: 'pattern', colMode: 'size' });
    expect(ordered(table.rows, rowOrder).at(-1)).toBe(MISSING);
  });

  it('returns empty orders for an empty table', () => {
    const empty = ContingencyTable.fromPoints([], 'row', 'col');
    expect(AxisOrdering.resolve(empty, { rowMode: 'pattern', colMode: 'pattern' })).toEqual({ rowOrder: [], colOrder: [] });
  });
});
