/**
 * Row and column orders for a {@link ContingencyTable} heatmap.
 *
 * Sort modes per axis: size (largest total first), name (the category list's
 * Name order), pattern (each value sits where it is over-represented along
 * the other axis, which lines the strongest cells up on a diagonal), and two
 * external orders: semantic (topics by embedding similarity, from the
 * backend) and value (the bins of a numeric field, low to high). Other and
 * Missing always come last. Orders are index permutations
 * into `table.rows` / `table.cols`.
 */

import { ContingencyTable } from './contingencyTable';
import { CategoryKey } from './categoryKey';
import { comparePeriods } from './temporalAnalysis';

export type AxisSortMode = 'size' | 'name' | 'pattern' | 'semantic' | 'value';

export interface AxisOrderRequest {
  rowMode: AxisSortMode;
  colMode: AxisSortMode;
  /** External order of the row values for the semantic and value modes; null or absent falls back to pattern. */
  rowExternal?: readonly string[] | null;
  /** External order of the column values for the semantic and value modes. */
  colExternal?: readonly string[] | null;
  /** Display label of a value of `field`, used by name order (defaults to the value). */
  labelOf?: (field: string, value: string) => string;
}

export interface AxisOrder {
  rowOrder: number[];
  colOrder: number[];
}

type Label = (value: string) => string;

/** Pinned keys in display order, after every real value. */
const PINNED_ORDER = [ContingencyTable.OTHER, CategoryKey.MISSING];

/** Most rounds of alternating sorts when both axes use pattern order. */
const PATTERN_ROUNDS = 8;

const identity: Label = value => value;

export class AxisOrdering {
  /** Largest total first, ties by name; pinned keys last. */
  static bySize(values: readonly string[], totals: readonly number[], label: Label = identity): number[] {
    const byName = AxisOrdering.nameComparator(values, label);
    return AxisOrdering.withPinnedLast(values, (a, b) => totals[b] - totals[a] || byName(a, b));
  }

  /** Name order of the display labels (numbers numerically); pinned keys last. */
  static byName(values: readonly string[], label: Label = identity): number[] {
    return AxisOrdering.withPinnedLast(values, AxisOrdering.nameComparator(values, label));
  }

  /**
   * Values listed in `order` first, in that order; unlisted values after them
   * by size; pinned keys last. Entries of `order` not in `values` are ignored.
   */
  static byExternal(
    values: readonly string[],
    totals: readonly number[],
    order: readonly string[],
    label: Label = identity,
  ): number[] {
    const rank = new Map<string, number>();
    order.forEach((value, position) => {
      if (!rank.has(value)) rank.set(value, position);
    });
    const byName = AxisOrdering.nameComparator(values, label);
    return AxisOrdering.withPinnedLast(values, (a, b) => {
      const ra = rank.get(values[a]) ?? Infinity;
      const rb = rank.get(values[b]) ?? Infinity;
      if (ra !== rb) return ra - rb;
      return totals[b] - totals[a] || byName(a, b);
    });
  }

  /** Row and column orders for the requested modes. */
  static resolve(table: ContingencyTable, request: AxisOrderRequest): AxisOrder {
    const labelOf = request.labelOf ?? ((_field: string, value: string) => value);
    const rowLabel: Label = value => labelOf(table.rowField, value);
    const colLabel: Label = value => labelOf(table.colField, value);

    const rowMode = AxisOrdering.effectiveMode(request.rowMode, request.rowExternal);
    const colMode = AxisOrdering.effectiveMode(request.colMode, request.colExternal);
    const fixedRows = () =>
      AxisOrdering.fixedOrder(table.rows, table.rowTotals, rowMode, request.rowExternal, rowLabel);
    const fixedCols = () =>
      AxisOrdering.fixedOrder(table.cols, table.colTotals, colMode, request.colExternal, colLabel);

    if (rowMode !== 'pattern' && colMode !== 'pattern') {
      return { rowOrder: fixedRows(), colOrder: fixedCols() };
    }
    if (colMode !== 'pattern') {
      const colOrder = fixedCols();
      return { rowOrder: AxisOrdering.barycentric(table, colOrder, rowLabel), colOrder };
    }
    if (rowMode !== 'pattern') {
      const rowOrder = fixedRows();
      return { rowOrder, colOrder: AxisOrdering.barycentric(table.transpose(), rowOrder, colLabel) };
    }

    // Both axes: alternate barycentric sorts from a size order until the rows settle.
    const transposed = table.transpose();
    let colOrder = AxisOrdering.bySize(table.cols, table.colTotals, colLabel);
    let rowOrder = AxisOrdering.barycentric(table, colOrder, rowLabel);
    for (let round = 0; round < PATTERN_ROUNDS; round++) {
      colOrder = AxisOrdering.barycentric(transposed, rowOrder, colLabel);
      const next = AxisOrdering.barycentric(table, colOrder, rowLabel);
      const settled = next.every((index, position) => index === rowOrder[position]);
      rowOrder = next;
      if (settled) break;
    }
    return { rowOrder, colOrder };
  }

  /** An external mode without its order degrades to pattern order. */
  private static effectiveMode(mode: AxisSortMode, external: readonly string[] | null | undefined): AxisSortMode {
    return AxisOrdering.isExternal(mode) && !external?.length ? 'pattern' : mode;
  }

  private static isExternal(mode: AxisSortMode): boolean {
    return mode === 'semantic' || mode === 'value';
  }

  private static fixedOrder(
    values: readonly string[],
    totals: readonly number[],
    mode: AxisSortMode,
    external: readonly string[] | null | undefined,
    label: Label,
  ): number[] {
    if (mode === 'name') return AxisOrdering.byName(values, label);
    if (AxisOrdering.isExternal(mode) && external) return AxisOrdering.byExternal(values, totals, external, label);
    return AxisOrdering.bySize(values, totals, label);
  }

  /**
   * Orders the rows of `table` by their barycentre along the columns placed in
   * `colOrder`. Each cell weighs by its excess over the independence
   * expectation, so a row lands where it is over-represented rather than where
   * the largest columns are; a row with no excess falls back to raw counts.
   * Pinned columns carry no weight. Ties go by size, then name.
   */
  private static barycentric(table: ContingencyTable, colOrder: readonly number[], label: Label): number[] {
    const position = new Array<number>(table.cols.length);
    colOrder.forEach((j, p) => {
      position[j] = p;
    });
    const scoredCols = table.cols
      .map((_, j) => j)
      .filter(j => !ContingencyTable.isPinned(table.cols[j]));

    const centre = table.rows.map((_, i) => {
      let weighted = 0;
      let weight = 0;
      for (const j of scoredCols) {
        const excess = Math.max(0, table.counts[i][j] - table.expected(i, j));
        weighted += excess * position[j];
        weight += excess;
      }
      if (weight === 0) {
        for (const j of scoredCols) {
          weighted += table.counts[i][j] * position[j];
          weight += table.counts[i][j];
        }
      }
      return weight > 0 ? weighted / weight : Infinity;
    });

    const byName = AxisOrdering.nameComparator(table.rows, label);
    return AxisOrdering.withPinnedLast(
      table.rows,
      (a, b) => centre[a] - centre[b] || table.rowTotals[b] - table.rowTotals[a] || byName(a, b),
    );
  }

  private static nameComparator(values: readonly string[], label: Label): (a: number, b: number) => number {
    return (a, b) => comparePeriods(label(values[a]), label(values[b]));
  }

  /** Sorts the real values with `compare` and appends the pinned keys present (Other, then Missing). */
  static withPinnedLast(values: readonly string[], compare: (a: number, b: number) => number): number[] {
    const free = values.map((_, i) => i).filter(i => !ContingencyTable.isPinned(values[i]));
    const pinned = PINNED_ORDER.map(key => values.indexOf(key)).filter(i => i >= 0);
    return [...free.sort(compare), ...pinned];
  }
}
