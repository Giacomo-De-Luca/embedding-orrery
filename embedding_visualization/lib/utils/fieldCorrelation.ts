/**
 * Correlation statistics between metadata fields on their raw values:
 * Pearson r, Spearman ρ (Pearson on average ranks), the bias-corrected
 * correlation ratio η between a categorical and a numeric field, and a
 * pairwise matrix over many numeric fields (pairwise-complete: each pair uses
 * the points where both fields have a value).
 */

import type { MetadataPoint } from '../types/types';
import { NumericBins } from './numericBins';

export type CorrelationMethod = 'spearman' | 'pearson';

export interface CorrelationMatrix {
  fields: string[];
  /** Coefficient per pair, 1 on the diagonal; null with fewer than three shared points or no variance. */
  values: (number | null)[][];
  /** Points where both fields have a value. */
  counts: number[][];
}

interface PairResult {
  value: number | null;
  count: number;
}

/** Fewest shared points a coefficient is reported for. */
const MIN_PAIRS = 3;

export class FieldCorrelation {
  /** Two decimals with a true minus sign; an en dash when undefined. */
  static format(value: number | null): string {
    if (value === null) return '–';
    return value.toFixed(2).replace('-', '−');
  }

  /** 1-based ranks; tied values share the average of their ranks. */
  static ranks(values: readonly number[]): number[] {
    const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
    const ranks = new Array<number>(values.length);
    for (let start = 0; start < order.length; ) {
      let end = start;
      while (end + 1 < order.length && values[order[end + 1]] === values[order[start]]) end++;
      const rank = (start + end) / 2 + 1;
      for (let k = start; k <= end; k++) ranks[order[k]] = rank;
      start = end + 1;
    }
    return ranks;
  }

  static pearson(xs: readonly number[], ys: readonly number[]): number | null {
    return xs.length === ys.length ? FieldCorrelation.pearsonWhere(xs, ys).value : null;
  }

  static spearman(xs: readonly number[], ys: readonly number[]): number | null {
    if (xs.length !== ys.length || xs.length < MIN_PAIRS) return null;
    return FieldCorrelation.pearson(FieldCorrelation.ranks(xs), FieldCorrelation.ranks(ys));
  }

  /**
   * Correlation ratio η: how much of the values' variance lies between the
   * groups (0 = same mean in every group, 1 = groups fully separate the values).
   * Bias-corrected (ε²): plain η² grows by about (groups − 1) / points even on
   * noise, which many small groups (topics, under a filter) would read as a
   * strong association. Null with fewer than two groups, no more points than
   * groups, or no variance.
   */
  static correlationRatio(groups: readonly string[], values: readonly number[]): number | null {
    const n = values.length;
    if (groups.length !== n) return null;
    const sums = new Map<string, { sum: number; count: number }>();
    values.forEach((v, i) => {
      const group = sums.get(groups[i]) ?? { sum: 0, count: 0 };
      group.sum += v;
      group.count += 1;
      sums.set(groups[i], group);
    });
    const k = sums.size;
    if (k < 2 || n <= k) return null;

    const mean = values.reduce((sum, v) => sum + v, 0) / n;
    const total = values.reduce((sum, v) => sum + (v - mean) ** 2, 0);
    if (total === 0) return null;
    let between = 0;
    for (const { sum, count } of sums.values()) between += count * (sum / count - mean) ** 2;
    const withinMeanSquare = (total - between) / (n - k);
    const epsilonSquared = (between - (k - 1) * withinMeanSquare) / total;
    return Math.sqrt(Math.min(1, Math.max(0, epsilonSquared)));
  }

  /** Column cache over the points not in `exclude`, for repeated matrices (see {@link CorrelationColumns}). */
  static columns(points: readonly MetadataPoint[], exclude?: ReadonlySet<number> | null): CorrelationColumns {
    return new CorrelationColumns(points, exclude);
  }

  /** Pairwise-complete coefficients between `fields` over the points not in `exclude`. */
  static matrix(
    points: readonly MetadataPoint[],
    fields: readonly string[],
    method: CorrelationMethod,
    exclude?: ReadonlySet<number> | null,
  ): CorrelationMatrix {
    return FieldCorrelation.columns(points, exclude).matrix(fields, method);
  }

  /** Pearson r over the positions where both arrays hold a number (NaN marks a gap), with that count. */
  static pearsonWhere(xs: ArrayLike<number>, ys: ArrayLike<number>): PairResult {
    let count = 0;
    let meanX = 0;
    let meanY = 0;
    for (let k = 0; k < xs.length; k++) {
      if (Number.isNaN(xs[k]) || Number.isNaN(ys[k])) continue;
      count += 1;
      meanX += xs[k];
      meanY += ys[k];
    }
    if (count < MIN_PAIRS) return { value: null, count };
    meanX /= count;
    meanY /= count;
    let sxx = 0;
    let syy = 0;
    let sxy = 0;
    for (let k = 0; k < xs.length; k++) {
      if (Number.isNaN(xs[k]) || Number.isNaN(ys[k])) continue;
      const dx = xs[k] - meanX;
      const dy = ys[k] - meanY;
      sxx += dx * dx;
      syy += dy * dy;
      sxy += dx * dy;
    }
    if (sxx === 0 || syy === 0) return { value: null, count };
    return { value: Math.min(1, Math.max(-1, sxy / Math.sqrt(sxx * syy))), count };
  }
}

/**
 * Numeric columns of the visible points for the correlation matrix. Each
 * field is extracted and sorted once; a pair's Spearman ranks over its shared
 * points come from one linear pass over that sorted order (no per-pair sort),
 * and pair results are cached, so adding a field only computes its new pairs.
 */
export class CorrelationColumns {
  private readonly visible: readonly MetadataPoint[];
  /** Field values aligned to the visible points; NaN where the field has no number. */
  private readonly columns = new Map<string, Float64Array>();
  /** Positions of a field's values, ascending by value. */
  private readonly orders = new Map<string, Uint32Array>();
  private readonly pairs = new Map<string, PairResult>();

  constructor(points: readonly MetadataPoint[], exclude?: ReadonlySet<number> | null) {
    this.visible = exclude?.size ? points.filter(p => !exclude.has(p.index)) : points;
  }

  matrix(fields: readonly string[], method: CorrelationMethod): CorrelationMatrix {
    const size = fields.length;
    const values = fields.map(() => new Array<number | null>(size).fill(null));
    const counts = fields.map(() => new Array<number>(size).fill(0));
    for (let a = 0; a < size; a++) {
      values[a][a] = 1;
      counts[a][a] = this.order(fields[a]).length;
      for (let b = a + 1; b < size; b++) {
        const { value, count } = this.pair(fields[a], fields[b], method);
        values[a][b] = values[b][a] = value;
        counts[a][b] = counts[b][a] = count;
      }
    }
    return { fields: [...fields], values, counts };
  }

  private pair(a: string, b: string, method: CorrelationMethod): PairResult {
    const [first, second] = a < b ? [a, b] : [b, a];
    const key = `${method}\u0000${first}\u0000${second}`;
    const cached = this.pairs.get(key);
    if (cached) return cached;

    const columnA = this.column(first);
    const columnB = this.column(second);
    const result = method === 'pearson'
      ? FieldCorrelation.pearsonWhere(columnA, columnB)
      : FieldCorrelation.pearsonWhere(
        CorrelationColumns.sharedRanks(this.order(first), columnA, columnB),
        CorrelationColumns.sharedRanks(this.order(second), columnB, columnA),
      );
    this.pairs.set(key, result);
    return result;
  }

  private column(field: string): Float64Array {
    let column = this.columns.get(field);
    if (!column) {
      column = new Float64Array(this.visible.length);
      this.visible.forEach((point, k) => {
        column![k] = NumericBins.numericValue(point.metadata?.[field]) ?? Number.NaN;
      });
      this.columns.set(field, column);
    }
    return column;
  }

  private order(field: string): Uint32Array {
    let order = this.orders.get(field);
    if (!order) {
      const column = this.column(field);
      const positions: number[] = [];
      column.forEach((value, k) => {
        if (!Number.isNaN(value)) positions.push(k);
      });
      order = Uint32Array.from(positions).sort((x, y) => column[x] - column[y]);
      this.orders.set(field, order);
    }
    return order;
  }

  /**
   * Average ranks of `column` over the positions where `other` also has a
   * value (NaN elsewhere), walking `column`'s sorted order once.
   */
  private static sharedRanks(order: Uint32Array, column: Float64Array, other: Float64Array): Float64Array {
    const ranks = new Float64Array(column.length).fill(Number.NaN);
    const shared: number[] = [];
    for (const k of order) if (!Number.isNaN(other[k])) shared.push(k);
    for (let start = 0; start < shared.length; ) {
      let end = start;
      while (end + 1 < shared.length && column[shared[end + 1]] === column[shared[start]]) end++;
      const rank = (start + end) / 2 + 1;
      for (let k = start; k <= end; k++) ranks[shared[k]] = rank;
      start = end + 1;
    }
    return ranks;
  }
}
