/**
 * Cross-tabulation of two categorical metadata fields.
 *
 * Counts how many points carry each pair of values and derives the statistics
 * the Analytics field-association view shows: shares within a row or column,
 * the count expected if the fields were unrelated, Pearson residuals
 * (enrichment), Cramér's V and the effective number of values a row spreads
 * over. Pure and React-free so it is unit-testable in the node environment.
 */

import type { MetadataPoint } from '../types/types';
import { CategoryKey } from './categoryKey';

export interface ContingencyTableOptions {
  /** Point indices to skip (the active text-search and temporal filters). */
  exclude?: ReadonlySet<number> | null;
  /** Row key of a raw value; defaults to {@link CategoryKey.of} (e.g. a bin label for numeric fields). */
  rowKey?: (raw: unknown) => string;
  /** Column key of a raw value; defaults to {@link CategoryKey.of}. */
  colKey?: (raw: unknown) => string;
  /** Row values to drop (e.g. unclustered topic noise). */
  rowExclude?: ReadonlySet<string>;
  /** Column values to drop. */
  colExclude?: ReadonlySet<string>;
  /**
   * Most values kept per axis, Other included; the smallest values beyond it
   * fold into {@link ContingencyTable.OTHER}. Missing is kept apart from the cap.
   */
  maxValues?: number;
}

type CountMap = Map<string, Map<string, number>>;

export class ContingencyTable {
  /** Reserved key for the values folded together past `maxValues`. */
  static readonly OTHER = '__other__';
  static readonly OTHER_LABEL = 'Other';

  private readonly rowIndex: Map<string, number>;
  private readonly colIndex: Map<string, number>;

  private constructor(
    readonly rowField: string,
    readonly colField: string,
    /** Row keys: kept values in string order, then Other, then Missing. */
    readonly rows: string[],
    /** Column keys, ordered like {@link rows}. */
    readonly cols: string[],
    /** `counts[i][j]`: points with row value `rows[i]` and column value `cols[j]`. */
    readonly counts: number[][],
    readonly rowTotals: number[],
    readonly colTotals: number[],
    readonly total: number,
    /** Raw row value → the row key it is counted under (itself, Other, or Missing). */
    private readonly rowFold: ReadonlyMap<string, string>,
    private readonly colFold: ReadonlyMap<string, string>,
  ) {
    this.rowIndex = new Map(rows.map((key, i) => [key, i]));
    this.colIndex = new Map(cols.map((key, j) => [key, j]));
  }

  /** Cross-tabulates `rowField` against `colField` over `points` in one pass. */
  static fromPoints(
    points: readonly MetadataPoint[],
    rowField: string,
    colField: string,
    options: ContingencyTableOptions = {},
  ): ContingencyTable {
    const { exclude, rowExclude, colExclude, maxValues = Infinity, rowKey = CategoryKey.of, colKey = CategoryKey.of } = options;
    const pairs: CountMap = new Map();
    const rowSums = new Map<string, number>();
    const colSums = new Map<string, number>();

    for (const point of points) {
      if (exclude?.has(point.index)) continue;
      const row = rowKey(point.metadata?.[rowField]);
      const col = colKey(point.metadata?.[colField]);
      if (rowExclude?.has(row) || colExclude?.has(col)) continue;

      let byCol = pairs.get(row);
      if (!byCol) pairs.set(row, (byCol = new Map()));
      byCol.set(col, (byCol.get(col) ?? 0) + 1);
      rowSums.set(row, (rowSums.get(row) ?? 0) + 1);
      colSums.set(col, (colSums.get(col) ?? 0) + 1);
    }

    const rowFold = ContingencyTable.foldMap(rowSums, maxValues);
    const colFold = ContingencyTable.foldMap(colSums, maxValues);
    const rows = ContingencyTable.axisKeys(rowFold);
    const cols = ContingencyTable.axisKeys(colFold);
    const rowAt = new Map(rows.map((key, i) => [key, i]));
    const colAt = new Map(cols.map((key, j) => [key, j]));

    const counts = rows.map(() => new Array<number>(cols.length).fill(0));
    const rowTotals = new Array<number>(rows.length).fill(0);
    const colTotals = new Array<number>(cols.length).fill(0);
    let total = 0;
    for (const [row, byCol] of pairs) {
      const i = rowAt.get(rowFold.get(row)!)!;
      for (const [col, count] of byCol) {
        const j = colAt.get(colFold.get(col)!)!;
        counts[i][j] += count;
        rowTotals[i] += count;
        colTotals[j] += count;
        total += count;
      }
    }

    return new ContingencyTable(rowField, colField, rows, cols, counts, rowTotals, colTotals, total, rowFold, colFold);
  }

  /**
   * Maps each value to the key it is counted under: itself, or OTHER when it
   * falls past the `maxValues` cap (largest values kept, ties in string order).
   */
  private static foldMap(sums: Map<string, number>, maxValues: number): Map<string, string> {
    const ranked = [...sums.keys()]
      .filter(key => key !== CategoryKey.MISSING)
      .sort((a, b) => sums.get(b)! - sums.get(a)! || (a < b ? -1 : a > b ? 1 : 0));
    const keep = ranked.length > maxValues ? Math.max(0, maxValues - 1) : ranked.length;
    const fold = new Map<string, string>();
    ranked.forEach((key, rank) => fold.set(key, rank < keep ? key : ContingencyTable.OTHER));
    if (sums.has(CategoryKey.MISSING)) fold.set(CategoryKey.MISSING, CategoryKey.MISSING);
    return fold;
  }

  /** Distinct folded keys: kept values in string order, then Other, then Missing. */
  private static axisKeys(fold: Map<string, string>): string[] {
    const keys = new Set(fold.values());
    const hasOther = keys.delete(ContingencyTable.OTHER);
    const sorted = CategoryKey.sort(keys);
    if (!hasOther) return sorted;
    const missingAt = sorted.indexOf(CategoryKey.MISSING);
    if (missingAt < 0) return [...sorted, ContingencyTable.OTHER];
    return [...sorted.slice(0, missingAt), ContingencyTable.OTHER, CategoryKey.MISSING];
  }

  /** True for Missing and Other, which sort after every real value. */
  static isPinned(key: string): boolean {
    return key === CategoryKey.MISSING || key === ContingencyTable.OTHER;
  }

  /** Row key a raw row value is counted under; undefined when it was excluded or never seen. */
  rowKeyFor(value: string): string | undefined {
    return this.rowFold.get(value);
  }

  /** Column key a raw column value is counted under; undefined when it was excluded or never seen. */
  colKeyFor(value: string): string | undefined {
    return this.colFold.get(value);
  }

  /** Count for a pair of keys; 0 when either key is not in the table. */
  countOf(row: string, col: string): number {
    const i = this.rowIndex.get(row);
    const j = this.colIndex.get(col);
    return i === undefined || j === undefined ? 0 : this.counts[i][j];
  }

  /** Count expected in a cell if the two fields were unrelated. */
  expected(i: number, j: number): number {
    return this.total > 0 ? (this.rowTotals[i] * this.colTotals[j]) / this.total : 0;
  }

  /** Pearson residual `(observed − expected) / √expected`: the enrichment colour scale. */
  residual(i: number, j: number): number {
    const expected = this.expected(i, j);
    return expected > 0 ? (this.counts[i][j] - expected) / Math.sqrt(expected) : 0;
  }

  /** Observed / expected ("3.4× expected"). */
  ratio(i: number, j: number): number {
    const expected = this.expected(i, j);
    return expected > 0 ? this.counts[i][j] / expected : 0;
  }

  /** Share of row `i`'s points that fall in column `j`. */
  rowShare(i: number, j: number): number {
    return this.rowTotals[i] > 0 ? this.counts[i][j] / this.rowTotals[i] : 0;
  }

  /** Share of column `j`'s points that fall in row `i`. */
  colShare(i: number, j: number): number {
    return this.colTotals[j] > 0 ? this.counts[i][j] / this.colTotals[j] : 0;
  }

  /**
   * Effective number of columns row `i` spreads over: `exp(entropy)` of its
   * shares. 1 when every point sits in one column, k when spread evenly over k.
   */
  effectiveCount(i: number): number {
    let entropy = 0;
    for (let j = 0; j < this.cols.length; j++) {
      const share = this.rowShare(i, j);
      if (share > 0) entropy -= share * Math.log(share);
    }
    return Math.exp(entropy);
  }

  /**
   * Cramér's V in [0, 1], bias-corrected (Bergsma 2013): plain V grows with
   * the table size over the sample size, so small or filtered samples would
   * show an association between unrelated fields. Null when either axis has
   * fewer than two values or the sample is too small to correct.
   */
  cramersV(): number | null {
    const r = this.rows.length;
    const c = this.cols.length;
    const n = this.total;
    if (Math.min(r, c) < 2 || n < 2) return null;
    let chiSquare = 0;
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < c; j++) chiSquare += this.residual(i, j) ** 2;
    }
    const phiSquared = Math.max(0, chiSquare / n - ((r - 1) * (c - 1)) / (n - 1));
    const rCorrected = r - (r - 1) ** 2 / (n - 1);
    const cCorrected = c - (c - 1) ** 2 / (n - 1);
    const denominator = Math.min(rCorrected, cCorrected) - 1;
    return denominator > 0 ? Math.min(1, Math.sqrt(phiSquared / denominator)) : null;
  }

  /** The same table with rows and columns swapped. */
  transpose(): ContingencyTable {
    const counts = this.cols.map((_, j) => this.rows.map((_, i) => this.counts[i][j]));
    return new ContingencyTable(
      this.colField,
      this.rowField,
      this.cols,
      this.rows,
      counts,
      this.colTotals,
      this.rowTotals,
      this.total,
      this.colFold,
      this.rowFold,
    );
  }
}
