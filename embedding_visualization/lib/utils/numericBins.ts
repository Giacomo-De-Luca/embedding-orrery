/**
 * The one binning utility of the frontend: ordered numeric bins with a fast
 * bin lookup, range labels and counting.
 *
 * Two families of bins:
 *
 * - **Axis bins** (`fromValues`), for a numeric field on a contingency-table
 *   axis next to categorical fields: equal-count bins (edges at quantiles, so
 *   each bin holds about the same number of points; tied values can merge
 *   bins) or equal-width bins. Each bin's key is its range label, e.g.
 *   "1.90–2.40" for [1.90, 2.40); whole-number fields (years, counts) are
 *   labelled by the values a bin holds, "1993–1996" or "2019", and get one bin
 *   per value when they have no more distinct values than bins. Each bin also
 *   has a percentile label ("10–20%": the share of points below its edges).
 * - **Histogram bins** (`histogramBins` / `histogramFor` / `histogram`):
 *   strictly equal-width bins, in linear or log10 space, for distribution
 *   charts (the continuous-scale legend, the SAE explorer's histograms).
 */

import { CategoryKey } from './categoryKey';
import type { HistogramBin } from '../types/types';

export type BinMethod = 'quantile' | 'width';
export type HistogramScale = 'linear' | 'log';

/** Most decimals an edge label gets. */
const MAX_DECIMALS = 10;
/** Most decimals a percentile label gets. */
const MAX_PERCENT_DECIMALS = 2;
/** A label may round its edge by at most this fraction of the narrowest bin. */
const LABEL_TOLERANCE = 0.05;
/** Rounding allowed on the label of a constant field. */
const CONSTANT_TOLERANCE = 1e-9;

export class NumericBins {
  static readonly DEFAULT_COUNT = 10;
  static readonly MIN_COUNT = 2;
  static readonly MAX_COUNT = 50;

  private readonly percentileByKey: Map<string, string> | null;

  private constructor(
    /**
     * Ascending bin edges: bin i covers [edges[i], edges[i + 1]), the last bin
     * is closed. For whole-number axis bins: each bin's lowest value, then the maximum.
     */
    readonly edges: number[],
    /** Range label of each bin, low to high (unique). */
    readonly keys: string[],
    /** Percentile label of each bin, parallel to `keys`; null for histogram bins. */
    percentiles: string[] | null = null,
  ) {
    this.percentileByKey = percentiles ? new Map(keys.map((key, i) => [key, percentiles[i]])) : null;
  }

  // ---- Axis bins ----

  /** Axis bins over `values` (non-finite entries ignored); `count` is clamped to the allowed range. */
  static fromValues(values: readonly number[], count: number, method: BinMethod): NumericBins {
    const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (sorted.length === 0) return new NumericBins([], [], []);
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    if (min === max) {
      const label = NumericBins.format(min, NumericBins.decimalsFor([min], CONSTANT_TOLERANCE));
      return new NumericBins([min, max], [label], ['0–100%']);
    }

    const bins = NumericBins.clampCount(count);
    if (sorted.every(Number.isInteger)) return NumericBins.wholeNumbers(sorted, bins, method);

    const edges = NumericBins.edgesFor(sorted, bins, method);
    return new NumericBins(edges, NumericBins.rangeKeys(edges), NumericBins.percentiles(sorted, edges.slice(0, -1)));
  }

  /** Edges from `min` to `max` (sorted, at least two distinct values); coinciding edges are kept once. */
  private static edgesFor(sorted: readonly number[], bins: number, method: BinMethod): number[] {
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const inner = Array.from({ length: bins - 1 }, (_, k) =>
      method === 'width'
        ? min + ((max - min) * (k + 1)) / bins
        : sorted[Math.floor(((k + 1) * sorted.length) / bins)],
    );
    // Ties (or a skewed field) can make quantile edges coincide.
    return [...new Set([min, ...inner, max])];
  }

  /** Axis bins of a whole-number field, labelled by the whole numbers each one holds. */
  private static wholeNumbers(sorted: readonly number[], bins: number, method: BinMethod): NumericBins {
    const distinct = [...new Set(sorted)];
    const max = distinct[distinct.length - 1];
    const lows = distinct.length <= bins
      ? distinct
      : [...new Set(NumericBins.edgesFor(sorted, bins, method).slice(0, -1).map(Math.ceil))];
    const keys = lows.map((low, i) => {
      const high = i + 1 < lows.length ? lows[i + 1] - 1 : max;
      return low === high ? String(low) : `${low}–${high}`;
    });
    return new NumericBins([...lows, max], keys, NumericBins.percentiles(sorted, lows));
  }

  /**
   * Percentile label of each bin: the share of `sorted` below its lower edge
   * and below the next bin's ("10–20%"), the last bin ending at 100%. Uses the
   * fewest decimals (up to two) that keep different shares apart.
   */
  private static percentiles(sorted: readonly number[], lowerEdges: readonly number[]): string[] {
    const n = sorted.length;
    const shares = [...lowerEdges.map(edge => (100 * NumericBins.countBelow(sorted, edge)) / n), 100];
    let decimals = 0;
    while (
      decimals < MAX_PERCENT_DECIMALS
      && shares.some((share, i) => i > 0 && share !== shares[i - 1]
        && share.toFixed(decimals) === shares[i - 1].toFixed(decimals))
    ) {
      decimals += 1;
    }
    return lowerEdges.map((_, i) => `${shares[i].toFixed(decimals)}–${shares[i + 1].toFixed(decimals)}%`);
  }

  /** Values of `sorted` strictly below `value` (binary search). */
  private static countBelow(sorted: readonly number[], value: number): number {
    let low = 0;
    let high = sorted.length;
    while (low < high) {
      const mid = (low + high) >> 1;
      if (sorted[mid] < value) low = mid + 1;
      else high = mid;
    }
    return low;
  }

  /** The bin's percentile label ("10–20%"); undefined for unknown keys and histogram bins. */
  percentileLabel(key: string): string | undefined {
    return this.percentileByKey?.get(key);
  }

  // ---- Histogram bins ----

  /**
   * `count` equal-width bins over [min, max], in value space or in log10
   * space (edges stay in value space; a range reaching zero or below is
   * shifted by |min| + 1 first). A constant range makes one bin.
   */
  static histogramBins(min: number, max: number, count: number, scale: HistogramScale = 'linear'): NumericBins {
    if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) return new NumericBins([], []);
    if (min === max) return new NumericBins([min, max], [NumericBins.tickLabel(min)]);

    const bins = NumericBins.clampCount(count);
    let edges: number[];
    if (scale === 'log') {
      const offset = min <= 0 ? Math.abs(min) + 1 : 0;
      const logMin = Math.log10(min + offset);
      const step = (Math.log10(max + offset) - logMin) / bins;
      edges = Array.from({ length: bins + 1 }, (_, k) => 10 ** (logMin + k * step) - offset);
    } else {
      edges = Array.from({ length: bins + 1 }, (_, k) => min + ((max - min) * k) / bins);
    }
    // Exact ends, so the minimum and the maximum always fall inside.
    edges[0] = min;
    edges[bins] = max;
    return new NumericBins(edges, NumericBins.rangeKeys(edges));
  }

  /** Histogram bins over the range of the numeric entries of `values`. */
  static histogramFor(values: Iterable<unknown>, count: number, scale: HistogramScale = 'linear'): NumericBins {
    let min = Infinity;
    let max = -Infinity;
    for (const raw of values) {
      const value = NumericBins.numericValue(raw);
      if (value === null) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
    return min > max ? new NumericBins([], []) : NumericBins.histogramBins(min, max, count, scale);
  }

  /** Histogram of `values`: equal-width bins over their range with each bin's count. */
  static histogram(values: readonly unknown[], count: number, scale: HistogramScale = 'linear'): HistogramBin[] {
    return NumericBins.histogramFor(values, count, scale).toHistogram(values);
  }

  /** Points per bin (numeric entries inside the range only). */
  counts(values: Iterable<unknown>): number[] {
    const counts = new Array<number>(this.keys.length).fill(0);
    for (const raw of values) {
      const value = NumericBins.numericValue(raw);
      const index = value === null ? -1 : this.indexOf(value);
      if (index >= 0) counts[index] += 1;
    }
    return counts;
  }

  /** Each bin's edges, count and tick label (its start, compactly formatted). */
  toHistogram(values: Iterable<unknown>): HistogramBin[] {
    const counts = this.counts(values);
    return this.keys.map((_, i) => ({
      binStart: this.edges[i],
      binEnd: this.edges[i + 1] ?? this.edges[i],
      count: counts[i],
      label: NumericBins.tickLabel(this.edges[i]),
    }));
  }

  /** Compact tick label: fewer decimals for larger magnitudes, exponent below 0.01. */
  static tickLabel(value: number): string {
    const magnitude = Math.abs(value);
    if (magnitude >= 100) return value.toFixed(0);
    if (magnitude >= 1) return value.toFixed(1);
    if (magnitude >= 0.01 || value === 0) return value.toFixed(2);
    return value.toExponential(1);
  }

  // ---- Lookup ----

  /** Bin index of a value; -1 outside [first edge, last edge] or when there are no bins. */
  indexOf(value: number): number {
    const last = this.edges.length - 1;
    if (this.keys.length === 0 || !(value >= this.edges[0] && value <= this.edges[last])) return -1;
    // Last edge at or below the value (binary search), clamped to a valid bin.
    let low = 0;
    let high = last;
    while (low < high) {
      const mid = (low + high + 1) >> 1;
      if (this.edges[mid] <= value) low = mid;
      else high = mid - 1;
    }
    return Math.min(low, this.keys.length - 1);
  }

  /** Bin key of a raw metadata value; Missing when it is not a number. Out-of-range values clamp to the end bins. */
  keyOf = (raw: unknown): string => {
    const value = NumericBins.numericValue(raw);
    if (value === null || this.keys.length === 0) return CategoryKey.MISSING;
    if (value < this.edges[0]) return this.keys[0];
    if (value > this.edges[this.edges.length - 1]) return this.keys[this.keys.length - 1];
    return this.keys[this.indexOf(value)];
  };

  // ---- Parsing and labels ----

  /** Bin count typed by the user: clamped when numeric, `fallback` when empty or not a number. */
  static parseCount(text: string, fallback: number): number {
    const trimmed = text.trim();
    const value = trimmed === '' ? Number.NaN : Number(trimmed);
    return Number.isFinite(value) ? NumericBins.clampCount(value) : fallback;
  }

  /** A whole number of bins within [MIN_COUNT, MAX_COUNT]; the default for anything unusable. */
  static clampCount(count: number): number {
    if (!Number.isFinite(count)) return NumericBins.DEFAULT_COUNT;
    return Math.min(NumericBins.MAX_COUNT, Math.max(NumericBins.MIN_COUNT, Math.round(count)));
  }

  /** Numbers, or strings that parse as numbers (as field analysis reads them); null otherwise. */
  static numericValue(raw: unknown): number | null {
    const value = typeof raw === 'number' ? raw : typeof raw === 'string' ? parseFloat(raw) : Number.NaN;
    return Number.isFinite(value) ? value : null;
  }

  /** "a–b" label of each bin, printed with the fewest accurate decimals; collisions get a counter. */
  private static rangeKeys(edges: readonly number[]): string[] {
    const narrowest = Math.min(...edges.slice(1).map((edge, i) => edge - edges[i]));
    const decimals = NumericBins.decimalsFor(edges, narrowest * LABEL_TOLERANCE);
    const keys = edges.slice(0, -1).map(
      (edge, i) => `${NumericBins.format(edge, decimals)}–${NumericBins.format(edges[i + 1], decimals)}`,
    );
    return NumericBins.uniqueKeys(keys);
  }

  /** Fewest decimals that keep every edge label distinct and within `tolerance` of its edge. */
  private static decimalsFor(edges: readonly number[], tolerance: number): number {
    for (let decimals = 0; decimals < MAX_DECIMALS; decimals++) {
      const labels = edges.map(edge => NumericBins.format(edge, decimals));
      const distinct = new Set(labels).size === labels.length;
      const accurate = edges.every((edge, i) => Math.abs(Number(labels[i]) - edge) <= tolerance);
      if (distinct && accurate) return decimals;
    }
    return MAX_DECIMALS;
  }

  /** Fixed decimals, never a negative zero. */
  private static format(value: number, decimals: number): string {
    const text = value.toFixed(decimals);
    return /^-0(\.0*)?$/.test(text) ? text.slice(1) : text;
  }

  /** Labels that still collide (edges closer than the printable precision) get a counter. */
  private static uniqueKeys(keys: readonly string[]): string[] {
    const seen = new Map<string, number>();
    return keys.map(key => {
      const n = (seen.get(key) ?? 0) + 1;
      seen.set(key, n);
      return n === 1 ? key : `${key} (${n})`;
    });
  }
}
