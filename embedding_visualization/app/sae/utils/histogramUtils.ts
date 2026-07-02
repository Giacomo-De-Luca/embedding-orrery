/**
 * Histogram binning utilities for SAE feature statistics.
 */

export interface HistogramBin {
  binStart: number;
  binEnd: number;
  count: number;
  label: string;
}

/**
 * Compute linear histogram bins using Sturges' rule for default bin count.
 */
export function computeHistogramBins(values: number[], numBins?: number): HistogramBin[] {
  if (values.length === 0) return [];

  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    return [{ binStart: min, binEnd: max, count: values.length, label: formatNum(min) }];
  }

  const n = numBins ?? Math.min(50, Math.max(10, Math.ceil(Math.log2(values.length)) + 1));
  const binWidth = (max - min) / n;
  const bins: HistogramBin[] = Array.from({ length: n }, (_, i) => ({
    binStart: min + i * binWidth,
    binEnd: min + (i + 1) * binWidth,
    count: 0,
    label: `${formatNum(min + i * binWidth)}`,
  }));

  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), n - 1);
    bins[idx].count++;
  }

  return bins;
}

/**
 * Compute log-scale histogram bins (for values spanning orders of magnitude like density).
 * Filters out non-positive values before binning.
 */
export function computeLogHistogramBins(values: number[], numBins?: number): HistogramBin[] {
  const positive = values.filter((v) => v > 0);
  if (positive.length === 0) return [];

  const logValues = positive.map(Math.log10);
  const logMin = Math.min(...logValues);
  const logMax = Math.max(...logValues);
  if (logMin === logMax) {
    return [{ binStart: positive[0], binEnd: positive[0], count: positive.length, label: formatSci(positive[0]) }];
  }

  const n = numBins ?? Math.min(30, Math.max(10, Math.ceil(Math.log2(positive.length)) + 1));
  const logWidth = (logMax - logMin) / n;
  const bins: HistogramBin[] = Array.from({ length: n }, (_, i) => {
    const start = Math.pow(10, logMin + i * logWidth);
    const end = Math.pow(10, logMin + (i + 1) * logWidth);
    return { binStart: start, binEnd: end, count: 0, label: formatSci(start) };
  });

  for (const v of positive) {
    const logV = Math.log10(v);
    const idx = Math.min(Math.floor((logV - logMin) / logWidth), n - 1);
    bins[idx].count++;
  }

  return bins;
}

/**
 * Find which bin index contains a given value (for highlighting).
 */
export function findBinIndex(bins: HistogramBin[], value: number): number {
  return bins.findIndex((b) => value >= b.binStart && value <= b.binEnd);
}

function formatNum(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(1);
  if (Math.abs(v) >= 0.01) return v.toFixed(2);
  return v.toExponential(1);
}

function formatSci(v: number): string {
  if (v >= 0.01 && v < 100) return v.toFixed(3);
  return v.toExponential(1);
}
