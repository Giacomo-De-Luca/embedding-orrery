import { describe, it, expect } from 'vitest';
import { NumericBins } from '../numericBins';
import { CategoryKey } from '../categoryKey';

const oneToHundred = Array.from({ length: 100 }, (_, i) => i + 1);

function countPerBin(bins: NumericBins, values: number[]): number[] {
  const counts = new Map(bins.keys.map(key => [key, 0]));
  for (const v of values) counts.set(bins.keyOf(v), (counts.get(bins.keyOf(v)) ?? 0) + 1);
  return bins.keys.map(key => counts.get(key)!);
}

describe('NumericBins equal-count', () => {
  it('splits evenly spread values into bins of equal size', () => {
    const bins = NumericBins.fromValues(oneToHundred, 10, 'quantile');
    expect(bins.keys).toHaveLength(10);
    expect(countPerBin(bins, oneToHundred)).toEqual(new Array(10).fill(10));
  });

  it('puts the extremes in the first and last bins', () => {
    const bins = NumericBins.fromValues(oneToHundred, 4, 'quantile');
    expect(bins.keyOf(1)).toBe(bins.keys[0]);
    expect(bins.keyOf(100)).toBe(bins.keys[3]);
  });

  it('merges bins whose edges coincide when many values tie', () => {
    const values = [...new Array(90).fill(0), ...Array.from({ length: 10 }, (_, i) => i + 1)];
    const bins = NumericBins.fromValues(values, 10, 'quantile');
    expect(bins.keys.length).toBeLessThan(10);
    expect(new Set(bins.keys).size).toBe(bins.keys.length);
    expect(countPerBin(bins, values).reduce((a, b) => a + b, 0)).toBe(100);
  });
});

describe('NumericBins equal-width', () => {
  it('cuts the range into bins of equal width', () => {
    const bins = NumericBins.fromValues([0, 1.5, 2, 3, 10], 5, 'width');
    expect(bins.edges).toEqual([0, 2, 4, 6, 8, 10]);
    expect(bins.keyOf(1)).toBe(bins.keys[0]);
    expect(bins.keyOf(2)).toBe(bins.keys[1]);
    expect(bins.keyOf(10)).toBe(bins.keys[4]);
  });
});

describe('NumericBins on whole numbers', () => {
  /** Lowest and highest whole number a label covers ("1993–1996" or "2019"). */
  const rangeOf = (label: string) => {
    const [lo, hi = lo] = label.split('–').map(Number);
    return { lo, hi };
  };

  it('labels bins by the whole numbers they hold, without overlap', () => {
    const years = Array.from({ length: 35 }, (_, i) => 1990 + i);
    for (const method of ['quantile', 'width'] as const) {
      const bins = NumericBins.fromValues(years, 10, method);
      for (const year of years) {
        const { lo, hi } = rangeOf(bins.keyOf(year));
        expect(lo).toBeLessThanOrEqual(year);
        expect(hi).toBeGreaterThanOrEqual(year);
      }
      const ranges = bins.keys.map(rangeOf);
      ranges.slice(1).forEach((range, i) => expect(range.lo).toBe(ranges[i].hi + 1));
      expect(ranges.at(-1)!.hi).toBe(2024);
    }
  });

  it('gives each value its own bin when there are no more values than bins', () => {
    expect(NumericBins.fromValues([1, 2, 3, 4, 5, 3], 10, 'quantile').keys).toEqual(['1', '2', '3', '4', '5']);
  });

  it('labels equal-width integer bins by their lowest and highest value', () => {
    const values = Array.from({ length: 101 }, (_, i) => i);
    expect(NumericBins.fromValues(values, 5, 'width').keys).toEqual(['0–19', '20–39', '40–59', '60–79', '80–100']);
  });
});

describe('NumericBins keys and labels', () => {
  it('labels each bin with its range, in value order', () => {
    const bins = NumericBins.fromValues([0, 2.5, 5], 2, 'width');
    expect(bins.keys).toEqual(['0.0–2.5', '2.5–5.0']);
  });

  it('keeps keys unique even when edges are too close to print apart', () => {
    const values = [0.1234567890123, 0.1234567890124, 0.1234567890125, 0.5];
    const bins = NumericBins.fromValues(values, 4, 'quantile');
    expect(new Set(bins.keys).size).toBe(bins.keys.length);
    expect(bins.keys.length).toBe(3);
  });

  it('never prints a negative zero', () => {
    const bins = NumericBins.fromValues([-0.001, 0.4, 1, 2], 2, 'width');
    expect(bins.keys.some(key => key.startsWith('-0'))).toBe(false);
  });

  it('uses enough decimals to keep close edges apart', () => {
    const bins = NumericBins.fromValues([0.1231, 0.1234, 0.1237, 0.124], 3, 'width');
    expect(new Set(bins.keys).size).toBe(3);
  });

  it('maps absent and non-numeric values to Missing and parses numeric strings', () => {
    const bins = NumericBins.fromValues(oneToHundred, 4, 'quantile');
    expect(bins.keyOf(undefined)).toBe(CategoryKey.MISSING);
    expect(bins.keyOf('n/a')).toBe(CategoryKey.MISSING);
    expect(bins.keyOf(Number.NaN)).toBe(CategoryKey.MISSING);
    expect(bins.keyOf('100')).toBe(bins.keys[3]);
  });

  it('clamps values outside the range into the end bins', () => {
    const bins = NumericBins.fromValues([0, 10], 2, 'width');
    expect(bins.keyOf(-5)).toBe(bins.keys[0]);
    expect(bins.keyOf(50)).toBe(bins.keys[1]);
  });

  it('makes one bin for a constant field and none for no values', () => {
    expect(NumericBins.fromValues([3, 3, 3], 10, 'quantile').keys).toEqual(['3']);
    const empty = NumericBins.fromValues([], 10, 'quantile');
    expect(empty.keys).toEqual([]);
    expect(empty.keyOf(1)).toBe(CategoryKey.MISSING);
  });

  it('reads numbers and numeric strings, nothing else', () => {
    expect(NumericBins.numericValue(2.5)).toBe(2.5);
    expect(NumericBins.numericValue('3')).toBe(3);
    expect(NumericBins.numericValue('abc')).toBeNull();
    expect(NumericBins.numericValue(null)).toBeNull();
    expect(NumericBins.numericValue(Number.POSITIVE_INFINITY)).toBeNull();
  });

  it('parses a typed bin count, falling back for empty or non-numeric text', () => {
    expect(NumericBins.parseCount('25', 10)).toBe(25);
    expect(NumericBins.parseCount(' 3.6 ', 10)).toBe(4);
    expect(NumericBins.parseCount('0', 10)).toBe(NumericBins.MIN_COUNT);
    expect(NumericBins.parseCount('', 10)).toBe(10);
    expect(NumericBins.parseCount('abc', 7)).toBe(7);
  });

  it('clamps the requested bin count to the allowed range', () => {
    expect(NumericBins.clampCount(0)).toBe(NumericBins.MIN_COUNT);
    expect(NumericBins.clampCount(999)).toBe(NumericBins.MAX_COUNT);
    expect(NumericBins.clampCount(7.6)).toBe(8);
    expect(NumericBins.clampCount(Number.NaN)).toBe(NumericBins.DEFAULT_COUNT);
  });
});

describe('NumericBins histograms', () => {
  it('cuts strictly equal-width bins, whole numbers included, and counts the values', () => {
    const values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const bins = NumericBins.histogramFor(values, 5);
    expect(bins.edges).toEqual([0, 2, 4, 6, 8, 10]);
    expect(bins.toHistogram(values).map(b => b.count)).toEqual([2, 2, 2, 2, 3]);
  });

  it('describes each bin by its edges, its count and a tick label', () => {
    const [first] = NumericBins.histogram([0.5, 1.5, 2.5, 3.5], 2);
    expect(first).toEqual({ binStart: 0.5, binEnd: 2, count: 2, label: '0.50' });
  });

  it('bins in log space for values spanning orders of magnitude', () => {
    const values = [1, 10, 100, 1000];
    const bins = NumericBins.histogramFor(values, 3, 'log');
    expect(bins.edges.map(e => Number(e.toPrecision(6)))).toEqual([1, 10, 100, 1000]);
    expect(bins.toHistogram(values).map(b => b.count)).toEqual([1, 1, 2]);
  });

  it('shifts a log range that reaches zero or below, keeping the edges in value space', () => {
    const bins = NumericBins.histogramBins(0, 99, 2, 'log');
    expect(bins.edges.map(e => Number(e.toFixed(6)))).toEqual([0, 9, 99]);
    expect(bins.toHistogram([0, 9, 99]).map(b => b.count)).toEqual([1, 2]);
  });

  it('makes one bin for constant values and none for no values', () => {
    expect(NumericBins.histogram([4, 4, 4], 10)).toEqual([{ binStart: 4, binEnd: 4, count: 3, label: '4.0' }]);
    expect(NumericBins.histogram([], 10)).toEqual([]);
  });

  it('finds the bin of a value, -1 outside the range', () => {
    const bins = NumericBins.histogramBins(0, 10, 5);
    expect(bins.indexOf(0)).toBe(0);
    expect(bins.indexOf(5)).toBe(2);
    expect(bins.indexOf(10)).toBe(4);
    expect(bins.indexOf(-1)).toBe(-1);
    expect(bins.indexOf(11)).toBe(-1);
  });

  it('skips values that are not numbers and handles large inputs', () => {
    const many = Array.from({ length: 300_000 }, (_, i) => i % 100);
    const bins = NumericBins.histogramFor(many, 10);
    expect(bins.counts([...many, 'n/a', null]).reduce((a, b) => a + b, 0)).toBe(300_000);
  });
});

describe('NumericBins percentile labels', () => {
  const percentiles = (bins: NumericBins) => bins.keys.map(key => bins.percentileLabel(key));

  it('labels equal-count bins by the share of points below their edges', () => {
    expect(percentiles(NumericBins.fromValues(oneToHundred, 4, 'quantile'))).toEqual(['0–25%', '25–50%', '50–75%', '75–100%']);
  });

  it('shows the uneven shares of equal-width bins', () => {
    const values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 9.5];
    expect(percentiles(NumericBins.fromValues(values, 2, 'width'))).toEqual(['0–90%', '90–100%']);
  });

  it('labels whole-number bins the same way', () => {
    expect(percentiles(NumericBins.fromValues([1, 1, 2, 3], 10, 'quantile'))).toEqual(['0–50%', '50–75%', '75–100%']);
  });

  it('adds a decimal when rounded shares would collide', () => {
    const values = [...new Array(997).fill(0), 1.5, 1.5, 3];
    expect(percentiles(NumericBins.fromValues(values, 2, 'width'))).toEqual(['0.0–99.7%', '99.7–100.0%']);
  });

  it('has no percentile label for keys it does not know', () => {
    expect(NumericBins.fromValues(oneToHundred, 4, 'quantile').percentileLabel(CategoryKey.MISSING)).toBeUndefined();
  });
});
