/**
 * Tests for the category row view-model builder.
 *
 * These tests cover the pure data logic behind the CategoryBarList
 * component: filter detection, sorting modes, name filtering,
 * percentage math, and bar-fraction normalization.
 */

import { describe, it, expect } from 'vitest';
import {
  isCategoryFilterActive,
  buildCategoryRows,
  type CategorySortMode,
} from '../categoryRowData';

/** Shorthand: build rows with sensible defaults. */
function build(overrides: {
  categoryValues?: string[];
  categoryCounts?: Record<string, number>;
  filteredCounts?: Record<string, number> | null;
  sortMode?: CategorySortMode;
  nameFilter?: string;
  getLabel?: (value: string) => string;
} = {}) {
  return buildCategoryRows({
    categoryValues: overrides.categoryValues ?? ['a', 'b', 'c'],
    categoryCounts: overrides.categoryCounts ?? { a: 60, b: 30, c: 10 },
    filteredCounts: overrides.filteredCounts,
    sortMode: overrides.sortMode ?? 'count',
    nameFilter: overrides.nameFilter,
    getLabel: overrides.getLabel,
  });
}

describe('isCategoryFilterActive', () => {
  it('is false for null, undefined, and empty object', () => {
    expect(isCategoryFilterActive(null)).toBe(false);
    expect(isCategoryFilterActive(undefined)).toBe(false);
    expect(isCategoryFilterActive({})).toBe(false);
  });

  it('is true for any non-empty object, even with zero values', () => {
    expect(isCategoryFilterActive({ a: 0 })).toBe(true);
    expect(isCategoryFilterActive({ a: 5, b: 0 })).toBe(true);
  });
});

describe('buildCategoryRows — no filter', () => {
  it('computes pctOfTotal and trackFraction from counts', () => {
    const { rows } = build();
    const byValue = Object.fromEntries(rows.map(r => [r.value, r]));
    expect(byValue.a.pctOfTotal).toBeCloseTo(0.6);
    expect(byValue.b.pctOfTotal).toBeCloseTo(0.3);
    expect(byValue.c.pctOfTotal).toBeCloseTo(0.1);
    expect(byValue.a.trackFraction).toBeCloseTo(1);
    expect(byValue.b.trackFraction).toBeCloseTo(0.5);
    expect(byValue.c.trackFraction).toBeCloseTo(1 / 6);
  });

  it('leaves filter-related fields null', () => {
    const { rows, summary } = build();
    for (const row of rows) {
      expect(row.filteredCount).toBeNull();
      expect(row.matchRate).toBeNull();
      expect(row.fillFraction).toBeNull();
    }
    expect(summary.totalFiltered).toBeNull();
    expect(summary.matchedCategoryCount).toBeNull();
  });

  it('treats a value missing from categoryCounts as count 0 without NaN', () => {
    const { rows } = build({ categoryValues: ['a', 'b'], categoryCounts: { a: 5 } });
    const b = rows.find(r => r.value === 'b')!;
    expect(b.count).toBe(0);
    expect(b.pctOfTotal).toBe(0);
    expect(b.trackFraction).toBe(0);
    expect(Number.isNaN(b.pctOfTotal)).toBe(false);
  });

  it('uses identity labels by default and resolves labels via getLabel', () => {
    const { rows: identityRows } = build();
    expect(identityRows.every(r => r.label === r.value)).toBe(true);

    const { rows } = build({ getLabel: v => (v === 'a' ? 'Alpha' : v.toUpperCase()) });
    const byValue = Object.fromEntries(rows.map(r => [r.value, r]));
    expect(byValue.a.label).toBe('Alpha');
    expect(byValue.b.label).toBe('B');
  });

  it('populates summary counts', () => {
    const { summary } = build();
    expect(summary.totalCount).toBe(100);
    expect(summary.visibleCategoryCount).toBe(3);
  });
});

describe('buildCategoryRows — filter active', () => {
  it('treats a missing filteredCounts key as 0 matches', () => {
    const { rows } = build({ filteredCounts: { a: 30 } });
    const byValue = Object.fromEntries(rows.map(r => [r.value, r]));
    expect(byValue.a.filteredCount).toBe(30);
    expect(byValue.b.filteredCount).toBe(0);
    expect(byValue.c.filteredCount).toBe(0);
  });

  it('computes matchRate as filteredCount / count, 0 for zero-count categories', () => {
    const { rows } = build({
      categoryValues: ['a', 'b', 'z'],
      categoryCounts: { a: 60, b: 30 },
      filteredCounts: { a: 30, b: 15, z: 4 },
    });
    const byValue = Object.fromEntries(rows.map(r => [r.value, r]));
    expect(byValue.a.matchRate).toBeCloseTo(0.5);
    expect(byValue.b.matchRate).toBeCloseTo(0.5);
    // z has count 0 — rate must be 0, never NaN/Infinity
    expect(byValue.z.matchRate).toBe(0);
    expect(Number.isFinite(byValue.z.matchRate!)).toBe(true);
  });

  it('scales fill and track by the same denominator so fill <= track', () => {
    const { rows } = build({ filteredCounts: { a: 30, b: 30, c: 5 } });
    const byValue = Object.fromEntries(rows.map(r => [r.value, r]));
    expect(byValue.a.trackFraction).toBeCloseTo(1);
    expect(byValue.a.fillFraction).toBeCloseTo(0.5);
    for (const row of rows) {
      expect(row.fillFraction!).toBeLessThanOrEqual(row.trackFraction);
    }
  });

  it('summary sums only categoryValues keys and counts matched categories', () => {
    const { summary } = build({
      filteredCounts: { a: 30, b: 0, Unclustered: 999 },
    });
    // 'Unclustered' is not in categoryValues — must be ignored
    expect(summary.totalFiltered).toBe(30);
    expect(summary.matchedCategoryCount).toBe(1);
  });
});

describe('buildCategoryRows — sorting', () => {
  it('sorts by count desc by default, tie-breaking by natural label order', () => {
    const { rows } = build({
      categoryValues: ['b', 'a', 'c'],
      categoryCounts: { b: 5, a: 5, c: 9 },
    });
    expect(rows.map(r => r.value)).toEqual(['c', 'a', 'b']);
  });

  it('sorts by filteredCount desc when a filter is active in count mode', () => {
    const { rows } = build({
      categoryCounts: { a: 60, b: 30, c: 10 },
      filteredCounts: { a: 1, b: 20, c: 9 },
      sortMode: 'count',
    });
    expect(rows.map(r => r.value)).toEqual(['b', 'c', 'a']);
  });

  it('breaks filteredCount ties by total count desc', () => {
    const { rows } = build({
      categoryCounts: { a: 10, b: 40, c: 100 },
      filteredCounts: { a: 5, b: 5, c: 5 },
      sortMode: 'count',
    });
    expect(rows.map(r => r.value)).toEqual(['c', 'b', 'a']);
  });

  it('sorts by matchRate desc in rate mode, ties by filteredCount desc', () => {
    const { rows } = build({
      categoryValues: ['a', 'b', 'c', 'd'],
      categoryCounts: { a: 2, b: 100, c: 10, d: 10 },
      filteredCounts: { a: 2, b: 50, c: 0, d: 10 },
      sortMode: 'rate',
    });
    // d: rate 1 (fc 10), a: rate 1 (fc 2), b: rate 0.5, c: rate 0
    expect(rows.map(r => r.value)).toEqual(['d', 'a', 'b', 'c']);
  });

  it('falls back to count order in rate mode when no filter is active', () => {
    const rateRows = build({ sortMode: 'rate' }).rows.map(r => r.value);
    const countRows = build({ sortMode: 'count' }).rows.map(r => r.value);
    expect(rateRows).toEqual(countRows);
  });

  it('sorts numerically in natural mode', () => {
    const { rows } = build({
      categoryValues: ['10', '2', '1'],
      categoryCounts: { '10': 1, '2': 2, '1': 3 },
      sortMode: 'natural',
    });
    expect(rows.map(r => r.value)).toEqual(['1', '2', '10']);
  });

  it('sorts non-numeric labels lexicographically in natural mode', () => {
    const { rows } = build({
      categoryValues: ['beta', 'alpha'],
      categoryCounts: { beta: 1, alpha: 2 },
      sortMode: 'natural',
    });
    expect(rows.map(r => r.value)).toEqual(['alpha', 'beta']);
  });

  it('sorts by resolved label, not raw value, in natural mode', () => {
    const { rows } = build({
      categoryValues: ['x', 'y'],
      categoryCounts: { x: 1, y: 2 },
      sortMode: 'natural',
      getLabel: v => (v === 'x' ? 'zebra' : 'apple'),
    });
    expect(rows.map(r => r.value)).toEqual(['y', 'x']);
  });

  it('keeps original relative order for duplicate labels (stable sort)', () => {
    const { rows } = build({
      categoryValues: ['first', 'second'],
      categoryCounts: { first: 1, second: 1 },
      sortMode: 'natural',
      getLabel: () => 'Same Label',
    });
    expect(rows.map(r => r.value)).toEqual(['first', 'second']);
  });
});

describe('buildCategoryRows — name filter', () => {
  it('matches resolved labels case-insensitively', () => {
    const { rows } = build({
      categoryValues: ['NN', 'VB'],
      categoryCounts: { NN: 10, VB: 5 },
      nameFilter: 'nou',
      getLabel: v => (v === 'NN' ? 'Noun' : 'Verb'),
    });
    expect(rows.map(r => r.value)).toEqual(['NN']);
  });

  it('ignores whitespace-only filters', () => {
    const { rows } = build({ nameFilter: '   ' });
    expect(rows).toHaveLength(3);
  });

  it('renormalizes trackFraction to the max of remaining rows, keeps global pctOfTotal', () => {
    const { rows } = build({ nameFilter: 'c' });
    expect(rows).toHaveLength(1);
    expect(rows[0].value).toBe('c');
    expect(rows[0].trackFraction).toBeCloseTo(1);
    expect(rows[0].pctOfTotal).toBeCloseTo(0.1);
  });

  it('keeps summary.totalCount global while visibleCategoryCount reflects the filter', () => {
    const { summary } = build({ nameFilter: 'c' });
    expect(summary.totalCount).toBe(100);
    expect(summary.visibleCategoryCount).toBe(1);
  });
});

describe('buildCategoryRows — name filter combined with active filter', () => {
  it('computes summary filtered totals before the name filter', () => {
    const { summary } = build({ nameFilter: 'c', filteredCounts: { a: 30, b: 5 } });
    expect(summary.totalFiltered).toBe(35);
    expect(summary.matchedCategoryCount).toBe(2);
    expect(summary.visibleCategoryCount).toBe(1);
  });

  it('renormalizes track and fill fractions to the visible rows', () => {
    const { rows } = build({ nameFilter: 'c', filteredCounts: { c: 5 } });
    // Only 'c' (count 10) remains visible: track 10/10, fill 5/10
    expect(rows).toHaveLength(1);
    expect(rows[0].trackFraction).toBeCloseTo(1);
    expect(rows[0].fillFraction).toBeCloseTo(0.5);
  });

  it('keeps natural order independent of filtered counts', () => {
    const { rows } = build({
      categoryValues: ['b', 'a'],
      categoryCounts: { b: 1, a: 2 },
      filteredCounts: { b: 100, a: 1 },
      sortMode: 'natural',
    });
    expect(rows.map(r => r.value)).toEqual(['a', 'b']);
  });
});

describe('buildCategoryRows — edge cases', () => {
  it('handles empty categoryValues', () => {
    const { rows, summary } = build({ categoryValues: [], categoryCounts: {} });
    expect(rows).toEqual([]);
    expect(summary.totalCount).toBe(0);
    expect(summary.visibleCategoryCount).toBe(0);
    expect(summary.totalFiltered).toBeNull();
    expect(summary.matchedCategoryCount).toBeNull();
  });

  it('handles all-zero counts without NaN', () => {
    const { rows } = build({ categoryCounts: { a: 0, b: 0, c: 0 } });
    for (const row of rows) {
      expect(row.pctOfTotal).toBe(0);
      expect(row.trackFraction).toBe(0);
      expect(Number.isNaN(row.trackFraction)).toBe(false);
    }
  });

  it('gives a single category full fractions', () => {
    const { rows } = build({ categoryValues: ['only'], categoryCounts: { only: 7 } });
    expect(rows[0].pctOfTotal).toBe(1);
    expect(rows[0].trackFraction).toBe(1);
  });
});
