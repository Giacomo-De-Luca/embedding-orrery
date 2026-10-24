import { describe, expect, it } from 'vitest';

import { HighlightedPoints } from '../highlightedPoints';

const points = Array.from({ length: 10 }, (_, index) => ({ index, name: `p${index}` }));

describe('HighlightedPoints', () => {
  it('returns the highlighted points in highlight order, looked up by index', () => {
    const highlights = new Map([[5, 0.9], [2, 0.8], [9, 0.1]]);
    expect(HighlightedPoints.lookup(points, highlights).map(p => p.name)).toEqual(['p5', 'p2', 'p9']);
  });

  it('skips indices outside the points', () => {
    expect(HighlightedPoints.lookup(points, new Map([[3, 1], [42, 1], [-1, 1]])).map(p => p.name)).toEqual(['p3']);
  });

  it('returns nothing without highlights', () => {
    expect(HighlightedPoints.lookup(points, null)).toEqual([]);
    expect(HighlightedPoints.lookup(points, new Map())).toEqual([]);
  });

  it('falls back to a scan when points are not stored at their index', () => {
    const subset = [points[2], points[5], points[9]];
    const highlights = new Map([[5, 0.9], [2, 0.8], [9, 0.1], [7, 0.5]]);
    expect(HighlightedPoints.lookup(subset, highlights).map(p => p.name)).toEqual(['p5', 'p2', 'p9']);
  });
});
