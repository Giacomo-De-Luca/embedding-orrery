import { describe, it, expect } from 'vitest';
import { FieldCorrelation } from '../fieldCorrelation';

const pointsOf = (rows: Record<string, unknown>[]) => rows.map((metadata, index) => ({ index, metadata }));

/** Deterministic pseudo-random numbers in [0, 1) (LCG), so the tests are reproducible. */
function random(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

describe('FieldCorrelation basics', () => {
  it('formats coefficients with two decimals and a true minus sign', () => {
    expect(FieldCorrelation.format(-0.0512)).toBe('−0.05');
    expect(FieldCorrelation.format(0.6)).toBe('0.60');
    expect(FieldCorrelation.format(null)).toBe('–');
  });

  it('gives tied values their average rank', () => {
    expect(FieldCorrelation.ranks([10, 20, 20, 30])).toEqual([1, 2.5, 2.5, 4]);
    expect(FieldCorrelation.ranks([3, 1, 2])).toEqual([3, 1, 2]);
  });

  it('computes Pearson r, null without variance or with fewer than three pairs', () => {
    expect(FieldCorrelation.pearson([1, 2, 3, 4], [2, 4, 6, 8])).toBeCloseTo(1);
    expect(FieldCorrelation.pearson([1, 2, 3, 4], [8, 6, 4, 2])).toBeCloseTo(-1);
    expect(FieldCorrelation.pearson([1, 2, 3], [5, 5, 5])).toBeNull();
    expect(FieldCorrelation.pearson([1, 2], [1, 2])).toBeNull();
  });

  it('computes Spearman rho on ranks, 1 for any increasing relation', () => {
    const xs = [1, 2, 3, 4, 5];
    const ys = xs.map(x => x ** 3);
    expect(FieldCorrelation.spearman(xs, ys)).toBeCloseTo(1);
    expect(FieldCorrelation.pearson(xs, ys)!).toBeLessThan(1);
  });

  it('computes the correlation ratio eta between groups and values', () => {
    expect(FieldCorrelation.correlationRatio(['a', 'a', 'b', 'b'], [1, 1, 5, 5])).toBeCloseTo(1);
    expect(FieldCorrelation.correlationRatio(['a', 'a', 'b', 'b'], [1, 5, 1, 5])).toBeCloseTo(0);
    expect(FieldCorrelation.correlationRatio(['a', 'b', 'c'], [2, 2, 2])).toBeNull();
  });

  it('has no eta with a single group, or with no more points than groups', () => {
    expect(FieldCorrelation.correlationRatio(['a', 'a', 'a'], [1, 2, 3])).toBeNull();
    expect(FieldCorrelation.correlationRatio(['a', 'b', 'c'], [1, 2, 3])).toBeNull();
  });

  it('corrects eta for the many-small-groups bias, so noise reads as no association', () => {
    const next = random(1);
    const groups = Array.from({ length: 300 }, (_, i) => `g${i % 100}`);
    const values = groups.map(() => next());
    expect(FieldCorrelation.correlationRatio(groups, values)!).toBeLessThan(0.25);
  });
});

describe('FieldCorrelation.matrix', () => {
  const points = pointsOf([
    { x: 1, y: 2, z: 9, w: 1 },
    { x: 2, y: 4, z: 7, w: 1 },
    { x: 3, y: 6, z: 8 },
    { x: 4, y: 8, z: 1, w: 4 },
    { x: 5, y: 10, z: 2, w: 2 },
  ]);

  it('is symmetric with a unit diagonal', () => {
    const m = FieldCorrelation.matrix(points, ['x', 'y', 'z'], 'spearman');
    expect(m.fields).toEqual(['x', 'y', 'z']);
    expect(m.values[0][0]).toBe(1);
    expect(m.values[0][1]).toBeCloseTo(1);
    expect(m.values[0][2]).toBeCloseTo(m.values[2][0]!);
    expect(m.values[0][2]!).toBeLessThan(0);
  });

  it('uses the pairs where both fields have a value', () => {
    const m = FieldCorrelation.matrix(points, ['x', 'w'], 'spearman');
    expect(m.counts[0][1]).toBe(4);
    expect(m.values[0][1]).toBeCloseTo(FieldCorrelation.spearman([1, 2, 4, 5], [1, 1, 4, 2])!);
  });

  it('skips the excluded point indices', () => {
    const m = FieldCorrelation.matrix(points, ['x', 'z'], 'pearson', new Set([3, 4]));
    expect(m.counts[0][1]).toBe(3);
    expect(m.values[0][1]).toBeCloseTo(FieldCorrelation.pearson([1, 2, 3], [9, 7, 8])!);
  });

  it('matches direct Spearman on each pair, with ties and missing values', () => {
    const next = random(7);
    const rows = Array.from({ length: 60 }, () => ({
      a: Math.floor(next() * 5),
      b: next() < 0.2 ? undefined : Math.floor(next() * 4),
      c: next() < 0.1 ? null : Math.round(next() * 3) / 2,
    }));
    const fields = ['a', 'b', 'c'];
    const m = FieldCorrelation.matrix(pointsOf(rows), fields, 'spearman');
    for (let i = 0; i < fields.length; i++) {
      for (let j = i + 1; j < fields.length; j++) {
        const shared = rows.filter(r => typeof r[fields[i] as 'a'] === 'number' && typeof r[fields[j] as 'a'] === 'number');
        const direct = FieldCorrelation.spearman(
          shared.map(r => r[fields[i] as 'a'] as number),
          shared.map(r => r[fields[j] as 'a'] as number),
        );
        expect(m.counts[i][j]).toBe(shared.length);
        expect(m.values[i][j]).toBeCloseTo(direct!, 10);
      }
    }
  });

  it('gives the same values when fields are added to a cached set of columns', () => {
    const next = random(3);
    const points = pointsOf(Array.from({ length: 40 }, () => ({ a: next(), b: next(), c: next() })));
    const columns = FieldCorrelation.columns(points);
    const small = columns.matrix(['a', 'b'], 'spearman');
    const large = columns.matrix(['a', 'b', 'c'], 'spearman');
    expect(large.values[0][1]).toBe(small.values[0][1]);
    expect(large.values[0][2]).toBeCloseTo(FieldCorrelation.matrix(points, ['a', 'c'], 'spearman').values[0][1]!, 12);
  });

  it('reports null for a pair without enough shared values', () => {
    const sparse = pointsOf([{ a: 1 }, { b: 2 }, { a: 3, b: 4 }]);
    const m = FieldCorrelation.matrix(sparse, ['a', 'b'], 'spearman');
    expect(m.values[0][1]).toBeNull();
    expect(m.counts[0][1]).toBe(1);
  });
});
