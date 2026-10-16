import { describe, it, expect } from 'vitest';
import { computeTemporalCrossTab, windowPeriodsByFraction } from '../temporalAnalysis';
import { CategoryKey } from '../categoryKey';

describe('computeTemporalCrossTab', () => {
  it('stacks points with a missing category under the Missing key', () => {
    const { MISSING } = CategoryKey;
    const points = [
      { index: 0, metadata: { year: 2020, category: 'a' } },
      { index: 1, metadata: { year: 2020 } },
      { index: 2, metadata: { year: 2021, category: '' } },
      { index: 3, metadata: { category: 'a' } },
    ];
    expect(computeTemporalCrossTab(points, 'category', 'year', ['a', MISSING])).toEqual([
      { period: '2020', a: 1, [MISSING]: 1 },
      { period: '2021', a: 0, [MISSING]: 1 },
    ]);
  });

  it('sorts periods numerically and ignores categories outside the requested list', () => {
    const points = [
      { index: 0, metadata: { year: 10, category: 'a' } },
      { index: 1, metadata: { year: 9, category: 'b' } },
      { index: 2, metadata: { year: 9, category: 'a' } },
    ];
    expect(computeTemporalCrossTab(points, 'category', 'year', ['a'])).toEqual([
      { period: '9', a: 1 },
      { period: '10', a: 1 },
    ]);
  });
});

describe('windowPeriodsByFraction', () => {
  // A 12-period span, e.g. EMNLP years 2014–2025.
  const twelve = Array.from({ length: 12 }, (_, i) => String(2014 + i));

  it('selects the earliest third as an inclusive [start, end] pair', () => {
    // floor(0·12)=0 .. ceil(1/3·12)-1 = 3  →  2014..2017 (4 of 12 periods)
    expect(windowPeriodsByFraction(twelve, 0, 1 / 3)).toEqual({
      startPeriod: '2014',
      endPeriod: '2017',
    });
  });

  it('selects a middle window', () => {
    // floor(1/3·12)=4 .. ceil(2/3·12)-1 = 7  →  2018..2021
    expect(windowPeriodsByFraction(twelve, 1 / 3, 2 / 3)).toEqual({
      startPeriod: '2018',
      endPeriod: '2021',
    });
  });

  it('returns null for the full range (no filtering needed)', () => {
    expect(windowPeriodsByFraction(twelve, 0, 1)).toBeNull();
  });

  it('returns null when there are fewer than two periods', () => {
    expect(windowPeriodsByFraction([], 0, 1 / 3)).toBeNull();
    expect(windowPeriodsByFraction(['2020'], 0, 1 / 3)).toBeNull();
  });

  it('clamps indices and keeps end ≥ start on tiny spans', () => {
    // Two periods, earliest third → a single-period window at the start.
    expect(windowPeriodsByFraction(['2019', '2020'], 0, 1 / 3)).toEqual({
      startPeriod: '2019',
      endPeriod: '2019',
    });
  });

  it('never produces an out-of-range or inverted window', () => {
    for (let n = 2; n <= 20; n++) {
      const periods = Array.from({ length: n }, (_, i) => String(i));
      for (const [from, to] of [
        [0, 1 / 3],
        [0.25, 0.75],
        [0.5, 1],
        [0.9, 1],
      ] as const) {
        const w = windowPeriodsByFraction(periods, from, to);
        if (w === null) continue;
        const si = periods.indexOf(w.startPeriod);
        const ei = periods.indexOf(w.endPeriod);
        expect(si).toBeGreaterThanOrEqual(0);
        expect(ei).toBeLessThanOrEqual(n - 1);
        expect(ei).toBeGreaterThanOrEqual(si);
      }
    }
  });
});
