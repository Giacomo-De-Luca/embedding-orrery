import { describe, it, expect } from 'vitest';
import {
  computePercent,
  formatElapsed,
  nextEtaState,
  type EtaState,
  type ProgressCounts,
} from '../jobProgress';

function counts(partial: Partial<ProgressCounts>): ProgressCounts {
  return { itemsProcessed: 0, totalItems: 0, currentBatch: 0, totalBatches: 0, ...partial };
}

describe('formatElapsed', () => {
  it('formats minutes and zero-padded seconds', () => {
    expect(formatElapsed(0)).toBe('0:00');
    expect(formatElapsed(5_000)).toBe('0:05');
    expect(formatElapsed(65_000)).toBe('1:05');
    expect(formatElapsed(600_000)).toBe('10:00');
  });
});

describe('computePercent', () => {
  it('returns 0 with no totals', () => {
    expect(computePercent(counts({}))).toBe(0);
  });

  it('computes item-only progress for single-stage jobs', () => {
    expect(computePercent(counts({ itemsProcessed: 250, totalItems: 1000 }))).toBe(25);
    expect(computePercent(counts({ itemsProcessed: 1000, totalItems: 1000, totalBatches: 1 }))).toBe(100);
  });

  it('computes stage-only progress when items are not meaningful', () => {
    expect(computePercent(counts({ currentBatch: 2, totalBatches: 4 }))).toBe(50);
  });

  it('blends stage and item progress for multi-stage jobs', () => {
    // Stage 1 of 4 (25% base) + half of the current stage's 25% width
    expect(
      computePercent(counts({ currentBatch: 1, totalBatches: 4, itemsProcessed: 50, totalItems: 100 }))
    ).toBe(38);
  });

  it('caps at 100', () => {
    expect(computePercent(counts({ currentBatch: 5, totalBatches: 4 }))).toBe(100);
  });
});

describe('nextEtaState', () => {
  const T0 = 1_000_000;

  it('stays unchanged without meaningful item progress', () => {
    expect(nextEtaState(null, counts({}), T0)).toBeNull();
    const prev: EtaState = { baseTime: T0, baseItems: 10, lastTotal: 100, lastItems: 10, etaMs: null };
    expect(nextEtaState(prev, counts({ totalBatches: 3, currentBatch: 1 }), T0 + 1000)).toBe(prev);
  });

  it('establishes a baseline on the first data point (no ETA yet)', () => {
    const s = nextEtaState(null, counts({ itemsProcessed: 10, totalItems: 100 }), T0);
    expect(s).toEqual({ baseTime: T0, baseItems: 10, lastTotal: 100, lastItems: 10, etaMs: null });
  });

  it('produces an ETA from the second data point', () => {
    const s1 = nextEtaState(null, counts({ itemsProcessed: 10, totalItems: 100 }), T0);
    // 10 more items in 10s → 1s/item → 80 remaining items → 80s
    const s2 = nextEtaState(s1, counts({ itemsProcessed: 20, totalItems: 100 }), T0 + 10_000);
    expect(s2?.etaMs).toBe(80_000);
    expect(s2?.lastItems).toBe(20);
  });

  it('resets the baseline when totalItems changes (new phase)', () => {
    const s1 = nextEtaState(null, counts({ itemsProcessed: 90, totalItems: 100 }), T0);
    const s2 = nextEtaState(s1, counts({ itemsProcessed: 5, totalItems: 50 }), T0 + 1000);
    expect(s2).toEqual({ baseTime: T0 + 1000, baseItems: 5, lastTotal: 50, lastItems: 5, etaMs: null });
  });

  it('resets the baseline when itemsProcessed regresses', () => {
    const s1 = nextEtaState(null, counts({ itemsProcessed: 90, totalItems: 100 }), T0);
    const s2 = nextEtaState(s1, counts({ itemsProcessed: 10, totalItems: 100 }), T0 + 1000);
    expect(s2?.baseItems).toBe(10);
    expect(s2?.etaMs).toBeNull();
  });

  it('recomputes ETA against the baseline, so a stall grows the estimate', () => {
    const s1 = nextEtaState(null, counts({ itemsProcessed: 10, totalItems: 100 }), T0);
    const s2 = nextEtaState(s1, counts({ itemsProcessed: 20, totalItems: 100 }), T0 + 10_000);
    // Same item count 10s later → avg time per item doubles → ETA doubles
    const s3 = nextEtaState(s2, counts({ itemsProcessed: 20, totalItems: 100 }), T0 + 20_000);
    expect(s3?.etaMs).toBe(160_000);
  });
});
