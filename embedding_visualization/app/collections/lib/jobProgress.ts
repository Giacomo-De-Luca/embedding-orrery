/**
 * Pure progress math shared by the job progress UI (modal, dock, strip).
 * Kept free of React/GraphQL imports so it is trivially unit-testable.
 */

/** Structural subset of JobProgress needed for the math. */
export interface ProgressCounts {
  itemsProcessed: number;
  totalItems: number;
  currentBatch: number;
  totalBatches: number;
}

export function formatElapsed(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

/**
 * Compute progress percentage using a blended stage+item approach.
 *
 * Multi-stage operations (totalBatches > 1): base progress from
 * currentBatch/totalBatches, plus fractional item progress within the current
 * stage when available. Single-stage operations: items-only progress.
 */
export function computePercent(p: ProgressCounts): number {
  const hasMeaningfulItems = p.totalItems > 0 && p.itemsProcessed > 0;
  const isMultiStage = p.totalBatches > 1;

  if (isMultiStage) {
    const stageWidth = 100 / p.totalBatches;
    const base = p.currentBatch * stageWidth;
    if (hasMeaningfulItems) {
      // Blend: stage base + fractional item progress within this stage
      const itemFraction = p.itemsProcessed / p.totalItems;
      return Math.min(100, Math.round(base + itemFraction * stageWidth));
    }
    return Math.min(100, Math.round(base));
  }

  if (p.totalItems > 0) {
    return Math.round((p.itemsProcessed / p.totalItems) * 100);
  }

  return 0;
}

/**
 * ETA estimation state machine. The baseline is the first meaningful progress
 * update; an ETA is produced once a second data point arrives. The baseline
 * resets when totalItems changes or itemsProcessed regresses (a new phase of
 * the same job).
 */
export interface EtaState {
  baseTime: number;
  baseItems: number;
  lastTotal: number;
  lastItems: number;
  /** Estimated remaining milliseconds, or null before two data points exist. */
  etaMs: number | null;
}

export function nextEtaState(
  prev: EtaState | null,
  p: ProgressCounts,
  now: number
): EtaState | null {
  // No meaningful item-level progress → state unchanged
  if (!(p.totalItems > 0 && p.itemsProcessed > 0)) return prev;

  // (Re)establish the baseline on first update, total change, or regression
  if (prev === null || p.totalItems !== prev.lastTotal || p.itemsProcessed < prev.lastItems) {
    return {
      baseTime: now,
      baseItems: p.itemsProcessed,
      lastTotal: p.totalItems,
      lastItems: p.itemsProcessed,
      etaMs: null,
    };
  }

  if (p.itemsProcessed > prev.baseItems) {
    const elapsedSinceBase = now - prev.baseTime;
    const itemsSinceBase = p.itemsProcessed - prev.baseItems;
    const avgTimePerItem = elapsedSinceBase / itemsSinceBase;
    return {
      ...prev,
      lastItems: p.itemsProcessed,
      etaMs: avgTimePerItem * (p.totalItems - p.itemsProcessed),
    };
  }

  return { ...prev, lastItems: p.itemsProcessed };
}
