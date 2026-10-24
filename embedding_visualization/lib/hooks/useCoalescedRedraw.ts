'use client';

import { useCallback, useState, type RefObject } from 'react';
import { CoalescedRedraw } from '../utils/coalescedRedraw';

/**
 * A stable function requesting one run of `renderRef.current` per burst of
 * requests (`CoalescedRedraw`: a microtask after the synchronous work that asked,
 * so after this commit's effects and before the browser paints). The run is
 * skipped while `isSuspended()` holds. `renderRef` holds the latest render
 * (`useLatestRef`), so the run never draws a stale closure.
 */
export function useCoalescedRedraw(
  renderRef: RefObject<(() => void) | null>,
  isSuspended?: () => boolean,
): () => void {
  const [redraw] = useState(() => new CoalescedRedraw());
  const run = useCallback(() => {
    if (!isSuspended?.()) renderRef.current?.();
  }, [renderRef, isSuspended]);
  return useCallback(() => redraw.request(run), [redraw, run]);
}
