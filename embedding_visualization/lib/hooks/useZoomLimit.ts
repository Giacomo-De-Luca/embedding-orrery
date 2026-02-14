'use client';

import { useEffect, RefObject } from 'react';

/**
 * Prevents Plotly scroll-zoom-out beyond a limit by intercepting wheel events
 * in the capture phase before Plotly processes them.
 *
 * Each caller provides its own `isAtZoomOutLimit` callback that reads live
 * state from Plotly internals (3D: camera eye distance, 2D: axis ranges).
 */
export function useZoomLimit(
  containerRef: RefObject<HTMLElement | null>,
  isAtZoomOutLimit: () => boolean
) {
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      // deltaY > 0 = scroll down = zoom out in Plotly
      if (e.deltaY > 0 && isAtZoomOutLimit()) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    container.addEventListener('wheel', handleWheel, { capture: true, passive: false });
    return () => container.removeEventListener('wheel', handleWheel, { capture: true } as EventListenerOptions);
  }, [containerRef, isAtZoomOutLimit]);
}
