'use client';

import { useEffect, RefObject } from 'react';
import type { TouchPhase, TouchPoint } from '../utils/pinchGesture';
import { PinchZoomSession, type PinchZoomTarget } from '../utils/plotPinchZoom';

/**
 * Two-finger pinch zoom on a plot container (Plotly has none on touch screens).
 *
 * Listens in the capture phase so a pinch reaches `target` before Plotly sees it;
 * one-finger touches pass through untouched (2D pan, 3D rotation). The gesture
 * logic lives in `PinchZoomSession`; this hook only adapts DOM touch events.
 * `target` must be stable.
 */
export function usePinchZoom(containerRef: RefObject<HTMLElement | null>, target: PinchZoomTarget) {
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const session = new PinchZoomSession(target, {
      request: callback => requestAnimationFrame(callback),
      cancel: id => cancelAnimationFrame(id),
    });

    // Fingers on this plot only: a finger resting on a panel is not half of a pinch.
    const touchesOnPlot = (e: TouchEvent): TouchPoint[] =>
      Array.from(e.touches)
        .filter(touch => touch.target instanceof Node && container.contains(touch.target))
        .map(touch => ({ id: touch.identifier, x: touch.clientX, y: touch.clientY }));

    const listener = (phase: TouchPhase) => (e: TouchEvent) => {
      if (session.handle(phase, touchesOnPlot(e))) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    const onStart = listener('start');
    const onMove = listener('move');
    const onEnd = listener('end');
    const options = { capture: true, passive: false };
    container.addEventListener('touchstart', onStart, options);
    container.addEventListener('touchmove', onMove, options);
    container.addEventListener('touchend', onEnd, options);
    container.addEventListener('touchcancel', onEnd, options);
    return () => {
      container.removeEventListener('touchstart', onStart, { capture: true });
      container.removeEventListener('touchmove', onMove, { capture: true });
      container.removeEventListener('touchend', onEnd, { capture: true });
      container.removeEventListener('touchcancel', onEnd, { capture: true });
      session.dispose();
    };
  }, [containerRef, target]);
}
