'use client';

import { useLayoutEffect, useRef, type RefObject } from 'react';

/**
 * A ref holding the latest `value`, written in a layout effect (never during
 * render), for identity-stable callbacks that must read current props or state
 * when they run outside React's commit: a queued `CoalescedRedraw` request, a
 * Plotly config function, a handler kept stable so it is never rebound.
 */
export function useLatestRef<T>(value: T): RefObject<T> {
  const ref = useRef(value);
  useLayoutEffect(() => {
    ref.current = value;
  });
  return ref;
}
