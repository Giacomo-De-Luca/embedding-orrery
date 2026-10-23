'use client';

import { useSyncExternalStore } from 'react';
import { FloatRenderTargets } from '../utils/floatRenderTargets';

// A browser's GPU capabilities never change during a page's life: nothing to subscribe to.
const subscribe = () => () => {};

/**
 * True when this browser can render into float textures, which both density
 * renderers need (3D density haze, 2D density overlay). False on the server
 * and during hydration, so both renders agree; React re-renders with the
 * probed value straight after.
 */
export function useFloatRenderTargets(): boolean {
  return useSyncExternalStore(subscribe, FloatRenderTargets.available, () => false);
}
