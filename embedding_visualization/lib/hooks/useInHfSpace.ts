'use client';

import { useSyncExternalStore } from 'react';
import { IS_DEMO } from '../utils/demoMode';
import { isEmbedded } from '../utils/hfSpaceUrlSync';

// Being framed never changes during a page's life: nothing to subscribe to.
const subscribe = () => () => {};

/**
 * True when this demo build runs inside the Hugging Face Space page's iframe,
 * whose own chrome floats over the app (see `.explore-shell` in globals.css).
 * False on the server and during hydration, so both renders agree; React
 * re-renders with the client value straight after.
 */
export function useInHfSpace(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => IS_DEMO && isEmbedded(window),
    () => false,
  );
}
