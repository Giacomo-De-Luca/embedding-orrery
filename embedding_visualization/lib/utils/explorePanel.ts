import { cn } from './utils';

/**
 * Shared off-canvas geometry for all Explore sidebars. The top edge follows
 * the header's height at each breakpoint (`--explore-panel-top`, published by
 * `.explore-shell` in app/globals.css).
 *
 * Phones get the full width between 1rem gutters (a capped panel left a
 * leftover sliver of map that read as a layout bug). The hidden offset must
 * clear the whole panel: a hidden panel is only transparent, so any part
 * left on-screen would still swallow taps meant for the plot.
 */
export function explorePanelClass(active: boolean): string {
  return cn(
    'pointer-events-auto absolute bottom-2 z-40 shadow-2xl transition-all duration-300 ease-in-out',
    'top-(--explore-panel-top) w-[calc(100vw-2rem)] sm:w-80',
    active ? 'left-4' : '-left-[100vw] sm:-left-[400px] opacity-0',
  );
}
