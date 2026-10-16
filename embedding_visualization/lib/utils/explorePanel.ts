import { cn } from './utils';

/** Shared off-canvas geometry for all Explore sidebars. */
export function explorePanelClass(active: boolean): string {
  return cn(
    'pointer-events-auto absolute bottom-2 z-40 shadow-2xl transition-all duration-300 ease-in-out',
    'top-34 sm:top-20 w-[min(20rem,calc(100vw-2rem))]',
    active ? 'left-4' : '-left-[400px] opacity-0',
  );
}
