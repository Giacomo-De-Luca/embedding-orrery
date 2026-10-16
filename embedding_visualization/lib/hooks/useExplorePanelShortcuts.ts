'use client';
import { useEffect, type Dispatch, type SetStateAction } from 'react';

export type ExplorePanel = 'controls' | 'search' | 'analytics' | null;

/** Explore panel shortcuts: ⌘/Ctrl+B controls, +K search, +J analytics (press again to close). */
export function useExplorePanelShortcuts(setPanel: Dispatch<SetStateAction<ExplorePanel>>) {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!event.metaKey && !event.ctrlKey) return;
      const panel = ({ b: 'controls', k: 'search', j: 'analytics' } as const)[event.key.toLowerCase() as 'b' | 'k' | 'j'];
      if (!panel) return;
      event.preventDefault();
      setPanel(current => current === panel ? null : panel);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [setPanel]);
}
