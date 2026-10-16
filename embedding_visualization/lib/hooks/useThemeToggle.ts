'use client';

import { useTheme } from 'next-themes';

/** Light/dark state plus a flip, for the header's theme button and menu item. */
export function useThemeToggle(): { isDark: boolean; toggle: () => void } {
  const { resolvedTheme, setTheme } = useTheme();
  const isDark = (resolvedTheme ?? 'light') === 'dark';
  return { isDark, toggle: () => setTheme(isDark ? 'light' : 'dark') };
}
