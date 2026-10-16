import { Orbit, Microscope, FolderOpen } from 'lucide-react';
import type { PillNavItem } from '../ui-primitives/pill-nav';
import { DEMO_DISABLED_MESSAGE } from './demoMode';

interface PageNavOptions {
  /** Context-carrying SAE deep link (e.g. /sae?modelId=...&saeId=...). */
  saeHref?: string;
  isDemo: boolean;
}

/**
 * The app's top-level pages, shared by the pill bar (`PageNav`) and the
 * Explore header's compact menu (`HeaderMenu`).
 *
 * Demo builds keep the Collections page visible but inert with an explanatory
 * reason (the route also redirects server-side as a backstop). The SAE
 * explorer is live in the demo: its read-only browse/search paths are served
 * from the seeded tables.
 */
export function buildPageNavItems({ saeHref, isDemo }: PageNavOptions): PillNavItem[] {
  const demoDisabled = isDemo
    ? { disabled: true, disabledReason: DEMO_DISABLED_MESSAGE }
    : {};
  return [
    { id: 'explore', label: 'Explore', icon: Orbit, href: '/', match: 'exact' },
    { id: 'collections', label: 'Collections', icon: FolderOpen, href: '/collections', ...demoDisabled },
    { id: 'sae', label: 'SAE', icon: Microscope, href: saeHref ?? '/sae' },
  ];
}
