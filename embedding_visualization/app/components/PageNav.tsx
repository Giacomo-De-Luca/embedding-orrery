'use client';

import { Orbit, Microscope, FolderOpen } from 'lucide-react';
import { PillNav, type PillNavItem } from '@/lib/ui-primitives/pill-nav';
import { IS_DEMO, DEMO_DISABLED_MESSAGE } from '@/lib/utils/demoMode';

interface PageNavProps {
  /** 'glass' floats over the plot (Explore header); 'solid' sits in document headers. */
  variant?: 'glass' | 'solid';
  size?: 'sm' | 'default';
  /** Context-carrying SAE deep link (e.g. /sae?modelId=...&saeId=...). */
  saeHref?: string;
}

/** The app's top-level page navigation, rendered by each page's header. */
export function PageNav({ variant = 'glass', size = 'default', saeHref }: PageNavProps) {
  // Demo builds keep the Collections tab visible but inert with an
  // explanatory tooltip (the route also redirects server-side as a backstop).
  // The SAE explorer is live in the demo — its read-only browse/search paths
  // are served from the seeded tables.
  const demoDisabled = IS_DEMO
    ? { disabled: true, disabledReason: DEMO_DISABLED_MESSAGE }
    : {};
  const items: PillNavItem[] = [
    { id: 'explore', label: 'Explore', icon: Orbit, href: '/', match: 'exact' },
    { id: 'collections', label: 'Collections', icon: FolderOpen, href: '/collections', ...demoDisabled },
    { id: 'sae', label: 'SAE', icon: Microscope, href: saeHref ?? '/sae' },
  ];
  return <PillNav items={items} variant={variant} size={size} aria-label="Page navigation" />;
}
