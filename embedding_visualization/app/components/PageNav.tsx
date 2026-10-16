'use client';

import { PillNav } from '@/lib/ui-primitives/pill-nav';
import { IS_DEMO } from '@/lib/utils/demoMode';
import { buildPageNavItems } from '@/lib/utils/pageNavItems';

interface PageNavProps {
  /** 'glass' floats over the plot (Explore header); 'solid' sits in document headers. */
  variant?: 'glass' | 'solid';
  size?: 'sm' | 'default';
  /** Context-carrying SAE deep link (e.g. /sae?modelId=...&saeId=...). */
  saeHref?: string;
}

/** The app's top-level page navigation, rendered by each page's header. */
export function PageNav({ variant = 'glass', size = 'default', saeHref }: PageNavProps) {
  const items = buildPageNavItems({ saeHref, isDemo: IS_DEMO });
  return <PillNav items={items} variant={variant} size={size} aria-label="Page navigation" />;
}
