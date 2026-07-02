'use client';

import { Orbit, Brain, FolderOpen } from 'lucide-react';
import { PillNav, type PillNavItem } from '@/lib/ui-primitives/pill-nav';

interface PageNavProps {
  /** 'glass' floats over the plot (Explore header); 'solid' sits in document headers. */
  variant?: 'glass' | 'solid';
  size?: 'sm' | 'default';
  /** Context-carrying SAE deep link (e.g. /sae?modelId=...&saeId=...). */
  saeHref?: string;
}

/** The app's top-level page navigation, rendered by each page's header. */
export function PageNav({ variant = 'glass', size = 'default', saeHref }: PageNavProps) {
  const items: PillNavItem[] = [
    { id: 'explore', label: 'Explore', icon: Orbit, href: '/', match: 'exact' },
    { id: 'sae', label: 'SAE', icon: Brain, href: saeHref ?? '/sae' },
    { id: 'collections', label: 'Collections', icon: FolderOpen, href: '/collections' },
  ];
  return <PillNav items={items} variant={variant} size={size} aria-label="Page navigation" />;
}
