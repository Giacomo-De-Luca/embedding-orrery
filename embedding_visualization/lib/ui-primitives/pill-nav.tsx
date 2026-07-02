'use client';

import * as React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, useReducedMotion } from 'motion/react';
import { cva, type VariantProps } from 'class-variance-authority';
import type { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils/utils';

/**
 * iOS-style pill tab bar: rounded-full container with icon+label items and a
 * capsule that marks the active item.
 *
 * The capsule slides between items (motion `layoutId`) only while an instance
 * stays mounted — e.g. controlled via `activeId`. When used as page navigation,
 * each route mounts its own instance, so across a route change the capsule
 * fades in on the new active item instead of sliding. A cross-page slide would
 * require hoisting the nav into the root layout, which the page headers don't
 * currently share.
 *
 * Reads only `usePathname()` for active detection — never `useSearchParams` —
 * so it can render outside a Suspense boundary.
 */

export interface PillNavItem {
  /** Stable key, also used to resolve the active item. */
  id: string;
  label: string;
  icon: LucideIcon;
  /** May carry a query string (caller-computed); it is ignored for matching. */
  href: string;
  /** 'prefix' (default) also matches nested paths; use 'exact' for '/'. */
  match?: 'exact' | 'prefix';
}

const pillNavVariants = cva('flex items-center gap-0.5 rounded-full p-1', {
  variants: {
    variant: {
      /** Floating over a canvas — frosted glass, matches the `circular` button variant. */
      glass: 'bg-secondary/30 backdrop-blur-md border-glass shadow-[var(--shadow-float)]',
      /** Inline in a document-style header — nothing behind it to blur. */
      solid: 'border bg-muted',
    },
  },
  defaultVariants: { variant: 'glass' },
});

const pillNavItemVariants = cva(
  'relative flex items-center rounded-full font-medium outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring',
  {
    variants: {
      size: {
        default: 'h-8 px-3 text-sm',
        sm: 'h-7 px-2.5 text-xs',
      },
      active: {
        true: 'text-foreground',
        false: 'text-muted-foreground hover:text-foreground',
      },
    },
    defaultVariants: { size: 'default', active: false },
  },
);

interface PillNavProps extends VariantProps<typeof pillNavVariants> {
  items: PillNavItem[];
  size?: 'sm' | 'default';
  /** Controlled active item — bypasses pathname matching (e.g. for non-route toggles). */
  activeId?: string;
  className?: string;
  'aria-label'?: string;
}

function isItemActive(item: PillNavItem, pathname: string): boolean {
  const path = item.href.split('?')[0];
  if ((item.match ?? 'prefix') === 'exact') return pathname === path;
  return pathname === path || pathname.startsWith(`${path}/`);
}

export function PillNav({
  items,
  variant = 'glass',
  size = 'default',
  activeId,
  className,
  'aria-label': ariaLabel = 'Page navigation',
}: PillNavProps) {
  const pathname = usePathname();
  const reduceMotion = useReducedMotion();
  // Instance-unique layoutId: two mounted PillNavs must never share a capsule.
  const layoutId = React.useId();

  const resolvedActiveId = activeId ?? items.find((item) => isItemActive(item, pathname))?.id;

  return (
    <nav aria-label={ariaLabel} className={cn(pillNavVariants({ variant }), className)}>
      {items.map((item) => {
        const active = item.id === resolvedActiveId;
        const Icon = item.icon;
        return (
          <Link
            key={item.id}
            href={item.href}
            aria-current={active ? 'page' : undefined}
            className={pillNavItemVariants({ size, active })}
          >
            {active && (
              <motion.span
                layoutId={layoutId}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className={cn(
                  'absolute inset-0 rounded-full shadow-sm',
                  variant === 'glass' ? 'bg-background/80 backdrop-blur-sm' : 'bg-background',
                )}
                transition={
                  reduceMotion
                    ? { duration: 0 }
                    : { type: 'spring', stiffness: 400, damping: 32, opacity: { duration: 0.15 } }
                }
              />
            )}
            <span className="relative z-10 flex items-center gap-1.5">
              <Icon className={size === 'sm' ? 'h-3.5 w-3.5' : 'h-4 w-4'} />
              <span className="hidden sm:inline">{item.label}</span>
            </span>
          </Link>
        );
      })}
    </nav>
  );
}
