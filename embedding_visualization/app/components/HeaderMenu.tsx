'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Check, Github, Menu, Moon, Sun } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/lib/ui-primitives/dropdown-menu';
import { isItemActive } from '@/lib/ui-primitives/pill-nav';
import { useThemeToggle } from '@/lib/hooks/useThemeToggle';
import { IS_DEMO } from '@/lib/utils/demoMode';
import { buildPageNavItems } from '@/lib/utils/pageNavItems';
import { REPO_URL } from '@/lib/utils/siteMetadata';

interface HeaderMenuProps {
  /** Context-carrying SAE deep link, as for `PageNav`. */
  saeHref?: string;
  className?: string;
}

/**
 * Compact stand-in for the Explore header's page nav, theme toggle and GitHub
 * link, for widths where those don't fit beside the search and collection
 * selector (AppHeader swaps the two at `xl`).
 */
export function HeaderMenu({ saeHref, className }: HeaderMenuProps) {
  const pathname = usePathname();
  const { isDark, toggle } = useThemeToggle();
  const pages = buildPageNavItems({ saeHref, isDemo: IS_DEMO });

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="circularghost"
          size="icon"
          className={className}
          aria-label="Menu: pages, theme and source code"
          title="Menu"
        >
          <Menu className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64">
        <DropdownMenuLabel className="text-xs font-normal text-muted-foreground">Pages</DropdownMenuLabel>
        {pages.map((item) => {
          const Icon = item.icon;
          // Touch has no hover tooltip, so the demo's reason is spelled out.
          // aria-disabled rather than `disabled`: the item stays focusable so
          // keyboard and screen-reader users reach the reason (as PillNav's
          // disabled pill does), and only the label is dimmed, not the reason.
          if (item.disabled) {
            return (
              <DropdownMenuItem
                key={item.id}
                aria-disabled="true"
                onSelect={(e) => e.preventDefault()}
                className="cursor-not-allowed items-start"
              >
                <Icon className="mt-0.5 opacity-50" />
                <span className="flex flex-col gap-0.5">
                  <span className="text-muted-foreground">{item.label}</span>
                  {item.disabledReason && (
                    <span className="text-xs text-muted-foreground">{item.disabledReason}</span>
                  )}
                </span>
              </DropdownMenuItem>
            );
          }
          const active = isItemActive(item, pathname);
          return (
            <DropdownMenuItem key={item.id} asChild>
              <Link href={item.href} aria-current={active ? 'page' : undefined}>
                <Icon />
                <span className="flex-1">{item.label}</span>
                {active && <Check />}
              </Link>
            </DropdownMenuItem>
          );
        })}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={toggle}>
          {isDark ? <Sun /> : <Moon />}
          {isDark ? 'Light mode' : 'Dark mode'}
        </DropdownMenuItem>
        {/* New tab for the same reason as the header's GitHub button: inside
            the HF Space iframe, GitHub's X-Frame-Options would blank the frame. */}
        <DropdownMenuItem asChild>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
            <Github />
            Source code on GitHub
          </a>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
