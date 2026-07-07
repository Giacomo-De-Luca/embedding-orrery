'use client';

import type { ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import { cn } from '@/lib/utils/utils';

interface CollapsibleSectionProps {
  title: string;
  /** Rendered as "(N)" after the title when set and > 0. */
  count?: number | null;
  defaultOpen?: boolean;
  children: ReactNode;
  className?: string;
}

/**
 * Card-chromed collapsible section shared by the feature detail pane
 * (Statistics, Similar Features, Activations). Content unmounts when closed
 * (Radix default), matching the previous hand-rolled behavior.
 */
export function CollapsibleSection({
  title,
  count,
  defaultOpen = false,
  children,
  className,
}: CollapsibleSectionProps) {
  return (
    <Collapsible defaultOpen={defaultOpen} className={cn('border rounded-lg bg-card', className)}>
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-4 py-2.5 text-left hover:bg-muted/50 transition-colors rounded-lg">
        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-90" />
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          {title}
          {count != null && count > 0 && <span className="ml-1">({count})</span>}
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent className="px-4 pb-4">{children}</CollapsibleContent>
    </Collapsible>
  );
}
