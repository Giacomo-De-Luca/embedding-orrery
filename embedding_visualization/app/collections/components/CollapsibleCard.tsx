'use client';

import type { ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';
import { Badge } from '@/lib/ui-primitives/badge';
import { Button } from '@/lib/ui-primitives/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/lib/ui-primitives/collapsible';
import { Label } from '@/lib/ui-primitives/label';

interface CollapsibleCardProps {
  title: string;
  /** After the title, also while collapsed (a status, a token budget). */
  badge?: ReactNode;
  description?: ReactNode;
  /** Under the description, also while collapsed (a load error). */
  headerExtra?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}

/**
 * A card whose header (chevron + title) expands its content: the optional
 * sections of the embed forms (Media Settings, Topic Extraction, SAE
 * Activations). Collapsed content unmounts; its state lives in the form.
 */
export function CollapsibleCard({
  title,
  badge,
  description,
  headerExtra,
  defaultOpen = false,
  children,
}: CollapsibleCardProps) {
  return (
    <Collapsible defaultOpen={defaultOpen}>
      <Card>
        <CardHeader className="pb-3">
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="group p-0 h-auto hover:bg-transparent justify-start">
              <div className="flex flex-wrap items-center gap-2">
                <ChevronRight className="h-4 w-4 transition-transform group-data-[state=open]:rotate-90" />
                <CardTitle>{title}</CardTitle>
                {badge}
              </div>
            </Button>
          </CollapsibleTrigger>
          {description && <CardDescription>{description}</CardDescription>}
          {headerExtra}
        </CardHeader>
        <CollapsibleContent>
          <CardContent className="space-y-4 pt-0">{children}</CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}

interface OptionalStepCardProps {
  title: string;
  description: ReactNode;
  /** The checkbox that turns the step on. */
  checkboxId: string;
  checkboxLabel: string;
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  /** The step's settings, shown while it is on. */
  children?: ReactNode;
}

/**
 * An optional step of the embed flow (Topic Extraction, SAE Activations):
 * collapsed while off, expanded with an "On" badge while on; its checkbox
 * turns the step on and reveals the settings.
 */
export function OptionalStepCard({
  title,
  description,
  checkboxId,
  checkboxLabel,
  enabled,
  onEnabledChange,
  children,
}: OptionalStepCardProps) {
  return (
    <CollapsibleCard
      title={title}
      badge={enabled ? <Badge variant="secondary" className="text-xs">On</Badge> : null}
      description={description}
      defaultOpen={enabled}
    >
      <div className="flex items-center gap-2">
        <Checkbox
          id={checkboxId}
          checked={enabled}
          onCheckedChange={(checked) => onEnabledChange(checked === true)}
        />
        <Label htmlFor={checkboxId} className="cursor-pointer">
          {checkboxLabel}
        </Label>
      </div>
      {enabled && children}
    </CollapsibleCard>
  );
}
