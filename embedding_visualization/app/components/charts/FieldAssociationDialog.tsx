'use client';

import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/lib/ui-primitives/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/lib/ui-primitives/tabs';
import type { FieldAssociation } from '@/lib/hooks/useFieldAssociation';
import type { AssociationFieldOption, BinSettings } from '@/lib/utils/fieldAxis';
import type { MetadataPoint } from '@/lib/types/types';
import { CrossTabPanel } from './CrossTabPanel';
import { CorrelationPanel } from './CorrelationPanel';

type AssociationTab = 'crosstab' | 'correlations';

interface FieldAssociationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The Group × Across cross-tab; null until two different fields are picked. */
  association: FieldAssociation | null;
  /** Fields the cross-tab can use (categorical, then binned numeric). */
  options: readonly AssociationFieldOption[];
  /** Numeric fields the correlation matrix can use. */
  numericOptions: readonly AssociationFieldOption[];
  onPickGroup: (field: string) => void;
  onPickAcross: (field: string) => void;
  bins: BinSettings;
  onBinsChange: (bins: BinSettings) => void;
  /** Per numeric field: shown as individual values (true) or binned (false); absent = default. */
  valueOverrides: Readonly<Record<string, boolean>>;
  onValuesChange: (field: string, asValues: boolean) => void;
  /** Similarity order of a topic field's values, keyed by field. */
  semanticOrders?: Readonly<Record<string, readonly string[] | null | undefined>>;
  points: MetadataPoint[];
  /** Points hidden by the text-search and temporal filters. */
  mutedIndices?: Set<number> | null;
  filtered: boolean;
}

/** "Open heatmap" dialog: a Cross-tab of two fields and a Correlations matrix of numeric fields. */
export function FieldAssociationDialog({
  open,
  onOpenChange,
  association,
  options,
  numericOptions,
  onPickGroup,
  onPickAcross,
  bins,
  onBinsChange,
  valueOverrides,
  onValuesChange,
  semanticOrders,
  points,
  mutedIndices,
  filtered,
}: FieldAssociationDialogProps) {
  const [tab, setTab] = useState<AssociationTab>('crosstab');
  const canCorrelate = numericOptions.length >= 2;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[min(90vh,56rem)] w-[min(75rem,calc(100vw-2rem))] max-w-none flex-col gap-3 sm:max-w-none">
        <Tabs
          value={tab}
          onValueChange={(value) => setTab(value as AssociationTab)}
          className="min-h-0 flex-1 gap-3"
        >
          <DialogHeader className="flex-row flex-wrap items-center justify-between gap-3 pr-8">
            <DialogTitle>Field associations</DialogTitle>
            <DialogDescription className="sr-only">
              Cross-tabulation of two fields, or correlations between numeric fields.
            </DialogDescription>
            <TabsList>
              <TabsTrigger value="crosstab">Cross-tab</TabsTrigger>
              <TabsTrigger
                value="correlations"
                disabled={!canCorrelate}
                title={canCorrelate ? undefined : 'Needs at least two numeric fields.'}
              >
                Correlations
              </TabsTrigger>
            </TabsList>
          </DialogHeader>

          {/* Both tabs stay mounted (hidden when inactive) so switching keeps their settings. */}
          <TabsContent value="crosstab" forceMount className="flex min-h-0 flex-col gap-3 data-[state=inactive]:hidden">
            {association ? (
              <CrossTabPanel
                association={association}
                options={options}
                onPickGroup={onPickGroup}
                onPickAcross={onPickAcross}
                bins={bins}
                onBinsChange={onBinsChange}
                valueOverrides={valueOverrides}
                onValuesChange={onValuesChange}
                semanticOrders={semanticOrders}
                filtered={filtered}
              />
            ) : (
              <p className="text-sm text-muted-foreground">Pick two different fields.</p>
            )}
          </TabsContent>

          <TabsContent value="correlations" forceMount className="flex min-h-0 flex-col gap-3 data-[state=inactive]:hidden">
            <CorrelationPanel
              points={points}
              options={numericOptions}
              mutedIndices={mutedIndices}
              filtered={filtered}
              active={tab === 'correlations'}
            />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
