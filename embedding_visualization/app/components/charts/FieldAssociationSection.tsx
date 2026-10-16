'use client';

import { useMemo, useState } from 'react';
import { ChevronRight, Maximize2 } from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { Button } from '@/lib/ui-primitives/button';
import { ScrollArea } from '@/lib/ui-primitives/scroll-area';
import { useFieldAssociation } from '@/lib/hooks/useFieldAssociation';
import { useCategoryData } from '@/lib/hooks/useCategoryData';
import { useCategoryColorMap } from '@/lib/hooks/useCategoryColorMap';
import { useTopicSimilarityOrder } from '@/lib/hooks/useTopicSimilarityOrder';
import { AssociationFields, AssociationView } from '@/lib/utils/associationView';
import { ContingencyTable } from '@/lib/utils/contingencyTable';
import { CategoricalFields, type ColorFieldOption } from '@/lib/utils/fieldAnalysis';
import { getCategoryDisplayName } from '@/lib/utils/categoryColors';
import { cn } from '@/lib/utils/utils';
import type { MetadataPoint } from '@/lib/types/types';
import { FieldAssociationDialog } from './FieldAssociationDialog';

const FALLBACK_COLOR = '#7f7f7f';

interface FieldAssociationSectionProps {
  points: MetadataPoint[];
  colorByField: string | null | undefined;
  colorFieldOptions?: ColorFieldOption[];
  /** Points hidden by the text-search and temporal filters. */
  mutedIndices?: Set<number> | null;
  categoricalPalette?: string;
  /** Collection whose topic-similarity orders back the heatmap's Similarity sort. */
  collectionName?: string | null;
}

function formatEffective(value: number): string {
  return value < 10 ? value.toFixed(1) : String(Math.round(value));
}

/**
 * "Field associations" block of the Analytics sidebar: how strongly two
 * categorical fields are related (Cramér's V), how many Across values each
 * Group value spreads over, and a button opening the full heatmap.
 */
export function FieldAssociationSection({
  points,
  colorByField,
  colorFieldOptions,
  mutedIndices,
  categoricalPalette,
  collectionName,
}: FieldAssociationSectionProps) {
  const [open, setOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [picks, setPicks] = useState<{ group: string | null; across: string | null }>({ group: null, across: null });

  const options = useMemo(() => CategoricalFields.options(colorFieldOptions), [colorFieldOptions]);
  const fieldNames = useMemo(() => options.map(o => o.field), [options]);
  const { group, across } = AssociationFields.resolve(picks.group, picks.across, fieldNames, colorByField);

  const active = open || dialogOpen;
  const table = useFieldAssociation(points, active ? group : null, active ? across : null, mutedIndices);

  // Colours come from the field's full value list, so each value keeps the colour it has in the plot and legend.
  const { categoryValues: groupValues } = useCategoryData(points, active ? group : null);
  const colorMap = useCategoryColorMap(group, groupValues, categoricalPalette);

  // Similarity orders for topic fields, fetched only once the heatmap opens.
  const groupSemantic = useTopicSimilarityOrder(collectionName, group, dialogOpen);
  const acrossSemantic = useTopicSimilarityOrder(collectionName, across, dialogOpen);
  const semanticOrders = useMemo(() => {
    const orders: Record<string, string[] | null> = {};
    if (group) orders[group] = groupSemantic;
    if (across) orders[across] = acrossSemantic;
    return orders;
  }, [group, across, groupSemantic, acrossSemantic]);

  const spread = useMemo(() => (table ? AssociationView.spreadRows(table) : []), [table]);
  const acrossCount = table ? table.cols.filter(key => !ContingencyTable.isPinned(key)).length : 0;
  const cramersV = table?.cramersV() ?? null;
  const strength = AssociationView.strength(cramersV);
  const filtered = !!mutedIndices && mutedIndices.size > 0;

  if (options.length < 2) return null;

  const groupName = getCategoryDisplayName(group);
  const acrossName = getCategoryDisplayName(across);

  // Picking the field already on the other side swaps the two.
  const pickGroup = (field: string) =>
    setPicks({ group: field, across: field === across ? group : across });
  const pickAcross = (field: string) =>
    setPicks({ group: field === group ? across : group, across: field });

  const fieldSelect = (label: string, value: string | null, onChange: (field: string) => void) => (
    <label className="flex min-w-0 flex-1 flex-col gap-1">
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</span>
      <Select value={value ?? ''} onValueChange={onChange}>
        <SelectTrigger size="sm" className="h-7 w-full text-xs">
          <SelectValue placeholder="Select field..." />
        </SelectTrigger>
        <SelectContent>
          {options.map(opt => (
            <SelectItem key={opt.field} value={opt.field}>
              <span className="text-xs">{opt.displayName} ({opt.uniqueCount})</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  );

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className="flex w-full items-center gap-1.5 text-left text-sm font-semibold"
          aria-label="Toggle field associations"
        >
          <ChevronRight className={cn('size-4 shrink-0 transition-transform', open && 'rotate-90')} />
          Field associations
        </button>
      </CollapsibleTrigger>

      <CollapsibleContent className="space-y-3 pt-3">
        <div className="flex gap-2">
          {fieldSelect('Group', group, pickGroup)}
          {fieldSelect('Across', across, pickAcross)}
        </div>

        {table && (
          <>
            <p
              className="text-xs text-muted-foreground"
              title="Cramér's V: 0 when the fields are unrelated, 1 when one field's value determines the other's."
            >
              {cramersV === null
                ? 'Needs at least two values on each side.'
                : <>Association <span className="font-medium text-foreground tabular-nums">{cramersV.toFixed(2)}</span> · {strength}{filtered ? ' (visible points)' : ''}</>}
            </p>

            <div className="space-y-1">
              <p className="text-[11px] text-muted-foreground">
                {acrossName} values each {groupName.toLowerCase()} spreads over (1 = all in one)
              </p>
              <ScrollArea className="[&>[data-radix-scroll-area-viewport]]:max-h-56 [&>[data-radix-scroll-area-viewport]>div]:block!">
                <ul className="space-y-0.5 pr-3">
                  {spread.map(row => {
                    const color = colorMap[row.key] ?? FALLBACK_COLOR;
                    const fraction = acrossCount > 0 ? Math.min(1, row.effective / acrossCount) : 0;
                    const effective = formatEffective(row.effective);
                    return (
                      <li
                        key={row.key}
                        className="relative flex items-center overflow-hidden rounded-sm px-1.5 py-1"
                        title={`${row.label}: ${row.count.toLocaleString()} points spread over about ${effective} of ${acrossCount} ${acrossName} values`}
                      >
                        <span
                          aria-hidden
                          className="absolute inset-y-0.5 left-0 rounded-r-sm"
                          style={{ width: `max(${(fraction * 100).toFixed(2)}%, 0.5rem)`, backgroundColor: color, opacity: 0.4 }}
                        />
                        <span className="relative z-10 min-w-0 flex-1 truncate text-xs">{row.label}</span>
                        <span className="relative z-10 shrink-0 pl-2 text-[11px] tabular-nums text-muted-foreground">
                          {effective} <span className="text-muted-foreground/60">of {acrossCount}</span>
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </ScrollArea>
            </div>

            <Button variant="outline" size="sm" className="h-7 w-full text-xs" onClick={() => setDialogOpen(true)}>
              <Maximize2 /> Open heatmap
            </Button>

            <FieldAssociationDialog
              open={dialogOpen}
              onOpenChange={setDialogOpen}
              table={table}
              semanticOrders={semanticOrders}
              filtered={filtered}
            />
          </>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
