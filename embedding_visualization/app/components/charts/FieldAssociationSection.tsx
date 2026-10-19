'use client';

import { useMemo, useState } from 'react';
import { ChevronRight, Maximize2 } from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import { Button } from '@/lib/ui-primitives/button';
import { ScrollArea } from '@/lib/ui-primitives/scroll-area';
import { useFieldAssociation } from '@/lib/hooks/useFieldAssociation';
import { useCategoryData } from '@/lib/hooks/useCategoryData';
import { useCategoryColorMap } from '@/lib/hooks/useCategoryColorMap';
import { useTopicSimilarityOrder } from '@/lib/hooks/useTopicSimilarityOrder';
import { AssociationView } from '@/lib/utils/associationView';
import { AssociationFields, AssociationStatistics, type BinSettings } from '@/lib/utils/fieldAxis';
import { NumericBins } from '@/lib/utils/numericBins';
import type { ColorFieldOption } from '@/lib/utils/fieldAnalysis';
import { getCategoryDisplayName } from '@/lib/utils/categoryColors';
import { cn } from '@/lib/utils/utils';
import type { MetadataPoint } from '@/lib/types/types';
import { AssociationFieldSelect } from './AssociationFieldSelect';
import { FieldAssociationDialog } from './FieldAssociationDialog';

const FALLBACK_COLOR = '#7f7f7f';
const DEFAULT_BINS: BinSettings = { count: NumericBins.DEFAULT_COUNT, method: 'quantile', labels: 'range' };

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
 * fields are related (Cramér's V, η or Spearman ρ by field kind), how many
 * Across values each Group value spreads over, and a button opening the
 * heatmap dialog (cross-tab + correlation matrix). The field picks and bin
 * settings live here, shared with the dialog.
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
  const [bins, setBins] = useState<BinSettings>(DEFAULT_BINS);
  // Numeric fields the user switched between bins and individual values (see AssociationFields.showsValues).
  const [valueOverrides, setValueOverrides] = useState<Record<string, boolean>>({});
  // Fetch similarity orders from the first opening on, so closing the dialog never re-sorts it mid-animation.
  const [similarityWanted, setSimilarityWanted] = useState(false);

  const options = useMemo(() => AssociationFields.options(colorFieldOptions), [colorFieldOptions]);
  const numericOptions = useMemo(() => AssociationFields.numericOptions(colorFieldOptions), [colorFieldOptions]);
  // Numeric fields shown as their values act as categorical fields from here on.
  const effectiveOptions = useMemo(
    () => AssociationFields.effective(options, valueOverrides),
    [options, valueOverrides],
  );
  const { group, across } = AssociationFields.resolve(picks.group, picks.across, effectiveOptions, colorByField);

  const active = open || dialogOpen;
  const association = useFieldAssociation(points, active ? group : null, active ? across : null, mutedIndices, bins);
  const table = association?.table ?? null;

  // Colours come from the field's full value list, so each value keeps the colour it has in the plot and legend.
  const groupIsCategorical = group?.kind === 'categorical';
  const { categoryValues: groupValues } = useCategoryData(points, active && groupIsCategorical ? group.field : null);
  const colorMap = useCategoryColorMap(groupIsCategorical ? group.field : null, groupValues, categoricalPalette);

  // Similarity orders for topic fields, fetched once the heatmap has been opened.
  const groupSemantic = useTopicSimilarityOrder(collectionName, group?.field ?? null, similarityWanted);
  const acrossSemantic = useTopicSimilarityOrder(collectionName, across?.field ?? null, similarityWanted);
  const semanticOrders = useMemo(() => {
    const orders: Record<string, string[] | null> = {};
    if (group) orders[group.field] = groupSemantic;
    if (across) orders[across.field] = acrossSemantic;
    return orders;
  }, [group, across, groupSemantic, acrossSemantic]);

  const groupAxis = association?.group ?? null;
  const spread = useMemo(
    () => (table && groupAxis ? AssociationView.spreadRows(table, key => groupAxis.labelOf(key, bins.labels)) : []),
    [table, groupAxis, bins.labels],
  );
  // Same column set as the effective count (Missing and Other included), so "N of M" never exceeds M.
  const acrossCount = table ? table.cols.length : 0;
  const filtered = !!mutedIndices && mutedIndices.size > 0;

  const groupName = getCategoryDisplayName(group?.field ?? null);
  const acrossName = getCategoryDisplayName(across?.field ?? null);
  const statistic = association?.statistic ?? null;
  const summary = statistic ? AssociationStatistics.summary(statistic) : null;

  // Picking the field already on the other side swaps the two.
  const pickGroup = (field: string) =>
    setPicks({ group: field, across: field === across?.field ? group?.field ?? null : across?.field ?? null });
  const pickAcross = (field: string) =>
    setPicks({ group: field === group?.field ? across?.field ?? null : group?.field ?? null, across: field });

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
          <AssociationFieldSelect label="Group" value={group?.field ?? null} options={options} onChange={pickGroup} />
          <AssociationFieldSelect label="Across" value={across?.field ?? null} options={options} onChange={pickAcross} />
        </div>

        {association && statistic && (
          <>
            <p className="text-xs text-muted-foreground" title={AssociationStatistics.explanation(statistic.kind)}>
              {summary === null
                ? 'Not enough values for a statistic.'
                : <>{summary}{filtered ? ' (visible points)' : ''}</>}
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
          </>
        )}

        <Button
          variant="outline"
          size="sm"
          className="h-7 w-full text-xs"
          onClick={() => {
            setSimilarityWanted(true);
            setDialogOpen(true);
          }}
        >
          <Maximize2 /> Open heatmap
        </Button>

        <FieldAssociationDialog
          open={dialogOpen}
          onOpenChange={setDialogOpen}
          association={association}
          options={options}
          numericOptions={numericOptions}
          onPickGroup={pickGroup}
          onPickAcross={pickAcross}
          bins={bins}
          onBinsChange={setBins}
          valueOverrides={valueOverrides}
          onValuesChange={(field, asValues) => setValueOverrides(current => ({ ...current, [field]: asValues }))}
          semanticOrders={semanticOrders}
          points={points}
          mutedIndices={mutedIndices}
          filtered={filtered}
        />
      </CollapsibleContent>
    </Collapsible>
  );
}
