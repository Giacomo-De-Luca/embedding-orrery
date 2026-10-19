'use client';

import { useMemo, useRef, useState } from 'react';
import { useTheme } from 'next-themes';
import { ListChecks } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Popover, PopoverContent, PopoverTrigger } from '@/lib/ui-primitives/popover';
import { ScrollArea } from '@/lib/ui-primitives/scroll-area';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import { useCorrelationMatrix } from '@/lib/hooks/useCorrelationMatrix';
import { useHeatmapExport } from '@/lib/hooks/useHeatmapExport';
import { CorrelationMatrixView } from '@/lib/utils/correlationView';
import type { CorrelationMethod } from '@/lib/utils/fieldCorrelation';
import type { AssociationFieldOption } from '@/lib/utils/fieldAxis';
import type { MetadataPoint } from '@/lib/types/types';
import { CorrelationCellTooltip } from './HeatmapTooltips';
import { HeatmapActions } from './HeatmapActions';
import { HeatmapGrid } from './HeatmapGrid';
import { HeatmapLegend } from './HeatmapLegend';

/** Fields selected when the tab first opens. */
const DEFAULT_FIELD_COUNT = 12;

const METHOD_LABELS: Record<CorrelationMethod, string> = {
  spearman: 'Spearman ρ',
  pearson: 'Pearson r',
};

interface CorrelationPanelProps {
  points: MetadataPoint[];
  /** Numeric fields that can be correlated. */
  options: readonly AssociationFieldOption[];
  mutedIndices?: Set<number> | null;
  filtered: boolean;
  /** Compute only while the tab is shown. */
  active: boolean;
}

/**
 * Correlations tab of the association dialog: a fields × fields matrix of
 * Spearman ρ or Pearson r between the chosen numeric fields, each pair over
 * the visible points where both fields have a value.
 */
export function CorrelationPanel({ points, options, mutedIndices, filtered, active }: CorrelationPanelProps) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';
  const [picked, setPicked] = useState<readonly string[] | null>(null);
  const [method, setMethod] = useState<CorrelationMethod>('spearman');
  const [showValues, setShowValues] = useState(true);
  const { exporting, exportPng } = useHeatmapExport();
  const headerRef = useRef<HTMLDivElement>(null);
  const legendRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  // Picks keep the options' order; stale picks (another collection) drop out.
  const fields = useMemo(() => {
    const available = options.map(o => o.field);
    if (picked === null) return available.slice(0, DEFAULT_FIELD_COUNT);
    return available.filter(field => picked.includes(field));
  }, [options, picked]);
  const displayName = useMemo(() => {
    const names = new Map(options.map(o => [o.field, o.displayName]));
    return (field: string) => names.get(field) ?? field;
  }, [options]);

  const matrix = useCorrelationMatrix(points, fields, method, mutedIndices, active);
  const view = useMemo(
    () => (matrix ? CorrelationMatrixView.of(matrix, isDark, displayName) : null),
    [matrix, isDark, displayName],
  );
  const order = useMemo(() => fields.map((_, i) => i), [fields]);
  const visibleCount = points.length - (mutedIndices?.size ?? 0);

  const toggleField = (field: string, checked: boolean) =>
    setPicked(checked ? [...fields, field] : fields.filter(f => f !== field));

  return (
    <>
      <div ref={headerRef} className="space-y-1">
        <h3 className="text-base font-semibold">Correlations between numeric fields</h3>
        <p className="text-sm text-muted-foreground">
          {METHOD_LABELS[method]} between {fields.length} fields over {visibleCount.toLocaleString()}{' '}
          {filtered ? 'visible ' : ''}points; each pair uses the points where both fields have a value.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="h-7 text-xs" disabled={exporting}>
              <ListChecks /> Fields ({fields.length} of {options.length})
            </Button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-72 p-2">
            <div className="flex justify-between gap-2 pb-2">
              <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={() => setPicked(options.map(o => o.field))}>
                Select all
              </Button>
              <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={() => setPicked([])}>
                Clear
              </Button>
            </div>
            <ScrollArea className="[&>[data-radix-scroll-area-viewport]]:max-h-64">
              <ul className="space-y-1 pr-3">
                {options.map(opt => (
                  <li key={opt.field}>
                    <label className="flex cursor-pointer items-center gap-2 rounded-sm px-1 py-0.5 text-xs hover:bg-muted/50">
                      <Checkbox
                        checked={fields.includes(opt.field)}
                        onCheckedChange={(checked) => toggleField(opt.field, checked === true)}
                      />
                      <span className="min-w-0 truncate">{opt.displayName}</span>
                    </label>
                  </li>
                ))}
              </ul>
            </ScrollArea>
          </PopoverContent>
        </Popover>
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={method}
          disabled={exporting}
          onValueChange={(value) => value && setMethod(value as CorrelationMethod)}
        >
          <ToggleGroupItem value="spearman" className="h-7 px-2 text-xs" title="Rank correlation: any consistent rise or fall">
            Spearman
          </ToggleGroupItem>
          <ToggleGroupItem value="pearson" className="h-7 px-2 text-xs" title="Linear correlation">
            Pearson
          </ToggleGroupItem>
        </ToggleGroup>
        <HeatmapActions
          showValues={showValues}
          onShowValuesChange={setShowValues}
          exporting={exporting}
          onExport={() => exportPng([headerRef.current, legendRef.current, gridRef.current], `correlations-${method}.png`)}
          exportDisabled={!view}
        />
      </div>

      {view ? (
        <>
          <HeatmapLegend
            ref={legendRef}
            title={METHOD_LABELS[method]}
            legend={view.legend()}
            emptyColor={view.emptyColor}
            emptyLabel="Not enough shared values"
          />
          <div className="min-h-0 flex-1">
            <HeatmapGrid
              key={String(showValues)}
              model={view}
              rowOrder={order}
              colOrder={order}
              showValues={showValues}
              ariaLabel={`${METHOD_LABELS[method]} between numeric fields`}
              renderTooltip={(i, j) => <CorrelationCellTooltip view={view} method={method} i={i} j={j} />}
              exportRef={gridRef}
            />
          </div>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">Choose at least two fields.</p>
      )}
    </>
  );
}
