'use client';

import { useMemo, useRef, useState } from 'react';
import { useTheme } from 'next-themes';
import { ArrowLeftRight, Download, Hash, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/lib/ui-primitives/dialog';
import { Button } from '@/lib/ui-primitives/button';
import { Toggle } from '@/lib/ui-primitives/toggle';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { AssociationView, type AssociationMode } from '@/lib/utils/associationView';
import { AxisOrdering, type AxisSortMode } from '@/lib/utils/axisOrdering';
import type { ContingencyTable } from '@/lib/utils/contingencyTable';
import { getCategoryDisplayName, isTopicLikeField } from '@/lib/utils/categoryColors';
import { HeatmapCapture } from '@/lib/utils/heatmapCapture';
import { downloadCanvasPng } from '@/lib/utils/downloadJson';
import { AssociationHeatmap } from './AssociationHeatmap';

/** View modes named by field, so they keep their meaning when the axes swap. */
type FieldMode = 'enrichment' | 'count' | 'withinGroup' | 'withinAcross';

const SORT_LABELS: Record<AxisSortMode, string> = {
  pattern: 'Pattern',
  semantic: 'Similarity',
  size: 'Size',
  name: 'Name',
};

interface FieldAssociationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Group values on rows, Across values on columns. */
  table: ContingencyTable;
  /** Similarity order of a field's values, keyed by field; fields without one offer no Similarity sort. */
  semanticOrders?: Readonly<Record<string, readonly string[] | null | undefined>>;
  /** True while text-search or temporal filters hide some points. */
  filtered: boolean;
}

export function FieldAssociationDialog({
  open,
  onOpenChange,
  table,
  semanticOrders,
  filtered,
}: FieldAssociationDialogProps) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';
  const [mode, setMode] = useState<FieldMode>('enrichment');
  const [swapped, setSwapped] = useState(false);
  const [showValues, setShowValues] = useState(false);
  const [sortPicks, setSortPicks] = useState<Record<string, AxisSortMode>>({});
  const [exporting, setExporting] = useState(false);
  const headerRef = useRef<HTMLDivElement>(null);
  const legendRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  const groupField = table.rowField;
  const acrossField = table.colField;
  const groupName = getCategoryDisplayName(groupField);
  const acrossName = getCategoryDisplayName(acrossField);

  // The field with more values goes on the scrolling vertical axis; Swap flips it.
  const groupOnRows = (table.rows.length >= table.cols.length) !== swapped;
  const displayTable = useMemo(() => (groupOnRows ? table : table.transpose()), [table, groupOnRows]);

  const semanticOf = (field: string) => semanticOrders?.[field] ?? null;
  const sortOf = (field: string): AxisSortMode => {
    const pick = sortPicks[field];
    if (pick && (pick !== 'semantic' || semanticOf(field))) return pick;
    return semanticOf(field) ? 'semantic' : 'pattern';
  };
  const rowField = displayTable.rowField;
  const colField = displayTable.colField;
  const rowSort = sortOf(rowField);
  const colSort = sortOf(colField);
  const rowSemantic = semanticOf(rowField);
  const colSemantic = semanticOf(colField);

  const { rowOrder, colOrder } = useMemo(
    () => AxisOrdering.resolve(displayTable, {
      rowMode: rowSort,
      colMode: colSort,
      rowSemantic,
      colSemantic,
      labelOf: AssociationView.axisLabel,
    }),
    [displayTable, rowSort, colSort, rowSemantic, colSemantic],
  );

  const axisMode: AssociationMode =
    mode === 'withinGroup'
      ? (groupOnRows ? 'rowShare' : 'colShare')
      : mode === 'withinAcross'
        ? (groupOnRows ? 'colShare' : 'rowShare')
        : mode;
  const view = useMemo(() => AssociationView.of(displayTable, axisMode, isDark), [displayTable, axisMode, isDark]);
  const legend = view.legend();

  const cramersV = table.cramersV();
  const strength = AssociationView.strength(cramersV);
  const modeLabel = mode === 'withinGroup' ? `% within ${groupName}`
    : mode === 'withinAcross' ? `% within ${acrossName}`
      : mode === 'count' ? 'Count' : 'Enrichment';

  const exportPng = async () => {
    if (exporting || !headerRef.current || !legendRef.current || !gridRef.current) return;
    setExporting(true);
    try {
      const background = getComputedStyle(headerRef.current.parentElement!).backgroundColor;
      const canvas = await HeatmapCapture.toCanvas(
        [headerRef.current, legendRef.current, gridRef.current], background,
      );
      const filename = `field-association-${groupField}-${acrossField}-${mode}.png`
        .replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_');
      if (!await downloadCanvasPng(canvas, filename)) throw new Error('PNG encoding failed.');
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      console.error('Heatmap PNG export failed:', error);
      toast.error('Could not export the heatmap. Please try again.');
    } finally {
      setExporting(false);
    }
  };

  const sortSelect = (field: string) => (
    <Select
      disabled={exporting}
      value={sortOf(field)}
      onValueChange={(value) => setSortPicks(picks => ({ ...picks, [field]: value as AxisSortMode }))}
    >
      <SelectTrigger size="sm" className="h-7 text-xs" aria-label={`Sort ${getCategoryDisplayName(field)}`}>
        <span className="text-muted-foreground">{getCategoryDisplayName(field)}:</span>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {(['pattern', 'semantic', 'size', 'name'] as const)
          .filter(sort => sort !== 'semantic' || semanticOf(field) || isTopicLikeField(field))
          .map(sort => {
            // Topic fields always list Similarity; it stays disabled until centroids exist.
            const unavailable = sort === 'semantic' && !semanticOf(field);
            return (
              <SelectItem
                key={sort}
                value={sort}
                disabled={unavailable}
                title={unavailable ? 'Topic centroids have not been computed for this collection.' : undefined}
              >
                <span className="text-xs">
                  {SORT_LABELS[sort]}{unavailable ? ' (unavailable)' : ''}
                </span>
              </SelectItem>
            );
          })}
      </SelectContent>
    </Select>
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[min(90vh,56rem)] w-[min(75rem,calc(100vw-2rem))] max-w-none flex-col gap-3 sm:max-w-none">
        <DialogHeader ref={headerRef}>
          <DialogTitle>{groupName} × {acrossName}</DialogTitle>
          <DialogDescription>
            {cramersV === null
              ? 'Needs at least two values on each side.'
              : `Cramér's V ${cramersV.toFixed(2)} (${strength} association) over ${table.total.toLocaleString()} ${filtered ? 'visible ' : ''}points.`}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-wrap items-center gap-2">
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={mode}
            disabled={exporting}
            onValueChange={(value) => value && setMode(value as FieldMode)}
          >
            <ToggleGroupItem value="enrichment" className="h-7 px-2 text-xs">Enrichment</ToggleGroupItem>
            <ToggleGroupItem value="count" className="h-7 px-2 text-xs">Count</ToggleGroupItem>
            <ToggleGroupItem value="withinGroup" className="h-7 px-2 text-xs">% within {groupName}</ToggleGroupItem>
            <ToggleGroupItem value="withinAcross" className="h-7 px-2 text-xs">% within {acrossName}</ToggleGroupItem>
          </ToggleGroup>
          {sortSelect(rowField)}
          {sortSelect(colField)}
          <Button variant="outline" size="sm" className="h-7 text-xs" disabled={exporting} onClick={() => setSwapped(s => !s)}>
            <ArrowLeftRight /> Swap axes
          </Button>
          <Toggle
            variant="outline"
            size="sm"
            className="h-7 text-xs"
            pressed={showValues}
            disabled={exporting}
            onPressedChange={setShowValues}
            aria-label="Show values in cells"
          >
            <Hash /> Values
          </Toggle>
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-xs"
            onClick={exportPng}
            disabled={exporting || table.total === 0}
            aria-busy={exporting}
          >
            {exporting ? <Loader2 className="animate-spin" /> : <Download />}
            {exporting ? 'Exporting…' : 'Export PNG'}
          </Button>
        </div>

        <div ref={legendRef} className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
          <span>{modeLabel}</span>
          <div className="flex items-center gap-2">
            <span>{legend.start}</span>
            <div className="relative h-2.5 w-44 rounded-full" style={{ background: legend.gradient }}>
              {legend.middle && (
                <span aria-hidden className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-foreground/40" />
              )}
            </div>
            <span>{legend.end}</span>
          </div>
          {legend.middle && <span>Centre: {legend.middle.toLowerCase()}</span>}
          {axisMode !== 'enrichment' && (
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ backgroundColor: view.emptyColor }} />
              No points
            </span>
          )}
        </div>

        <div className="min-h-0 flex-1">
          <AssociationHeatmap
            key={`${groupOnRows}-${showValues}`}
            view={view}
            rowOrder={rowOrder}
            colOrder={colOrder}
            showValues={showValues}
            exportRef={gridRef}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
