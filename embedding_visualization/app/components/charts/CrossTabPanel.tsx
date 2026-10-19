'use client';

import { useMemo, useRef, useState } from 'react';
import { useTheme } from 'next-themes';
import { ArrowLeftRight } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import { Input } from '@/lib/ui-primitives/input';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { useHeatmapExport } from '@/lib/hooks/useHeatmapExport';
import type { FieldAssociation } from '@/lib/hooks/useFieldAssociation';
import { AssociationView, type AssociationMode } from '@/lib/utils/associationView';
import { AxisOrdering, type AxisSortMode } from '@/lib/utils/axisOrdering';
import {
  AssociationFields,
  AssociationStatistics,
  type AssociationFieldOption,
  type BinLabelStyle,
  type BinSettings,
} from '@/lib/utils/fieldAxis';
import { ContingencyTable } from '@/lib/utils/contingencyTable';
import { NumericBins, type BinMethod } from '@/lib/utils/numericBins';
import { getCategoryDisplayName } from '@/lib/utils/categoryColors';
import { AssociationFieldSelect } from './AssociationFieldSelect';
import { HeatmapActions } from './HeatmapActions';
import { AssociationCellTooltip } from './HeatmapTooltips';
import { HeatmapGrid } from './HeatmapGrid';
import { HeatmapLegend } from './HeatmapLegend';

/** View modes named by field, so they keep their meaning when the axes swap. */
type FieldMode = 'enrichment' | 'count' | 'withinGroup' | 'withinAcross';

const SORT_LABELS: Record<AxisSortMode, string> = {
  pattern: 'Pattern',
  semantic: 'Similarity',
  size: 'Size',
  name: 'Name',
  value: 'Value',
};

const CATEGORICAL_SORTS: readonly AxisSortMode[] = ['pattern', 'semantic', 'size', 'name'];
/** Sorts of a categorical axis whose values are numbers (Value is also their name order). */
const VALUE_SORTS: readonly AxisSortMode[] = ['value', 'pattern', 'size'];

interface CrossTabPanelProps {
  association: FieldAssociation;
  options: readonly AssociationFieldOption[];
  onPickGroup: (field: string) => void;
  onPickAcross: (field: string) => void;
  bins: BinSettings;
  onBinsChange: (bins: BinSettings) => void;
  /** Per numeric field: shown as individual values (true) or binned (false); absent = default. */
  valueOverrides: Readonly<Record<string, boolean>>;
  onValuesChange: (field: string, asValues: boolean) => void;
  /** Similarity order of a topic field's values, keyed by field. */
  semanticOrders?: Readonly<Record<string, readonly string[] | null | undefined>>;
  /** True while text-search or temporal filters hide some points. */
  filtered: boolean;
}

/**
 * Cross-tab tab of the association dialog: the Group × Across heatmap with its
 * field pickers, bin settings (when a field is numeric), view modes, per-axis
 * sorts, swap, values toggle, legend and PNG export.
 */
export function CrossTabPanel({
  association,
  options,
  onPickGroup,
  onPickAcross,
  bins,
  onBinsChange,
  valueOverrides,
  onValuesChange,
  semanticOrders,
  filtered,
}: CrossTabPanelProps) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';
  const [mode, setMode] = useState<FieldMode>('enrichment');
  const [swapped, setSwapped] = useState(false);
  const [showValues, setShowValues] = useState(false);
  const [sortPicks, setSortPicks] = useState<Record<string, AxisSortMode>>({});
  const { exporting, exportPng } = useHeatmapExport();
  const headerRef = useRef<HTMLDivElement>(null);
  const legendRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  const { table, statistic } = association;
  const groupField = table.rowField;
  const acrossField = table.colField;
  const groupName = getCategoryDisplayName(groupField);
  const acrossName = getCategoryDisplayName(acrossField);
  const hasNumeric = association.group.kind === 'numeric' || association.across.kind === 'numeric';

  // The field with more values goes on the scrolling vertical axis; Swap flips it.
  const groupOnRows = (table.rows.length >= table.cols.length) !== swapped;
  const displayTable = useMemo(() => (groupOnRows ? table : table.transpose()), [table, groupOnRows]);

  const axisOf = (field: string) => (field === groupField ? association.group : association.across);
  /** A numeric axis follows its bins; a topic axis may follow the backend similarity order. */
  const externalOf = (field: string) => {
    const semantic = semanticOrders?.[field];
    return axisOf(field).valueOrder ?? (semantic?.length ? semantic : null);
  };
  const sortOf = (field: string): AxisSortMode => {
    const axis = axisOf(field);
    if (axis.kind === 'numeric') return 'value';
    const pick = sortPicks[field];
    const pickValid = pick && (pick === 'value' ? !!axis.valueOrder : pick !== 'semantic' || externalOf(field));
    if (pickValid) return pick;
    // Numbers shown as categories read best in value order (e.g. centuries).
    if (axis.valueOrder) return 'value';
    return externalOf(field) ? 'semantic' : 'pattern';
  };
  const rowField = displayTable.rowField;
  const colField = displayTable.colField;
  const rowSort = sortOf(rowField);
  const colSort = sortOf(colField);
  const rowExternal = externalOf(rowField);
  const colExternal = externalOf(colField);

  const { rowOrder, colOrder } = useMemo(
    () => AxisOrdering.resolve(displayTable, {
      rowMode: rowSort,
      colMode: colSort,
      rowExternal,
      colExternal,
      labelOf: AssociationView.axisLabel,
    }),
    [displayTable, rowSort, colSort, rowExternal, colExternal],
  );

  const axisMode: AssociationMode =
    mode === 'withinGroup'
      ? (groupOnRows ? 'rowShare' : 'colShare')
      : mode === 'withinAcross'
        ? (groupOnRows ? 'colShare' : 'rowShare')
        : mode;
  // Numeric axes show their bins as value ranges or percentile bands (display only).
  const labelStyle = bins.labels;
  const view = useMemo(() => {
    const rowAxis = displayTable.rowField === groupField ? association.group : association.across;
    const colAxis = rowAxis === association.group ? association.across : association.group;
    return AssociationView.of(displayTable, axisMode, isDark, {
      row: key => rowAxis.labelOf(key, labelStyle),
      col: key => colAxis.labelOf(key, labelStyle),
    });
  }, [displayTable, axisMode, isDark, association, groupField, labelStyle]);
  /** With percentile labels, the tooltip still names a numeric bin's value range. */
  const rangeNote = (field: string, key: string) =>
    labelStyle === 'percentile' && axisOf(field).kind === 'numeric' && !ContingencyTable.isPinned(key)
      ? key
      : undefined;
  const legend = view.legend();
  const modeLabel = mode === 'withinGroup' ? `% within ${groupName}`
    : mode === 'withinAcross' ? `% within ${acrossName}`
      : mode === 'count' ? 'Count' : 'Enrichment';
  const summary = AssociationStatistics.summary(statistic);
  // Whole-number fields with few distinct values get fewer bins than asked for: say so.
  const binNotes = [association.group, association.across]
    .filter(axis => axis.bins && axis.bins.keys.length !== bins.count)
    .map(axis => `${getCategoryDisplayName(axis.field)}: ${axis.bins!.keys.length} bins`)
    .join(' · ');

  /** Bins | Values switch for a numeric field (by its original kind, before any switch). */
  const valuesToggle = (field: string) => {
    const option = options.find(o => o.field === field);
    if (option?.kind !== 'numeric') return null;
    const asValues = AssociationFields.showsValues(option, valueOverrides);
    return (
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={asValues ? 'values' : 'bins'}
        disabled={exporting}
        onValueChange={(value) => value && onValuesChange(field, value === 'values')}
        aria-label={`How to show ${getCategoryDisplayName(field)}`}
      >
        <ToggleGroupItem value="bins" className="h-7 px-2 text-xs" title="Group the values into ranges">
          Bins
        </ToggleGroupItem>
        <ToggleGroupItem
          value="values"
          className="h-7 px-2 text-xs"
          title={`Each distinct value as its own category, like a categorical field (up to ${AssociationFields.MAX_AXIS_VALUES}; the rest fold into Other)`}
        >
          Values
        </ToggleGroupItem>
      </ToggleGroup>
    );
  };

  const sortSelect = (field: string) => {
    const axis = axisOf(field);
    const numeric = axis.kind === 'numeric';
    const sorts: readonly AxisSortMode[] = numeric
      ? ['value']
      : axis.valueOrder
        ? VALUE_SORTS
        : CATEGORICAL_SORTS.filter(sort => sort !== 'semantic' || AssociationFields.supportsSimilarity(field));
    return (
      <Select
        disabled={exporting || numeric}
        value={sortOf(field)}
        onValueChange={(value) => setSortPicks(picks => ({ ...picks, [field]: value as AxisSortMode }))}
      >
        <SelectTrigger size="sm" className="h-7 text-xs" aria-label={`Sort ${getCategoryDisplayName(field)}`}>
          <span className="text-muted-foreground">Sort {getCategoryDisplayName(field)}:</span>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {sorts.map(sort => {
            // Topic fields always list Similarity; it stays disabled until their centroids exist.
            const unavailable = sort === 'semantic' && !externalOf(field);
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
  };

  return (
    <>
      <div ref={headerRef} className="space-y-1">
        <h3 className="text-base font-semibold">{groupName} × {acrossName}</h3>
        <p className="text-sm text-muted-foreground" title={AssociationStatistics.explanation(statistic.kind)}>
          {summary === null
            ? 'Not enough values for a statistic.'
            : `${summary}, over ${statistic.n.toLocaleString()} ${filtered ? 'visible ' : ''}points.`}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1">
          <AssociationFieldSelect
            variant="inline"
            label="Group"
            value={groupField}
            options={options}
            onChange={onPickGroup}
            disabled={exporting}
          />
          {valuesToggle(groupField)}
        </div>
        <div className="flex items-center gap-1">
          <AssociationFieldSelect
            variant="inline"
            label="Across"
            value={acrossField}
            options={options}
            onChange={onPickAcross}
            disabled={exporting}
          />
          {valuesToggle(acrossField)}
        </div>
        {hasNumeric && <BinControls bins={bins} onChange={onBinsChange} disabled={exporting} />}
        {hasNumeric && binNotes && <span className="text-[11px] text-muted-foreground">{binNotes}</span>}
      </div>

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
        <HeatmapActions
          showValues={showValues}
          onShowValuesChange={setShowValues}
          exporting={exporting}
          onExport={() => exportPng(
            [headerRef.current, legendRef.current, gridRef.current],
            `field-association-${groupField}-${acrossField}-${mode}.png`,
          )}
          exportDisabled={table.total === 0}
        />
      </div>

      <HeatmapLegend
        ref={legendRef}
        title={modeLabel}
        legend={legend}
        emptyColor={axisMode !== 'enrichment' ? view.emptyColor : null}
        emptyLabel="No points"
      />

      <div className="min-h-0 flex-1">
        <HeatmapGrid
          key={`${groupOnRows}-${showValues}`}
          model={view}
          rowOrder={rowOrder}
          colOrder={colOrder}
          showValues={showValues}
          ariaLabel={`${getCategoryDisplayName(rowField)} by ${getCategoryDisplayName(colField)}`}
          renderTooltip={(i, j) => (
            <AssociationCellTooltip
              view={view}
              i={i}
              j={j}
              rowNote={rangeNote(rowField, displayTable.rows[i])}
              colNote={rangeNote(colField, displayTable.cols[j])}
            />
          )}
          exportRef={gridRef}
        />
      </div>
    </>
  );
}

/**
 * Number of bins (applied on blur or Enter; invalid text restores the current
 * count), the binning method, and whether bins are labelled by range or percentile.
 */
function BinControls({
  bins,
  onChange,
  disabled,
}: {
  bins: BinSettings;
  onChange: (bins: BinSettings) => void;
  disabled: boolean;
}) {
  const [draft, setDraft] = useState(String(bins.count));
  const commit = (text: string) => {
    const count = NumericBins.parseCount(text, bins.count);
    setDraft(String(count));
    if (count !== bins.count) onChange({ ...bins, count });
  };

  return (
    <div className="flex items-center gap-2 text-xs">
      <label className="flex items-center gap-1.5">
        <span className="text-muted-foreground">Bins</span>
        <Input
          type="number"
          inputMode="numeric"
          min={NumericBins.MIN_COUNT}
          max={NumericBins.MAX_COUNT}
          value={draft}
          disabled={disabled}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={(e) => commit(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commit(e.currentTarget.value);
          }}
          className="h-7 w-16 px-2 text-xs"
          aria-label={`Number of bins (${NumericBins.MIN_COUNT} to ${NumericBins.MAX_COUNT})`}
        />
      </label>
      <Select
        value={bins.method}
        disabled={disabled}
        onValueChange={(value) => onChange({ ...bins, method: value as BinMethod })}
      >
        <SelectTrigger size="sm" className="h-7 text-xs" aria-label="Binning method">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="quantile">
            <span className="text-xs">Equal count</span>
          </SelectItem>
          <SelectItem value="width">
            <span className="text-xs">Equal width</span>
          </SelectItem>
        </SelectContent>
      </Select>
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={bins.labels}
        disabled={disabled}
        onValueChange={(value) => value && onChange({ ...bins, labels: value as BinLabelStyle })}
        aria-label="Bin labels"
      >
        <ToggleGroupItem value="range" className="h-7 px-2 text-xs" title="Label bins by their value ranges">
          Ranges
        </ToggleGroupItem>
        <ToggleGroupItem value="percentile" className="h-7 px-2 text-xs" title="Label bins by the share of points below their edges">
          Percentiles
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  );
}
