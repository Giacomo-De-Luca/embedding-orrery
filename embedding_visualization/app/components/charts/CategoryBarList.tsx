'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/lib/ui-primitives/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { Input } from '@/lib/ui-primitives/input';
import { ScrollArea } from '@/lib/ui-primitives/scroll-area';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import { buildCategoryColorMap, getCategoryLabel, getCategoryDisplayName } from '@/lib/utils/categoryColors';
import {
  buildCategoryRows,
  isCategoryFilterActive,
  resolveSortMode,
  type CategorySortMode,
} from '@/lib/utils/categoryRowData';
import { useVisualizationStore } from '@/lib/stores/useVisualizationStore';
import { cn } from '@/lib/utils/utils';
import type { ColorFieldOption } from '@/lib/utils/fieldAnalysis';

const FOLLOW_COLOR_FIELD = '__follow__';
const NAME_FILTER_THRESHOLD = 15;
const FALLBACK_COLOR = '#7f7f7f';

function formatPct(fraction: number): string {
  if (fraction <= 0) return '0%';
  const pct = Math.round(fraction * 100);
  return pct < 1 ? '<1%' : `${pct}%`;
}

/** Proportional bar width, floored to a small chip so tiny categories don't look clipped. */
function barWidth(fraction: number): string {
  if (fraction <= 0) return '0px';
  return `max(${(fraction * 100).toFixed(2)}%, 0.5rem)`;
}

interface CategoryBarListProps {
  categoryField: string | null;
  categoryValues: string[];
  categoryCounts: Record<string, number>;
  categoricalPalette?: string;
  /** Per-category counts surviving active filters (text search + temporal). */
  filteredCounts?: Record<string, number> | null;
  colorFieldOptions?: ColorFieldOption[];
  analysisField?: string | null;
  onAnalysisFieldChange?: (field: string | null) => void;
  mutedCategories?: string[];
  onCategoryToggle?: (category: string, shiftKey: boolean) => void;
}

export function CategoryBarList({
  categoryField,
  categoryValues,
  categoryCounts,
  categoricalPalette,
  filteredCounts,
  colorFieldOptions,
  analysisField,
  onAnalysisFieldChange,
  mutedCategories = [],
  onCategoryToggle,
}: CategoryBarListProps) {
  const [nameFilter, setNameFilter] = useState('');
  const [sortMode, setSortMode] = useState<CategorySortMode>('count');

  // Reset local controls when the analyzed field changes
  useEffect(() => {
    setNameFilter('');
    setSortMode('count');
  }, [categoryField]);

  const colorOverrides = useVisualizationStore(
    (s) => categoryField ? s.categoryColorOverrides[categoryField] : undefined
  );
  const colorMap = useMemo(
    () => buildCategoryColorMap(categoryField, categoryValues, categoricalPalette, colorOverrides),
    [categoryField, categoryValues, categoricalPalette, colorOverrides]
  );

  const filterActive = isCategoryFilterActive(filteredCounts);
  const effectiveSort = resolveSortMode(sortMode, filterActive);

  const getLabel = useCallback(
    (value: string) => getCategoryLabel(categoryField, value),
    [categoryField]
  );

  const { rows, summary } = useMemo(
    () => buildCategoryRows({
      categoryValues,
      categoryCounts,
      filteredCounts,
      sortMode: effectiveSort,
      nameFilter,
      getLabel,
    }),
    [categoryValues, categoryCounts, filteredCounts, effectiveSort, nameFilter, getLabel]
  );

  const mutedSet = useMemo(() => new Set(mutedCategories), [mutedCategories]);

  const categoricalOptions = useMemo(() => {
    if (!colorFieldOptions) return [];
    return colorFieldOptions.filter(o => o.recommendedScale === 'categorical');
  }, [colorFieldOptions]);

  const hasData = categoryValues.length > 0;
  const displayName = getCategoryDisplayName(categoryField);
  const showFieldSelector = categoricalOptions.length > 0 && !!onAnalysisFieldChange;

  if (!hasData && !showFieldSelector) return null;

  return (
    <Card className="gap-0 border-0 bg-transparent py-0 shadow-none">
      <CardHeader className="gap-2 px-0 pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="min-w-0 truncate text-sm">
            {hasData ? `${displayName} Distribution` : 'Distribution'}
          </CardTitle>
          {hasData && (
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              className="shrink-0"
              value={effectiveSort}
              onValueChange={(v) => v && setSortMode(v as CategorySortMode)}
            >
              <ToggleGroupItem value="count" className="h-6 px-2 text-[10px]" aria-label="Sort by count">
                Count
              </ToggleGroupItem>
              {filterActive && (
                <ToggleGroupItem value="rate" className="h-6 px-2 text-[10px]" aria-label="Sort by match rate">
                  Rate
                </ToggleGroupItem>
              )}
              <ToggleGroupItem value="natural" className="h-6 px-2 text-[10px]" aria-label="Sort by name">
                Name
              </ToggleGroupItem>
            </ToggleGroup>
          )}
        </div>
        {showFieldSelector && (
          <Select
            value={analysisField ?? FOLLOW_COLOR_FIELD}
            onValueChange={(val) => onAnalysisFieldChange!(val === FOLLOW_COLOR_FIELD ? null : val)}
          >
            <SelectTrigger size="sm" className="h-7 w-full text-xs">
              <SelectValue placeholder="Select field..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={FOLLOW_COLOR_FIELD}>
                <span className="text-xs text-muted-foreground">Follow color field</span>
              </SelectItem>
              <SelectSeparator />
              {categoricalOptions.map(opt => (
                <SelectItem key={opt.field} value={opt.field}>
                  <span className="text-xs">{opt.displayName} ({opt.uniqueCount})</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        {hasData ? (
          <CardDescription className="text-xs">
            {filterActive
              ? `${(summary.totalFiltered ?? 0).toLocaleString()} visible across ${summary.matchedCategoryCount ?? 0} categories`
              : `${summary.totalCount.toLocaleString()} points across ${summary.visibleCategoryCount} categories`
            }
          </CardDescription>
        ) : (
          <CardDescription className="text-xs">
            Select a categorical field to view its distribution.
          </CardDescription>
        )}
        {categoryValues.length > NAME_FILTER_THRESHOLD && (
          <Input
            placeholder="Filter categories..."
            aria-label="Filter categories"
            value={nameFilter}
            onChange={(e) => setNameFilter(e.target.value)}
            className="h-7 text-xs"
          />
        )}
      </CardHeader>

      {hasData && (
        <CardContent className="px-0">
          {rows.length > 0 ? (
            <ScrollArea className="[&>[data-radix-scroll-area-viewport]]:max-h-56">
              <ul className="space-y-0.5 pr-3">
                {rows.map(row => {
                  const color = colorMap[row.value] ?? FALLBACK_COLOR;
                  const isMuted = mutedSet.has(row.value);
                  const countText = filterActive
                    ? `${(row.filteredCount ?? 0).toLocaleString()} / ${row.count.toLocaleString()}`
                    : `${row.count.toLocaleString()} (${formatPct(row.pctOfTotal)})`;
                  const fillFraction = filterActive ? (row.fillFraction ?? 0) : row.trackFraction;

                  return (
                    <li key={row.value}>
                      <button
                        type="button"
                        title={`${row.label} — ${countText}`}
                        aria-pressed={isMuted}
                        onClick={(e) => onCategoryToggle?.(row.value, e.shiftKey)}
                        className={cn(
                          'relative flex w-full items-center overflow-hidden rounded-sm px-1.5 py-1 text-left',
                          onCategoryToggle ? 'cursor-pointer hover:bg-muted/50' : 'cursor-default',
                          isMuted && 'opacity-40'
                        )}
                      >
                        {filterActive && (
                          <span
                            aria-hidden
                            className="absolute inset-y-0 left-0 rounded-l-sm rounded-r-full transition-[width] duration-300"
                            style={{ width: barWidth(row.trackFraction), backgroundColor: color, opacity: 0.15 }}
                          />
                        )}
                        <span
                          aria-hidden
                          className="absolute inset-y-0 left-0 rounded-l-sm rounded-r-full transition-[width] duration-300"
                          style={{ width: barWidth(fillFraction), backgroundColor: color, opacity: 0.35 }}
                        />
                        <span className={cn('relative z-10 min-w-0 flex-1 truncate text-xs', isMuted && 'line-through')}>
                          {row.label}
                        </span>
                        <span className="relative z-10 shrink-0 pl-2 text-[11px] tabular-nums text-muted-foreground">
                          {filterActive ? (
                            countText
                          ) : (
                            <>
                              {row.count.toLocaleString()}{' '}
                              <span className="text-muted-foreground/60">({formatPct(row.pctOfTotal)})</span>
                            </>
                          )}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </ScrollArea>
          ) : (
            <p className="text-xs text-muted-foreground">No categories match the filter.</p>
          )}
          {onCategoryToggle && rows.length > 0 && (
            <p className="pt-2 text-[10px] text-muted-foreground/70">
              Click to isolate · Shift+click to toggle
            </p>
          )}
        </CardContent>
      )}
    </Card>
  );
}
