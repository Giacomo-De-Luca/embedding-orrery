'use client';

import * as React from 'react';
import { useMemo } from 'react';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from '@/lib/ui-primitives/sidebar';
import { Separator } from '@/lib/ui-primitives/separator';
import { ScrollBar } from '@/lib/ui-primitives/scroll-area';
import { CategoryBarChart } from './charts/CategoryBarChart';
import { TemporalFilterChart } from './charts/TemporalFilterChart';
import { useTemporalData } from '../../lib/hooks/useTemporalData';
import { CATEGORY_PRESETS } from '../../lib/utils/categoryColors';
import type { Point2D, Point3D, TemporalRange } from '../../lib/types/types';

interface AnalyticsSidebarProps extends React.ComponentProps<typeof Sidebar> {
  points: (Point2D | Point3D)[];
  colorByField: string | null | undefined;
  categoryValues: string[];
  categoryCounts: Record<string, number>;
  availableFields: string[];
  categoricalPalette?: string;
  mutedCategories?: string[];
  temporalRange?: TemporalRange | null;
  onTemporalRangeChange?: (range: TemporalRange | null) => void;
}

export function AnalyticsSidebar({
  points,
  colorByField,
  categoryValues,
  categoryCounts,
  availableFields,
  categoricalPalette,
  mutedCategories = [],
  temporalRange,
  onTemporalRangeChange,
  className,
  ...props
}: AnalyticsSidebarProps) {
  // Filter out unclustered noise points from topic fields.
  const filteredCategoryValues = useMemo(() => {
    const preset = colorByField ? CATEGORY_PRESETS[colorByField.toLowerCase()] : null;
    if (!preset) return categoryValues;
    const noiseValues = new Set<string>();
    for (const [key, label] of Object.entries(preset.labels ?? {})) {
      if (label === 'Unclustered') {
        noiseValues.add(key);
        noiseValues.add(label);
      }
    }
    if (noiseValues.size === 0) return categoryValues;
    return categoryValues.filter(v => !noiseValues.has(v));
  }, [colorByField, categoryValues]);

  const filteredCounts = useMemo(() => {
    if (filteredCategoryValues.length === categoryValues.length) return categoryCounts;
    const counts: Record<string, number> = {};
    for (const v of filteredCategoryValues) {
      if (categoryCounts[v] !== undefined) counts[v] = categoryCounts[v];
    }
    return counts;
  }, [filteredCategoryValues, categoryValues.length, categoryCounts]);

  const { temporalField, crossTabData, temporalCounts, allPeriods } = useTemporalData(
    points,
    colorByField,
    filteredCategoryValues,
    availableFields
  );

  const hasCategoricalData = colorByField && filteredCategoryValues.length > 0;
  const hasTemporalData = temporalField && allPeriods.length >= 2;
  const hasStackedTemporalData = hasTemporalData && crossTabData.length >= 2 && hasCategoricalData;

  // Compute brush indices from temporalRange
  const brushStartIndex = useMemo(() => {
    if (!temporalRange || !hasTemporalData) return undefined;
    const idx = allPeriods.indexOf(temporalRange.startPeriod);
    return idx >= 0 ? idx : undefined;
  }, [temporalRange, hasTemporalData, allPeriods]);

  const brushEndIndex = useMemo(() => {
    if (!temporalRange || !hasTemporalData) return undefined;
    const idx = allPeriods.indexOf(temporalRange.endPeriod);
    return idx >= 0 ? idx : undefined;
  }, [temporalRange, hasTemporalData, allPeriods]);

  const handleBrushChange = React.useCallback((startPeriod: string, endPeriod: string, periods: string[]) => {
    if (!onTemporalRangeChange || !temporalField) return;
    if (startPeriod === periods[0] && endPeriod === periods[periods.length - 1]) {
      onTemporalRangeChange(null);
      return;
    }
    onTemporalRangeChange({
      field: temporalField,
      startPeriod,
      endPeriod,
      allPeriods: periods,
    });
  }, [onTemporalRangeChange, temporalField]);

  return (
    <Sidebar
      collapsible="offcanvas"
      className={className}
      {...props}
    >
      <SidebarHeader className="border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold">Analytics</span>
        </div>
      </SidebarHeader>

      <SidebarContent className="gap-0">
        <div className="p-4 space-y-6">
          {hasCategoricalData && (
            <CategoryBarChart
              categoryField={colorByField}
              categoryValues={filteredCategoryValues}
              categoryCounts={filteredCounts}
              categoricalPalette={categoricalPalette}
            />
          )}

          {hasTemporalData && (
            <>
              {hasCategoricalData && <Separator />}
              <TemporalFilterChart
                temporalField={temporalField!}
                allPeriods={allPeriods}
                onBrushChange={handleBrushChange}
                brushStartIndex={brushStartIndex}
                brushEndIndex={brushEndIndex}
                temporalCounts={temporalCounts}
                categoryField={hasStackedTemporalData ? colorByField : null}
                categoryValues={hasStackedTemporalData ? filteredCategoryValues : undefined}
                categoryCounts={hasStackedTemporalData ? filteredCounts : undefined}
                crossTabData={hasStackedTemporalData ? crossTabData : undefined}
                categoricalPalette={categoricalPalette}
                mutedCategories={mutedCategories}
              />
            </>
          )}

          {!hasCategoricalData && !hasTemporalData && (
            <p className="text-sm text-muted-foreground">
              Select a categorical color field to view distribution charts.
            </p>
          )}
        </div>
        <ScrollBar orientation="vertical" />
      </SidebarContent>

      <SidebarFooter className="border-t px-4 py-3">
        <div className="text-xs text-muted-foreground text-center">
          Press{' '}
          <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium">
            <span className="text-xs">⌘</span>J
          </kbd>{' '}
          to toggle
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
