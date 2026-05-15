'use client';

import * as React from 'react';
import { useMemo, useState, useEffect } from 'react';
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
import { useCategoryData } from '../../lib/hooks/useCategoryData';
import { CATEGORY_PRESETS } from '../../lib/utils/categoryColors';
import type { Point2D, Point3D, TemporalRange } from '../../lib/types/types';
import type { ColorFieldOption } from '../../lib/utils/fieldAnalysis';

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
  /** Pre-computed filtered counts from DashboardPanel (keyed by colorByField). */
  sharedFilteredCounts?: Record<string, number> | null;
  /** Combined muted indices (text search + temporal) for local fallback computation. */
  combinedMutedIndices?: Set<number> | null;
  colorFieldOptions?: ColorFieldOption[];
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
  sharedFilteredCounts,
  combinedMutedIndices,
  colorFieldOptions,
  className,
  ...props
}: AnalyticsSidebarProps) {
  // Independent analysis field: null means "follow colorByField"
  const [analysisField, setAnalysisField] = useState<string | null>(null);

  // Reset analysisField when colorByField changes (so it follows by default)
  useEffect(() => {
    setAnalysisField(null);
  }, [colorByField]);

  const effectiveAnalysisField = analysisField ?? colorByField;

  // Compute category data for the analysis field (may differ from colorByField)
  const isOverridden = analysisField !== null && analysisField !== colorByField;
  const { categoryValues: analysisCategoryValues, categoryCounts: analysisCategoryCounts } =
    useCategoryData(points, isOverridden ? analysisField : null);

  // Use analysis-specific data when overridden, otherwise use parent-provided data
  const activeCategoryValues = isOverridden ? analysisCategoryValues : categoryValues;
  const activeCategoryCounts = isOverridden ? analysisCategoryCounts : categoryCounts;

  // Filter out unclustered noise points from topic fields.
  const filteredCategoryValues = useMemo(() => {
    const preset = effectiveAnalysisField ? CATEGORY_PRESETS[effectiveAnalysisField.toLowerCase()] : null;
    if (!preset) return activeCategoryValues;
    const noiseValues = new Set<string>();
    for (const [key, label] of Object.entries(preset.labels ?? {})) {
      if (label === 'Unclustered') {
        noiseValues.add(key);
        noiseValues.add(label);
      }
    }
    if (noiseValues.size === 0) return activeCategoryValues;
    return activeCategoryValues.filter(v => !noiseValues.has(v));
  }, [effectiveAnalysisField, activeCategoryValues]);

  const filteredCounts = useMemo(() => {
    if (filteredCategoryValues.length === activeCategoryValues.length) return activeCategoryCounts;
    const counts: Record<string, number> = {};
    for (const v of filteredCategoryValues) {
      if (activeCategoryCounts[v] !== undefined) counts[v] = activeCategoryCounts[v];
    }
    return counts;
  }, [filteredCategoryValues, activeCategoryValues.length, activeCategoryCounts]);

  // Filtered counts per category: reuse shared computation when field matches, compute locally otherwise.
  // Uses combinedMutedIndices (text search + temporal) for consistency with scatter plot muting.
  const searchMatchCounts = useMemo(() => {
    // When analysis field follows colorByField, reuse the pre-computed counts from DashboardPanel
    if (!isOverridden && sharedFilteredCounts) return sharedFilteredCounts;
    // Otherwise compute locally using combinedMutedIndices (covers both text + temporal filters)
    if (!combinedMutedIndices || combinedMutedIndices.size === 0 || !effectiveAnalysisField) return null;
    const counts: Record<string, number> = {};
    for (const p of points) {
      if (combinedMutedIndices.has(p.index)) continue;
      const value = p.metadata?.[effectiveAnalysisField];
      if (value === null || value === undefined || value === '') continue;
      const key = String(value);
      counts[key] = (counts[key] ?? 0) + 1;
    }
    return counts;
  }, [isOverridden, sharedFilteredCounts, combinedMutedIndices, effectiveAnalysisField, points]);

  const { temporalField, crossTabData, temporalCounts, allPeriods } = useTemporalData(
    points,
    colorByField,
    filteredCategoryValues,
    availableFields
  );

  const hasCategoricalData = effectiveAnalysisField && filteredCategoryValues.length > 0;
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

  const handleAnalysisFieldChange = React.useCallback((field: string | null) => {
    setAnalysisField(field);
  }, []);

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
              categoryField={effectiveAnalysisField}
              categoryValues={filteredCategoryValues}
              categoryCounts={filteredCounts}
              categoricalPalette={categoricalPalette}
              searchMatchCounts={searchMatchCounts}
              colorFieldOptions={colorFieldOptions}
              analysisField={analysisField}
              onAnalysisFieldChange={handleAnalysisFieldChange}
            />
          )}

          {!hasCategoricalData && colorFieldOptions && colorFieldOptions.length > 0 && (
            <CategoryBarChart
              categoryField={null}
              categoryValues={[]}
              categoryCounts={{}}
              categoricalPalette={categoricalPalette}
              colorFieldOptions={colorFieldOptions}
              analysisField={analysisField}
              onAnalysisFieldChange={handleAnalysisFieldChange}
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

          {!hasCategoricalData && !hasTemporalData && (!colorFieldOptions || colorFieldOptions.length === 0) && (
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
