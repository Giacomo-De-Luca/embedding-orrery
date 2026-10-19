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
import { CategoryBarList } from './charts/CategoryBarList';
import { FieldAssociationSection } from './charts/FieldAssociationSection';
import { TemporalFilterChart } from './charts/TemporalFilterChart';
import { ProbeSection } from './ProbeSection';
import { IS_DEMO } from '@/lib/utils/demoMode';
import { useTemporalData } from '../../lib/hooks/useTemporalData';
import { useCategoryData } from '../../lib/hooks/useCategoryData';
import { getUnclusteredValues } from '../../lib/utils/categoryColors';
import { CategoryKey } from '../../lib/utils/categoryKey';
import type { MetadataPoint, TemporalRange } from '../../lib/types/types';
import { CategoricalFields, type ColorFieldOption } from '../../lib/utils/fieldAnalysis';
import { AssociationFields } from '../../lib/utils/fieldAxis';

/** Stable empty array so useTemporalData memo deps don't churn. */
const NO_CATEGORIES: string[] = [];
import type { UseProbesReturn } from '../../lib/hooks/useProbes';

interface AnalyticsSidebarProps extends React.ComponentProps<typeof Sidebar> {
  points: MetadataPoint[];
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
  probes?: UseProbesReturn | null;
  onCategoryToggle?: (category: string, shiftKey: boolean) => void;
  /** Active collection, for the field-association heatmap's topic-similarity sort. */
  collectionName?: string | null;
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
  probes,
  onCategoryToggle,
  collectionName,
  className,
  ...props
}: AnalyticsSidebarProps) {
  // Independent analysis field: null means "follow colorByField"
  const [analysisField, setAnalysisField] = useState<string | null>(null);
  // Temporal field override: null means auto-detect
  const [temporalFieldOverride, setTemporalFieldOverride] = useState<string | null>(null);

  // Reset analysisField when colorByField changes (so it follows by default)
  useEffect(() => {
    setAnalysisField(null);
  }, [colorByField]);

  const effectiveAnalysisField = analysisField ?? colorByField ?? null;

  // Compute category data for the analysis field (may differ from colorByField)
  const isOverridden = analysisField !== null && analysisField !== colorByField;
  const { categoryValues: analysisCategoryValues, categoryCounts: analysisCategoryCounts } =
    useCategoryData(points, isOverridden ? analysisField : null);

  // Use analysis-specific data when overridden, otherwise use parent-provided data
  const activeCategoryValues = isOverridden ? analysisCategoryValues : categoryValues;
  const activeCategoryCounts = isOverridden ? analysisCategoryCounts : categoryCounts;

  // Filter out unclustered noise points from topic and subtopic fields.
  const filteredCategoryValues = useMemo(() => {
    const noiseValues = getUnclusteredValues(effectiveAnalysisField);
    if (noiseValues.size === 0) return activeCategoryValues;
    return activeCategoryValues.filter(v => !noiseValues.has(v));
  }, [effectiveAnalysisField, activeCategoryValues]);

  const denoisedCategoryCounts = useMemo(() => {
    if (filteredCategoryValues.length === activeCategoryValues.length) return activeCategoryCounts;
    const counts: Record<string, number> = {};
    for (const v of filteredCategoryValues) {
      if (activeCategoryCounts[v] !== undefined) counts[v] = activeCategoryCounts[v];
    }
    return counts;
  }, [filteredCategoryValues, activeCategoryValues.length, activeCategoryCounts]);

  // Filtered counts per category: reuse shared computation when field matches, compute locally otherwise.
  // Uses combinedMutedIndices (text search + temporal) for consistency with scatter plot muting.
  const activeFilterCounts = useMemo(() => {
    // When analysis field follows colorByField, reuse the pre-computed counts from DashboardPanel
    if (!isOverridden && sharedFilteredCounts) return sharedFilteredCounts;
    // Otherwise compute locally using combinedMutedIndices (covers both text + temporal filters)
    if (!combinedMutedIndices || combinedMutedIndices.size === 0 || !effectiveAnalysisField) return null;
    return CategoryKey.count(points, effectiveAnalysisField, combinedMutedIndices).categoryCounts;
  }, [isOverridden, sharedFilteredCounts, combinedMutedIndices, effectiveAnalysisField, points]);

  // Guard against continuous color fields (sequential/diverging scales) exploding
  // into one "category" per distinct float value.
  const isCategoricalField = useMemo(
    () => CategoricalFields.includes(effectiveAnalysisField, colorFieldOptions, filteredCategoryValues.length),
    [effectiveAnalysisField, colorFieldOptions, filteredCategoryValues.length]
  );

  const hasCategoricalData = Boolean(
    isCategoricalField && filteredCategoryValues.length > 0
  );

  // The association view compares two fields (categorical, or numeric binned on the axis).
  const hasAssociationFields = AssociationFields.options(colorFieldOptions).length >= 2;

  const { temporalField, temporalFieldCandidates, crossTabData, temporalCounts, allPeriods } = useTemporalData(
    points,
    colorByField,
    hasCategoricalData ? filteredCategoryValues : NO_CATEGORIES,
    availableFields,
    temporalFieldOverride
  );

  const hasTemporalData = temporalField && allPeriods.length >= 2;
  const showTemporalSection = hasTemporalData || availableFields.length > 0;
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

  const handleTemporalFieldChange = React.useCallback((field: string | null) => {
    setTemporalFieldOverride(field);
    onTemporalRangeChange?.(null); // clear brush when switching fields
  }, [onTemporalRangeChange]);

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
            <CategoryBarList
              categoryField={effectiveAnalysisField ?? null}
              categoryValues={filteredCategoryValues}
              categoryCounts={denoisedCategoryCounts}
              categoricalPalette={categoricalPalette}
              filteredCounts={activeFilterCounts}
              colorFieldOptions={colorFieldOptions}
              analysisField={analysisField}
              onAnalysisFieldChange={handleAnalysisFieldChange}
              mutedCategories={mutedCategories}
              onCategoryToggle={onCategoryToggle}
            />
          )}

          {!hasCategoricalData && colorFieldOptions && colorFieldOptions.length > 0 && (
            <CategoryBarList
              categoryField={null}
              categoryValues={[]}
              categoryCounts={{}}
              categoricalPalette={categoricalPalette}
              colorFieldOptions={colorFieldOptions}
              analysisField={analysisField}
              onAnalysisFieldChange={handleAnalysisFieldChange}
            />
          )}

          {hasAssociationFields && (
            <>
              <Separator />
              <FieldAssociationSection
                points={points}
                colorByField={colorByField}
                colorFieldOptions={colorFieldOptions}
                mutedIndices={combinedMutedIndices}
                categoricalPalette={categoricalPalette}
                collectionName={collectionName}
              />
            </>
          )}

          {showTemporalSection && (
            <>
              {(hasCategoricalData || hasAssociationFields) && <Separator />}
              {/* data-tour: spotlight target for the demo tour's temporal step */}
              <div data-tour="temporal-chart">
                <TemporalFilterChart
                  temporalField={temporalField ?? ''}
                  allPeriods={allPeriods}
                  onBrushChange={handleBrushChange}
                  brushStartIndex={brushStartIndex}
                  brushEndIndex={brushEndIndex}
                  temporalCounts={temporalCounts}
                  categoryField={hasStackedTemporalData ? colorByField : null}
                  categoryValues={hasStackedTemporalData ? filteredCategoryValues : undefined}
                  categoryCounts={hasStackedTemporalData ? denoisedCategoryCounts : undefined}
                  crossTabData={hasStackedTemporalData ? crossTabData : undefined}
                  categoricalPalette={categoricalPalette}
                  mutedCategories={mutedCategories}
                  availableFields={availableFields}
                  temporalFieldOverride={temporalFieldOverride}
                  onTemporalFieldChange={handleTemporalFieldChange}
                />
              </div>
            </>
          )}

          {!hasCategoricalData && !showTemporalSection && (!colorFieldOptions || colorFieldOptions.length === 0) && (
            <p className="text-sm text-muted-foreground">
              Select a categorical color field to view distribution charts.
            </p>
          )}

          {/* Probe training is a write op — demo builds render the section
              read-only: shipped probes stay browsable/colorable (all client-
              side), only fitting and deleting disappear. The data-tour
              wrapper gives the probing tour's final step its anchor. */}
          {probes && (
            <div data-tour="probe-section" className="space-y-6">
              {(hasCategoricalData || hasAssociationFields || showTemporalSection) && <Separator />}
              <ProbeSection
                probes={probes}
                colorFieldOptions={colorFieldOptions}
                readOnly={IS_DEMO}
              />
            </div>
          )}
        </div>
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
