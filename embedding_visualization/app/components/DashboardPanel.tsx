'use client';

import { useCallback, useMemo } from 'react';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/lib/ui-primitives/resizable';
import { ScatterPlot2D } from './ScatterPlot2D';
import { ScatterPlot3D } from './ScatterPlot3D';
import { Legend } from './Legend';
import { SimilarItemsTable } from './SimilarItemsTable';
import { EmbeddingSidebar } from './EmbeddingSidebar';
import { SearchSidebar } from './SearchSidebar';
import type { Point2D, Point3D, VisualizationState, SemanticSearchResult, HighlightMap } from '../../lib/types/types';
import type { ColorFieldOption } from '../../lib/utils/fieldAnalysis';
import { ScrollArea, ScrollBar } from '@/lib/ui-primitives/scroll-area';
import { cn } from '@/lib/utils/utils';
import { useCategoryData } from '../../lib/hooks/useCategoryData';

export type ActivePanel = 'controls' | 'search' | null;

interface DashboardPanelProps {
  state: VisualizationState;
  points2d: Point2D[];
  points3d: Point3D[];
  highlightedIndices?: HighlightMap;
  onPointClick: (point: Point2D | Point3D) => void;
  onStateChange: (newState: Partial<VisualizationState>) => void;
  selectedPoint?: Point2D | Point3D | null;
  // Search results panel props
  semanticSearchResults: SemanticSearchResult[] | null;
  searchQueryLabel: string | null;
  // Controls sidebar props
  embeddingDim: number;
  metadata: {
    pca_2d_variance?: number[];
    pca_3d_variance?: number[];
  };
  colorFieldOptions?: ColorFieldOption[];
  // Search sidebar props
  searchQuery?: string;
  highlightedCount?: number;
  textSearchResults?: (Point2D | Point3D)[];
  onTextResultClick?: (point: Point2D | Point3D) => void;
  // Panel state
  activePanel: ActivePanel;
  // Query prompt configuration
  queryPromptName?: string | null;
  onQueryPromptNameChange?: (value: string | null) => void;
}

export function DashboardPanel({
  state,
  points2d,
  points3d,
  highlightedIndices,
  onPointClick,
  onStateChange,
  selectedPoint,
  semanticSearchResults,
  searchQueryLabel,
  embeddingDim,
  metadata,
  colorFieldOptions = [],
  searchQuery,
  highlightedCount,
  textSearchResults,
  onTextResultClick,
  activePanel,
  queryPromptName,
  onQueryPromptNameChange,
}: DashboardPanelProps) {
  const isExpanded = activePanel !== null;
  const is2D = state.mode === '2d';
  const colorByField = state.colorByField;

  // Compute category values and counts from points
  const points = is2D ? points2d : points3d;
  const { categoryValues, categoryCounts } = useCategoryData(points, colorByField);

  // Check if we're using a continuous scale
  const isContinuousScale = state.colorScaleType === 'sequential' || state.colorScaleType === 'diverging' || state.colorScaleType === 'monochrome';

  // Compute numeric range for continuous scales
  const numericRange = useMemo(() => {
    if (!isContinuousScale || !colorByField || points.length === 0) return undefined;

    let min: number | undefined;
    let max: number | undefined;

    for (const point of points) {
      const value = point.metadata?.[colorByField];
      if (value === null || value === undefined || value === '') continue;

      const numValue = typeof value === 'number' ? value : parseFloat(String(value));
      if (isNaN(numValue)) continue;

      if (min === undefined || numValue < min) min = numValue;
      if (max === undefined || numValue > max) max = numValue;
    }

    if (min !== undefined && max !== undefined) {
      return { min, max };
    }
    return undefined;
  }, [isContinuousScale, colorByField, points]);

  // Toggle handler for muting/unmuting categories
  const handleCategoryToggle = useCallback((category: string) => {
    const muted = state.mutedCategories ?? [];
    const newMuted = muted.includes(category)
      ? muted.filter(c => c !== category)
      : [...muted, category];
    onStateChange({ mutedCategories: newMuted });
  }, [state.mutedCategories, onStateChange]);

  // Show legend for categorical scales with values, or continuous scales with numeric range
  const showLegend = colorByField && (
    (categoryValues.length > 0 && !isContinuousScale) ||
    (isContinuousScale && numericRange !== undefined)
  );
  const showResultsTable = semanticSearchResults && semanticSearchResults.length > 0;

  // Convert colorByField to the 'category' | 'none' format expected by scatter plots
  const colorBy = colorByField ? 'category' : 'none';

  const plot = is2D ? (
    <ScatterPlot2D
      points={points2d}
      colorBy={colorBy}
      categoryField={colorByField}
      categoryValues={categoryValues}
      colorScaleType={state.colorScaleType}
      monochromeColor={state.monochromeColor}
      highlightedIndices={highlightedIndices}
      selectedPoint={selectedPoint as Point2D | null}
      onPointClick={onPointClick}
      showOnlyHighlighted={state.showOnlyHighlighted}
      showLabels={state.showLabels}
      mutedCategories={state.mutedCategories}
    />
  ) : (
    <ScatterPlot3D
      points={points3d}
      colorBy={colorBy}
      categoryField={colorByField}
      categoryValues={categoryValues}
      colorScaleType={state.colorScaleType}
      monochromeColor={state.monochromeColor}
      highlightedIndices={highlightedIndices}
      selectedPoint={selectedPoint as Point3D | null}
      onPointClick={onPointClick}
      showOnlyHighlighted={state.showOnlyHighlighted}
      showLabels={state.showLabels}
      mutedCategories={state.mutedCategories}
    />
  );

  return (

    <div className="relative h-full w-full overflow-hidden">

      {/* 1. LAYER: Plot Background (Z-0) */}
      <div className="absolute inset-0 z-0">
        <div className="h-full w-full rounded-lg text-card-foreground shadow-sm">
          {plot}
        </div>
      </div>

      {/* 2. LAYER: Legend Overlay (Z-10) */}
      {showLegend && (
        <div className="absolute top-30 right-4 z-10 pointer-events-none">

          {/* Horizontal Spacer *
        <div className="absolute right-10 bottom-0 left-0 top-40 z-10 pointer-events-none">
          <ResizablePanelGroup direction="horizontal" className="h-full w-full">
            <ResizablePanel defaultSize={80} minSize={50} className="bg-transparent" />

            <ResizableHandle className="bg-transparent hover:bg-border/30 w-2 pointer-events-auto" />

            <ResizablePanel defaultSize={20} minSize={15} maxSize={50} className="pointer-events-none"> */}
          <ScrollArea className="overflow-y-auto pointer-events-auto">
            <Legend
              categoryField={colorByField}
              categoryValues={categoryValues}
              categoryCounts={categoryCounts}
              mutedCategories={state.mutedCategories}
              onCategoryToggle={handleCategoryToggle}
              colorScaleType={state.colorScaleType}
              numericRange={numericRange}
            />
          </ScrollArea>
          {/* Horizontal Spacer 
            </ResizablePanel>
          </ResizablePanelGroup> */}
        </div>
      )}


      {/* 3. LAYER: Table Overlay (Z-20) */}
      {showResultsTable && (
        <div className={cn(
          "absolute inset-0 z-20 pointer-events-none transition-all duration-300 ease-in-out",
          isExpanded ? "pl-84" : "pl-0"
        )}>
          <ResizablePanelGroup direction="vertical" className="h-full w-full">

            {/* Vertical Spacer - Allows clicking through to the plot above */}
            <ResizablePanel defaultSize={70} minSize={10} className="bg-transparent" />

            {/* Handle - Needs pointer-events-auto to be draggable */}
            <ResizableHandle className="bg-transparent hover:bg-border/30 h-2 pointer-events-auto" />

            {/* Table Panel - Bottom Floating Panel */}
            <ResizablePanel
              defaultSize={30}
              minSize={5}
              maxSize={120}
              className="pointer-events-auto" // Re-enable clicks for the table
            >
              <div className="h-full w-full px-2 pb-2">
                {/* Added background/blur so text is readable over the plot points */}
                <ScrollArea className="h-full overflow-y-auto border rounded-md shadow-lg">
                  <SimilarItemsTable
                    results={semanticSearchResults}
                    queryLabel={searchQueryLabel}
                    categoryField={colorByField}
                  />
                  <ScrollBar orientation="vertical" />
                </ScrollArea>
              </div>
            </ResizablePanel>

          </ResizablePanelGroup>
        </div>
      )}

      {/* 4. LAYER: Sidebar Overlay (Z-40) */}
      <div className="absolute inset-y-0 left-0 z-40 pointer-events-none">
        {/* Controls Sidebar */}
        <EmbeddingSidebar
          state={state}
          onStateChange={onStateChange}
          embeddingDim={embeddingDim}
          metadata={metadata}
          selectedPoint={selectedPoint || null}
          colorFieldOptions={colorFieldOptions}
          variant="floating"
          className={cn(
            "pointer-events-auto absolute top-20 bottom-2 z-40 w-80 shadow-2xl transition-all duration-300 ease-in-out",
            activePanel === 'controls' ? "left-4" : "-left-[400px] opacity-0"
          )}
        />

        {/* Search Sidebar */}
        <SearchSidebar
          searchQuery={searchQuery ?? ''}
          onSearchChange={(value) => onStateChange({ searchQuery: value })}
          showOnlyHighlighted={state.showOnlyHighlighted ?? false}
          onShowOnlyHighlightedChange={(checked) => onStateChange({ showOnlyHighlighted: checked })}
          showLabels={state.showLabels ?? false}
          onShowLabelsChange={(checked) => onStateChange({ showLabels: checked })}
          hasHighlights={Boolean(highlightedCount && highlightedCount > 0)}
          textSearchResults={textSearchResults}
          selectedPointId={selectedPoint?.id}
          onResultClick={onTextResultClick}
          categoryField={colorByField}
          queryPromptName={queryPromptName}
          onQueryPromptNameChange={onQueryPromptNameChange}
          variant="floating"
          className={cn(
            "pointer-events-auto absolute top-20 bottom-2 z-40 w-80 shadow-2xl transition-all duration-300 ease-in-out",
            activePanel === 'search' ? "left-4" : "-left-[400px] opacity-0"
          )}
        />
      </div>
    </div>
  );
}