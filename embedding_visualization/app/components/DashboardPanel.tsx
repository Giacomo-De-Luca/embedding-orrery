'use client';

import { useCallback, useMemo, useState, useEffect, useDeferredValue } from 'react';
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
import { AnalyticsSidebar } from './AnalyticsSidebar';
import { Table2, Palette } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import type { Point2D, Point3D, SemanticSearchResult, HighlightMap, TopicInfo, TemporalRange } from '../../lib/types/types';
import type { TopicSearchMode, TopicSearchResult } from '../../lib/hooks/useTopicSearch';
import type { ColorFieldOption } from '../../lib/utils/fieldAnalysis';
import { cn } from '@/lib/utils/utils';
import { useCategoryData } from '../../lib/hooks/useCategoryData';
import { useNestedCategoryData } from '../../lib/hooks/useNestedCategoryData';
import { useVerticalResize } from '../../lib/hooks/useVerticalResize';
import { useVisualizationStore } from '../../lib/stores/useVisualizationStore';

export type ActivePanel = 'controls' | 'search' | 'analytics' | null;

interface DashboardPanelProps {
  // Data
  points2d: Point2D[];
  points3d: Point3D[];
  highlightedIndices?: HighlightMap;
  onPointClick: (point: Point2D | Point3D) => void;
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
  textSearchHighlights?: Set<number>;
  textSearchLoading?: boolean;
  onTextResultClick?: (point: Point2D | Point3D) => void;
  // Panel state
  activePanel: ActivePanel;
  // Query prompt configuration
  queryPromptName?: string | null;
  onQueryPromptNameChange?: (value: string | null) => void;
  // Tooltip configuration
  availableFields?: string[];
  // Topic search props (threaded to SearchSidebar)
  topics?: TopicInfo[];
  topicSearchMode?: TopicSearchMode;
  onTopicSearchModeChange?: (mode: TopicSearchMode) => void;
  topicDirectQuery?: string;
  onTopicDirectQueryChange?: (q: string) => void;
  topicFilteredTopics?: TopicInfo[];
  topicSemanticQuery?: string;
  onTopicSemanticQueryChange?: (q: string) => void;
  onTopicSemanticSearch?: () => void;
  topicSemanticResults?: TopicSearchResult[];
  topicSemanticLoading?: boolean;
  selectedTopicIds?: Set<number>;
  onToggleTopic?: (id: number) => void;
  onSelectAllTopics?: () => void;
  onClearAllTopics?: () => void;
}

export function DashboardPanel({
  points2d,
  points3d,
  highlightedIndices,
  onPointClick,
  selectedPoint,
  semanticSearchResults,
  searchQueryLabel,
  embeddingDim,
  metadata,
  colorFieldOptions = [],
  searchQuery,
  highlightedCount,
  textSearchResults,
  textSearchHighlights,
  textSearchLoading,
  onTextResultClick,
  activePanel,
  queryPromptName,
  onQueryPromptNameChange,
  availableFields = [],
  // Topic search props
  topics,
  topicSearchMode,
  onTopicSearchModeChange,
  topicDirectQuery,
  onTopicDirectQueryChange,
  topicFilteredTopics,
  topicSemanticQuery,
  onTopicSemanticQueryChange,
  onTopicSemanticSearch,
  topicSemanticResults,
  topicSemanticLoading,
  selectedTopicIds,
  onToggleTopic,
  onSelectAllTopics,
  onClearAllTopics,
}: DashboardPanelProps) {
  const isExpanded = activePanel !== null;

  // --- Visualization state from Zustand store ---
  const mode = useVisualizationStore((s) => s.mode);
  const colorByField = useVisualizationStore((s) => s.colorByField);
  const colorScale = useVisualizationStore((s) => s.colorScale);
  const categoricalPalette = useVisualizationStore((s) => s.categoricalPalette);
  const nestedColorMode = useVisualizationStore((s) => s.nestedColorMode);
  const mutedCategories = useVisualizationStore((s) => s.mutedCategories);
  const temporalRange = useVisualizationStore((s) => s.temporalRange);
  const showOnlyHighlighted = useVisualizationStore((s) => s.showOnlyHighlighted);
  const showLabels = useVisualizationStore((s) => s.showLabels);
  const tooltipFields = useVisualizationStore((s) => s.tooltipFields);
  const hideUnclustered = useVisualizationStore((s) => s.hideUnclustered);
  const nebulaMode = useVisualizationStore((s) => s.nebulaMode);
  const showClusterLabels = useVisualizationStore((s) => s.showClusterLabels);
  const hideFilteredPoints = useVisualizationStore((s) => s.hideFilteredPoints);
  const mutedPointOpacity = useVisualizationStore((s) => s.mutedPointOpacity);
  const setMutedCategories = useVisualizationStore((s) => s.setMutedCategories);
  const setTemporalRange = useVisualizationStore((s) => s.setTemporalRange);

  const is2D = mode === '2d';

  // Compute category values and counts from points
  const points = is2D ? points2d : points3d;
  const { categoryValues, categoryCounts } = useCategoryData(points, colorByField);

  // Nested topic/subtopic color mapping
  const { available: nestedColorAvailable, nestedColorMap } = useNestedCategoryData(
    points, colorByField, nestedColorMode, categoricalPalette
  );

  // Topic-mode detection: when coloring by topic_label, selectedTopicIds drives everything
  const isTopicColorField = colorByField === 'topic_label';

  // Maps between topic labels and topic IDs for legend ↔ topic selection sync
  const topicLabelToIdMap = useMemo(() => {
    if (!isTopicColorField || !topics?.length) return null;
    const map = new Map<string, number>();
    for (const t of topics) {
      if (t.label) map.set(t.label, t.topicId);
    }
    return map;
  }, [isTopicColorField, topics]);

  const topicIdToLabelMap = useMemo(() => {
    if (!isTopicColorField || !topics?.length) return null;
    const map = new Map<number, string>();
    for (const t of topics) {
      if (t.label) map.set(t.topicId, t.label);
    }
    return map;
  }, [isTopicColorField, topics]);

  // Derive effective muted categories: in topic mode, derive from selectedTopicIds (no state writes)
  const effectiveMutedCategories = useMemo(() => {
    if (!isTopicColorField || !selectedTopicIds || selectedTopicIds.size === 0 || !topicIdToLabelMap) {
      return mutedCategories ?? [];
    }
    // Compute the set of labels that should NOT be muted (selected topics + their subtopics)
    const unmuted = new Set<string>();
    for (const id of selectedTopicIds) {
      const label = topicIdToLabelMap.get(id);
      if (label) {
        unmuted.add(label);
        // In nested mode, include subtopics of selected topics
        if (nestedColorMap?.hierarchy?.[label]) {
          for (const sub of nestedColorMap.hierarchy[label]) {
            unmuted.add(sub);
          }
        }
      }
    }
    // All categories not in unmuted set are muted
    const allCategories = nestedColorMap
      ? [...Object.keys(nestedColorMap.hierarchy), ...Object.values(nestedColorMap.hierarchy).flat()]
      : categoryValues;
    return allCategories.filter(c => !unmuted.has(c));
  }, [isTopicColorField, selectedTopicIds, topicIdToLabelMap, mutedCategories, nestedColorMap, categoryValues]);

  // Temporal range filtering: compute indices of points outside the selected time range
  const deferredTemporalRange = useDeferredValue(temporalRange);

  const temporallyMutedIndices = useMemo(() => {
    const range = deferredTemporalRange;
    if (!range) return null;
    const startIdx = range.allPeriods.indexOf(range.startPeriod);
    const endIdx = range.allPeriods.indexOf(range.endPeriod);
    if (startIdx < 0 || endIdx < 0) return null;
    // Build period→index map for O(1) lookups instead of O(P) indexOf per point
    const periodIndexMap = new Map<string, number>();
    for (let i = 0; i < range.allPeriods.length; i++) {
      periodIndexMap.set(range.allPeriods[i], i);
    }
    const set = new Set<number>();
    for (const p of points) {
      const val = String(p.metadata?.[range.field] ?? '');
      const periodIdx = periodIndexMap.get(val) ?? -1;
      if (periodIdx < startIdx || periodIdx > endIdx || periodIdx === -1) {
        set.add(p.index);
      }
    }
    return set.size > 0 ? set : null;
  }, [deferredTemporalRange, points]);

  // Text search muting: points not matching the text query get muted (like temporal filtering)
  const searchMutedIndices = useMemo(() => {
    if (!textSearchHighlights || textSearchHighlights.size === 0) return null;
    const set = new Set<number>();
    for (const p of points) {
      if (!textSearchHighlights.has(p.index)) set.add(p.index);
    }
    return set.size > 0 ? set : null;
  }, [textSearchHighlights, points]);

  // Combine temporal and search muting into a single set for scatter plots
  const combinedMutedIndices = useMemo(() => {
    if (!temporallyMutedIndices && !searchMutedIndices) return null;
    if (temporallyMutedIndices && !searchMutedIndices) return temporallyMutedIndices;
    if (!temporallyMutedIndices && searchMutedIndices) return searchMutedIndices;
    const set = new Set<number>();
    for (const i of temporallyMutedIndices!) set.add(i);
    for (const i of searchMutedIndices!) set.add(i);
    return set.size > 0 ? set : null;
  }, [temporallyMutedIndices, searchMutedIndices]);

  // Check if we're using a continuous scale
  const isContinuousScale = colorScale.type === 'sequential' || colorScale.type === 'diverging' || colorScale.type === 'monochrome';

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

  // Select-only handler: click isolates a category, shift+click toggles multi-select
  const handleCategoryToggle = useCallback((category: string, shiftKey: boolean) => {
    // In topic color mode, delegate to selectedTopicIds (always multi-select toggle;
    // shiftKey distinction is not applicable — topics always use toggle behavior)
    if (isTopicColorField && topicLabelToIdMap && onToggleTopic) {
      // Look up topic ID from the category label
      let topicId = topicLabelToIdMap.get(category);
      // If not found directly, it might be a subtopic — find parent topic
      if (topicId === undefined && nestedColorMap) {
        const parentLabel = Object.entries(nestedColorMap.hierarchy)
          .find(([, subs]) => subs.includes(category))?.[0];
        if (parentLabel) topicId = topicLabelToIdMap.get(parentLabel);
      }
      if (topicId !== undefined) onToggleTopic(topicId);
      return;
    }

    // Non-topic mode: existing mutedCategories behavior
    const muted = mutedCategories ?? [];
    const subtopics = nestedColorMap?.hierarchy?.[category];

    // In nested mode, allCategories must include both topics and subtopics
    const allCategories = nestedColorMap
      ? [...Object.keys(nestedColorMap.hierarchy), ...Object.values(nestedColorMap.hierarchy).flat()]
      : categoryValues;

    if (shiftKey) {
      // Shift+click: toggle individual category (add/remove from muted)
      if (subtopics && subtopics.length > 0) {
        const allRelated = [category, ...subtopics];
        const isCurrentlyMuted = muted.includes(category);
        if (isCurrentlyMuted) {
          setMutedCategories(muted.filter(c => !allRelated.includes(c)));
        } else {
          setMutedCategories([...new Set([...muted, ...allRelated])]);
        }
      } else {
        const newMuted = muted.includes(category)
          ? muted.filter(c => c !== category)
          : [...muted, category];
        setMutedCategories(newMuted);
      }
    } else {
      // Normal click: select only this category (mute all others)
      // Determine what should stay unmuted
      let keepUnmuted: Set<string>;
      if (subtopics && subtopics.length > 0) {
        // Clicked a topic header: keep topic + all its subtopics
        keepUnmuted = new Set([category, ...subtopics]);
      } else if (nestedColorMap) {
        // Clicked a subtopic: keep it + its parent topic (so subtopics stay visible in legend)
        const parentTopic = Object.entries(nestedColorMap.hierarchy)
          .find(([, subs]) => subs.includes(category))?.[0];
        keepUnmuted = parentTopic
          ? new Set([category, parentTopic])
          : new Set([category]);
      } else {
        keepUnmuted = new Set([category]);
      }

      // If already isolated on this exact selection, toggle back to show all
      const selected = allCategories.filter(c => !muted.includes(c));
      const isAlreadyIsolated = selected.length > 0 && selected.every(c => keepUnmuted.has(c));

      if (isAlreadyIsolated) {
        setMutedCategories([]);
      } else {
        setMutedCategories(allCategories.filter(c => !keepUnmuted.has(c)));
      }
    }
  }, [isTopicColorField, topicLabelToIdMap, onToggleTopic, nestedColorMap, mutedCategories, setMutedCategories, categoryValues]);

  // Temporal range change handler
  const handleTemporalRangeChange = useCallback((range: TemporalRange | null) => {
    setTemporalRange(range);
  }, [setTemporalRange]);

  // Double-click reset: show all categories
  const handleCategoryReset = useCallback(() => {
    if (isTopicColorField && onClearAllTopics) {
      onClearAllTopics();
    } else {
      setMutedCategories([]);
    }
  }, [isTopicColorField, onClearAllTopics, setMutedCategories]);

  // Show legend for categorical scales with values, or continuous scales with numeric range
  const showLegend = colorByField && (
    (categoryValues.length > 0 && !isContinuousScale) ||
    (isContinuousScale && numericRange !== undefined)
  );
  const showResultsTable = semanticSearchResults && semanticSearchResults.length > 0;

  // Collapse/expand state for the results panel
  const [resultsCollapsed, setResultsCollapsed] = useState(false);

  // Auto-expand when new results arrive
  useEffect(() => {
    if (showResultsTable) setResultsCollapsed(false);
  }, [showResultsTable]);

  // Collapse/expand state for the legend
  const [legendCollapsed, setLegendCollapsed] = useState(false);
  const {
    height: legendHeight,
    handleRef: legendDragRef,
    isDragging: legendDragging,
    reset: resetLegendHeight,
  } = useVerticalResize({
    initialHeight: 256,
    minHeight: 140,
    maxHeight: 600,
    onCollapse: () => setLegendCollapsed(true),
  });

  const legendDragHandle = (
    <div
      ref={legendDragRef}
      className="h-3 w-full cursor-ns-resize flex items-center justify-center group pointer-events-auto"
    >
      <div className="h-0.5 w-8 rounded-full bg-border group-hover:bg-foreground/30 transition-colors" />
    </div>
  );

  const plot = is2D ? (
    <ScatterPlot2D
      points={points2d}
      categoryField={colorByField}
      categoryValues={categoryValues}
      colorScale={colorScale}
      highlightedIndices={highlightedIndices}
      selectedPoint={selectedPoint as Point2D | null}
      onPointClick={onPointClick}
      showOnlyHighlighted={showOnlyHighlighted}
      showLabels={showLabels}
      mutedCategories={effectiveMutedCategories}
      tooltipFields={tooltipFields}
      hideUnclustered={hideUnclustered}
      categoricalPalette={categoricalPalette}
      nestedColorMap={nestedColorMap}
      combinedMutedIndices={combinedMutedIndices}
      hideFilteredPoints={hideFilteredPoints}
      mutedPointOpacity={mutedPointOpacity}
      showClusterLabels={showClusterLabels}
      onClusterLabelClick={isTopicColorField ? onToggleTopic : undefined}
      topicLabelToIdMap={isTopicColorField ? topicLabelToIdMap : undefined}
    />
  ) : (
    <ScatterPlot3D
      points={points3d}
      categoryField={colorByField}
      categoryValues={categoryValues}
      colorScale={colorScale}
      highlightedIndices={highlightedIndices}
      selectedPoint={selectedPoint as Point3D | null}
      onPointClick={onPointClick}
      showOnlyHighlighted={showOnlyHighlighted}
      showLabels={showLabels}
      mutedCategories={effectiveMutedCategories}
      tooltipFields={tooltipFields}
      hideUnclustered={hideUnclustered}
      categoricalPalette={categoricalPalette}
      nestedColorMap={nestedColorMap}
      nebulaMode={nebulaMode}
      showClusterLabels={showClusterLabels}
      onClusterLabelClick={isTopicColorField ? onToggleTopic : undefined}
      topicLabelToIdMap={isTopicColorField ? topicLabelToIdMap : undefined}
      combinedMutedIndices={combinedMutedIndices}
      hideFilteredPoints={hideFilteredPoints}
      mutedPointOpacity={mutedPointOpacity}
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
      {showLegend && !legendCollapsed && (
        <div className={cn(
          "absolute top-30 right-4 z-10 pointer-events-none",
          legendDragging && "select-none"
        )}>
          <Legend
            categoryField={colorByField}
            categoryValues={categoryValues}
            categoryCounts={categoryCounts}
            mutedCategories={effectiveMutedCategories}
            onCategoryToggle={handleCategoryToggle}
            onCategoryReset={handleCategoryReset}
            colorScale={colorScale}
            numericRange={numericRange}
            categoricalPalette={categoricalPalette}
            nestedColorMap={nestedColorMap}
            maxHeight={legendHeight}
            dragHandle={legendDragHandle}
          />
        </div>
      )}

      {/* 2b. LAYER: Collapsed Legend Pill (Z-10) */}
      {showLegend && legendCollapsed && (
        <div className="absolute top-30 right-4 z-10">
          <Button
            variant="circularghost"
            size="icon"
            onClick={() => { resetLegendHeight(); setLegendCollapsed(false); }}
            aria-label="Show legend"
          >
            <Palette className="h-4 w-4" />
          </Button>
        </div>
      )}


      {/* 3. LAYER: Table Overlay (Z-20) */}
      {showResultsTable && !resultsCollapsed && (
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
              className="pointer-events-auto"
              onResize={(size) => { if (size < 8) setResultsCollapsed(true); }}
            >
              <div className={cn("h-full w-full px-2 pb-4",
              isExpanded? "pr-4" : ""
              )}
              >
                {/* Added background/blur so text is readable over the plot points */}
                <div className="h-full border rounded-md shadow-lg overflow-hidden">
                  <SimilarItemsTable
                    results={semanticSearchResults}
                    queryLabel={searchQueryLabel}
                    categoryField={colorByField}
                    onClose={() => setResultsCollapsed(true)}
                  />
                </div>
              </div>
            </ResizablePanel>

          </ResizablePanelGroup>
        </div>
      )}

      {/* 3b. LAYER: Collapsed Results Chip (Z-20) */}
      {showResultsTable && resultsCollapsed && (
        <div className="absolute bottom-4 right-4 z-20">
          <Button
            variant="circularghost"
            size="icon"
            onClick={() => setResultsCollapsed(false)}
            aria-label="Show search results"
          >
            <Table2 className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* 4. LAYER: Sidebar Overlay (Z-40) */}
      <div className="absolute inset-y-0 left-0 z-40 pointer-events-none">
        {/* Controls Sidebar */}
        <EmbeddingSidebar
          embeddingDim={embeddingDim}
          metadata={metadata}
          selectedPoint={selectedPoint || null}
          colorFieldOptions={colorFieldOptions}
          availableFields={availableFields}
          nestedColorAvailable={nestedColorAvailable}
          variant="floating"
          className={cn(
            "pointer-events-auto absolute top-20 bottom-2 z-40 w-80 shadow-2xl transition-all duration-300 ease-in-out",
            activePanel === 'controls' ? "left-4" : "-left-[400px] opacity-0"
          )}
        />

        {/* Search Sidebar */}
        <SearchSidebar
          searchQuery={searchQuery ?? ''}
          onSearchChange={(value) => useVisualizationStore.getState().setSearchQuery(value)}
          showOnlyHighlighted={showOnlyHighlighted ?? false}
          onShowOnlyHighlightedChange={(checked) => useVisualizationStore.getState().setFlag('showOnlyHighlighted', checked)}
          showLabels={showLabels ?? false}
          onShowLabelsChange={(checked) => useVisualizationStore.getState().setFlag('showLabels', checked)}
          hasHighlights={Boolean(highlightedCount && highlightedCount > 0)}
          textSearchResults={textSearchResults}
          selectedPointId={selectedPoint?.id}
          onResultClick={onTextResultClick}
          categoryField={colorByField}
          queryPromptName={queryPromptName}
          onQueryPromptNameChange={onQueryPromptNameChange}
          topics={topics}
          topicSearchMode={topicSearchMode}
          onTopicSearchModeChange={onTopicSearchModeChange}
          topicDirectQuery={topicDirectQuery}
          onTopicDirectQueryChange={onTopicDirectQueryChange}
          topicFilteredTopics={topicFilteredTopics}
          topicSemanticQuery={topicSemanticQuery}
          onTopicSemanticQueryChange={onTopicSemanticQueryChange}
          onTopicSemanticSearch={onTopicSemanticSearch}
          topicSemanticResults={topicSemanticResults}
          topicSemanticLoading={topicSemanticLoading}
          selectedTopicIds={selectedTopicIds}
          onToggleTopic={onToggleTopic}
          onSelectAllTopics={onSelectAllTopics}
          onClearAllTopics={onClearAllTopics}
          categoricalPalette={categoricalPalette}
          textSearchLoading={textSearchLoading}
          availableFields={availableFields}
          variant="floating"
          className={cn(
            "pointer-events-auto absolute top-20 bottom-2 z-40 w-80 shadow-2xl transition-all duration-300 ease-in-out",
            activePanel === 'search' ? "left-4" : "-left-[400px] opacity-0"
          )}
        />

        {/* Analytics Sidebar */}
        <AnalyticsSidebar
          points={points}
          colorByField={colorByField}
          categoryValues={categoryValues}
          categoryCounts={categoryCounts}
          availableFields={availableFields}
          categoricalPalette={categoricalPalette}
          mutedCategories={effectiveMutedCategories}
          temporalRange={temporalRange}
          onTemporalRangeChange={handleTemporalRangeChange}
          textSearchHighlights={textSearchHighlights}
          colorFieldOptions={colorFieldOptions}
          variant="floating"
          className={cn(
            "pointer-events-auto absolute top-20 bottom-2 z-40 w-80 shadow-2xl transition-all duration-300 ease-in-out",
            activePanel === 'analytics' ? "left-4" : "-left-[400px] opacity-0"
          )}
        />
      </div>
    </div>
  );
}