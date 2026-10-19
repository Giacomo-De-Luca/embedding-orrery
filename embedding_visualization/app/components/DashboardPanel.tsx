'use client';

import { useCallback, useMemo, useState, useEffect, useDeferredValue, useRef } from 'react';
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
import { Table2, Palette, ExternalLink } from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/lib/ui-primitives/button';
import type { Point2D, Point3D, SemanticSearchResult, HighlightMap, TopicInfo, TemporalRange } from '../../lib/types/types';
import type { TopicSearchMode, TopicSearchResult } from '../../lib/hooks/useTopicSearch';
import type { ColorFieldOption } from '../../lib/utils/fieldAnalysis';
import type { UseDocumentFeatureSearchReturn } from '../../lib/hooks/useDocumentFeatureSearch';
import type { UseProbesReturn } from '../../lib/hooks/useProbes';
import type { CameraViewAdjustment } from '../../lib/utils/tourSteps';
import { cn } from '@/lib/utils/utils';
import { SAE_FEATURE_INDEX_FIELD } from '../../lib/utils/saeCollections';
import { useCategoryData } from '../../lib/hooks/useCategoryData';
import { useCategoryColorMap } from '../../lib/hooks/useCategoryColorMap';
import { useNestedCategoryData } from '../../lib/hooks/useNestedCategoryData';
import { useVerticalResize } from '../../lib/hooks/useVerticalResize';
import { MOBILE_BREAKPOINT } from '../../lib/hooks/use-mobile';
import { useVisualizationStore } from '../../lib/stores/useVisualizationStore';
import { drawLegendOverlay } from '../../lib/utils/plotCapture';
import { downloadCanvasPng } from '../../lib/utils/downloadJson';
import { toast } from 'sonner';

export type ActivePanel = 'controls' | 'search' | 'analytics' | null;

import { CategorySelection } from '@/lib/utils/categorySelection';
import { CategoryKey } from '@/lib/utils/categoryKey';
import { MutedIndices } from '@/lib/utils/mutedIndices';
import { explorePanelClass as panelClass } from '@/lib/utils/explorePanel';
import { NumericBins } from '@/lib/utils/numericBins';

/** Bars in the continuous-scale legend's histogram. */
const LEGEND_HISTOGRAM_BINS = 30;

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
  // Active collection (for saving its default colour scheme)
  collectionName?: string | null;
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
  itemMetadata?: Record<string, unknown>[];
  // SAE cross-link (when collection is an SAE embedding collection)
  saeInfo?: { modelId: string; saeId: string } | null;
  // SAE prompt highlight
  promptHighlightStatus?: 'idle' | 'loading_model' | 'running' | 'error';
  promptHighlightError?: string | null;
  promptHighlightActivePrompt?: string | null;
  onPromptHighlightSubmit?: (prompt: string) => void;
  onPromptHighlightClear?: () => void;
  promptHighlightResults?: SemanticSearchResult[] | null;
  promptMaxDensity?: number | null;
  onPromptMaxDensityChange?: (value: number | null) => void;
  // Feature search (document activations) — combobox multi-select
  featureSearch?: UseDocumentFeatureSearchReturn | null;
  onFeatureSearchResultClick?: (rowIndex: number) => void;
  // Embedding-space probes (threaded to AnalyticsSidebar)
  probes?: UseProbesReturn | null;
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
  topicSemanticResults?: TopicSearchResult[] | null;
  topicSemanticLoading?: boolean;
  topicSemanticError?: string | null;
  selectedTopicIds?: Set<number>;
  onToggleTopic?: (id: number) => void;
  onSelectAllTopics?: () => void;
  onClearAllTopics?: () => void;
  /** Bump to animate the 3D camera back to its default framing (demo tour). */
  cameraResetSignal?: number;
  /** Optional orbit/tilt/zoom/pan applied on top of the reset framing. */
  cameraResetView?: CameraViewAdjustment | null;
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
  collectionName,
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
  itemMetadata,
  // SAE cross-link
  saeInfo,
  // SAE prompt highlight
  promptHighlightStatus,
  promptHighlightError,
  promptHighlightActivePrompt,
  onPromptHighlightSubmit,
  onPromptHighlightClear,
  promptHighlightResults,
  promptMaxDensity,
  onPromptMaxDensityChange,
  // Feature search (document activations)
  featureSearch,
  onFeatureSearchResultClick,
  // Embedding-space probes
  probes,
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
  topicSemanticError,
  selectedTopicIds,
  onToggleTopic,
  onSelectAllTopics,
  onClearAllTopics,
  cameraResetSignal,
  cameraResetView,
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
  const showAxes = useVisualizationStore((s) => s.showAxes);
  const selectedDimensions = useVisualizationStore((s) => s.selectedDimensions);
  const showClusterLabels = useVisualizationStore((s) => s.showClusterLabels);
  const hideFilteredPoints = useVisualizationStore((s) => s.hideFilteredPoints);
  const mutedPointOpacity = useVisualizationStore((s) => s.mutedPointOpacity);
  const pointOpacity = useVisualizationStore((s) => s.pointOpacity);
  const customNumericRange = useVisualizationStore((s) => s.customNumericRange);
  const setCustomNumericRange = useVisualizationStore((s) => s.setCustomNumericRange);
  const setMutedCategories = useVisualizationStore((s) => s.setMutedCategories);
  const setTemporalRange = useVisualizationStore((s) => s.setTemporalRange);
  const setCategoryColorOverride = useVisualizationStore((s) => s.setCategoryColorOverride);
  const clearCategoryColorOverrides = useVisualizationStore((s) => s.clearCategoryColorOverrides);
  const colorOverrides = useVisualizationStore(
    (s) => colorByField ? s.categoryColorOverrides[colorByField] : undefined
  );

  const is2D = mode === '2d';

  // --- Context menu for SAE feature cross-link ---
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; featureIndex: number } | null>(null);

  const handlePointContextMenu = useCallback((point: Point2D | Point3D, event: MouseEvent) => {
    if (!saeInfo || point.metadata?.[SAE_FEATURE_INDEX_FIELD] == null) return;
    setContextMenu({
      x: event.clientX,
      y: event.clientY,
      featureIndex: Number(point.metadata[SAE_FEATURE_INDEX_FIELD]),
    });
  }, [saeInfo]);

  // Close context menu on click anywhere or escape
  useEffect(() => {
    if (!contextMenu) return;
    const close = () => setContextMenu(null);
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close(); };
    window.addEventListener('click', close);
    window.addEventListener('keydown', handleKey);
    return () => { window.removeEventListener('click', close); window.removeEventListener('keydown', handleKey); };
  }, [contextMenu]);

  // Compute category values and counts from points
  const points = is2D ? points2d : points3d;
  const { categoryValues, categoryCounts } = useCategoryData(points, colorByField);

  // Nested topic/subtopic color mapping
  const { available: nestedColorAvailable, nestedColorMap } = useNestedCategoryData(
    points, colorByField, nestedColorMode, categoricalPalette
  );

  // Topic-mode detection: when coloring by topic_label, selectedTopicIds drives everything
  const isTopicColorField = colorByField === 'topic_label';

  // Colour dots of the Search panel's topic rows: resolved exactly as the plots
  // and legend colour topic_label (same values, palette and overrides; nested
  // mode's topic hues when active), so each dot matches its topic's points —
  // also while the plot is coloured by another field.
  const { categoryValues: otherFieldTopicValues } = useCategoryData(
    points, isTopicColorField ? null : 'topic_label'
  );
  const flatTopicColorMap = useCategoryColorMap(
    'topic_label', isTopicColorField ? categoryValues : otherFieldTopicValues, categoricalPalette
  );
  const topicColorMap = nestedColorMap?.topicColors ?? flatTopicColorMap;

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
  // (topic selection mutes categories instead: effectiveMutedCategories)
  const combinedMutedIndices = useMemo(
    () => MutedIndices.union([temporallyMutedIndices, searchMutedIndices]),
    [temporallyMutedIndices, searchMutedIndices],
  );

  // Signals for the Controls sidebar: gate the filtered-point and highlight-dependent controls
  const hasActiveFilter = combinedMutedIndices !== null || effectiveMutedCategories.length > 0;
  const hasHighlights = Boolean(highlightedCount && highlightedCount > 0);

  // If highlights disappear while "Show only highlighted" is on, its (now hidden) checkbox
  // could no longer be unchecked and the plot would stay empty — clear the flag instead.
  // hideFilteredPoints deliberately gets no such reset: a stale `true` only re-applies once
  // a new filter mutes points again (desirable continuity), never empties the plot.
  useEffect(() => {
    if (!hasHighlights && showOnlyHighlighted) {
      useVisualizationStore.getState().setFlag('showOnlyHighlighted', false);
    }
  }, [hasHighlights, showOnlyHighlighted]);

  // Compute per-category filtered counts (points surviving all filters) in a single pass.
  // Shared between Legend and AnalyticsSidebar to avoid duplicate iteration.
  const { filteredCategoryCounts, filteredTopicCounts, filteredSubtopicCounts } = useMemo(() => {
    if (!combinedMutedIndices || !colorByField) {
      return { filteredCategoryCounts: null, filteredTopicCounts: null, filteredSubtopicCounts: null };
    }
    const catCounts: Record<string, number> = {};
    const topCounts: Record<string, number> = {};
    const subCounts: Record<string, number> = {};
    const isNested = !!nestedColorMap;
    for (const p of points) {
      if (combinedMutedIndices.has(p.index)) continue;
      const key = CategoryKey.of(p.metadata?.[colorByField]);
      catCounts[key] = (catCounts[key] ?? 0) + 1;
      if (isNested) {
        const { topic, subtopic } = CategoryKey.nested(p.metadata);
        topCounts[topic] = (topCounts[topic] ?? 0) + 1;
        subCounts[subtopic] = (subCounts[subtopic] ?? 0) + 1;
      }
    }
    return {
      filteredCategoryCounts: catCounts,
      filteredTopicCounts: isNested ? topCounts : null,
      filteredSubtopicCounts: isNested ? subCounts : null,
    };
  }, [combinedMutedIndices, colorByField, points, nestedColorMap]);

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

  // The plots draw category traces (where the Missing group can be faded) unless a
  // continuous scale has a numeric range to map, matching their numericData rule
  const drawsCategoryTraces = !numericRange || numericRange.min === numericRange.max;

  // Histogram of the colour field under the continuous-scale legend (equal-width bins,
  // in log10 space when the scale is logarithmic).
  const isLogScale = customNumericRange?.logScale === true;
  const histogramBins = useMemo(() => {
    if (!numericRange || !colorByField) return undefined;
    const { min, max } = numericRange;
    if (min === max) return undefined;
    return NumericBins
      .histogramBins(min, max, LEGEND_HISTOGRAM_BINS, isLogScale ? 'log' : 'linear')
      .toHistogram(points.map(p => p.metadata?.[colorByField]));
  }, [numericRange, colorByField, points, isLogScale]);

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
      if (topicId !== undefined) {
        onToggleTopic(topicId);
        return;
      }
      // Not a topic (e.g. the Missing entry): drop the topic selection so the category muting below applies
      if (selectedTopicIds?.size) onClearAllTopics?.();
    }

    setMutedCategories(CategorySelection.toggle(category, shiftKey, mutedCategories ?? [], categoryValues, nestedColorMap));
  }, [isTopicColorField, topicLabelToIdMap, onToggleTopic, selectedTopicIds, onClearAllTopics, nestedColorMap, mutedCategories, setMutedCategories, categoryValues]);

  // Temporal range change handler
  const handleTemporalRangeChange = useCallback((range: TemporalRange | null) => {
    setTemporalRange(range);
  }, [setTemporalRange]);

  // Double-click reset: show all categories
  const handleCategoryReset = useCallback(() => {
    if (isTopicColorField && onClearAllTopics) onClearAllTopics();
    // Topic mode can also hold category muting from non-topic entries (Missing)
    setMutedCategories([]);
  }, [isTopicColorField, onClearAllTopics, setMutedCategories]);

  const handleColorOverride = useCallback((category: string, color: string) => {
    if (colorByField) setCategoryColorOverride(colorByField, category, color);
  }, [colorByField, setCategoryColorOverride]);

  const handleColorOverrideClear = useCallback(() => {
    if (colorByField) clearCategoryColorOverrides(colorByField);
  }, [colorByField, clearCategoryColorOverrides]);

  // Show legend for categorical scales with values, or continuous scales with numeric range
  const showLegend = colorByField && (
    (categoryValues.length > 0 && !isContinuousScale) ||
    (isContinuousScale && numericRange !== undefined)
  );
  const hasSemanticResults = semanticSearchResults && semanticSearchResults.length > 0;
  const hasPromptResults = promptHighlightResults && promptHighlightResults.length > 0;
  const showResultsTable = hasSemanticResults || hasPromptResults;
  const tableResults = hasPromptResults ? promptHighlightResults : semanticSearchResults;
  const tableQueryLabel = hasPromptResults ? promptHighlightActivePrompt : searchQueryLabel;

  // Collapse/expand state for the results panel
  const [resultsCollapsed, setResultsCollapsed] = useState(false);

  // Auto-expand when new results arrive
  useEffect(() => {
    if (showResultsTable) setResultsCollapsed(false);
  }, [showResultsTable]);

  // Collapse/expand state for the legend
  const [legendCollapsed, setLegendCollapsed] = useState(false);

  // On phone-width viewports the legend card covers most of the plot, so it
  // starts collapsed to its pill. Latched once at mount rather than bound to
  // the live width: once the user expands it, a rotation or resize must not
  // fold it away again. Runs in an effect (not a lazy initializer) so the SSR
  // markup and the first client render agree — the card shows for one frame.
  useEffect(() => {
    if (window.innerWidth < MOBILE_BREAKPOINT) setLegendCollapsed(true);
  }, []);

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
      data-export-exclude
      className="h-3 w-full cursor-ns-resize flex items-center justify-center group pointer-events-auto"
    >
      <div className="h-0.5 w-8 rounded-full bg-border group-hover:bg-foreground/30 transition-colors" />
    </div>
  );

  // Modebar screenshot: the plot component delivers its composited canvas;
  // rasterize the live legend DOM onto it at its on-screen position, then download.
  const legendOverlayRef = useRef<HTMLDivElement>(null);
  const handleScreenshot = useCallback(async (plotCanvas: HTMLCanvasElement) => {
    try {
      if (legendOverlayRef.current) {
        await drawLegendOverlay(plotCanvas, legendOverlayRef.current);
      }
    } catch (err) {
      console.error('Legend capture failed:', err);
      toast.error('Legend capture failed — exported the plot only');
    }
    downloadCanvasPng(plotCanvas, `${collectionName ?? 'embedding'}-${mode}.png`);
  }, [collectionName, mode]);

  const plot = is2D ? (
    <ScatterPlot2D
      points={points2d}
      categoryField={colorByField}
      categoryValues={categoryValues}
      colorScale={colorScale}
      highlightedIndices={highlightedIndices}
      selectedPoint={selectedPoint as Point2D | null}
      onPointClick={onPointClick}
      onPointContextMenu={saeInfo ? handlePointContextMenu : undefined}
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
      pointOpacity={pointOpacity}
      showClusterLabels={showClusterLabels}
      onClusterLabelClick={isTopicColorField ? onToggleTopic : undefined}
      topicLabelToIdMap={isTopicColorField ? topicLabelToIdMap : undefined}
      customNumericRange={customNumericRange}
      showAxes={showAxes}
      onScreenshot={handleScreenshot}
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
      pointOpacity={pointOpacity}
      customNumericRange={customNumericRange}
      onPointContextMenu={saeInfo ? handlePointContextMenu : undefined}
      showAxes={showAxes}
      selectedDimensions={selectedDimensions}
      onScreenshot={handleScreenshot}
      cameraResetSignal={cameraResetSignal}
      cameraResetView={cameraResetView}
    />
  );

  return (

    // Floating chrome (modebar, legend, panels, corner chips) hangs off the
    // header geometry published as --explore-* in globals.css.
    <div className="relative h-full w-full overflow-hidden">

      {/* 1. LAYER: Plot Background (Z-0) */}
      <div className="absolute inset-0 z-0" data-tour="plot">
        <div className="h-full w-full rounded-lg text-card-foreground shadow-sm">
          {plot}
        </div>
        {/* Invisible tour anchor: whole-plot steps hang their card off the
            plot's right edge (placement "left") so it never covers the view
            it narrates. Sits at 62% height, not center — the legend occupies
            the top-right (under the header + its resizable height), and a vertically
            centered card overlapped it; this parks cards just below. Lives
            inside the plot layer so it unmounts with it — joyride's
            target-wait then covers collection loads. */}
        <div
          data-tour="plot-side"
          aria-hidden
          className="pointer-events-none absolute right-6 top-[62%] h-px w-px"
        />
      </div>

      {/* 2. LAYER: Legend Overlay (Z-10) — sits under the modebar, both
          below the header (--explore-legend-top in globals.css). */}
      {showLegend && !legendCollapsed && (
        <div ref={legendOverlayRef} className={cn(
          "absolute right-4 top-(--explore-legend-top) z-10 pointer-events-none",
          legendDragging && "select-none"
        )}>
          <Legend
            categoryField={colorByField}
            categoryValues={categoryValues}
            categoryCounts={categoryCounts}
            filteredCounts={filteredCategoryCounts}
            filteredTopicCounts={filteredTopicCounts}
            filteredSubtopicCounts={filteredSubtopicCounts}
            mutedCategories={effectiveMutedCategories}
            onCategoryToggle={handleCategoryToggle}
            onCategoryReset={handleCategoryReset}
            colorScale={colorScale}
            numericRange={numericRange}
            histogramBins={histogramBins}
            customNumericRange={customNumericRange}
            onCustomRangeChange={setCustomNumericRange}
            categoricalPalette={categoricalPalette}
            nestedColorMap={nestedColorMap}
            maxHeight={legendHeight}
            dragHandle={legendDragHandle}
            onColorOverride={handleColorOverride}
            onColorOverrideClear={handleColorOverrideClear}
            categoryColorOverrides={colorOverrides}
          />
        </div>
      )}

      {/* 2b. LAYER: Collapsed Legend Pill (Z-10) */}
      {showLegend && legendCollapsed && (
        <div className="absolute right-4 top-(--explore-legend-top) z-10">
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
              <div className={cn("h-full w-full px-2 pb-(--explore-corner-inset)",
              isExpanded? "pr-4" : ""
              )}
              >
                {/* Added background/blur so text is readable over the plot points */}
                <div className="h-full border rounded-md shadow-lg overflow-hidden">
                  <SimilarItemsTable
                    results={tableResults ?? null}
                    queryLabel={tableQueryLabel ?? null}
                    categoryField={colorByField}
                    onClose={() => setResultsCollapsed(true)}
                    isActivationResults={!!hasPromptResults}
                  />
                </div>
              </div>
            </ResizablePanel>

          </ResizablePanelGroup>
        </div>
      )}

      {/* 3b. LAYER: Collapsed Results Chip (Z-20) — lifted above the HF
          Space's corner button on embedded phones (--explore-corner-inset). */}
      {showResultsTable && resultsCollapsed && (
        <div className="absolute bottom-(--explore-corner-inset) right-4 z-20">
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

      {/* 3c. LAYER: SAE Context Menu (Z-50) — live in every build; the parent
          passes saeInfo only for genuine feature collections, and the SAE
          explorer's read paths are served in the demo too. */}
      {contextMenu && saeInfo && (
        <div
          className="fixed z-50 min-w-40 rounded-md border bg-popover p-1 shadow-md"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <Link
            href={`/sae?modelId=${encodeURIComponent(saeInfo.modelId)}&saeId=${encodeURIComponent(saeInfo.saeId)}&featureIndex=${contextMenu.featureIndex}`}
            className="flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent cursor-pointer"
            onClick={() => setContextMenu(null)}
          >
            <ExternalLink className="h-3.5 w-3.5" />
            View Feature #{contextMenu.featureIndex}
          </Link>
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
          hasMissingValues={categoryValues.includes(CategoryKey.MISSING)}
          canFadeMissing={drawsCategoryTraces}
          collectionName={collectionName}
          hasActiveFilter={hasActiveFilter}
          hasHighlights={hasHighlights}
          variant="floating"
          mobileSheet={false}
          data-tour="panel-controls"
          className={panelClass(activePanel === 'controls')}
        />

        {/* Search Sidebar */}
        <SearchSidebar
          searchQuery={searchQuery ?? ''}
          onSearchChange={(value) => useVisualizationStore.getState().setSearchQuery(value)}
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
          topicSemanticError={topicSemanticError}
          selectedTopicIds={selectedTopicIds}
          onToggleTopic={onToggleTopic}
          onSelectAllTopics={onSelectAllTopics}
          onClearAllTopics={onClearAllTopics}
          topicColorMap={topicColorMap}
          textSearchLoading={textSearchLoading}
          availableFields={availableFields}
          itemMetadata={itemMetadata}
          saeInfo={saeInfo}
          promptHighlightStatus={promptHighlightStatus}
          promptHighlightError={promptHighlightError}
          promptHighlightActivePrompt={promptHighlightActivePrompt}
          onPromptHighlightSubmit={onPromptHighlightSubmit}
          onPromptHighlightClear={onPromptHighlightClear}
          promptMaxDensity={promptMaxDensity}
          onPromptMaxDensityChange={onPromptMaxDensityChange}
          featureSearch={featureSearch}
          onFeatureSearchResultClick={onFeatureSearchResultClick}
          variant="floating"
          mobileSheet={false}
          data-tour="panel-search"
          className={panelClass(activePanel === 'search')}
        />

        {/* Analytics Sidebar */}
        <AnalyticsSidebar
          points={points}
          collectionName={collectionName}
          colorByField={colorByField}
          categoryValues={categoryValues}
          categoryCounts={categoryCounts}
          availableFields={availableFields}
          categoricalPalette={categoricalPalette}
          mutedCategories={effectiveMutedCategories}
          temporalRange={temporalRange}
          onTemporalRangeChange={handleTemporalRangeChange}
          sharedFilteredCounts={filteredCategoryCounts}
          combinedMutedIndices={combinedMutedIndices}
          colorFieldOptions={colorFieldOptions}
          probes={probes}
          onCategoryToggle={handleCategoryToggle}
          variant="floating"
          mobileSheet={false}
          data-tour="panel-analytics"
          className={panelClass(activePanel === 'analytics')}
        />
      </div>
    </div>
  );
}