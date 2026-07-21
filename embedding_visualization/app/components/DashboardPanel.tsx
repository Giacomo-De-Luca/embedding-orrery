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
import { cn } from '@/lib/utils/utils';
import { IS_DEMO, DEMO_DISABLED_MESSAGE } from '@/lib/utils/demoMode';
import { SAE_FEATURE_INDEX_FIELD } from '../../lib/utils/saeCollections';
import { useCategoryData } from '../../lib/hooks/useCategoryData';
import { useNestedCategoryData } from '../../lib/hooks/useNestedCategoryData';
import { useVerticalResize } from '../../lib/hooks/useVerticalResize';
import { useVisualizationStore } from '../../lib/stores/useVisualizationStore';
import { drawLegendOverlay } from '../../lib/utils/plotCapture';
import { downloadCanvasPng } from '../../lib/utils/downloadJson';
import { toast } from 'sonner';

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
      const val = p.metadata?.[colorByField];
      if (val != null && val !== '') catCounts[String(val)] = (catCounts[String(val)] ?? 0) + 1;
      if (isNested) {
        const t = p.metadata?.['topic_label'];
        const s = p.metadata?.['subtopic_label'];
        if (t != null && t !== '') topCounts[String(t)] = (topCounts[String(t)] ?? 0) + 1;
        if (s != null && s !== '') subCounts[String(s)] = (subCounts[String(s)] ?? 0) + 1;
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

  // Compute histogram bins for continuous scale legend
  const isLogScale = customNumericRange?.logScale === true;
  const histogramBins = useMemo(() => {
    if (!numericRange || !colorByField) return undefined;
    const { min, max } = numericRange;
    if (min === max) return undefined;
    const BIN_COUNT = 30;

    if (isLogScale) {
      // Log-space binning: equal-width bins in log10 space, edges in linear space
      const offset = min <= 0 ? Math.abs(min) + 1 : 0;
      const logMin = Math.log10(min + offset + 1e-10);
      const logMax = Math.log10(max + offset);
      const logBinWidth = (logMax - logMin) / BIN_COUNT;
      const counts = new Array(BIN_COUNT).fill(0);
      for (const p of points) {
        const val = p.metadata?.[colorByField];
        const num = typeof val === 'number' ? val : parseFloat(String(val));
        if (isNaN(num)) continue;
        const logVal = Math.log10(num + offset + 1e-10);
        const idx = Math.min(Math.floor((logVal - logMin) / logBinWidth), BIN_COUNT - 1);
        counts[idx]++;
      }
      return counts.map((count: number, i: number) => ({
        binStart: Math.pow(10, logMin + i * logBinWidth) - offset,
        binEnd: Math.pow(10, logMin + (i + 1) * logBinWidth) - offset,
        count,
      }));
    }

    // Linear binning
    const binWidth = (max - min) / BIN_COUNT;
    const counts = new Array(BIN_COUNT).fill(0);
    for (const p of points) {
      const val = p.metadata?.[colorByField];
      const num = typeof val === 'number' ? val : parseFloat(String(val));
      if (isNaN(num)) continue;
      const idx = Math.min(Math.floor((num - min) / binWidth), BIN_COUNT - 1);
      counts[idx]++;
    }
    return counts.map((count: number, i: number) => ({
      binStart: min + i * binWidth,
      binEnd: min + (i + 1) * binWidth,
      count,
    }));
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
    />
  );

  return (

    <div className="relative h-full w-full overflow-hidden">

      {/* 1. LAYER: Plot Background (Z-0) */}
      <div className="absolute inset-0 z-0" data-tour="plot">
        <div className="h-full w-full rounded-lg text-card-foreground shadow-sm">
          {plot}
        </div>
      </div>

      {/* 2. LAYER: Legend Overlay (Z-10) */}
      {showLegend && !legendCollapsed && (
        <div ref={legendOverlayRef} className={cn(
          "absolute top-30 right-4 z-10 pointer-events-none",
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

      {/* 3c. LAYER: SAE Context Menu (Z-50) — in demo builds the item is inert
          with an explanatory tooltip (/sae also redirects server-side) */}
      {contextMenu && saeInfo && (
        <div
          className="fixed z-50 min-w-40 rounded-md border bg-popover p-1 shadow-md"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          {IS_DEMO ? (
            <span
              aria-disabled="true"
              title={DEMO_DISABLED_MESSAGE}
              className="flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm cursor-not-allowed opacity-50"
            >
              <ExternalLink className="h-3.5 w-3.5" />
              View Feature #{contextMenu.featureIndex}
            </span>
          ) : (
            <Link
              href={`/sae?modelId=${encodeURIComponent(saeInfo.modelId)}&saeId=${encodeURIComponent(saeInfo.saeId)}&featureIndex=${contextMenu.featureIndex}`}
              className="flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent cursor-pointer"
              onClick={() => setContextMenu(null)}
            >
              <ExternalLink className="h-3.5 w-3.5" />
              View Feature #{contextMenu.featureIndex}
            </Link>
          )}
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
          collectionName={collectionName}
          hasActiveFilter={hasActiveFilter}
          hasHighlights={hasHighlights}
          variant="floating"
          data-tour="panel-controls"
          className={cn(
            "pointer-events-auto absolute top-20 bottom-2 z-40 w-80 shadow-2xl transition-all duration-300 ease-in-out",
            activePanel === 'controls' ? "left-4" : "-left-[400px] opacity-0"
          )}
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
          selectedTopicIds={selectedTopicIds}
          onToggleTopic={onToggleTopic}
          onSelectAllTopics={onSelectAllTopics}
          onClearAllTopics={onClearAllTopics}
          categoricalPalette={categoricalPalette}
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
          data-tour="panel-search"
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
          sharedFilteredCounts={filteredCategoryCounts}
          combinedMutedIndices={combinedMutedIndices}
          colorFieldOptions={colorFieldOptions}
          probes={probes}
          onCategoryToggle={handleCategoryToggle}
          variant="floating"
          data-tour="panel-analytics"
          className={cn(
            "pointer-events-auto absolute top-20 bottom-2 z-40 w-80 shadow-2xl transition-all duration-300 ease-in-out",
            activePanel === 'analytics' ? "left-4" : "-left-[400px] opacity-0"
          )}
        />
      </div>
    </div>
  );
}