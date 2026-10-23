'use client';

import React, { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { useTheme } from 'next-themes';
import type { PlotParams } from 'react-plotly.js';
import type {
  PlotData,
  Layout,
  Config,
  PlotMouseEvent,
  PlotHoverEvent,
  PlotRelayoutEvent,
} from 'plotly.js';
import type { Point2D, HighlightMap, NestedColorMap, ColorScale, CustomNumericRange } from '../../lib/types/types';
import { getCategoryLabel, getSequentialScale, getDivergingScale, getMonochromeScale, desaturateHex, type SequentialScaleName, type DivergingScaleName } from '../../lib/utils/categoryColors';
import { CategoryKey } from '../../lib/utils/categoryKey';
import { HiddenPoints } from '../../lib/utils/hiddenPoints';
import { isCrameriScale, getCrameriPlotlyScale } from '../../lib/colorMaps/crameriScales';
import { useColorScaleReady } from '../../lib/hooks/useColorScaleReady';
import { calculateMarkerStyle, calculateHighlightScale, calculateSimilarityColors } from '../../lib/utils/plotUtils';
import { useContainerDimensions } from '../../lib/hooks/useContainerDimensions';
import { useZoomLimit } from '../../lib/hooks/useZoomLimit';
import { useLatestRef } from '../../lib/hooks/useLatestRef';
import { usePinchZoom } from '../../lib/hooks/usePinchZoom';
import { CartesianPinchZoom } from '../../lib/utils/plotPinchZoom';
import { CoalescedRedraw } from '../../lib/utils/coalescedRedraw';
import { ZoomMarkerStyle, type OpacityTrace } from '../../lib/utils/zoomMarkerStyle';
import { formatHoverText } from '../utils/rendeding';
import { groupPointsByCluster, type ClusterData } from '../../lib/utils/clusterGeometry';
import { useVisualizationStore } from '../../lib/stores/useVisualizationStore';
import { useCategoryColorMap } from '../../lib/hooks/useCategoryColorMap';
import {
  computeClusterLabelPlacements,
  computeCurrentScale,
  plotAreaFromFullLayout,
  projectDataToScreen,
  type AxisRanges,
  type ClusterLabelPlacement,
} from '../utils/labelPlacement2D';
import { useDensityOverlay } from '../../lib/hooks/useDensityOverlay';
import { useDensityClusterLabels } from '../../lib/hooks/useDensityClusterLabels';
import { assignCategoryChannels, type ActiveCategory } from '../../lib/utils/densityRenderer/densityMath';

import { FrostedTooltip, type TooltipData } from './FrostedTooltip';
import { build2DModeBarButtons } from '../../lib/utils/plotlyIcons';
import { capture2DPlot } from '../../lib/utils/plotCapture';
import { plotlyLoader } from '../../lib/utils/plotlyLoader';
import { toast } from 'sonner';

// `meta` is a Plotly trace attribute that @types/plotly.js lacks
type PlotlyData = Partial<PlotData> & { meta?: string };
// Fork-only config options: forked/plotly_changes.md change 6
type MarkerScale = (trace: OpacityTrace, xRange: number[], yRange: number[]) => number;
type ForkConfig = Partial<Config> & {
  scatterglOpacityScale?: MarkerScale;
  scatterglSizeScale?: MarkerScale;
};

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic<PlotParams>(() => import('react-plotly.js'), { ssr: false });

interface ScatterPlot2DProps {
  points: Point2D[];
  categoryField?: string | null;  // Field to color by (used for both categorical AND numeric)
  categoryValues?: string[];
  colorScale?: ColorScale;
  highlightedIndices?: HighlightMap;
  selectedPoint?: Point2D | null;
  onPointClick?: (point: Point2D) => void;
  /** Optional className for the container */
  className?: string;
  /** When true, only show highlighted points (hide non-highlighted) */
  showOnlyHighlighted?: boolean;
  /** When true, show text labels on highlighted points */
  showLabels?: boolean;
  /** Categories to gray out (muted) in the visualization */
  mutedCategories?: string[];
  /** Extra metadata fields to show in hover tooltip */
  tooltipFields?: string[];
  /** When true, hide points with topic_id = -1 (unclustered/noise) */
  hideUnclustered?: boolean;
  /** Crameri categorical palette name for category coloring */
  categoricalPalette?: string;
  /** Nested topic/subtopic color map for hierarchical coloring */
  nestedColorMap?: NestedColorMap | null;
  /** Combined indices to mute (temporal + text search) */
  combinedMutedIndices?: Set<number> | null;
  /** Remove muted points entirely instead of graying out */
  hideFilteredPoints?: boolean;
  /** 0-1 multiplier applied to the base opacity for muted points (default 0.20) */
  mutedPointOpacity?: number;
  /** 0-1 multiplier applied to the base opacity for normal points (default 1.0) */
  pointOpacity?: number;
  /** Show topic/subtopic names at cluster centroids */
  showClusterLabels?: boolean;
  /** Callback when a cluster label is clicked (topic toggle) */
  onClusterLabelClick?: (topicId: number) => void;
  /** Map from topic label string → topic ID for click handling */
  topicLabelToIdMap?: Map<string, number> | null;
  /**
   * Cluster-label clicks without a topic id (e.g. a topic-tree preview's
   * groups): toggle the category; returns whether the label was one.
   */
  onClusterCategoryClick?: (category: string, shiftKey: boolean) => boolean;
  /** Custom numeric range overrides for cmin/cmax */
  customNumericRange?: CustomNumericRange | null;
  /** Callback when a point is right-clicked (contextmenu) */
  onPointContextMenu?: (point: Point2D, event: MouseEvent) => void;
  /** Show axis lines, grid, and tick labels */
  showAxes?: boolean;
  /** Called with the composited plot canvas when the modebar screenshot button is clicked */
  onScreenshot?: (canvas: HTMLCanvasElement) => void;
}


export const ScatterPlot2D = React.memo(function ScatterPlot2D({
  points,
  categoryField = null,
  categoryValues = [],
  colorScale = { type: 'categorical' },
  highlightedIndices,
  selectedPoint,
  onPointClick,
  className,
  showOnlyHighlighted = false,
  showLabels = false,
  mutedCategories = [],
  tooltipFields,
  hideUnclustered = false,
  categoricalPalette,
  nestedColorMap,
  combinedMutedIndices,
  hideFilteredPoints = false,
  mutedPointOpacity,
  pointOpacity,
  showClusterLabels = false,
  onClusterLabelClick,
  topicLabelToIdMap,
  onClusterCategoryClick,
  customNumericRange,
  onPointContextMenu,
  showAxes = false,
  onScreenshot,
}: ScatterPlot2DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphDivRef = useRef<any>(null);
  const hoveredPointRef = useRef<Point2D | null>(null);
  const { width, height } = useContainerDimensions(containerRef, { width: 800, height: 600 });

  // Deferred selected point: only sync to traces when highlightedIndices changes,
  // keeping plotData stable and avoiding expensive Plotly.react on point click
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const renderedSelectedPoint = useMemo(() => selectedPoint, [highlightedIndices]);

  // Data bounds for zoom-out limit
  const bounds = useMemo(() => {
    if (points.length === 0) return null;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    return { minX, maxX, minY, maxY };
  }, [points]);

  // Block zoom-out when visible axes exceed 2x the data extent
  const MAX_RANGE_MULTIPLIER = 2.0;
  const isAtZoomOutLimit2D = useCallback(() => {
    const gd = graphDivRef.current;
    if (!gd?._fullLayout || !bounds) return false;
    const xRange = gd._fullLayout.xaxis?.range;
    const yRange = gd._fullLayout.yaxis?.range;
    if (!xRange || !yRange) return false;
    const visibleX = Math.abs(xRange[1] - xRange[0]);
    const visibleY = Math.abs(yRange[1] - yRange[0]);
    const dataX = (bounds.maxX - bounds.minX) || 1;
    const dataY = (bounds.maxY - bounds.minY) || 1;
    return visibleX >= dataX * MAX_RANGE_MULTIPLIER || visibleY >= dataY * MAX_RANGE_MULTIPLIER;
  }, [bounds]);
  useZoomLimit(containerRef, isAtZoomOutLimit2D);

  // Two-finger pinch replays as wheel zoom, so the zoom-out limit above covers it too
  const [pinchZoom] = useState(() => new CartesianPinchZoom(graphDivRef));
  usePinchZoom(containerRef, pinchZoom);

  // Points grow larger and more opaque as you zoom in. The fork's scatterglSizeScale and
  // scatterglOpacityScale config options apply it before every draw, and toImage copies
  // them into screenshots. Read through a ref because the config memo must stay stable:
  // a config change replots. useLatestRef's layout effect is early enough: react-plotly
  // runs Plotly.react in a promise continuation.
  const zoomMarkerStyle = useMemo(() => new ZoomMarkerStyle(points.length, bounds), [points.length, bounds]);
  const zoomMarkerStyleRef = useLatestRef(zoomMarkerStyle);

  // --- Cluster labels: canvas overlay refs and state ---
  const labelCanvasRef = useRef<HTMLCanvasElement>(null);
  const clusterPlacementsRef = useRef<ClusterLabelPlacement[]>([]);
  const initialRangesRef = useRef<AxisRanges | null>(null);
  const currentRangesRef = useRef<AxisRanges | null>(null);
  // Store drawn label bounding boxes for click hit-testing
  const drawnLabelBoxesRef = useRef<{ loX: number; loY: number; hiX: number; hiY: number; label: string | null }[]>([]);

  const { resolvedTheme } = useTheme();
  const theme = resolvedTheme ?? 'light';
  const isDark = theme === 'dark';

  // Load Plotly lib reference through the shared loader (the Explore page
  // preloads it; react-plotly.js requires the same module, so this is a cache hit)
  const plotlyLibRef = useRef<any>(null);
  const [plotlyReady, setPlotlyReady] = useState(false);
  useEffect(() => {
    plotlyLoader.load().then((lib) => { plotlyLibRef.current = lib; setPlotlyReady(true); });
  }, []);

  // Modebar screenshot: routed through a ref (reassigned every render) because
  // the config memo only rebuilds on [plotlyReady], so a plain closure would go stale.
  const screenshotHandlerRef = useRef<((gd: any) => void) | null>(null);
  screenshotHandlerRef.current = (gd: any) => {
    if (!onScreenshot || !containerRef.current || !plotlyLibRef.current) return;
    capture2DPlot({
      gd: gd ?? graphDivRef.current,
      plotlyLib: plotlyLibRef.current,
      container: containerRef.current,
      densityCanvas: densityCanvasRef.current,
      labelCanvas: labelCanvasRef.current,
      isDark,
    })
      .then(onScreenshot)
      .catch((err) => {
        console.error('Screenshot failed:', err);
        toast.error('Screenshot failed');
      });
  };

  const axisColor = isDark ? '#e2e8f0' : '#0f172a';
  const gridColor = isDark ? '#334155' : '#e5e7eb';
  const plotBg = isDark ? 'rgba(0,0,0,0)' : '#ffffff';
  const paperBg = isDark ? 'rgba(0,0,0,0)' : '#ffffff';
  const legendBg = isDark ? 'rgba(2,6,23,0.85)' : 'rgba(255,255,255,0.85)';

  // Build color map based on category values (with custom overrides / aliases)
  const colorMap = useCategoryColorMap(categoryField, categoryValues, categoricalPalette);

  const densityMode = useVisualizationStore((s) => s.densityMode);
  const densityIntensity = useVisualizationStore((s) => s.densityIntensity);
  const hideMissing = useVisualizationStore((s) => s.hideMissing);
  const missingPointOpacity = useVisualizationStore((s) => s.missingPointOpacity);
  const hiddenOptions = useMemo(() => ({ hideUnclustered, hideMissing }), [hideUnclustered, hideMissing]);

  // --- Visibly active points: shared input for cluster labels and the density
  // overlay, so both track the same filters (mirrors the 3D clusterDataMap).
  const visiblePoints = useMemo(() => {
    if (!showClusterLabels && !densityMode) return [] as Point2D[];

    let displayPoints = HiddenPoints.filter(points, categoryField, hiddenOptions);

    if (categoryField && mutedCategories.length > 0) {
      const mutedSet = new Set(mutedCategories);
      displayPoints = displayPoints.filter(p => !mutedSet.has(CategoryKey.of(p.metadata?.[categoryField])));
    }

    // Exclude points muted by text search / temporal filters.
    if (combinedMutedIndices && combinedMutedIndices.size > 0) {
      displayPoints = displayPoints.filter(p => !combinedMutedIndices.has(p.index));
    }

    return displayPoints;
  }, [showClusterLabels, densityMode, points, categoryField, hiddenOptions, mutedCategories, combinedMutedIndices]);

  // --- Cluster label data (shared with 3D pattern via groupPointsByCluster) ---
  const clusterDataMap = useMemo(() => {
    if (!showClusterLabels || !categoryField) return new Map<string, ClusterData>();
    // groupPointsByCluster works with Point3D but Point2D is compatible (z defaults via centroid calc)
    return groupPointsByCluster(visiblePoints as any, categoryField, colorMap, nestedColorMap);
  }, [showClusterLabels, visiblePoints, categoryField, colorMap, nestedColorMap]);

  // --- Density overlay: category → RGBA channel assignment (≤4 active
  // categorical categories get per-category colors, else muted tint) ---
  const channelAssignment = useMemo(() => {
    const isCategoricalField =
      colorScale.type === 'categorical' && categoryField != null && categoryValues.length > 0;
    let active: ActiveCategory[] = [];
    if (densityMode && isCategoricalField && categoryField) {
      const counts = new Map<string, number>();
      for (const p of visiblePoints) {
        const cat = CategoryKey.of(p.metadata?.[categoryField]);
        counts.set(cat, (counts.get(cat) ?? 0) + 1);
      }
      active = Array.from(counts, ([name, count]) => ({
        name,
        count,
        color: (nestedColorMap?.topicColors[name] ?? colorMap[name]) || '#7f7f7f',
      }));
    }
    return assignCategoryChannels(active, isCategoricalField, isDark);
  }, [densityMode, visiblePoints, categoryField, categoryValues.length, colorScale.type, colorMap, nestedColorMap, isDark]);

  const densityCanvasRef = useRef<HTMLCanvasElement>(null);
  const [plotReady, setPlotReady] = useState(false);

  // Initial (full-extent) axis ranges: data bounds + Plotly's ~5% auto-pad.
  const initialRanges = useMemo((): AxisRanges | null => {
    if (!bounds) return null;
    const padX = (bounds.maxX - bounds.minX) * 0.05 || 0.5;
    const padY = (bounds.maxY - bounds.minY) * 0.05 || 0.5;
    return {
      xRange: [bounds.minX - padX, bounds.maxX + padX],
      yRange: [bounds.minY - padY, bounds.maxY + padY],
    };
  }, [bounds]);

  const { scheduleRender: scheduleDensityRender, supported: densitySupported } = useDensityOverlay({
    enabled: densityMode && plotReady,
    canvasRef: densityCanvasRef,
    graphDivRef,
    visiblePoints,
    categoryField,
    channelAssignment,
    initialRanges,
    intensity: densityIntensity,
    isDark,
    width,
    height,
  });

  // Density-blob cluster labels (hybrid text): only attempted in density mode;
  // null falls back to the category-centroid placements below.
  const hasTopicLabels = useMemo(
    () => points.some(p => p.metadata?.['topic_label'] != null),
    [points],
  );
  const densityPlacements = useDensityClusterLabels({
    enabled: densityMode && showClusterLabels && plotReady,
    visiblePoints,
    labelField: colorScale.type === 'categorical' && categoryField
      ? categoryField
      : hasTopicLabels ? 'topic_label' : null,
    colorMap,
    nestedColorMap,
    isDark,
    initialRanges,
    graphDivRef,
    width,
    height,
  });

  // --- Cluster label font constant ---
  const CLUSTER_FONT = 'bold 13px Geist Mono, monospace';
  const CLUSTER_FONT_SIZE = 13;

  // --- Recompute Apple placements when clusters or layout change ---
  useEffect(() => {
    if (!showClusterLabels || !initialRanges) {
      clusterPlacementsRef.current = [];
      return;
    }
    initialRangesRef.current = initialRanges;

    // Density-blob labels take over when the pipeline produced them.
    if (densityPlacements && densityPlacements.length > 0) {
      clusterPlacementsRef.current = densityPlacements;
      return;
    }

    if (clusterDataMap.size === 0) {
      clusterPlacementsRef.current = [];
      return;
    }

    // Use a temporary canvas for text measurement
    const canvas = labelCanvasRef.current ?? document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const plotArea = plotAreaFromFullLayout(graphDivRef.current?._fullLayout, width, height);

    clusterPlacementsRef.current = computeClusterLabelPlacements(
      clusterDataMap,
      ctx,
      CLUSTER_FONT,
      initialRanges,
      plotArea,
    );
  }, [showClusterLabels, clusterDataMap, densityPlacements, initialRanges, width, height]);

  // --- Render cluster labels on the canvas overlay ---
  const renderClusterLabels = useCallback(() => {
    const canvas = labelCanvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = width;
    const cssH = height;

    // Resize canvas backing store
    const bw = Math.round(cssW * dpr);
    const bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    // Clear hit-test boxes
    drawnLabelBoxesRef.current = [];

    const placements = clusterPlacementsRef.current;
    const initRanges = initialRangesRef.current;
    if (!placements.length || !initRanges) return;

    // Get current axis ranges from Plotly's internal layout
    const gd = graphDivRef.current;
    const xRange = gd?._fullLayout?.xaxis?.range;
    const yRange = gd?._fullLayout?.yaxis?.range;
    if (!xRange || !yRange) return;

    const curRanges: AxisRanges = {
      xRange: [xRange[0], xRange[1]],
      yRange: [yRange[0], yRange[1]],
    };
    currentRangesRef.current = curRanges;

    const scale = computeCurrentScale(curRanges, initRanges);

    // Get the plot area from Plotly's internal layout for accurate coordinate mapping
    const plotArea = plotAreaFromFullLayout(gd?._fullLayout, cssW, cssH);

    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';

    const boxes: { loX: number; loY: number; hiX: number; hiY: number; label: string | null }[] = [];

    for (const cl of placements) {
      // Check Apple visibility: placement must exist and current scale must be in range
      if (!cl.placement) continue;
      if (scale < cl.placement.minScale || scale > cl.placement.maxScale) continue;

      // Project data centroid to current screen position
      const screen = projectDataToScreen(cl.dataX, cl.dataY, curRanges, plotArea);

      // Clip to plot area
      if (screen.x < plotArea.left - 20 || screen.x > plotArea.left + plotArea.width + 20) continue;
      if (screen.y < plotArea.top - 20 || screen.y > plotArea.top + plotArea.height + 20) continue;

      // Density labels carry per-level font sizes; topic-centroid labels use the default
      const fontSize = cl.fontSize ?? CLUSTER_FONT_SIZE;
      ctx.font = fontSize === CLUSTER_FONT_SIZE ? CLUSTER_FONT : `bold ${fontSize}px Geist Mono, monospace`;

      const textWidth = ctx.measureText(cl.label).width;
      const pad = 4;
      const box = {
        loX: screen.x - textWidth / 2 - pad,
        loY: screen.y - fontSize * 0.6 - pad,
        hiX: screen.x + textWidth / 2 + pad,
        hiY: screen.y + fontSize * 0.6 + pad,
        // Explicit null disables topic clicks for other fields and summaries,
        // even when their text happens to match a topic name.
        label: cl.topicLabel === undefined ? cl.label : cl.topicLabel,
      };

      // Draw label with stroke outline for readability (same style as 3D)
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = isDark ? 'rgba(15, 23, 42, 0.85)' : 'rgba(255, 255, 255, 0.6)';
      ctx.lineWidth = isDark ? 4 : 3;
      ctx.lineJoin = 'round';
      ctx.strokeText(cl.label, screen.x, screen.y);
      ctx.fillStyle = desaturateHex(cl.color, isDark ? 0.3 : 0.45, isDark);
      ctx.fillText(cl.label, screen.x, screen.y);

      boxes.push(box);
    }

    drawnLabelBoxesRef.current = boxes;
    ctx.globalAlpha = 1.0;
  }, [width, height, isDark]);

  const labelsActive = showClusterLabels &&
    (clusterDataMap.size > 0 || (densityPlacements?.length ?? 0) > 0);

  // --- Handle Plotly relayout (zoom/pan) to re-render the overlays.
  // `relayouting` fires continuously during drag/scroll so the density map and
  // labels track the gesture live; `relayout` fires once at the end. A pinch
  // frame emits several, so both overlays coalesce them into one redraw.
  // The handler stays stable (no rebinding) and reads the label state through a
  // ref, which the queued redraw also reads when it runs.
  const labelStateRef = useLatestRef({ labelsActive, renderClusterLabels });
  const [labelRedraw] = useState(() => new CoalescedRedraw());
  const renderLatestLabels = useCallback(() => labelStateRef.current.renderClusterLabels(), [labelStateRef]);
  const handleRelayout = useCallback((_event: PlotRelayoutEvent) => {
    scheduleDensityRender();
    if (labelStateRef.current.labelsActive) labelRedraw.request(renderLatestLabels);
  }, [labelStateRef, labelRedraw, renderLatestLabels, scheduleDensityRender]);

  // Re-render labels when cluster data, size, or theme changes
  useEffect(() => {
    if (labelsActive) {
      // Small delay to ensure Plotly has laid out
      const timer = setTimeout(() => renderClusterLabels(), 50);
      return () => clearTimeout(timer);
    } else if (labelCanvasRef.current) {
      const ctx = labelCanvasRef.current.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, labelCanvasRef.current.width, labelCanvasRef.current.height);
      drawnLabelBoxesRef.current = [];
    }
  }, [labelsActive, clusterDataMap, densityPlacements, width, height, renderClusterLabels]);

  // --- Click handler for cluster labels (intercept before Plotly) ---
  const handleContainerClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const topicClicks = !!onClusterLabelClick && !!topicLabelToIdMap;
    if ((!topicClicks && !onClusterCategoryClick) || drawnLabelBoxesRef.current.length === 0) return;

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    for (const box of drawnLabelBoxesRef.current) {
      if (box.label === null) continue;
      if (x >= box.loX && x <= box.hiX && y >= box.loY && y <= box.hiY) {
        const topicId = topicLabelToIdMap?.get(box.label);
        if (onClusterLabelClick && topicId !== undefined) {
          e.stopPropagation();
          onClusterLabelClick(topicId);
          return;
        }
        if (onClusterCategoryClick?.(box.label, e.shiftKey)) {
          e.stopPropagation();
          return;
        }
      }
    }
  }, [onClusterLabelClick, topicLabelToIdMap, onClusterCategoryClick]);

  // Extract raw numeric data for native Plotly colorscales
  const numericData = useMemo(() => {
    if (colorScale.type === 'categorical' || !categoryField) return null;

    const values: (number | null)[] = points.map(p => {
      const val = p.metadata?.[categoryField];
      if (typeof val === 'number') return val;
      if (typeof val === 'string') {
        const parsed = parseFloat(val);
        return isNaN(parsed) ? null : parsed;
      }
      return null;
    });

    const validValues = values.filter((v): v is number => v !== null);
    if (validValues.length === 0) return null;

    let min = Infinity;
    let max = -Infinity;
    for (const v of validValues) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (min === max) return null;

    return {
      values,
      min,
      max,
      cleanValues: values.map(v => (v === null) ? NaN : v)
    };
  }, [colorScale.type, categoryField, points]);

  // Log transform: when logScale is active, transform cleanValues through log10
  const logData = useMemo(() => {
    if (!numericData || !customNumericRange?.logScale) return null;
    const offset = numericData.min <= 0 ? Math.abs(numericData.min) + 1 : 0;
    const toLog = (v: number) => Math.log10(v + offset + 1e-10);
    return {
      ...numericData,
      cleanValues: numericData.cleanValues.map(v => isNaN(v) ? NaN : toLog(v)),
      min: toLog(numericData.min),
      max: Math.log10(numericData.max + offset),
      toLog,  // expose for effectiveRange
    };
  }, [numericData, customNumericRange?.logScale]);
  const activeNumericData = logData ?? numericData;

  // Merge custom range overrides with auto-detected data range
  const effectiveRange = useMemo(() => {
    if (!activeNumericData) return null;
    const toLog = logData?.toLog;
    let effMin = customNumericRange?.min != null
      ? (toLog ? toLog(customNumericRange.min) : customNumericRange.min)
      : activeNumericData.min;
    let effMax = customNumericRange?.max != null
      ? (toLog ? toLog(customNumericRange.max) : customNumericRange.max)
      : activeNumericData.max;
    if (customNumericRange?.center !== undefined) {
      const c = toLog ? toLog(customNumericRange.center) : customNumericRange.center;
      const deviation = Math.max(Math.abs(effMax - c), Math.abs(effMin - c));
      effMin = c - deviation;
      effMax = c + deviation;
    }
    return { min: effMin, max: effMax };
  }, [activeNumericData, logData, customNumericRange]);

  // Generate Plotly-compatible colorscale array.
  // Lazy-load the backing Crameri/strip colormap (only the active one); the tick
  // forces a recompute once it's available, else custom strips fall back to viridis.
  const colormapTick = useColorScaleReady(colorScale);
  const plotlyColorScale = useMemo(() => {
    if (colorScale.type === 'categorical') return undefined;

    // For Crameri scales, use the pre-computed 256-step Plotly array directly
    const scaleName = colorScale.type === 'diverging' || colorScale.type === 'sequential'
      ? colorScale.scaleName : undefined;
    if (scaleName && isCrameriScale(scaleName)) {
      const crameriScale = getCrameriPlotlyScale(scaleName);
      if (crameriScale) return crameriScale;
      // Fall through to D3 sampling if not loaded yet
    }

    let scaleFunc: (t: number) => string;
    if (colorScale.type === 'monochrome') {
      scaleFunc = getMonochromeScale(colorScale.baseColor, [0, 1]);
    } else if (colorScale.type === 'diverging') {
      scaleFunc = getDivergingScale([0, 0.5, 1], colorScale.scaleName);
    } else if (colorScale.type === 'sequential') {
      scaleFunc = getSequentialScale([0, 1], colorScale.scaleName);
    } else {
      return undefined;
    }

    // Sample the scale to create Plotly gradient definition
    const steps = 20;
    return Array.from({ length: steps + 1 }, (_, i) => {
      const t = i / steps;
      return [t, scaleFunc(t)] as [number, string];
    });
  }, [colorScale, colormapTick]);

  // Calculate dynamic marker style based on point count
  const markerStyleRaw = useMemo(() => {
    return calculateMarkerStyle(points.length);
  }, [points.length]);
  // Light backgrounds wash out colored points — boost opacity
  const markerStyle = useMemo(() => ({
    ...markerStyleRaw,
    opacity: Math.min(markerStyleRaw.opacity * (isDark ? 1.0 : 1.6) * (pointOpacity ?? 1.0), 1.0),
  }), [markerStyleRaw, isDark, pointOpacity]);

  // Calculate dynamic highlight scaling based on point count
  const highlightScale = useMemo(() => {
    return calculateHighlightScale(points.length);
  }, [points.length]);

  const plotData = useMemo((): PlotlyData[] => {
    let traces: PlotlyData[] = [];

    // If there are highlighted points, separate them into two traces
    if (highlightedIndices && highlightedIndices.size > 0) {
      const unhighlightedPoints = points.filter(p => !highlightedIndices.has(p.index));

      // Apply the hide toggles (unclustered noise, missing values) to background points
      const filteredUnhighlightedPoints = HiddenPoints.filter(unhighlightedPoints, categoryField, hiddenOptions);

      const highlightedPoints = points.filter(p => highlightedIndices.has(p.index));

      // A. Background Points (dimmed) - skip if showOnlyHighlighted is true
      // Color priority: numericData > categorical > default gold
      // Preserve the user's color mode, just apply dim factor
      if (filteredUnhighlightedPoints.length > 0 && !showOnlyHighlighted) {
        const dimOpacity = markerStyle.opacity * 0.3; // Consistent dim factor
        const mutedOp = dimOpacity * (mutedPointOpacity ?? 0.20);

        if (activeNumericData && plotlyColorScale) {
          // MODE: NATIVE COLORSCALE (GPU ACCELERATED) - preserve colors with dimming
          const unhighlightedNumericValues = filteredUnhighlightedPoints.map(p => activeNumericData.cleanValues[p.index]);
          traces.push({
            x: filteredUnhighlightedPoints.map(p => p.x),
            y: filteredUnhighlightedPoints.map(p => p.y),
            mode: 'markers' as const,
            type: 'scattergl' as const,
            name: 'Context',
            marker: {
              size: markerStyle.size,
              color: unhighlightedNumericValues,
              colorscale: plotlyColorScale as any,
              cmin: effectiveRange!.min,
              cmax: effectiveRange!.max,
              opacity: dimOpacity,
              showscale: false,
            },
            text: filteredUnhighlightedPoints.map(formatHoverText),
            hovertemplate: '<b>%{text}</b><extra></extra>',
            customdata: filteredUnhighlightedPoints as any,
            showlegend: false,
          } satisfies PlotlyData);
        } else if (categoryField != null && categoryValues.length > 0) {
          // MODE: CATEGORICAL - preserve category colors with dimming
          if (nestedColorMap) {
            // NESTED: group by (topic, subtopic), color from nestedColorMap
            const pointsBySub: Record<string, Point2D[]> = {};
            filteredUnhighlightedPoints.forEach(point => {
              (pointsBySub[CategoryKey.nestedKey(point.metadata)] ??= []).push(point);
            });

            Object.entries(pointsBySub).forEach(([key, subPoints]) => {
              // Check if subtopic or its parent topic is muted
              const { topic: parentTopic, subtopic: sub } = CategoryKey.splitSubtopicKey(key);
              const isMuted = mutedCategories.includes(sub) || mutedCategories.includes(parentTopic);
              const fade = CategoryKey.missingFade(parentTopic, missingPointOpacity);
              if (hideFilteredPoints && isMuted) return;
              if (combinedMutedIndices && combinedMutedIndices.size > 0) {
                const activePoints = subPoints.filter(p => !combinedMutedIndices.has(p.index));
                const mutedPts = subPoints.filter(p => combinedMutedIndices.has(p.index));
                if (activePoints.length > 0) {
                  traces.push({
                    x: activePoints.map(p => p.x),
                    y: activePoints.map(p => p.y),
                    mode: 'markers' as const,
                    type: 'scattergl' as const,
                    name: sub,
                    marker: {
                      size: markerStyle.size,
                      color: isMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[key] || '#7f7f7f'),
                      opacity: (isMuted ? mutedOp : dimOpacity) * fade,
                    },
                    text: activePoints.map(formatHoverText),
                    hovertemplate: '<b>%{text}</b><extra></extra>',
                    customdata: activePoints as any,
                    showlegend: false,
                  } satisfies PlotlyData);
                }
                if (mutedPts.length > 0 && !hideFilteredPoints) {
                  traces.push({
                    x: mutedPts.map(p => p.x),
                    y: mutedPts.map(p => p.y),
                    mode: 'markers' as const,
                    type: 'scattergl' as const,
                    name: sub,
                    marker: {
                      size: markerStyle.size,
                      color: '#9ca3af',
                      opacity: mutedOp * fade,
                    },
                    text: mutedPts.map(formatHoverText),
                    hovertemplate: '<b>%{text}</b><extra></extra>',
                    customdata: mutedPts as any,
                    showlegend: false,
                  } satisfies PlotlyData);
                }
              } else {
                traces.push({
                  x: subPoints.map(p => p.x),
                  y: subPoints.map(p => p.y),
                  mode: 'markers' as const,
                  type: 'scattergl' as const,
                  name: sub,
                  marker: {
                    size: markerStyle.size,
                    color: isMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[key] || '#7f7f7f'),
                    opacity: (isMuted ? mutedOp : dimOpacity) * fade,
                  },
                  text: subPoints.map(formatHoverText),
                  hovertemplate: '<b>%{text}</b><extra></extra>',
                  customdata: subPoints as any,
                  showlegend: false,
                } satisfies PlotlyData);
              }
            });
          } else {
            const pointsByCategory: Record<string, Point2D[]> = {};
            filteredUnhighlightedPoints.forEach(point => {
              const cat = CategoryKey.of(categoryField ? point.metadata?.[categoryField] : undefined);
              if (!pointsByCategory[cat]) {
                pointsByCategory[cat] = [];
              }
              pointsByCategory[cat].push(point);
            });

            Object.entries(pointsByCategory).forEach(([cat, catPoints]) => {
              const isMuted = mutedCategories.includes(cat);
              const fade = CategoryKey.missingFade(cat, missingPointOpacity);
              if (hideFilteredPoints && isMuted) return;
              if (combinedMutedIndices && combinedMutedIndices.size > 0) {
                const activePoints = catPoints.filter(p => !combinedMutedIndices.has(p.index));
                const mutedPts = catPoints.filter(p => combinedMutedIndices.has(p.index));
                if (activePoints.length > 0) {
                  traces.push({
                    x: activePoints.map(p => p.x),
                    y: activePoints.map(p => p.y),
                    mode: 'markers' as const,
                    type: 'scattergl' as const,
                    name: getCategoryLabel(categoryField, cat),
                    marker: {
                      size: markerStyle.size,
                      color: isMuted ? '#9ca3af' : (colorMap[cat] || '#7f7f7f'),
                      opacity: (isMuted ? mutedOp : dimOpacity) * fade,
                    },
                    text: activePoints.map(formatHoverText),
                    hovertemplate: '<b>%{text}</b><extra></extra>',
                    customdata: activePoints as any,
                    showlegend: false,
                  } satisfies PlotlyData);
                }
                if (mutedPts.length > 0 && !hideFilteredPoints) {
                  traces.push({
                    x: mutedPts.map(p => p.x),
                    y: mutedPts.map(p => p.y),
                    mode: 'markers' as const,
                    type: 'scattergl' as const,
                    name: getCategoryLabel(categoryField, cat),
                    marker: {
                      size: markerStyle.size,
                      color: '#9ca3af',
                      opacity: mutedOp * fade,
                    },
                    text: mutedPts.map(formatHoverText),
                    hovertemplate: '<b>%{text}</b><extra></extra>',
                    customdata: mutedPts as any,
                    showlegend: false,
                  } satisfies PlotlyData);
                }
              } else {
                traces.push({
                  x: catPoints.map(p => p.x),
                  y: catPoints.map(p => p.y),
                  mode: 'markers' as const,
                  type: 'scattergl' as const,
                  name: getCategoryLabel(categoryField, cat),
                  marker: {
                    size: markerStyle.size,
                    color: isMuted ? '#9ca3af' : (colorMap[cat] || '#7f7f7f'),
                    opacity: (isMuted ? mutedOp : dimOpacity) * fade,
                  },
                  text: catPoints.map(formatHoverText),
                  hovertemplate: '<b>%{text}</b><extra></extra>',
                  customdata: catPoints as any,
                  showlegend: false,
                } satisfies PlotlyData);
              }
            });
          }
        } else {
          // MODE: NO COLORING - use gold fallback
          if (combinedMutedIndices && combinedMutedIndices.size > 0) {
            const activePoints = filteredUnhighlightedPoints.filter(p => !combinedMutedIndices.has(p.index));
            const mutedPts = filteredUnhighlightedPoints.filter(p => combinedMutedIndices.has(p.index));
            if (activePoints.length > 0) {
              traces.push({
                x: activePoints.map(p => p.x),
                y: activePoints.map(p => p.y),
                mode: 'markers' as const,
                type: 'scattergl' as const,
                name: 'Other items',
                marker: {
                  size: markerStyle.size,
                  color: '#e5a819ff',
                  opacity: dimOpacity,
                },
                text: activePoints.map(formatHoverText),
                hovertemplate: '<b>%{text}</b><extra></extra>',
                customdata: activePoints as any,
                showlegend: false,
              } satisfies PlotlyData);
            }
            if (mutedPts.length > 0 && !hideFilteredPoints) {
              traces.push({
                x: mutedPts.map(p => p.x),
                y: mutedPts.map(p => p.y),
                mode: 'markers' as const,
                type: 'scattergl' as const,
                name: 'Other items',
                marker: {
                  size: markerStyle.size,
                  color: '#9ca3af',
                  opacity: mutedOp,
                },
                text: mutedPts.map(formatHoverText),
                hovertemplate: '<b>%{text}</b><extra></extra>',
                customdata: mutedPts as any,
                showlegend: false,
              } satisfies PlotlyData);
            }
          } else {
            traces.push({
              x: filteredUnhighlightedPoints.map(p => p.x),
              y: filteredUnhighlightedPoints.map(p => p.y),
              mode: 'markers' as const,
              type: 'scattergl' as const,
              name: 'Other items',
              marker: {
                size: markerStyle.size,
                color: '#e5a819ff',
                opacity: dimOpacity,
              },
              text: filteredUnhighlightedPoints.map(formatHoverText),
              hovertemplate: '<b>%{text}</b><extra></extra>',
              customdata: filteredUnhighlightedPoints as any,
              showlegend: false,
            } satisfies PlotlyData);
          }
        }
      }

      // B. Constellation Lines - from selected point to each result
      if (renderedSelectedPoint && highlightedPoints.length > 0) {
        const lineX: number[] = [];
        const lineY: number[] = [];

        highlightedPoints.forEach(p => {
          // Don't draw line to itself
          if (p.index !== renderedSelectedPoint.index) {
            lineX.push(renderedSelectedPoint.x, p.x, null as any);
            lineY.push(renderedSelectedPoint.y, p.y, null as any);
          }
        });

        if (lineX.length > 0) {
          traces.push({
            x: lineX,
            y: lineY,
            mode: 'lines' as const,
            type: 'scattergl' as const,
            name: 'Connections',
            line: {
              color: isDark ? 'rgba(130, 160, 200, 0.12)' : 'rgba(71, 105, 155, 0.35)',
              width: isDark ? 0.5 : 0.8,
            },
            hoverinfo: 'skip' as any,
            showlegend: false,
          } satisfies PlotlyData);
        }
      }

      // C. Highlighted Points - triple-layer bluish glow
      if (highlightedPoints.length > 0) {
        // Separate selected point from other highlights
        const otherHighlights = renderedSelectedPoint
          ? highlightedPoints.filter(p => p.index !== renderedSelectedPoint.index)
          : highlightedPoints;

        if (otherHighlights.length > 0) {
          // OPTIMIZED: Collect all highlight data into arrays, then create 3 traces total
          // Instead of 3 traces per point (O(n) → O(1) trace count)
          // Colours vary with similarity; every glow point has the same size, so
          // sizes stay single values (the fork scales those for zoom by direct write)
          const hX: number[] = [];
          const hY: number[] = [];
          const outerColors: string[] = [];
          const innerColors: string[] = [];
          const coreColors: string[] = [];
          const coreLineColors: string[] = [];
          const coreTexts: string[] = [];
          const coreCustomData: any[] = [];

          otherHighlights.forEach(point => {
            const similarity = highlightedIndices!.get(point.index) ?? 1.0;
            const colors = calculateSimilarityColors(similarity, isDark);

            hX.push(point.x);
            hY.push(point.y);
            outerColors.push(colors.outerGlow);
            innerColors.push(colors.glowColor);
            coreColors.push(colors.coreColor);
            coreLineColors.push(colors.glowColor);
            coreTexts.push(formatHoverText(point));
            coreCustomData.push(point);
          });

          // Layer 1: Outer glow (single trace for all points)
          traces.push({
            x: hX,
            y: hY,
            mode: 'markers' as const,
            type: 'scattergl' as const,
            hoverinfo: 'skip',
            meta: ZoomMarkerStyle.FIXED_OPACITY_META,
            marker: {
              size: markerStyle.size * highlightScale.outerMultiplier,
              color: outerColors,
              opacity: 0.15, // Use average, scattergl doesn't support per-point opacity arrays well
              line: { width: 0 },
            },
            showlegend: false,
          } satisfies PlotlyData);

          // Layer 2: Inner glow (single trace for all points)
          traces.push({
            x: hX,
            y: hY,
            mode: 'markers' as const,
            type: 'scattergl' as const,
            hoverinfo: 'skip',
            meta: ZoomMarkerStyle.FIXED_OPACITY_META,
            marker: {
              size: markerStyle.size * highlightScale.innerMultiplier,
              color: innerColors,
              opacity: 0.3,
              line: { width: 0 },
            },
            showlegend: false,
          } satisfies PlotlyData);

          // Layer 3: Bright core (single trace for all points)
          traces.push({
            x: hX,
            y: hY,
            mode: 'markers' as const,
            type: 'scattergl' as const,
            name: 'Search results',
            meta: ZoomMarkerStyle.FIXED_OPACITY_META,
            marker: {
              size: Math.max(markerStyle.size * highlightScale.coreMultiplier, 3),
              color: coreColors,
              opacity: 1,
              line: { color: coreLineColors, width: 1 },
            },
            text: coreTexts,
            hovertemplate: '<b>%{text}</b><extra></extra>',
            customdata: coreCustomData as any,
            showlegend: false,
          } satisfies PlotlyData);
        }

        // D. Selected Point - golden tint to distinguish it
        if (renderedSelectedPoint) {
          // Outer golden glow
          traces.push({
            x: [renderedSelectedPoint.x],
            y: [renderedSelectedPoint.y],
            mode: 'markers' as const,
            type: 'scattergl' as const,
            hoverinfo: 'skip',
            meta: ZoomMarkerStyle.FIXED_OPACITY_META,
            marker: {
              size: markerStyle.size * highlightScale.selectedOuterMultiplier,
              color: 'rgba(255, 215, 140, 0.15)',
              opacity: 0.3,
              line: { width: 0 },
            },
            showlegend: false,
          } satisfies PlotlyData);

          // Inner golden glow
          traces.push({
            x: [renderedSelectedPoint.x],
            y: [renderedSelectedPoint.y],
            mode: 'markers' as const,
            type: 'scattergl' as const,
            hoverinfo: 'skip',
            meta: ZoomMarkerStyle.FIXED_OPACITY_META,
            marker: {
              size: Math.max(markerStyle.size * highlightScale.selectedInnerMultiplier, 8),
              color: 'rgba(255, 223, 160, 0.35)',
              opacity: 0.5,
              line: { width: 0 },
            },
            showlegend: false,
          } satisfies PlotlyData);

          // Golden core
          traces.push({
            x: [renderedSelectedPoint.x],
            y: [renderedSelectedPoint.y],
            mode: 'markers' as const,
            type: 'scattergl' as const,
            name: 'Selected',
            meta: ZoomMarkerStyle.FIXED_OPACITY_META,
            marker: {
              size: Math.max(markerStyle.size * highlightScale.selectedCoreMultiplier, 4),
              color: '#fff8e8',
              opacity: 1,
              line: { color: 'rgba(255, 200, 100, 0.6)', width: 1.5 },
            },
            text: [formatHoverText(renderedSelectedPoint)],
            hovertemplate: '<b>%{text}</b><extra></extra>',
            customdata: [renderedSelectedPoint] as any,
            showlegend: false,
          } satisfies PlotlyData);
        }
      }
    }
    // No highlighting - render normally
    else {
      // Apply the hide toggles (unclustered noise, missing values)
      const displayPoints = HiddenPoints.filter(points, categoryField, hiddenOptions);

      const mutedOp = markerStyle.opacity * (mutedPointOpacity ?? 0.20);

      if (activeNumericData && plotlyColorScale) {
        // MODE: NATIVE COLORSCALE (GPU ACCELERATED)
        if (combinedMutedIndices && combinedMutedIndices.size > 0) {
          const activePoints = displayPoints.filter(p => !combinedMutedIndices.has(p.index));
          const mutedPts = displayPoints.filter(p => combinedMutedIndices.has(p.index));
          if (activePoints.length > 0) {
            traces.push({
              x: activePoints.map(p => p.x),
              y: activePoints.map(p => p.y),
              mode: 'markers' as const,
              type: 'scattergl' as const,
              name: 'Data',
              marker: {
                size: markerStyle.size,
                color: activePoints.map(p => activeNumericData.cleanValues[p.index]),
                colorscale: plotlyColorScale as any,
                cmin: numericData!.min,
                cmax: numericData!.max,
                opacity: markerStyle.opacity,
                showscale: false,
              },
              text: activePoints.map(formatHoverText),
              hovertemplate: '<b>%{text}</b><extra></extra>',
              customdata: activePoints as any,
            } satisfies PlotlyData);
          }
          if (mutedPts.length > 0 && !hideFilteredPoints) {
            traces.push({
              x: mutedPts.map(p => p.x),
              y: mutedPts.map(p => p.y),
              mode: 'markers' as const,
              type: 'scattergl' as const,
              name: 'Outside range',
              marker: {
                size: markerStyle.size,
                color: '#9ca3af',
                opacity: mutedOp,
              },
              text: mutedPts.map(formatHoverText),
              hovertemplate: '<b>%{text}</b><extra></extra>',
              customdata: mutedPts as any,
              showlegend: false,
            } satisfies PlotlyData);
          }
        } else {
          traces = [{
            x: displayPoints.map(p => p.x),
            y: displayPoints.map(p => p.y),
            mode: 'markers' as const,
            type: 'scattergl' as const,
            name: 'Data',
            marker: {
              size: markerStyle.size,
              color: displayPoints.map(p => activeNumericData.cleanValues[p.index]),
              colorscale: plotlyColorScale as any,
              cmin: effectiveRange!.min,
              cmax: effectiveRange!.max,
              opacity: markerStyle.opacity,
              showscale: false, // Disabled - using custom Legend component instead
            },
            text: displayPoints.map(formatHoverText),
            hovertemplate: '<b>%{text}</b><extra></extra>',
            customdata: displayPoints as any,
          } satisfies PlotlyData];
        }
      } else if (categoryField != null && categoryValues.length > 0) {
        if (nestedColorMap) {
          // NESTED: group by (topic, subtopic)
          const pointsBySub: Record<string, Point2D[]> = {};
          displayPoints.forEach(point => {
            (pointsBySub[CategoryKey.nestedKey(point.metadata)] ??= []).push(point);
          });

          Object.entries(pointsBySub).forEach(([key, subPoints]) => {
            const { topic: parentTopic, subtopic: sub } = CategoryKey.splitSubtopicKey(key);
            const isMuted = mutedCategories.includes(sub) || mutedCategories.includes(parentTopic);
            const fade = CategoryKey.missingFade(parentTopic, missingPointOpacity);
            if (hideFilteredPoints && isMuted) return;
            if (combinedMutedIndices && combinedMutedIndices.size > 0) {
              const activePoints = subPoints.filter(p => !combinedMutedIndices.has(p.index));
              const mutedPts = subPoints.filter(p => combinedMutedIndices.has(p.index));
              if (activePoints.length > 0) {
                traces.push({
                  x: activePoints.map(p => p.x),
                  y: activePoints.map(p => p.y),
                  mode: 'markers' as const,
                  type: 'scattergl' as const,
                  name: sub,
                  marker: {
                    size: markerStyle.size,
                    color: isMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[key] || '#7f7f7f'),
                    opacity: (isMuted ? mutedOp : markerStyle.opacity) * fade,
                  },
                  text: activePoints.map(formatHoverText),
                  hovertemplate: '<b>%{text}</b><extra></extra>',
                  customdata: activePoints as any,
                } satisfies PlotlyData);
              }
              if (mutedPts.length > 0 && !hideFilteredPoints) {
                traces.push({
                  x: mutedPts.map(p => p.x),
                  y: mutedPts.map(p => p.y),
                  mode: 'markers' as const,
                  type: 'scattergl' as const,
                  name: sub,
                  marker: {
                    size: markerStyle.size,
                    color: '#9ca3af',
                    opacity: mutedOp * fade,
                  },
                  text: mutedPts.map(formatHoverText),
                  hovertemplate: '<b>%{text}</b><extra></extra>',
                  customdata: mutedPts as any,
                  showlegend: false,
                } satisfies PlotlyData);
              }
            } else {
              traces.push({
                x: subPoints.map(p => p.x),
                y: subPoints.map(p => p.y),
                mode: 'markers' as const,
                type: 'scattergl' as const,
                name: sub,
                marker: {
                  size: markerStyle.size,
                  color: isMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[key] || '#7f7f7f'),
                  opacity: (isMuted ? mutedOp : markerStyle.opacity) * fade,
                },
                text: subPoints.map(formatHoverText),
                hovertemplate: '<b>%{text}</b><extra></extra>',
                customdata: subPoints as any,
              } satisfies PlotlyData);
            }
          });
        } else {
          const pointsByCategory: Record<string, Point2D[]> = {};
          displayPoints.forEach(point => {
            const cat = CategoryKey.of(categoryField ? point.metadata?.[categoryField] : undefined);
            if (!pointsByCategory[cat]) {
              pointsByCategory[cat] = [];
            }
            pointsByCategory[cat].push(point);
          });

          Object.entries(pointsByCategory).forEach(([cat, catPoints]) => {
            const isMuted = mutedCategories.includes(cat);
            const fade = CategoryKey.missingFade(cat, missingPointOpacity);
            if (hideFilteredPoints && isMuted) return;
            if (combinedMutedIndices && combinedMutedIndices.size > 0) {
              const activePoints = catPoints.filter(p => !combinedMutedIndices.has(p.index));
              const mutedPts = catPoints.filter(p => combinedMutedIndices.has(p.index));
              if (activePoints.length > 0) {
                traces.push({
                  x: activePoints.map(p => p.x),
                  y: activePoints.map(p => p.y),
                  mode: 'markers' as const,
                  type: 'scattergl' as const,
                  name: getCategoryLabel(categoryField, cat),
                  marker: {
                    size: markerStyle.size,
                    color: isMuted ? '#9ca3af' : (colorMap[cat] || '#7f7f7f'),
                    opacity: (isMuted ? mutedOp : markerStyle.opacity) * fade,
                  },
                  text: activePoints.map(formatHoverText),
                  hovertemplate: '<b>%{text}</b><extra></extra>',
                  customdata: activePoints as any,
                } satisfies PlotlyData);
              }
              if (mutedPts.length > 0 && !hideFilteredPoints) {
                traces.push({
                  x: mutedPts.map(p => p.x),
                  y: mutedPts.map(p => p.y),
                  mode: 'markers' as const,
                  type: 'scattergl' as const,
                  name: getCategoryLabel(categoryField, cat),
                  marker: {
                    size: markerStyle.size,
                    color: '#9ca3af',
                    opacity: mutedOp * fade,
                  },
                  text: mutedPts.map(formatHoverText),
                  hovertemplate: '<b>%{text}</b><extra></extra>',
                  customdata: mutedPts as any,
                  showlegend: false,
                } satisfies PlotlyData);
              }
            } else {
              traces.push({
                x: catPoints.map(p => p.x),
                y: catPoints.map(p => p.y),
                mode: 'markers' as const,
                type: 'scattergl' as const,
                name: getCategoryLabel(categoryField, cat),
                marker: {
                  size: markerStyle.size,
                  color: isMuted ? '#9ca3af' : (colorMap[cat] || '#7f7f7f'),
                  opacity: (isMuted ? mutedOp : markerStyle.opacity) * fade,
                },
                text: catPoints.map(formatHoverText),
                hovertemplate: '<b>%{text}</b><extra></extra>',
                customdata: catPoints as any,
              } satisfies PlotlyData);
            }
          });
        }
      } else {
        if (combinedMutedIndices && combinedMutedIndices.size > 0) {
          const activePoints = displayPoints.filter(p => !combinedMutedIndices.has(p.index));
          const mutedPts = displayPoints.filter(p => combinedMutedIndices.has(p.index));
          if (activePoints.length > 0) {
            traces.push({
              x: activePoints.map(p => p.x),
              y: activePoints.map(p => p.y),
              mode: 'markers' as const,
              type: 'scattergl' as const,
              marker: {
                size: markerStyle.size,
                color: '#1f77b4',
                opacity: markerStyle.opacity,
              },
              text: activePoints.map(formatHoverText),
              hovertemplate: '<b>%{text}</b><extra></extra>',
              customdata: activePoints as any,
            } satisfies PlotlyData);
          }
          if (mutedPts.length > 0 && !hideFilteredPoints) {
            traces.push({
              x: mutedPts.map(p => p.x),
              y: mutedPts.map(p => p.y),
              mode: 'markers' as const,
              type: 'scattergl' as const,
              marker: {
                size: markerStyle.size,
                color: '#9ca3af',
                opacity: mutedOp,
              },
              text: mutedPts.map(formatHoverText),
              hovertemplate: '<b>%{text}</b><extra></extra>',
              customdata: mutedPts as any,
              showlegend: false,
            } satisfies PlotlyData);
          }
        } else {
          traces = [{
            x: displayPoints.map(p => p.x),
            y: displayPoints.map(p => p.y),
            mode: 'markers' as const,
            type: 'scattergl' as const,
            marker: {
              size: markerStyle.size,
              color: '#1f77b4',
              opacity: markerStyle.opacity,
            },
            text: displayPoints.map(formatHoverText),
            hovertemplate: '<b>%{text}</b><extra></extra>',
            customdata: displayPoints as any,
          } satisfies PlotlyData];
        }
      }
    }

    // E. Text Labels for highlighted points (rendered last, on top)
    if (showLabels && highlightedIndices && highlightedIndices.size > 0) {
      const highlightedPoints = points.filter(p => highlightedIndices.has(p.index));
      if (highlightedPoints.length > 0) {
        traces.push({
          x: highlightedPoints.map(p => p.x),
          y: highlightedPoints.map(p => p.y),
          mode: 'text' as const,
          type: 'scattergl' as const,
          text: highlightedPoints.map(p => p.label || p.id),
          textposition: 'top center' as const,
          textfont: {
            size: 11,
            color: isDark ? '#e2e8f0' : '#1e293b',
          },
          hoverinfo: 'skip' as const,
          showlegend: false,
        } satisfies PlotlyData);
      }
    }

    return traces;
  }, [points, categoryField, categoryValues, colorMap, activeNumericData, effectiveRange, plotlyColorScale, highlightedIndices, renderedSelectedPoint, isDark, markerStyle.size, markerStyle.opacity, highlightScale, showOnlyHighlighted, showLabels, mutedCategories, hiddenOptions, nestedColorMap, combinedMutedIndices, hideFilteredPoints, mutedPointOpacity, missingPointOpacity]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      width,
      height,
      uirevision: 'true', // Preserve zoom/pan state on resize
      dragmode: 'pan',
      hovermode: 'closest' as const,
      showlegend: false, 
      // showlegend: categoryField != null && categoryValues.length > 0,
      plot_bgcolor: plotBg,
      paper_bgcolor: paperBg,
      font: { color: axisColor },
      legend: {
        bgcolor: legendBg,
        bordercolor: gridColor,
        font: { color: axisColor },
      },
      xaxis: {
        title: { text: '', font: { color: axisColor } },
        gridcolor: gridColor,
        zerolinecolor: axisColor,
        zerolinewidth: showAxes ? 2 : 0,
        tickfont: { color: axisColor },
        showgrid: false,
        zeroline: showAxes,
        showspikes: false,
        showticklabels: showAxes,
        showline: false,
        showtitle: false,
      },
      yaxis: {
        title: { text: '', font: { color: axisColor } },
        gridcolor: gridColor,
        zerolinecolor: axisColor,
        zerolinewidth: showAxes ? 2 : 0,
        tickfont: { color: axisColor },
        scaleanchor: 'x',
        scaleratio: 1,
        showgrid: false,
        zeroline: showAxes,
        showspikes: false,
        showticklabels: showAxes,
        showline: false,
        showtitle: false,
      },
      // No decorative border. When axes are shown, keep just enough room for
      // tick labels (left/bottom); otherwise let the plot fill the container.
      margin: showAxes ? { l: 40, r: 8, t: 8, b: 30 } : { l: 0, r: 0, t: 0, b: 0 },
    }),
    [axisColor, categoryField, categoryValues.length, gridColor, height, legendBg, paperBg, plotBg, width, showAxes]
  );

   
  const config = useMemo<ForkConfig>(() => ({
    displayModeBar: true,
    displaylogo: false,
    responsive: true,
    scrollZoom: true,
    modeBarButtons: build2DModeBarButtons(plotlyLibRef.current, onScreenshot ? screenshotHandlerRef : undefined),
    scatterglOpacityScale: (trace, xRange, yRange) => zoomMarkerStyleRef.current.opacityScale(trace, xRange, yRange),
    scatterglSizeScale: (_trace, xRange, yRange) => zoomMarkerStyleRef.current.sizeScale(xRange, yRange),
  }), [plotlyReady]); // eslint-disable-line react-hooks/exhaustive-deps

  // Plotly handlers: react-plotly rebinds a handler whenever its identity changes
  // (fork change 2), so they are memoized to rebind only when their inputs change.
  const handleClick = useCallback((event: PlotMouseEvent) => {
    if (!onPointClick || !event.points || event.points.length === 0) {
      return;
    }

    const point = event.points[0];
    // Skip glow layers and lines (they don't have valid customdata)
    if (point.customdata && typeof point.customdata === 'object') {
      onPointClick(point.customdata as unknown as Point2D);
    }
  }, [onPointClick]);

  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);

  const handleHover = useCallback((event: PlotHoverEvent) => {
    if (event.points && event.points.length > 0) {
      const pt = event.points[0];
      // Only show tooltip for points with valid customdata (skip glow layers)
      if (pt.customdata && typeof pt.customdata === 'object') {
        const point = pt.customdata as unknown as Point2D;
        hoveredPointRef.current = point;
        const mouseEvent = event.event as MouseEvent;
        // Get position relative to container
        const containerRect = containerRef.current?.getBoundingClientRect();
        const x = mouseEvent.clientX - (containerRect?.left ?? 0);
        const y = mouseEvent.clientY - (containerRect?.top ?? 0);
        setTooltipData({
          x,
          y,
          label: point.label || point.id,
          document: point.document,
          visible: true,
          metadata: point.metadata,
          tooltipFields,
        });
      }
    }
  }, [tooltipFields]);

  const handleUnhover = useCallback(() => {
    hoveredPointRef.current = null;
    setTooltipData(null);
  }, []);

  const hasClusterLabels = labelsActive;

return (
  <div
    ref={containerRef}
    className={className ?? 'h-full w-full'}
    // touch-action: the plot handles every gesture (pan, pinch), never the page
    style={{ position: 'relative', touchAction: 'none' }}
    onClickCapture={hasClusterLabels ? handleContainerClick : undefined}
    onContextMenu={(e) => {
      if (onPointContextMenu && hoveredPointRef.current) {
        e.preventDefault();
        onPointContextMenu(hoveredPointRef.current, e.nativeEvent);
      }
    }}
  >
    <Plot
      data={plotData}
      layout={layout}
      config={config}
      onClick={plotReady ? handleClick : undefined}
      onHover={handleHover}
      onUnhover={handleUnhover}
      onRelayout={handleRelayout}
      // Fires continuously during pan/zoom gestures; supported by react-plotly
      // at runtime but missing from its type declarations.
      {...({ onRelayouting: handleRelayout } as Partial<PlotParams>)}
      // After every Plotly.react: the density effects can run before a queued react lands
      onUpdate={scheduleDensityRender}
      onInitialized={(_figure, graphDiv) => { graphDivRef.current = graphDiv; setPlotReady(true); }}
    />
    {densityMode && densitySupported && (
      <canvas
        ref={densityCanvasRef}
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          pointerEvents: 'none',
          zIndex: 5,
        }}
      />
    )}
    {hasClusterLabels && (
      <canvas
        ref={labelCanvasRef}
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: `${width}px`,
          height: `${height}px`,
          pointerEvents: 'none',
          zIndex: 10,
        }}
      />
    )}
    <FrostedTooltip data={tooltipData} />
  </div>
);
});
