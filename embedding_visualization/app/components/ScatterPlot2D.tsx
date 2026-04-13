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
import type { Point2D, HighlightMap, ColorScaleType, NestedColorMap } from '../../lib/types/types';
import { buildCategoryColorMap, getCategoryLabel, getSequentialScale, getDivergingScale, getMonochromeScale, desaturateHex, type SequentialScaleName, type DivergingScaleName } from '../../lib/utils/categoryColors';
import { isCrameriScale, getCrameriPlotlyScale } from '../../lib/colorMaps/crameriScales';
import { calculateMarkerStyle, calculateLuminosity, calculateHighlightScale, calculateSimilarityColors } from '../../lib/utils/plotUtils';
import { useContainerDimensions } from '../../lib/hooks/useContainerDimensions';
import { useZoomLimit } from '../../lib/hooks/useZoomLimit';
import { formatHoverText } from '../utils/rendeding';
import { groupPointsByCluster, type ClusterData } from '../../lib/utils/clusterGeometry';
import {
  computeClusterLabelPlacements,
  computeCurrentScale,
  projectDataToScreen,
  type AxisRanges,
  type PlotArea,
  type ClusterLabelPlacement,
} from '../utils/labelPlacement2D';

import { FrostedTooltip, type TooltipData } from './FrostedTooltip';

type PlotlyData = Partial<PlotData>;

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic<PlotParams>(() => import('react-plotly.js'), { ssr: false });

interface ScatterPlot2DProps {
  points: Point2D[];
  colorBy?: 'category' | 'none';
  categoryField?: string | null;  // Field to color by (used for both categorical AND numeric)
  categoryValues?: string[];
  colorScaleType?: ColorScaleType;
  monochromeColor?: string;
  sequentialScaleName?: SequentialScaleName;
  divergingScaleName?: DivergingScaleName;
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
  /** 0-1, opacity for muted points (default 0.15) */
  mutedPointOpacity?: number;
  /** Show topic/subtopic names at cluster centroids */
  showClusterLabels?: boolean;
  /** Callback when a cluster label is clicked (topic toggle) */
  onClusterLabelClick?: (topicId: number) => void;
  /** Map from topic label string → topic ID for click handling */
  topicLabelToIdMap?: Map<string, number> | null;
}


export const ScatterPlot2D = React.memo(function ScatterPlot2D({
  points,
  colorBy = 'none',
  categoryField = null,
  categoryValues = [],
  colorScaleType = 'categorical',
  monochromeColor = '#1f77b4',
  sequentialScaleName = 'sinebow',
  divergingScaleName = 'blueGold',
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
  showClusterLabels = false,
  onClusterLabelClick,
  topicLabelToIdMap,
}: ScatterPlot2DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphDivRef = useRef<any>(null);
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

  // --- Cluster labels: canvas overlay refs and state ---
  const labelCanvasRef = useRef<HTMLCanvasElement>(null);
  const clusterPlacementsRef = useRef<ClusterLabelPlacement[]>([]);
  const initialRangesRef = useRef<AxisRanges | null>(null);
  const currentRangesRef = useRef<AxisRanges | null>(null);
  // Store drawn label bounding boxes for click hit-testing
  const drawnLabelBoxesRef = useRef<{ loX: number; loY: number; hiX: number; hiY: number; label: string }[]>([]);

  const { resolvedTheme } = useTheme();
  const theme = resolvedTheme ?? 'light';
  const isDark = theme === 'dark';
  const axisColor = isDark ? '#e2e8f0' : '#0f172a';
  const gridColor = isDark ? '#334155' : '#e5e7eb';
  const plotBg = isDark ? "rgba(0,0,0,0" : "rgba(0,0,0,0)";
  const paperBg = isDark ? "rgba(0,0,0,0" : "rgba(0,0,0,0";
  const legendBg = isDark ? 'rgba(2,6,23,0.85)' : 'rgba(255,255,255,0.85)';

  // Build color map based on category values
  const colorMap = useMemo(() => {
    return buildCategoryColorMap(categoryField, categoryValues, categoricalPalette);
  }, [categoryField, categoryValues, categoricalPalette]);

  // --- Cluster label data (shared with 3D pattern via groupPointsByCluster) ---
  const clusterDataMap = useMemo(() => {
    if (!showClusterLabels || !categoryField) return new Map<string, ClusterData>();

    let displayPoints: Point2D[] = hideUnclustered
      ? points.filter(p => {
          const topicId = p.metadata?.['topic_id'];
          if (topicId === '-1' || topicId === -1) return false;
          const topicLabel = p.metadata?.['topic_label'];
          if (topicLabel === 'Unclustered' || topicLabel === 'unclustered') return false;
          return true;
        })
      : points;

    if (mutedCategories.length > 0) {
      const mutedSet = new Set(mutedCategories);
      displayPoints = displayPoints.filter(p => {
        const category = p.metadata?.[categoryField];
        return category == null || !mutedSet.has(String(category));
      });
    }

    // groupPointsByCluster works with Point3D but Point2D is compatible (z defaults via centroid calc)
    return groupPointsByCluster(displayPoints as any, categoryField, colorMap, nestedColorMap);
  }, [showClusterLabels, points, categoryField, colorMap, nestedColorMap, hideUnclustered, mutedCategories]);

  // --- Cluster label font constant ---
  const CLUSTER_FONT = 'bold 13px Geist Mono, monospace';
  const CLUSTER_FONT_SIZE = 13;

  // --- Recompute Apple placements when clusters or layout change ---
  useEffect(() => {
    if (!showClusterLabels || clusterDataMap.size === 0 || !bounds) {
      clusterPlacementsRef.current = [];
      return;
    }

    // Use a temporary canvas for text measurement
    const canvas = labelCanvasRef.current ?? document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Compute initial ranges from data bounds with some padding (Plotly auto-pads ~5%)
    const padX = (bounds.maxX - bounds.minX) * 0.05 || 0.5;
    const padY = (bounds.maxY - bounds.minY) * 0.05 || 0.5;
    const initRanges: AxisRanges = {
      xRange: [bounds.minX - padX, bounds.maxX + padX],
      yRange: [bounds.minY - padY, bounds.maxY + padY],
    };
    initialRangesRef.current = initRanges;

    // Use the Plotly layout area (approximate — margins are 50px each side)
    const plotArea: PlotArea = {
      left: 50,
      top: 50,
      width: Math.max(width - 100, 100),
      height: Math.max(height - 100, 100),
    };

    clusterPlacementsRef.current = computeClusterLabelPlacements(
      clusterDataMap,
      ctx,
      CLUSTER_FONT,
      initRanges,
      plotArea,
    );
  }, [showClusterLabels, clusterDataMap, bounds, width, height]);

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
    const fl = gd?._fullLayout;
    const plotArea: PlotArea = {
      left: fl?.margin?.l ?? 50,
      top: fl?.margin?.t ?? 50,
      width: (fl?.width ?? cssW) - (fl?.margin?.l ?? 50) - (fl?.margin?.r ?? 50),
      height: (fl?.height ?? cssH) - (fl?.margin?.t ?? 50) - (fl?.margin?.b ?? 50),
    };

    ctx.font = CLUSTER_FONT;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';

    const boxes: { loX: number; loY: number; hiX: number; hiY: number; label: string }[] = [];

    for (const cl of placements) {
      // Check Apple visibility: placement must exist and current scale must be in range
      if (!cl.placement) continue;
      if (scale < cl.placement.minScale || scale > cl.placement.maxScale) continue;

      // Project data centroid to current screen position
      const screen = projectDataToScreen(cl.dataX, cl.dataY, curRanges, plotArea);

      // Clip to plot area
      if (screen.x < plotArea.left - 20 || screen.x > plotArea.left + plotArea.width + 20) continue;
      if (screen.y < plotArea.top - 20 || screen.y > plotArea.top + plotArea.height + 20) continue;

      const textWidth = ctx.measureText(cl.label).width;
      const pad = 4;
      const box = {
        loX: screen.x - textWidth / 2 - pad,
        loY: screen.y - CLUSTER_FONT_SIZE * 0.6 - pad,
        hiX: screen.x + textWidth / 2 + pad,
        hiY: screen.y + CLUSTER_FONT_SIZE * 0.6 + pad,
        label: cl.label,
      };

      // Draw label with stroke outline for readability (same style as 3D)
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = isDark ? 'rgba(15, 23, 42, 0.85)' : 'rgba(255, 255, 255, 0.85)';
      ctx.lineWidth = 4;
      ctx.lineJoin = 'round';
      ctx.strokeText(cl.label, screen.x, screen.y);
      ctx.fillStyle = desaturateHex(cl.color, 0.3, isDark);
      ctx.fillText(cl.label, screen.x, screen.y);

      boxes.push(box);
    }

    drawnLabelBoxesRef.current = boxes;
    ctx.globalAlpha = 1.0;
  }, [width, height, isDark]);

  // --- Handle Plotly relayout (zoom/pan) to re-render cluster labels ---
  const handleRelayout = useCallback((_event: PlotRelayoutEvent) => {
    if (showClusterLabels && clusterDataMap.size > 0) {
      // Use rAF to let Plotly finish its layout update first
      requestAnimationFrame(() => renderClusterLabels());
    }
  }, [showClusterLabels, clusterDataMap.size, renderClusterLabels]);

  // Re-render labels when cluster data, size, or theme changes
  useEffect(() => {
    if (showClusterLabels && clusterDataMap.size > 0) {
      // Small delay to ensure Plotly has laid out
      const timer = setTimeout(() => renderClusterLabels(), 50);
      return () => clearTimeout(timer);
    } else if (labelCanvasRef.current) {
      const ctx = labelCanvasRef.current.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, labelCanvasRef.current.width, labelCanvasRef.current.height);
      drawnLabelBoxesRef.current = [];
    }
  }, [showClusterLabels, clusterDataMap, width, height, renderClusterLabels]);

  // --- Click handler for cluster labels (intercept before Plotly) ---
  const handleContainerClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!onClusterLabelClick || !topicLabelToIdMap || drawnLabelBoxesRef.current.length === 0) return;

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    for (const box of drawnLabelBoxesRef.current) {
      if (x >= box.loX && x <= box.hiX && y >= box.loY && y <= box.hiY) {
        const topicId = topicLabelToIdMap.get(box.label);
        if (topicId !== undefined) {
          e.stopPropagation();
          onClusterLabelClick(topicId);
          return;
        }
      }
    }
  }, [onClusterLabelClick, topicLabelToIdMap]);

  // Extract raw numeric data for native Plotly colorscales
  const numericData = useMemo(() => {
    if (colorScaleType === 'categorical' || !categoryField) return null;

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
  }, [colorScaleType, categoryField, points]);

  // Generate Plotly-compatible colorscale array
  const plotlyColorScale = useMemo(() => {
    if (colorScaleType === 'categorical') return undefined;

    // For Crameri scales, use the pre-computed 256-step Plotly array directly
    const scaleName = colorScaleType === 'diverging' ? divergingScaleName : sequentialScaleName;
    if (scaleName && isCrameriScale(scaleName)) {
      const crameriScale = getCrameriPlotlyScale(scaleName);
      if (crameriScale) return crameriScale;
      // Fall through to D3 sampling if not loaded yet
    }

    let scaleFunc: (t: number) => string;
    if (colorScaleType === 'monochrome') {
      scaleFunc = getMonochromeScale(monochromeColor, [0, 1]);
    } else if (colorScaleType === 'diverging') {
      scaleFunc = getDivergingScale([0, 0.5, 1], divergingScaleName);
    } else {
      scaleFunc = getSequentialScale([0, 1], sequentialScaleName);
    }

    // Sample the scale to create Plotly gradient definition
    const steps = 20;
    return Array.from({ length: steps + 1 }, (_, i) => {
      const t = i / steps;
      return [t, scaleFunc(t)] as [number, string];
    });
  }, [colorScaleType, monochromeColor, sequentialScaleName, divergingScaleName]);

  // Calculate dynamic marker style based on point count
  const markerStyle = useMemo(() => {
    return calculateMarkerStyle(points.length);
  }, [points.length]);

  // Calculate dynamic highlight scaling based on point count
  const highlightScale = useMemo(() => {
    return calculateHighlightScale(points.length);
  }, [points.length]);

  //const { generateClusters, isLoaded: isClusteringLoaded } = useDensityClustering();
  //const [clusters, setClusters] = useState<DensityCluster[]>([]);

  //useEffect(() => {
   // if (isClusteringLoaded && points.length > 0) {
      // Run in a timeout to avoid blocking render
  //    const timer = setTimeout(() => {
  //      const result = generateClusters(points);
  //      setClusters(result);
  //    }, 100);
  //    return () => clearTimeout(timer);
  //  } else {
  //    setClusters([]);
  //  }
  //}, [isClusteringLoaded, points, generateClusters]);
  const plotData = useMemo((): PlotlyData[] => {
    let traces: PlotlyData[] = [];

    // If there are highlighted points, separate them into two traces
    if (highlightedIndices && highlightedIndices.size > 0) {
      const unhighlightedPoints = points.filter(p => !highlightedIndices.has(p.index));

      // Apply hideUnclustered filter to background points
      // Check topic_id for -1 OR topic_label for "Unclustered" to handle all dataset shapes
      const filteredUnhighlightedPoints = hideUnclustered
        ? unhighlightedPoints.filter(p => {
            const topicId = p.metadata?.['topic_id'];
            if (topicId === '-1' || topicId === -1) return false;
            const topicLabel = p.metadata?.['topic_label'];
            if (topicLabel === 'Unclustered' || topicLabel === 'unclustered') return false;
            return true;
          })
        : unhighlightedPoints;

      const highlightedPoints = points.filter(p => highlightedIndices.has(p.index));

      // A. Background Points (dimmed) - skip if showOnlyHighlighted is true
      // Color priority: numericData > categorical > default gold
      // Preserve the user's color mode, just apply dim factor
      if (filteredUnhighlightedPoints.length > 0 && !showOnlyHighlighted) {
        const dimOpacity = markerStyle.opacity * 0.3; // Consistent dim factor

        if (numericData && plotlyColorScale) {
          // MODE: NATIVE COLORSCALE (GPU ACCELERATED) - preserve colors with dimming
          const unhighlightedNumericValues = filteredUnhighlightedPoints.map(p => numericData.cleanValues[p.index]);
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
              cmin: numericData.min,
              cmax: numericData.max,
              opacity: dimOpacity,
              showscale: false,
            },
            text: filteredUnhighlightedPoints.map(formatHoverText),
            hovertemplate: '<b>%{text}</b><extra></extra>',
            customdata: filteredUnhighlightedPoints as any,
            showlegend: false,
          } satisfies PlotlyData);
        } else if (colorBy === 'category' && categoryValues.length > 0) {
          // MODE: CATEGORICAL - preserve category colors with dimming
          if (nestedColorMap) {
            // NESTED: group by subtopic_label, color from nestedColorMap
            const pointsBySub: Record<string, Point2D[]> = {};
            filteredUnhighlightedPoints.forEach(point => {
              const sub = String(point.metadata?.['subtopic_label'] ?? point.metadata?.['topic_label'] ?? 'unknown');
              if (!pointsBySub[sub]) pointsBySub[sub] = [];
              pointsBySub[sub].push(point);
            });

            Object.entries(pointsBySub).forEach(([sub, subPoints]) => {
              // Check if subtopic or its parent topic is muted
              const parentTopic = String(subPoints[0]?.metadata?.['topic_label'] ?? 'unknown');
              const isMuted = mutedCategories.includes(sub) || mutedCategories.includes(parentTopic);
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
                      color: isMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[sub] || '#7f7f7f'),
                      opacity: isMuted ? (mutedPointOpacity ?? 0.15) : dimOpacity,
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
                      opacity: mutedPointOpacity ?? 0.15,
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
                    color: isMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[sub] || '#7f7f7f'),
                    opacity: isMuted ? (mutedPointOpacity ?? 0.15) : dimOpacity,
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
              const raw = categoryField ? point.metadata?.[categoryField] : undefined;
              const cat = (raw !== null && raw !== undefined && raw !== '') ? String(raw) : 'unknown';
              if (!pointsByCategory[cat]) {
                pointsByCategory[cat] = [];
              }
              pointsByCategory[cat].push(point);
            });

            Object.entries(pointsByCategory).forEach(([cat, catPoints]) => {
              const isMuted = mutedCategories.includes(cat);
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
                      opacity: isMuted ? (mutedPointOpacity ?? 0.15) : dimOpacity,
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
                      opacity: mutedPointOpacity ?? 0.15,
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
                    opacity: isMuted ? (mutedPointOpacity ?? 0.15) : dimOpacity,
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
                  opacity: mutedPointOpacity ?? 0.15,
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
              color: isDark ? 'rgba(130, 160, 200, 0.12)' : 'rgba(100, 130, 170, 0.15)',
              width: 0.5,
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
          const hX: number[] = [];
          const hY: number[] = [];
          const outerSizes: number[] = [];
          const outerColors: string[] = [];
          const outerOpacities: number[] = [];
          const innerSizes: number[] = [];
          const innerColors: string[] = [];
          const innerOpacities: number[] = [];
          const coreSizes: number[] = [];
          const coreColors: string[] = [];
          const coreOpacities: number[] = [];
          const coreLineColors: string[] = [];
          const coreTexts: string[] = [];
          const coreCustomData: any[] = [];

          otherHighlights.forEach(point => {
            const similarity = highlightedIndices!.get(point.index) ?? 1.0;
            const luminosity = calculateLuminosity(similarity);
            const colors = calculateSimilarityColors(similarity);

            hX.push(point.x);
            hY.push(point.y);

            outerSizes.push(markerStyle.size * highlightScale.outerMultiplier);
            outerColors.push(colors.outerGlow);
            outerOpacities.push(luminosity.outer);

            innerSizes.push(markerStyle.size * highlightScale.innerMultiplier);
            innerColors.push(colors.glowColor);
            innerOpacities.push(luminosity.inner);

            coreSizes.push(Math.max(markerStyle.size * highlightScale.coreMultiplier, 3));
            coreColors.push(colors.coreColor);
            coreOpacities.push(luminosity.core);
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
            marker: {
              size: outerSizes,
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
            marker: {
              size: innerSizes,
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
            marker: {
              size: coreSizes,
              color: coreColors,
              opacity: 1,
              line: { color: coreLineColors[0] || 'rgba(100, 150, 255, 0.6)', width: 1 },
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
      // Apply hideUnclustered filter
      // Check topic_id for -1 OR topic_label for "Unclustered" to handle all dataset shapes
      const displayPoints = hideUnclustered
        ? points.filter(p => {
            const topicId = p.metadata?.['topic_id'];
            if (topicId === '-1' || topicId === -1) return false;
            const topicLabel = p.metadata?.['topic_label'];
            if (topicLabel === 'Unclustered' || topicLabel === 'unclustered') return false;
            return true;
          })
        : points;

      if (numericData && plotlyColorScale) {
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
                color: activePoints.map(p => numericData.cleanValues[p.index]),
                colorscale: plotlyColorScale as any,
                cmin: numericData.min,
                cmax: numericData.max,
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
                opacity: mutedPointOpacity ?? 0.15,
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
              color: displayPoints.map(p => numericData.cleanValues[p.index]),
              colorscale: plotlyColorScale as any,
              cmin: numericData.min,
              cmax: numericData.max,
              opacity: markerStyle.opacity,
              showscale: false, // Disabled - using custom Legend component instead
            },
            text: displayPoints.map(formatHoverText),
            hovertemplate: '<b>%{text}</b><extra></extra>',
            customdata: displayPoints as any,
          } satisfies PlotlyData];
        }
      } else if (colorBy === 'category' && categoryValues.length > 0) {
        if (nestedColorMap) {
          // NESTED: group by subtopic_label
          const pointsBySub: Record<string, Point2D[]> = {};
          displayPoints.forEach(point => {
            const sub = String(point.metadata?.['subtopic_label'] ?? point.metadata?.['topic_label'] ?? 'unknown');
            if (!pointsBySub[sub]) pointsBySub[sub] = [];
            pointsBySub[sub].push(point);
          });

          Object.entries(pointsBySub).forEach(([sub, subPoints]) => {
            const parentTopic = String(subPoints[0]?.metadata?.['topic_label'] ?? 'unknown');
            const isMuted = mutedCategories.includes(sub) || mutedCategories.includes(parentTopic);
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
                    color: isMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[sub] || '#7f7f7f'),
                    opacity: isMuted ? (mutedPointOpacity ?? 0.15) : markerStyle.opacity,
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
                    opacity: mutedPointOpacity ?? 0.15,
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
                  color: isMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[sub] || '#7f7f7f'),
                  opacity: isMuted ? (mutedPointOpacity ?? 0.15) : markerStyle.opacity,
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
            const raw = categoryField ? point.metadata?.[categoryField] : undefined;
            const cat = (raw !== null && raw !== undefined && raw !== '') ? String(raw) : 'unknown';
            if (!pointsByCategory[cat]) {
              pointsByCategory[cat] = [];
            }
            pointsByCategory[cat].push(point);
          });

          Object.entries(pointsByCategory).forEach(([cat, catPoints]) => {
            const isMuted = mutedCategories.includes(cat);
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
                    opacity: isMuted ? (mutedPointOpacity ?? 0.15) : markerStyle.opacity,
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
                    opacity: mutedPointOpacity ?? 0.15,
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
                  opacity: isMuted ? (mutedPointOpacity ?? 0.15) : markerStyle.opacity,
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
                opacity: mutedPointOpacity ?? 0.15,
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

    // Add cluster traces
    //if (clusters.length > 0) {
    //   const x: (number | null)[] = [];
    //   const y: (number | null)[] = [];
       
    //   clusters.forEach(cluster => {
    //     cluster.contour.forEach(p => {
    //       x.push(p.x);
    //       y.push(p.y);
    //     });
         // Close the loop
    /*     if (cluster.contour.length > 0) {
            x.push(cluster.contour[0].x);
            y.push(cluster.contour[0].y);
         }
         // Separator
         x.push(null);
         y.push(null);
       });

       traces.push({
         x,
         y,
         mode: 'lines',
         type: 'scattergl',
         name: 'Clusters',
         line: {
           color: isDark ? 'rgba(255, 255, 255, 0.5)' : 'rgba(0, 0, 0, 0.5)',
           width: 1
         },
         hoverinfo: 'skip',
         showlegend: false
       });
    }
        */

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
  }, [points, colorBy, categoryField, categoryValues, colorMap, numericData, plotlyColorScale, highlightedIndices, renderedSelectedPoint, isDark, markerStyle.size, markerStyle.opacity, highlightScale, showOnlyHighlighted, showLabels, mutedCategories, hideUnclustered, nestedColorMap, combinedMutedIndices, hideFilteredPoints, mutedPointOpacity]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      width,
      height,
      uirevision: 'true', // Preserve zoom/pan state on resize
      dragmode: 'pan',
      hovermode: 'closest' as const,
      showlegend: false, 
      // showlegend: colorBy === 'category' && categoryValues.length > 0,
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
        zerolinecolor: gridColor,
        tickfont: { color: axisColor },
        showgrid: false, 
        zeroline: false,
        showspikes: false,
        showticklabels: false,
        showtitle: false,
      },
      yaxis: {
        title: { text: '', font: { color: axisColor } },
        gridcolor: gridColor,
        zerolinecolor: gridColor,
        tickfont: { color: axisColor },
        scaleanchor: 'x',
        scaleratio: 1,
        showgrid: false, 
        zeroline: false,
        showspikes: false,
        showticklabels: false,
        showtitle: false,
      },
      margin: { l: 50, r: 50, t: 50, b: 50 },
    }),
    [axisColor, colorBy, categoryValues.length, gridColor, height, legendBg, paperBg, plotBg, width]
  );

  const config = useMemo<Partial<Config>>(() => ({
    displayModeBar: true,
    displaylogo: false,
    responsive: true,
    scrollZoom: true,
  }), []);

  const handleClick = (event: PlotMouseEvent) => {
    if (!onPointClick || !event.points || event.points.length === 0) {
      return;
    }

    const point = event.points[0];
    // Skip glow layers and lines (they don't have valid customdata)
    if (point.customdata && typeof point.customdata === 'object') {
      onPointClick(point.customdata as unknown as Point2D);
    }
  };

  const [plotReady, setPlotReady] = useState(false);
  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);

  const handleHover = (event: PlotHoverEvent) => {
    if (event.points && event.points.length > 0) {
      const pt = event.points[0];
      // Only show tooltip for points with valid customdata (skip glow layers)
      if (pt.customdata && typeof pt.customdata === 'object') {
        const point = pt.customdata as unknown as Point2D;
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
  };

  const handleUnhover = () => {
    setTooltipData(null);
  };

  const hasClusterLabels = showClusterLabels && clusterDataMap.size > 0;

return (
  <div
    ref={containerRef}
    className={className ?? 'h-full w-full'}
    style={{ position: 'relative' }}
    onClickCapture={hasClusterLabels ? handleContainerClick : undefined}
  >
    <Plot
      data={plotData}
      layout={layout}
      config={config}
      onClick={plotReady ? handleClick : undefined}
      onHover={handleHover}
      onUnhover={handleUnhover}
      onRelayout={handleRelayout}
      onInitialized={(_figure, graphDiv) => { graphDivRef.current = graphDiv; setPlotReady(true); }}
    />
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