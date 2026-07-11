'use client';

import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';
import type { PlotData, Layout, Config, PlotMouseEvent, PlotRelayoutEvent } from 'plotly.js';
import type { Point3D, HighlightMap, NestedColorMap, ColorScale, CustomNumericRange } from '../../lib/types/types';
import { useTheme } from 'next-themes';
import { buildCategoryColorMap, getCategoryLabel, getSequentialScale, getDivergingScale, getMonochromeScale, desaturateHex } from '../../lib/utils/categoryColors';
import { isCrameriScale, getCrameriPlotlyScale } from '../../lib/colorMaps/crameriScales';
import { useColorScaleReady } from '../../lib/hooks/useColorScaleReady';
import { calculateMarkerStyle, calculateHighlightScale, calculateSimilarityColors } from '../../lib/utils/plotUtils';
import { useContainerDimensions } from '../../lib/hooks/useContainerDimensions';
import { FrostedTooltip, type TooltipData } from './FrostedTooltip';
import { useCameraFlyTo, animateCameraToRegion } from '../../lib/hooks/cameraAnimation';
import type { DataBounds } from '../utils/rendeding';
import { groupPointsByCluster, type ClusterData } from '../../lib/utils/clusterGeometry';
import { HazeRenderer } from '../../lib/utils/hazeRenderer';
import { computeMVP, projectToScreen, multiplyMat4Vec4 } from '../utils/labelPlacement';
import { buildDataToSceneMatrix, getSceneNormalization } from '../utils/rendeding';
import { computeOverlayLayout, overlayLayoutEqual, type OverlayLayout } from '../../lib/utils/overlayLayout';
import { CollisionGrid, type BoundingBox } from '../../lib/utils/collisionGrid';
import { useVisualizationStore } from '../../lib/stores/useVisualizationStore';
import { build3DModeBarButtons } from '../../lib/utils/plotlyIcons';
import { capture3DPlot } from '../../lib/utils/plotCapture';
import { toast } from 'sonner';

type PlotlyData = Partial<PlotData>;

/** Build a scatter3d trace from a point subset with the given marker options. */
function buildScatter3dTrace(
  pts: Point3D[],
  opts: {
    name: string;
    size: number;
    color: string | number[];
    opacity: number;
    colorscaleOpts?: { colorscale: any; cmin: number; cmax: number };
  },
): PlotlyData {
  const trace: PlotlyData = {
    x: pts.map(p => p.x),
    y: pts.map(p => p.y),
    z: pts.map(p => p.z),
    mode: 'markers',
    type: 'scatter3d',
    name: opts.name,
    marker: {
      sizemode: 'diameter',
      size: opts.size,
      color: opts.color as any,
      opacity: opts.opacity,
      ...(opts.colorscaleOpts && {
        colorscale: opts.colorscaleOpts.colorscale,
        cmin: opts.colorscaleOpts.cmin,
        cmax: opts.colorscaleOpts.cmax,
        showscale: false,
      }),
    },
    hoverinfo: 'none',
    customdata: pts.map(p => p.index) as any,
    showlegend: false,
  };
  return trace;
}

/** Build a scatter3d trace from indexed subsets of pre-computed arrays (for numeric colorscale mode). */
function buildIndexedScatter3dTrace(
  allX: ArrayLike<number>, allY: ArrayLike<number>, allZ: ArrayLike<number>,
  indices: number[],
  pointIndices: number[],
  opts: {
    name: string;
    size: number;
    color: string | number[];
    opacity: number;
    colorscaleOpts?: { colorscale: any; cmin: number; cmax: number };
  },
): PlotlyData {
  const trace: PlotlyData = {
    x: indices.map(i => allX[i]),
    y: indices.map(i => allY[i]),
    z: indices.map(i => allZ[i]),
    mode: 'markers',
    type: 'scatter3d',
    name: opts.name,
    marker: {
      sizemode: 'diameter',
      size: opts.size,
      color: opts.color as any,
      opacity: opts.opacity,
      ...(opts.colorscaleOpts && {
        colorscale: opts.colorscaleOpts.colorscale,
        cmin: opts.colorscaleOpts.cmin,
        cmax: opts.colorscaleOpts.cmax,
        showscale: false,
      }),
    },
    hoverinfo: 'none',
    customdata: indices.map(i => pointIndices[i]) as any,
    showlegend: false,
  };
  return trace;
}

interface ScatterPlot3DProps {
  points: Point3D[];
  categoryField?: string | null;  // Field to color by (used for both categorical AND numeric)
  categoryValues?: string[];
  colorScale?: ColorScale;
  highlightedIndices?: HighlightMap;
  selectedPoint?: Point3D | null;
  onPointClick?: (point: Point3D) => void;
  className?: string;
  showOnlyHighlighted?: boolean;
  showLabels?: boolean;
  mutedCategories?: string[];
  /** Extra metadata fields to show in hover tooltip */
  tooltipFields?: string[];
  /** When true, hide points with topic_id = -1 (unclustered/noise) */
  hideUnclustered?: boolean;
  /** Crameri categorical palette name for category coloring */
  categoricalPalette?: string;
  /** Nested topic/subtopic color map for hierarchical coloring */
  nestedColorMap?: NestedColorMap | null;
  /** Enable nebula haze effects around topic clusters */
  nebulaMode?: boolean;
  /** Show topic/subtopic names at cluster centroids */
  showClusterLabels?: boolean;
  /** Callback when a cluster label is clicked (topic toggle) */
  onClusterLabelClick?: (topicId: number) => void;
  /** Map from topic label string → topic ID for click handling */
  topicLabelToIdMap?: Map<string, number> | null;
  /** Combined indices to mute (temporal + text search) */
  combinedMutedIndices?: Set<number> | null;
  /** Remove muted points entirely instead of graying out */
  hideFilteredPoints?: boolean;
  /** 0-1 multiplier applied to the base opacity for muted points (default 0.20) */
  mutedPointOpacity?: number;
  /** 0-1 multiplier applied to the base opacity for normal points (default 1.0) */
  pointOpacity?: number;
  /** Custom numeric range overrides for cmin/cmax */
  customNumericRange?: CustomNumericRange | null;
  /** Callback when a point is right-clicked (contextmenu) */
  onPointContextMenu?: (point: Point3D, event: MouseEvent) => void;
  /** Show axis lines from origin */
  showAxes?: boolean;
  /** Selected dimension indices for axis labels (manual mode) */
  selectedDimensions?: number[];
  /** Called with the composited plot canvas when the modebar screenshot button is clicked */
  onScreenshot?: (canvas: HTMLCanvasElement) => void;
}

interface PlotlyGraphDiv extends HTMLDivElement {
  data?: PlotData[];
  _fullLayout?: {
    scene?: {
      camera?: any;
      _scene?: {
        glplot?: {
          camera?: any;
          draw?: () => void;
        };
      };
    };
  };
}

export const ScatterPlot3D = React.memo(function ScatterPlot3D({
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
  nebulaMode = false,
  showClusterLabels = false,
  combinedMutedIndices,
  hideFilteredPoints = false,
  mutedPointOpacity,
  pointOpacity,
  customNumericRange,
  onPointContextMenu,
  showAxes = false,
  selectedDimensions,
  onScreenshot,
}: ScatterPlot3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const hoveredPointRef = useRef<Point3D | null>(null);
  const { width, height } = useContainerDimensions(containerRef, { width: 800, height: 600 });

  // Deferred selected point: only sync to traces when highlightedIndices changes,
  // keeping plotData stable during camera fly-to animation (avoids expensive Plotly.react)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const renderedSelectedPoint = useMemo(() => selectedPoint, [highlightedIndices]);

  const { resolvedTheme } = useTheme();
  const theme = resolvedTheme ?? 'light';
  const isDark = theme === 'dark';
  // Theme colors
  const axisColor = isDark ? '#e2e8f0' : '#0f172a';
  const sceneBg = isDark ? 'rgba(0,0,0,0)' : '#ffffff';
  const paperBg = isDark ? 'rgba(0,0,0,0)' : '#ffffff';

  // --- Camera & Bounds Logic ---
  const bounds = useMemo(() => {
    if (points.length === 0) return null;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    points.forEach(p => {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
      if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z;
    });

    return { minX, maxX, minY, maxY, minZ, maxZ };
  }, [points]);


  //const defaultEye = { x: 0.9, y: 0.9, z: 0.9 };
  const defaultCenter = { x: 0, y: 0, z: 0 };
  const pendingFlyToRef = useRef<Point3D | null>(null);
  const graphDivRef = useRef<PlotlyGraphDiv | null>(null);
  const pointsRef = useRef(points);
  pointsRef.current = points;
  const [plotReady, setPlotReady] = useState(false);
  const [plotlyLoaded, setPlotlyLoaded] = useState(false);
  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);
  const plotlyLibRef = useRef<any>(null);

  // Calculate point count once for all zoom calculations
  const pointCount = points.length;

  const defaultEye = useMemo(() => {
    if (pointCount === 0) return { x: 2.5, y: 2.5, z: 2.5 };

    // INVERSE Logarithmic scaling:
    // Fewer points (<100) -> Start Far Away (e.g., 2.0)
    // Many points (>10k) -> Start Very Close (e.g., 0.6)
    
    const startDistance = 2.5; // Maximum distance (for sparse data)
    const zoomInRate = 0.4;    // How fast to zoom in per power of 10
    
    // Formula: MaxDist - (Rate * log10(count))
    // Example:
    // 10 pts   (Log 1) -> 2.5 - 0.4 = 2.1 (Far)
    // 1000 pts (Log 3) -> 2.5 - 1.2 = 1.3 (Medium)
    // 100k pts (Log 5) -> 2.5 - 2.0 = 0.5 (Close)
    const calculatedZoom = startDistance - (zoomInRate * Math.log10(pointCount));

    // Clamp: Never go closer than 0.1 (inside the points) or further than 2.5
    const zoom = Math.min(Math.max(calculatedZoom, 0.1), 1.5);

    return { x: zoom, y: zoom, z: zoom };
  }, [pointCount]);

  const currentCameraRef = useRef({ eye: defaultEye, center: defaultCenter });

  const defaultDistance = useMemo(() => {
    const { x, y, z } = defaultEye;
    return Math.sqrt(x * x + y * y + z * z);
  }, [defaultEye]);

  const labelCanvasRef = useRef<HTMLCanvasElement>(null);
  const hazeCanvasRef = useRef<HTMLCanvasElement>(null);
  const labelRenderDataRef = useRef<{
    points: { x: number; y: number; z: number; label: string; index: number }[];
    similarities: Map<number, number>;
    selectedIndex: number | null;
  } | null>(null);
  const renderLabelsRef = useRef<(() => void) | null>(null);

  const { startFlyTo, isAnimatingRef, animationFrameRef } = useCameraFlyTo(
    bounds, graphDivRef, currentCameraRef, plotlyLibRef, renderLabelsRef, labelCanvasRef,
  );

  // Load Plotly library and create initial plot imperatively (bypasses react-plotly.js
  // which does an O(n) deep-equality diff of all trace data on every React render)
  useEffect(() => {
    let cancelled = false;
    // NOTE: importing from 'plotly.js' (source build) rather than 'plotly.js-dist-min'
    // so that our patch to plotly.js/src/plots/gl3d/scene.js (fast-path skip of
    // trace.update when input refs unchanged) actually takes effect. The minified
    // pre-built bundle would bypass it.
    import('plotly.js').then((lib) => {
      if (cancelled) return;
      plotlyLibRef.current = lib.default;
      // Trigger initial plot creation via the base-trace effect
      setPlotReady(false);   // reset so newPlot runs
      setPlotlyLoaded(true);
    });
    return () => { cancelled = true; };
  }, []);


  // Ref to track bounds for projection (avoids stale closure)
  const boundsRef = useRef(bounds);
  boundsRef.current = bounds;

  // Lazy cache for the data→scene model matrix. Axis ranges are pinned
  // (autorange:false), so this matrix is constant during camera interaction and
  // only changes when `bounds` changes. Recompute only on a bounds-identity miss,
  // avoiding a Float32Array alloc + layout read + cbrt on every rendered frame.
  const modelMatrixCacheRef = useRef<{ bounds: DataBounds | null; matrix: Float32Array } | null>(null);
  const getModelMatrix = useCallback((currentBounds: DataBounds | null): Float32Array => {
    const cache = modelMatrixCacheRef.current;
    if (cache && cache.bounds === currentBounds) return cache.matrix;
    const matrix = buildDataToSceneMatrix(graphDivRef.current, currentBounds ?? undefined);
    // Only cache once Plotly's scene axis ranges exist: before that,
    // buildDataToSceneMatrix falls back to unpadded raw bounds, and caching that
    // would lock in a ~5%-off matrix until `bounds` next changes. Recompute
    // (cheaply) each frame until the ranges are populated.
    const rangesReady = !!(graphDivRef.current as any)?._fullLayout?.scene?.xaxis?.range;
    if (rangesReady) modelMatrixCacheRef.current = { bounds: currentBounds, matrix };
    return matrix;
  }, []);




  const handleRelayout = useCallback((e: Readonly<PlotRelayoutEvent>) => {
    if (isAnimatingRef.current) return;
    const sceneCamera = (e as any)['scene.camera'];
    if (sceneCamera) {
      if (sceneCamera.eye) currentCameraRef.current.eye = sceneCamera.eye;
      if (sceneCamera.center) currentCameraRef.current.center = sceneCamera.center;
    }
  }, []);

  const colorOverrides = useVisualizationStore(
    (s) => categoryField ? s.categoryColorOverrides[categoryField] : undefined
  );
  const colorMap = useMemo(() => {
    return buildCategoryColorMap(categoryField, categoryValues, categoricalPalette, colorOverrides);
  }, [categoryField, categoryValues, categoricalPalette, colorOverrides]);

  // --- 1. OPTIMIZED DATA EXTRACTION (Raw Numbers) ---
  const numericData = useMemo(() => {
    if (colorScale.type === 'categorical' || !categoryField) return null;

    // Extract raw numbers instead of mapping to hex strings
    const values: (number | null)[] = points.map(p => {
      const val = p.metadata?.[categoryField];
      if (typeof val === 'number') return val;
      if (typeof val === 'string') return parseFloat(val);
      return null;
    });

    const validValues = values.filter((v): v is number => v !== null && !isNaN(v));

    // Fallback if no valid numbers
    if (validValues.length === 0) return null;

    let min = Infinity;
    let max = -Infinity;
    for (const v of validValues) {
      if (v < min) min = v;
      if (v > max) max = v;
    }

    return {
      values,
      min,
      max,
      // Pass nulls as NaN for Plotly to render as transparent/grey if needed
      cleanValues: values.map(v => (v === null || isNaN(v)) ? NaN : v)
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
      toLog,
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

  // --- 2. GENERATE PLOTLY NATIVE COLORSCALE ---
  // Bridge the ColorScale union to a Plotly array [[0, 'hex'], [1, 'hex']]
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

    // Request a normalized interpolator (0 to 1) from your utils
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

    // Sample the function to create a gradient definition
    const steps = 20;
    return Array.from({ length: steps + 1 }, (_, i) => {
      const t = i / steps;
      return [t, scaleFunc(t)]; // [0.1, '#ff0000']
    });
  }, [colorScale, colormapTick]);

  const markerStyle = useMemo(() => calculateMarkerStyle(points.length), [points.length]);
  const highlightScale = useMemo(() => calculateHighlightScale(points.length), [points.length]);

  // Shared filtered points: used by both baseTraces and clusterDataMap
  const displayPoints = useMemo(() => {
    if (!hideUnclustered) return points;
    return points.filter(p => {
      const topicId = p.metadata?.['topic_id'];
      if (topicId === '-1' || topicId === -1) return false;
      const topicLabel = p.metadata?.['topic_label'];
      if (topicLabel === 'Unclustered' || topicLabel === 'unclustered') return false;
      return true;
    });
  }, [points, hideUnclustered]);

  // --- OPTIMIZED TRACES ---
  const baseTraces = useMemo((): PlotlyData[] => {
    if (showOnlyHighlighted) return [];

    const traces: PlotlyData[] = [];
    // Light backgrounds wash out colored points — boost opacity
    const dimOpacity = Math.min(markerStyle.opacity * (isDark ? 1.0 : 1.6) * (pointOpacity ?? 1.0), 1.0);
    const dimSize = Math.max(markerStyle.size * 0.7, 2);
    const mutedOp = dimOpacity * (mutedPointOpacity ?? 0.20);

    // Helper: split a point array into active/muted subsets
    const splitByMuted = (pts: Point3D[], mutedSet: Set<number>) => {
      const active: Point3D[] = [];
      const muted: Point3D[] = [];
      for (const p of pts) {
        if (mutedSet.has(p.index)) muted.push(p);
        else active.push(p);
      }
      return { active, muted };
    };

    // Helper: push active + muted traces for a categorical group
    const pushCategoricalTraces = (pts: Point3D[], name: string, color: string, isMuted: boolean) => {
      if (hideFilteredPoints && isMuted) return;
      const catColor = isMuted ? '#9ca3af' : color;
      const catOpacity = isMuted ? mutedOp : dimOpacity;

      if (combinedMutedIndices) {
        const { active, muted } = splitByMuted(pts, combinedMutedIndices);
        if (active.length > 0) {
          traces.push(buildScatter3dTrace(active, { name, size: dimSize, color: catColor, opacity: catOpacity }));
        }
        if (muted.length > 0 && !hideFilteredPoints) {
          traces.push(buildScatter3dTrace(muted, { name, size: dimSize, color: '#9ca3af', opacity: mutedOp }));
        }
      } else {
        traces.push(buildScatter3dTrace(pts, { name, size: dimSize, color: catColor, opacity: catOpacity }));
      }
    };

    if (activeNumericData && plotlyColorScale) {
      // --- MODE: NATIVE COLORSCALE (GPU ACCELERATED) ---
      const csOpts = { colorscale: plotlyColorScale as any, cmin: effectiveRange!.min, cmax: effectiveRange!.max };
      const n = displayPoints.length;
      const allX = new Float64Array(n);
      const allY = new Float64Array(n);
      const allZ = new Float64Array(n);
      const pointIndices = new Array<number>(n);
      for (let i = 0; i < n; i++) {
        const p = displayPoints[i];
        allX[i] = p.x;
        allY[i] = p.y;
        allZ[i] = p.z;
        pointIndices[i] = p.index;
      }

      if (combinedMutedIndices) {
        const activeIdx: number[] = [];
        const mutedIdx: number[] = [];
        displayPoints.forEach((p, i) => {
          if (combinedMutedIndices.has(p.index)) mutedIdx.push(i);
          else activeIdx.push(i);
        });
        if (activeIdx.length > 0) {
          traces.push(buildIndexedScatter3dTrace(allX, allY, allZ, activeIdx, pointIndices, {
            name: 'Data', size: dimSize,
            color: activeIdx.map(i => activeNumericData.cleanValues[i]),
            opacity: dimOpacity, colorscaleOpts: csOpts,
          }));
        }
        if (mutedIdx.length > 0 && !hideFilteredPoints) {
          traces.push(buildIndexedScatter3dTrace(allX, allY, allZ, mutedIdx, pointIndices, {
            name: 'Temporal (muted)', size: dimSize, color: '#9ca3af', opacity: mutedOp,
          }));
        }
      } else {
        traces.push({
          x: allX, y: allY, z: allZ,
          mode: 'markers', type: 'scatter3d', name: 'Data',
          marker: {
            sizemode: 'diameter', size: dimSize,
            color: activeNumericData.cleanValues as any,
            colorscale: csOpts.colorscale, cmin: csOpts.cmin, cmax: csOpts.cmax,
            opacity: dimOpacity, showscale: false,
          },
          hoverinfo: 'none', customdata: pointIndices as any, showlegend: false,
        });
      }
    } else if (categoryField != null && categoryValues.length > 0) {
      if (nestedColorMap) {
        // --- MODE: NESTED CATEGORICAL ---
        const pointsBySub: Record<string, Point3D[]> = {};
        for (const point of displayPoints) {
          const sub = String(point.metadata?.['subtopic_label'] ?? point.metadata?.['topic_label'] ?? 'unknown');
          (pointsBySub[sub] ??= []).push(point);
        }
        for (const [sub, subPoints] of Object.entries(pointsBySub)) {
          const parentTopic = String(subPoints[0]?.metadata?.['topic_label'] ?? 'unknown');
          const isMuted = mutedCategories.includes(sub) || mutedCategories.includes(parentTopic);
          pushCategoricalTraces(subPoints, sub, nestedColorMap.subtopicColors[sub] || '#7f7f7f', isMuted);
        }
      } else {
        // --- MODE: CATEGORICAL (Standard) ---
        const pointsByCategory: Record<string, Point3D[]> = {};
        for (const point of displayPoints) {
          const raw = categoryField ? point.metadata?.[categoryField] : undefined;
          const cat = (raw !== null && raw !== undefined && raw !== '') ? String(raw) : 'unknown';
          (pointsByCategory[cat] ??= []).push(point);
        }
        for (const [cat, catPoints] of Object.entries(pointsByCategory)) {
          const isMuted = mutedCategories.includes(cat);
          pushCategoricalTraces(catPoints, getCategoryLabel(categoryField, cat), colorMap[cat] || '#7f7f7f', isMuted);
        }
      }
    } else {
      // --- MODE: NO COLORING ---
      if (combinedMutedIndices) {
        const { active, muted } = splitByMuted(displayPoints, combinedMutedIndices);
        if (active.length > 0) {
          traces.push(buildScatter3dTrace(active, { name: 'Data', size: dimSize, color: '#1f77b4', opacity: dimOpacity }));
        }
        if (muted.length > 0 && !hideFilteredPoints) {
          traces.push(buildScatter3dTrace(muted, { name: 'Temporal (muted)', size: dimSize, color: '#9ca3af', opacity: mutedOp }));
        }
      } else {
        traces.push(buildScatter3dTrace(displayPoints, { name: 'Data', size: dimSize, color: '#1f77b4', opacity: dimOpacity }));
      }
    }

    // Axis lines from origin when showAxes is enabled
    if (showAxes && bounds) {
      const lineColor = isDark ? '#94a3b8' : '#475569';
      const dims = selectedDimensions ?? [0, 1, 2];

      // Full axis lines spanning the data range (null-separated segments)
      traces.push({
        x: [bounds.minX, bounds.maxX, null as any, 0, 0, null as any, 0, 0],
        y: [0, 0, null as any, bounds.minY, bounds.maxY, null as any, 0, 0],
        z: [0, 0, null as any, 0, 0, null as any, bounds.minZ, bounds.maxZ],
        mode: 'lines' as const, type: 'scatter3d' as const,
        line: { width: 5, color: lineColor },
        hoverinfo: 'skip' as any, showlegend: false,
      });

      // Dimension labels at the positive ends
      traces.push({
        x: [bounds.maxX, 0, 0], y: [0, bounds.maxY, 0], z: [0, 0, bounds.maxZ],
        mode: 'text' as const, type: 'scatter3d' as const,
        text: [String(dims[0]), String(dims[1]), String(dims[2])],
        textposition: 'top center' as any,
        textfont: { family: 'Geist Mono, monospace', size: 14, color: lineColor },
        hoverinfo: 'skip' as any, showlegend: false,
      });
    }

    return traces;
  }, [
    displayPoints, markerStyle, showOnlyHighlighted,
    categoryValues, colorMap, activeNumericData, effectiveRange, plotlyColorScale, categoryField,
    mutedCategories, nestedColorMap, combinedMutedIndices, hideFilteredPoints, mutedPointOpacity, pointOpacity, isDark,
    showAxes, bounds, selectedDimensions
  ]);

  // Bounds of active (non-muted) points — computed directly from muting state
  const activeBounds = useMemo<DataBounds | null>(() => {
    if (!bounds) return null;
    const hasMuting = mutedCategories.length > 0 || (combinedMutedIndices && combinedMutedIndices.size > 0);
    if (!hasMuting) return bounds;

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    let hasData = false;

    for (const p of displayPoints) {
      if (combinedMutedIndices?.has(p.index)) continue;
      if (mutedCategories.length > 0) {
        if (nestedColorMap) {
          const sub = String(p.metadata?.['subtopic_label'] ?? p.metadata?.['topic_label'] ?? 'unknown');
          const parent = String(p.metadata?.['topic_label'] ?? 'unknown');
          if (mutedCategories.includes(sub) || mutedCategories.includes(parent)) continue;
        } else if (categoryField) {
          const cat = String(p.metadata?.[categoryField] ?? 'unknown');
          if (mutedCategories.includes(cat)) continue;
        }
      }
      hasData = true;
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
      if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z;
    }
    if (!hasData) return bounds;
    return { minX, maxX, minY, maxY, minZ, maxZ };
  }, [displayPoints, bounds, mutedCategories, combinedMutedIndices, categoryField, nestedColorMap]);

  // Track whether camera was refitted so we can animate back when muting clears
  const wasRefittedRef = useRef(false);

  // Auto-refit camera to frame active points when muting is active
  useEffect(() => {
    if (!plotReady || !plotlyLibRef.current || !graphDivRef.current || !activeBounds || !bounds) return;
    if (isAnimatingRef.current) return;

    const hasMuting = activeBounds !== bounds; // different ref = muting active

    if (!hasMuting && !wasRefittedRef.current) return; // initial state, nothing to do

    if (!hasMuting && wasRefittedRef.current) {
      // Muting cleared — animate back to default view
      wasRefittedRef.current = false;
      animateCameraToRegion({
        targetEye: defaultEye, targetCenter: defaultCenter, duration: 2000,
        graphDivRef, currentCameraRef, plotlyLibRef,
        isAnimatingRef, animationFrameRef,
        renderLabelsRef, labelCanvasRef,
      });
      return;
    }

    // Muting active — animate to frame visible points
    wasRefittedRef.current = true;

    const cx = (activeBounds.minX + activeBounds.maxX) / 2;
    const cy = (activeBounds.minY + activeBounds.maxY) / 2;
    const cz = (activeBounds.minZ + activeBounds.maxZ) / 2;

    const visExtentX = activeBounds.maxX - activeBounds.minX;
    const visExtentY = activeBounds.maxY - activeBounds.minY;
    const visExtentZ = activeBounds.maxZ - activeBounds.minZ;
    const fullExtentX = bounds.maxX - bounds.minX;
    const fullExtentY = bounds.maxY - bounds.minY;
    const fullExtentZ = bounds.maxZ - bounds.minZ;

    const extentRatio = Math.max(
      visExtentX / (fullExtentX || 1),
      visExtentY / (fullExtentY || 1),
      visExtentZ / (fullExtentZ || 1),
      0.05,
    );

    const baseDistance = Math.sqrt(defaultEye.x ** 2 + defaultEye.y ** 2 + defaultEye.z ** 2);
    const factor = Math.max(extentRatio, 0.3);

    const norm = getSceneNormalization(graphDivRef.current, bounds);
    const targetCenter = norm
      ? { x: (cx - norm.centerX) / norm.normRange, y: (cy - norm.centerY) / norm.normRange, z: (cz - norm.centerZ) / norm.normRange }
      : { x: 0, y: 0, z: 0 };

    const targetEye = {
      x: targetCenter.x + defaultEye.x * factor,
      y: targetCenter.y + defaultEye.y * factor,
      z: targetCenter.z + defaultEye.z * factor,
    };

    animateCameraToRegion({
      targetEye, targetCenter, duration: 2000,
      graphDivRef, currentCameraRef, plotlyLibRef,
      isAnimatingRef, animationFrameRef,
      renderLabelsRef, labelCanvasRef,
    });
  }, [activeBounds, plotReady, bounds, defaultEye, isAnimatingRef, animationFrameRef]);

  // Pre-compute highlighted points via direct index lookup — O(k) not O(n)
  const highlightedPoints = useMemo(() => {
    if (!highlightedIndices || highlightedIndices.size === 0) return [];
    const result: Point3D[] = [];
    for (const idx of highlightedIndices.keys()) {
      if (points[idx]) result.push(points[idx]);
    }
    return result;
  }, [highlightedIndices, points]);

  // All overlay traces in a single memo — search result glows, selected point, and
  // connection lines are computed atomically so Plotly.redraw() fires once.
  // The selected point is included in the same glow traces (similarity 1.0 = max brightness).
  const overlayTraces = useMemo((): PlotlyData[] => {
    const hasHighlights = highlightedIndices && highlightedIndices.size > 0 && highlightedPoints.length > 0;
    if (!hasHighlights && !renderedSelectedPoint) return [];

    const traces: PlotlyData[] = [];

    // Build combined point list: highlighted points + selected point (at similarity 1.0)
    const allOverlayPoints: Point3D[] = hasHighlights ? [...highlightedPoints] : [];
    const selectedIdx = renderedSelectedPoint
      ? allOverlayPoints.findIndex(p => p.index === renderedSelectedPoint.index)
      : -1;
    // Add selected point if not already in highlights
    if (renderedSelectedPoint && selectedIdx === -1) {
      allOverlayPoints.push(renderedSelectedPoint);
    }

    if (allOverlayPoints.length > 0) {
      const hX = allOverlayPoints.map(p => p.x);
      const hY = allOverlayPoints.map(p => p.y);
      const hZ = allOverlayPoints.map(p => p.z);

      const outerSizes: number[] = [];
      const outerColors: string[] = [];
      const innerSizes: number[] = [];
      const innerColors: string[] = [];
      const coreSizes: number[] = [];
      const coreColors: string[] = [];
      const coreIndices: number[] = [];

      allOverlayPoints.forEach(point => {
        const isSelected = renderedSelectedPoint && point.index === renderedSelectedPoint.index;
        const similarity = isSelected ? 1.0 : (highlightedIndices?.get(point.index) ?? 1.0);
        const colors = calculateSimilarityColors(similarity, isDark);

        outerSizes.push(Math.max(markerStyle.size * (isSelected ? highlightScale.selectedOuterMultiplier : highlightScale.outerMultiplier), isSelected ? 35 : 30));
        outerColors.push(colors.outerGlow);

        innerSizes.push(Math.max(markerStyle.size * (isSelected ? highlightScale.selectedInnerMultiplier : highlightScale.innerMultiplier), isSelected ? 20 : 18));
        innerColors.push(colors.glowColor);

        coreSizes.push(Math.max(markerStyle.size * (isSelected ? highlightScale.selectedCoreMultiplier : highlightScale.coreMultiplier), isSelected ? 10 : 9));
        coreColors.push(colors.coreColor);
        coreIndices.push(point.index);
      });

      // --- Pass 1: All point overlays ---

      // 1. Outer Glow
      traces.push({
        x: hX, y: hY, z: hZ, mode: 'markers', type: 'scatter3d',
        marker: { sizemode: 'diameter', size: outerSizes, color: outerColors, opacity: 0.15, line: { width: 0 } },
        hoverinfo: 'skip', showlegend: false
      });

      // 2. Inner Glow
      traces.push({
        x: hX, y: hY, z: hZ, mode: 'markers', type: 'scatter3d',
        marker: { sizemode: 'diameter', size: innerSizes, color: innerColors, opacity: 0.3, line: { width: 0 } },
        hoverinfo: 'skip', showlegend: false
      });

      // 3. Core markers (hoverable)
      traces.push({
        x: hX, y: hY, z: hZ, mode: 'markers', type: 'scatter3d',
        marker: { sizemode: 'diameter', size: coreSizes, color: coreColors, opacity: 1, line: { color: innerColors, width: 1 } },
        hoverinfo: 'none', customdata: coreIndices as any, showlegend: false
      });

      // --- Pass 2: Connection lines ---
      if (renderedSelectedPoint && hasHighlights) {
        const lineX: number[] = [], lineY: number[] = [], lineZ: number[] = [];
        highlightedPoints.forEach(p => {
          if (p.index !== renderedSelectedPoint.index) {
            lineX.push(renderedSelectedPoint.x, p.x, null as any);
            lineY.push(renderedSelectedPoint.y, p.y, null as any);
            lineZ.push(renderedSelectedPoint.z, p.z, null as any);
          }
        });
        if (lineX.length > 0) {
          traces.push({
            x: lineX, y: lineY, z: lineZ, mode: 'lines' as const, type: 'scatter3d' as const,
            name: 'Connections',
            line: { color: isDark ? 'rgba(130, 160, 200, 0.60)' : 'rgba(71, 105, 155, 0.45)', width: isDark ? 0.1 : 0.75 },
            hoverinfo: 'skip' as any, showlegend: false
          });
        }
      }
    }

    return traces;
  }, [renderedSelectedPoint, highlightedIndices, highlightedPoints, markerStyle, highlightScale, isDark]);

  // Queue fly-to target from selectedPoint (not renderedSelectedPoint, which lags behind
  // for text searches because the auto-select effect in page.tsx fires after highlightedIndices).
  // The actual animation starts from the overlay-trace update effect (after Plotly.redraw).
  useEffect(() => {
    pendingFlyToRef.current = selectedPoint ?? null;
  }, [selectedPoint]);

  // Populate label render data (no React state — just a ref for the canvas renderer)
  useEffect(() => {
    if (!showLabels || !highlightedIndices || highlightedIndices.size === 0) {
      labelRenderDataRef.current = null;
      return;
    }
    if (highlightedPoints.length === 0) {
      labelRenderDataRef.current = null;
      return;
    }

    labelRenderDataRef.current = {
      points: highlightedPoints.map(p => ({
        x: p.x, y: p.y, z: p.z,
        label: p.label || p.id,
        index: p.index,
      })),
      similarities: highlightedIndices,
      selectedIndex: selectedPoint?.index ?? null,
    };
  }, [showLabels, highlightedIndices, highlightedPoints, selectedPoint]);

  // --- NEBULA / CLUSTER: Cluster data for nebula effects and cluster labels ---
  const clusterDataMap = useMemo(() => {
    if ((!nebulaMode && !showClusterLabels) || !categoryField) return new Map<string, ClusterData>();

    // Further filter by muted categories (displayPoints already handles hideUnclustered)
    let clusterPoints = displayPoints;
    if (mutedCategories.length > 0) {
      const mutedSet = new Set(mutedCategories);
      clusterPoints = displayPoints.filter(p => {
        const category = p.metadata?.[categoryField];
        return category == null || !mutedSet.has(String(category));
      });
    }

    // Exclude points muted by text search / temporal filters, so haze and
    // cluster labels track the visibly active points.
    if (combinedMutedIndices && combinedMutedIndices.size > 0) {
      clusterPoints = clusterPoints.filter(p => !combinedMutedIndices.has(p.index));
    }

    return groupPointsByCluster(clusterPoints, categoryField, colorMap, nestedColorMap);
  }, [nebulaMode, showClusterLabels, displayPoints, categoryField, colorMap, nestedColorMap, mutedCategories, combinedMutedIndices]);

  // --- Cluster label data for canvas overlay ---
  const clusterLabelDataRef = useRef<{
    labels: { x: number; y: number; z: number; label: string; color: string; pointCount: number }[];
  } | null>(null);

  useEffect(() => {
    if (!showClusterLabels || clusterDataMap.size === 0) {
      clusterLabelDataRef.current = null;
      return;
    }
    const labels: { x: number; y: number; z: number; label: string; color: string; pointCount: number }[] = [];
    for (const [key, cluster] of clusterDataMap) {
      labels.push({
        x: cluster.centroid.x,
        y: cluster.centroid.y,
        z: cluster.centroid.z,
        label: key,
        color: cluster.color,
        pointCount: cluster.points.length,
      });
    }
    clusterLabelDataRef.current = { labels };
  }, [showClusterLabels, clusterDataMap]);

  // --- Canvas label overlay ---
  const hasPointLabels = showLabels && highlightedIndices && highlightedIndices.size > 0;
  const hasClusterLabels = showClusterLabels && clusterDataMap.size > 0;
  const hasAnyLabels = hasPointLabels || hasClusterLabels;

  // Imperative: draw labels on the canvas overlay using CollisionGrid
  const renderLabels = useCallback(() => {
    const canvas = labelCanvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = width;
    const cssH = height;

    // Resize canvas backing store to match CSS size * DPR
    const bw = Math.round(cssW * dpr);
    const bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const pointData = labelRenderDataRef.current;
    const clusterData = clusterLabelDataRef.current;
    const currentBounds = boundsRef.current;
    const hasPointData = pointData && pointData.points.length > 0;
    const hasClusterData = clusterData && clusterData.labels.length > 0;
    if ((!hasPointData && !hasClusterData) || !currentBounds) return;

    // Get MVP from glplot internals
    const sceneLayout = (graphDivRef.current as any)?._fullLayout?.scene;
    const glplot = sceneLayout?._scene?.glplot;
    const cameraParams = glplot?.cameraParams || glplot?.camera;
    const projection = cameraParams?.projection || cameraParams?._projection;
    const view = cameraParams?.view || cameraParams?._view;
    if (!projection || !view) return;

    const model = getModelMatrix(currentBounds);
    const mvp = computeMVP(projection, view, model);

    // Get the actual GL canvas position relative to our overlay container.
    // Plotly's GL canvas may be offset within the plot div (margins, modebar, etc.)
    const gl = glplot?.gl as WebGLRenderingContext | null;
    const glCanvas = gl?.canvas as HTMLCanvasElement | undefined;

    let glOffsetX = 0;
    let glOffsetY = 0;
    let vpW: number;
    let vpH: number;

    if (glCanvas && gl) {
      // Offset + viewport size of the GL canvas within our overlay container (CSS px)
      const layout = computeOverlayLayout(container.getBoundingClientRect(), glCanvas.getBoundingClientRect());
      glOffsetX = layout.offsetX;
      glOffsetY = layout.offsetY;
      vpW = layout.cssW;
      vpH = layout.cssH;
    } else {
      vpW = cssW;
      vpH = cssH;
    }

    // Create collision grid over the full overlay canvas
    const gridBound: BoundingBox = { loX: 0, loY: 0, hiX: cssW, hiY: cssH };
    const grid = new CollisionGrid(gridBound, Math.max(cssW / 25, 1), Math.max(cssH / 50, 1));

    // --- Pass 1: Cluster labels (highest priority — inserted first) ---
    if (hasClusterData) {
      const clusterFontSize = 13;
      const clusterFontStr = `bold ${clusterFontSize}px Geist Mono, monospace`;
      ctx.font = clusterFontStr;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'center';

      const showAll = useVisualizationStore.getState().showAllClusterLabels;

      // In "show all" mode, iterate all labels at full opacity; otherwise sort by distance and cap
      const camEye = glplot.camera.eye as number[];
      const sorted = showAll
        ? clusterData!.labels.map(cl => ({ ...cl, dist: 0 }))
        : clusterData!.labels
            .map(cl => {
              const sp = multiplyMat4Vec4(model, [cl.x, cl.y, cl.z, 1]);
              const dx = sp[0] - camEye[0], dy = sp[1] - camEye[1], dz = sp[2] - camEye[2];
              return { ...cl, dist: Math.sqrt(dx * dx + dy * dy + dz * dz) };
            })
            .sort((a, b) => a.dist - b.dist || b.pointCount - a.pointCount);

      const total = sorted.length;
      const maxLabels = showAll ? total : Math.min(total, Math.max(8, Math.floor(Math.sqrt(total) * 2.5)));

      const minD = sorted[0]?.dist ?? 0;
      const maxD = sorted[Math.min(maxLabels, total) - 1]?.dist ?? 1;
      const dRange = maxD - minD || 1;

      for (let i = 0; i < maxLabels; i++) {
        const cl = sorted[i];
        const screen = projectToScreen(cl.x, cl.y, cl.z, mvp, vpW, vpH);
        if (!screen) continue;

        const sx = screen.x + glOffsetX;
        const sy = screen.y + glOffsetY - clusterFontSize * 1.5;

        const textWidth = ctx.measureText(cl.label).width;
        const pad = 4;
        const realBox: BoundingBox = {
          loX: sx - textWidth / 2 - pad,
          loY: sy - clusterFontSize * 0.6 - pad,
          hiX: sx + textWidth / 2 + pad,
          hiY: sy + clusterFontSize * 0.6 + pad,
        };
        if (!grid.insert(realBox)) continue;

        ctx.globalAlpha = showAll ? 1.0 : 1.0 - 0.35 * ((cl.dist - minD) / dRange);
        ctx.font = clusterFontStr;
        ctx.textAlign = 'center';
        ctx.strokeStyle = isDark ? 'rgba(15, 23, 42, 0.85)' : 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = isDark ? 4 : 3;
        ctx.lineJoin = 'round';
        ctx.strokeText(cl.label, sx, sy);
        ctx.fillStyle = desaturateHex(cl.color, isDark ? 0.1 : 0.35, isDark);
        ctx.fillText(cl.label, sx, sy);
      }
    }

    // --- Pass 2: Point labels (existing behavior) ---
    if (hasPointData) {
      const data = pointData!;
      // Sort candidates: selected first, then by similarity descending
      const sorted = data.points.map((p, i) => ({ ...p, i }));
      sorted.sort((a, b) => {
        const aSelected = a.index === data.selectedIndex ? 1 : 0;
        const bSelected = b.index === data.selectedIndex ? 1 : 0;
        if (aSelected !== bSelected) return bSelected - aSelected;
        return (data.similarities.get(b.index) ?? 0) - (data.similarities.get(a.index) ?? 0);
      });

      const fontSize = 11;
      const fontStr = `${fontSize}px Geist Mono, monospace`;
      ctx.font = fontStr;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      const offsetX = 8; // px right of projected point

      for (const candidate of sorted) {
        // Project to GL viewport coordinates, then shift by GL canvas offset
        const screen = projectToScreen(candidate.x, candidate.y, candidate.z, mvp, vpW, vpH);
        if (!screen) continue;

        // Convert from GL-viewport-local coords to overlay-canvas coords
        const sx = screen.x + glOffsetX;
        const sy = screen.y + glOffsetY;

        const isSelected = candidate.index === data.selectedIndex;
        const similarity = data.similarities.get(candidate.index) ?? 0;

        // Two-pass collision test (TensorBoard pattern):
        // Pass 1: thin box (width=1px) to cheaply reject dense areas
        const thinBox: BoundingBox = {
          loX: sx + offsetX,
          loY: sy - fontSize * 0.6,
          hiX: sx + offsetX + 1,
          hiY: sy + fontSize * 0.6,
        };
        if (!grid.insert(thinBox, true)) continue;

        // Pass 2: measure actual text width, build real bounding box
        const textWidth = ctx.measureText(candidate.label).width;
        const realBox: BoundingBox = {
          loX: sx + offsetX - 2,
          loY: sy - fontSize * 0.6 - 1,
          hiX: sx + offsetX + textWidth + 2,
          hiY: sy + fontSize * 0.6 + 1,
        };
        if (!grid.insert(realBox)) continue;

        // Draw label with stroke outline for contrast
        const alpha = isSelected ? 1.0 : 0.4 + similarity * 0.6;
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = isDark ? 'rgba(15, 23, 42, 0.8)' : 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = 3;
        ctx.lineJoin = 'round';
        ctx.strokeText(candidate.label, sx + offsetX, sy);
        ctx.fillStyle = isDark ? '#e2e8f0' : '#1e293b';
        ctx.fillText(candidate.label, sx + offsetX, sy);
      }
    }

    ctx.globalAlpha = 1.0;
  }, [width, height, isDark, showClusterLabels, clusterDataMap, getModelMatrix]);

  // Keep ref in sync so the camera animation can call renderLabels on completion
  renderLabelsRef.current = renderLabels;

  // Modebar screenshot: routed through a ref (reassigned every render) because
  // the config memo only rebuilds on [plotlyLoaded], so a plain closure would go stale.
  const screenshotHandlerRef = useRef<((gd: any) => void) | null>(null);
  screenshotHandlerRef.current = (gd: any) => {
    if (!onScreenshot || !containerRef.current) return;
    capture3DPlot({
      gd: gd ?? graphDivRef.current,
      container: containerRef.current,
      hazeCanvas: hazeCanvasRef.current,
      labelCanvas: labelCanvasRef.current,
      isDark,
    })
      .then(onScreenshot)
      .catch((err) => {
        console.error('Screenshot failed:', err);
        toast.error('Screenshot failed');
      });
  };

  // rAF-based camera polling: detect camera changes during 3D rotation/zoom/pan
  // (onRelayout does NOT fire during 3D mouse interaction)
  useEffect(() => {
    if (!hasAnyLabels || !plotReady || !graphDivRef.current) return;

    let rafId: number;
    const lastCam = { ex: 0, ey: 0, ez: 0, cx: 0, cy: 0, cz: 0 };

    const pollCamera = () => {
      const sceneLayout = (graphDivRef.current as any)?._fullLayout?.scene;
      const glplot = sceneLayout?._scene?.glplot;
      const camera = glplot?.camera;

      if (camera) {
        let ex: number, ey: number, ez: number, ccx: number, ccy: number, ccz: number;
        if (Array.isArray(camera.eye)) {
          [ex, ey, ez] = camera.eye;
          [ccx, ccy, ccz] = camera.center || [0, 0, 0];
        } else {
          ex = camera.eye?.x ?? 0; ey = camera.eye?.y ?? 0; ez = camera.eye?.z ?? 0;
          ccx = camera.center?.x ?? 0; ccy = camera.center?.y ?? 0; ccz = camera.center?.z ?? 0;
        }

        const changed =
          Math.abs(ex - lastCam.ex) > 1e-6 || Math.abs(ey - lastCam.ey) > 1e-6 ||
          Math.abs(ez - lastCam.ez) > 1e-6 || Math.abs(ccx - lastCam.cx) > 1e-6 ||
          Math.abs(ccy - lastCam.cy) > 1e-6 || Math.abs(ccz - lastCam.cz) > 1e-6;

        if (changed) {
          lastCam.ex = ex; lastCam.ey = ey; lastCam.ez = ez;
          lastCam.cx = ccx; lastCam.cy = ccy; lastCam.cz = ccz;

          // Keep currentCameraRef in sync
          currentCameraRef.current.eye = { x: ex, y: ey, z: ez };
          currentCameraRef.current.center = { x: ccx, y: ccy, z: ccz };

          // Skip expensive label rendering during fly-to animation to avoid stutter
          if (!isAnimatingRef.current) {
            renderLabels();
          }
        }
      }
      rafId = requestAnimationFrame(pollCamera);
    };

    // Render once immediately (unless animating), then start polling
    if (!isAnimatingRef.current) renderLabels();
    rafId = requestAnimationFrame(pollCamera);
    return () => cancelAnimationFrame(rafId);
  }, [hasAnyLabels, plotReady, renderLabels]);

  // Clear canvas when labels toggled off
  useEffect(() => {
    if (!hasAnyLabels && labelCanvasRef.current) {
      const ctx = labelCanvasRef.current.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, labelCanvasRef.current.width, labelCanvasRef.current.height);
    }
  }, [hasAnyLabels]);

  // Re-render labels on resize or when label data changes (e.g. search results arrive while camera is stationary)
  useEffect(() => {
    if (hasAnyLabels && !isAnimatingRef.current) renderLabels();
  }, [width, height, hasAnyLabels, renderLabels, showLabels, highlightedIndices, selectedPoint]);

  // Pin axis ranges to full bounds so Plotly doesn't auto-rerange when filtered traces shrink.
  // This keeps the scene coordinate space stable — only camera animation controls framing.
  const axisRange = useMemo(() => {
    if (!bounds) return null;
    const pad = 0.05; // 5% padding
    const px = (bounds.maxX - bounds.minX) * pad || 0.1;
    const py = (bounds.maxY - bounds.minY) * pad || 0.1;
    const pz = (bounds.maxZ - bounds.minZ) * pad || 0.1;
    return {
      x: [bounds.minX - px, bounds.maxX + px] as [number, number],
      y: [bounds.minY - py, bounds.maxY + py] as [number, number],
      z: [bounds.minZ - pz, bounds.maxZ + pz] as [number, number],
    };
  }, [bounds]);

  const layout = useMemo<Partial<Layout>>(() => ({
    width, height, autosize: true, uirevision: 'true', hovermode: 'closest', showlegend: false,
    paper_bgcolor: paperBg,
    font: { family: 'Courier New, monospace', color: axisColor },
    scene: {
      aspectmode: 'data',
      camera: {
        eye: defaultEye,
        center: defaultCenter,
        up: { x: 0, y: 0, z: 1 }
      },
      xaxis: { title: { text: '' }, backgroundcolor: sceneBg, showbackground: !showAxes, showgrid: false, zeroline: false, showticklabels: false, showspikes: false, ...(axisRange && { range: axisRange.x, autorange: false }) },
      yaxis: { title: { text: '' }, backgroundcolor: sceneBg, showbackground: !showAxes, showgrid: false, zeroline: false, showticklabels: false, showspikes: false, ...(axisRange && { range: axisRange.y, autorange: false }) },
      zaxis: { title: { text: '' }, backgroundcolor: sceneBg, showbackground: !showAxes, showgrid: false, zeroline: false, showticklabels: false, showspikes: false, ...(axisRange && { range: axisRange.z, autorange: false }) },
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
  }), [axisColor, axisRange, defaultEye, height, paperBg, sceneBg, width, showAxes]);

   
  const config = useMemo<Partial<Config>>(() => ({
    displayModeBar: true,
    displaylogo: false,
    responsive: true,
    modeBarButtons: build3DModeBarButtons(plotlyLibRef.current, onScreenshot ? screenshotHandlerRef : undefined),
  }), [plotlyLoaded]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- FULLY IMPERATIVE PLOTLY MANAGEMENT ---
  // Bypasses react-plotly.js which does an O(n) deep-equality diff on every React render.
  // With 200k points that diff alone freezes the main thread for seconds.

  const overlayTraceCountRef = useRef(0);
  const plotInitializedRef = useRef(false);

  // 1. Create the plot once when Plotly loads and the container is ready
  useEffect(() => {
    if (!plotlyLoaded || !plotlyLibRef.current || !graphDivRef.current) return;
    if (plotInitializedRef.current) return;

    const Plotly = plotlyLibRef.current;
    const gd = graphDivRef.current;

    Plotly.newPlot(gd, baseTraces as PlotData[], layout, config).then(() => {
      plotInitializedRef.current = true;
      overlayTraceCountRef.current = 0;
      setPlotReady(true);
    });

    return () => {
      if (plotInitializedRef.current && plotlyLibRef.current && graphDivRef.current) {
        plotlyLibRef.current.purge(graphDivRef.current);
        plotInitializedRef.current = false;
      }
    };
  // Only run on mount/unmount — subsequent updates handled by dedicated effects below
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plotlyLoaded]);

  // 2. Update base traces when they change (ref-equality check via deps — instant, not deep)
  useEffect(() => {
    if (!plotReady || !plotlyLibRef.current || !graphDivRef.current) return;
    const Plotly = plotlyLibRef.current;
    const gd = graphDivRef.current;

    // Rebuild all traces: base + current overlays
    const allTraces = [...baseTraces, ...overlayTraces];
    overlayTraceCountRef.current = overlayTraces.length;

    // Read the live glplot camera directly — currentCameraRef can be stale if the
    // rAF poll loop isn't running (no labels) or hasn't fired since the last zoom.
    const sceneLayout = gd._fullLayout?.scene;
    const glplotCam = (sceneLayout?._scene?.glplot as any)?.camera;
    if (glplotCam?.eye) {
      if (Array.isArray(glplotCam.eye)) {
        currentCameraRef.current.eye = { x: glplotCam.eye[0], y: glplotCam.eye[1], z: glplotCam.eye[2] };
        const c = glplotCam.center || [0, 0, 0];
        currentCameraRef.current.center = { x: c[0], y: c[1], z: c[2] };
      } else {
        currentCameraRef.current.eye = { x: glplotCam.eye.x, y: glplotCam.eye.y, z: glplotCam.eye.z };
        currentCameraRef.current.center = glplotCam.center
          ? { x: glplotCam.center.x, y: glplotCam.center.y, z: glplotCam.center.z }
          : { x: 0, y: 0, z: 0 };
      }
    }

    // Preserve current camera position so Plotly.react doesn't reset the view
    // (layout contains defaultEye/defaultCenter which would snap the camera back)
    const cam = currentCameraRef.current;
    const preservedLayout = {
      ...layout,
      scene: {
        ...layout.scene,
        camera: {
          eye: cam.eye,
          center: cam.center,
          up: { x: 0, y: 0, z: 1 },
        },
      },
    };
    Plotly.react(gd, allTraces as PlotData[], preservedLayout, config);
  }, [baseTraces, layout, config, plotReady]); // eslint-disable-line react-hooks/exhaustive-deps

  // 3. Update overlay traces (highlights + selected) — single atomic redraw
  useEffect(() => {
    if (!plotReady || !graphDivRef.current || !plotlyLibRef.current) return;
    const Plotly = plotlyLibRef.current;
    const gd = graphDivRef.current;

    const oldCount = overlayTraceCountRef.current;
    const newCount = overlayTraces.length;

    if (oldCount === 0 && newCount === 0) return;

    // Plotly.redraw runs supplyDefaults ASYNCHRONOUSLY (setTimeout(0)),
    // which copies gd.layout.scene.camera → _fullLayout, then the 3D
    // scene resets glplot.camera to match. If gd.layout still holds the
    // camera written by Plotly.relayout at the end of the PREVIOUS
    // fly-to animation, the async redraw teleports glplot back to that
    // old position — even though our synchronous code has moved on.
    //
    // Fix: write the live glplot camera into gd.layout.scene.camera
    // BEFORE Plotly.redraw, so the async supplyDefaults reads the
    // correct value and the scene doesn't reset the camera.
    const sceneLayout = gd._fullLayout?.scene;
    const glplotForSync = (sceneLayout?._scene?.glplot) as any;
    if (glplotForSync?.camera?.eye) {
      const cam = glplotForSync.camera;
      const eye = Array.isArray(cam.eye)
        ? { x: cam.eye[0], y: cam.eye[1], z: cam.eye[2] }
        : { x: cam.eye.x, y: cam.eye.y, z: cam.eye.z };
      const center = cam.center
        ? (Array.isArray(cam.center)
          ? { x: cam.center[0], y: cam.center[1], z: cam.center[2] }
          : { x: cam.center.x, y: cam.center.y, z: cam.center.z })
        : { x: 0, y: 0, z: 0 };
      const gdAny = gd as any;
      if (!gdAny.layout) gdAny.layout = {};
      if (!gdAny.layout.scene) gdAny.layout.scene = {};
      gdAny.layout.scene.camera = { eye, center, up: { x: 0, y: 0, z: 1 } };
    }

    // Splice overlay traces in place and redraw once
    const baseCount = (gd.data?.length ?? 0) - oldCount;
    gd.data?.splice(baseCount, oldCount, ...(overlayTraces as PlotData[]));
    overlayTraceCountRef.current = newCount;
    Plotly.redraw(gd);

    // startFlyTo captures the live glplot camera eagerly.
    const flyTarget = pendingFlyToRef.current;
    if (flyTarget) {
      pendingFlyToRef.current = null;
      startFlyTo(flyTarget);
    }
  }, [overlayTraces, plotReady, startFlyTo]);

  const mouseDownPosRef = useRef<{ x: number; y: number } | null>(null);
  const deliveredClickRef = useRef<{ press: object; idx: number } | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    // pointerdown covers mouse, touch, and pen — one fresh position object per
    // press; the drag guard and per-press click dedup below both key off it.
    const handlePointerDown = (e: PointerEvent) => { mouseDownPosRef.current = { x: e.clientX, y: e.clientY }; };
    container.addEventListener('pointerdown', handlePointerDown);
    return () => container.removeEventListener('pointerdown', handlePointerDown);
  }, []);

  const handleClick = useCallback((event: PlotMouseEvent) => {
    if (!onPointClick || !event.points || event.points.length === 0) return;

    // Drag detection: ignore clicks where the mouse moved significantly from mousedown
    const downPos = mouseDownPosRef.current;
    if (downPos && event.event) {
      const dx = (event.event as MouseEvent).clientX - downPos.x;
      const dy = (event.event as MouseEvent).clientY - downPos.y;
      if (dx * dx + dy * dy > 25) return; // >5px movement = drag
    }

    const point = event.points[0];
    if (point.customdata == null) return;

    const idx = point.customdata as unknown as number;

    // gl3d emits plotly_click from the per-frame pick pass, so one physical
    // press held over a point fires the event once per GL frame. Deliver only
    // the first per press (the mousedown position object identifies the press).
    if (downPos) {
      const last = deliveredClickRef.current;
      if (last && last.press === downPos && last.idx === idx) return;
      deliveredClickRef.current = { press: downPos, idx };
    }

    const clickedPoint = pointsRef.current[idx];
    if (!clickedPoint) return;
    onPointClick(clickedPoint);
  }, [onPointClick]);

  useEffect(() => {
    if (!plotReady || !graphDivRef.current) return;
    const graphDiv = graphDivRef.current as any;
    const handlePlotlyHover = (data: any) => {
      if (data.points && data.points.length > 0) {
        const pt = data.points[0];
        if (pt.customdata == null) return;

        const idx = pt.customdata as unknown as number;
        const point = pointsRef.current[idx];
        if (!point) return;

        const containerRect = containerRef.current?.getBoundingClientRect();

        // For 3D, try to use event coordinates or fallback to bbox
        let x: number, y: number;
        const mouseEvent = data.event as MouseEvent | undefined;

        if (mouseEvent && mouseEvent.clientX !== undefined) {
          x = mouseEvent.clientX - (containerRect?.left ?? 0);
          y = mouseEvent.clientY - (containerRect?.top ?? 0);
        } else if (pt.bbox) {
          x = pt.bbox.x0 + (pt.bbox.x1 - pt.bbox.x0) / 2;
          y = pt.bbox.y0;
        } else {
          x = pt.xaxis?.l2p?.(pt.x) ?? (containerRect?.width ?? 400) / 2;
          y = pt.yaxis?.l2p?.(pt.y) ?? (containerRect?.height ?? 300) / 2;
        }

        hoveredPointRef.current = point;
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
    };
    const handlePlotlyUnhover = () => { hoveredPointRef.current = null; setTooltipData(null); };
    if (typeof graphDiv.on === 'function') {
      graphDiv.on('plotly_hover', handlePlotlyHover);
      graphDiv.on('plotly_unhover', handlePlotlyUnhover);
      graphDiv.on('plotly_click', handleClick);
      graphDiv.on('plotly_relayout', handleRelayout);
    }
    return () => {
      if (typeof graphDiv.removeListener === 'function') {
        graphDiv.removeListener('plotly_hover', handlePlotlyHover);
        graphDiv.removeListener('plotly_unhover', handlePlotlyUnhover);
        graphDiv.removeListener('plotly_click', handleClick);
        graphDiv.removeListener('plotly_relayout', handleRelayout);
      }
    };
  }, [plotReady, tooltipFields, handleClick, handleRelayout]);

  // --- NEBULA: Haze sprites (separate overlay canvas) ---
  const hazeRendererRef = useRef<HazeRenderer | null>(null);
  // Set by the lifecycle effect; lets the cluster-update effect force a single
  // repaint after new sprites are built, so muting/filtering updates the haze
  // immediately without waiting for Plotly's next GL render.
  const renderHazeOnceRef = useRef<(() => void) | null>(null);
  // Identity of the clusterDataMap last pushed into the renderer, so the
  // cluster-update effect can no-op when the lifecycle effect already applied
  // the current map in the same commit (avoids a double Poisson resample).
  const appliedClusterMapRef = useRef<Map<string, ClusterData> | null>(null);

  // Whether there is anything to draw — a primitive boolean so the renderer
  // lifecycle effect only re-runs when crossing the empty/non-empty boundary,
  // not on every cluster-content change (mute / temporal filter / recolor).
  const hasNebulaClusters = clusterDataMap.size > 0;

  // 1. Renderer lifecycle: create/dispose the Three.js renderer and hook the GL
  //    render loop. Deps deliberately EXCLUDE clusterDataMap so cluster-content
  //    changes never tear down and re-instantiate the WebGL renderer — that
  //    teardown (plus its Poisson-disk resample) was the interaction hitch.
  useEffect(() => {
    if (!nebulaMode || !plotReady || !graphDivRef.current || !hasNebulaClusters) {
      if (hazeRendererRef.current) {
        hazeRendererRef.current.dispose();
        hazeRendererRef.current = null;
      }
      renderHazeOnceRef.current = null;
      appliedClusterMapRef.current = null;
      return;
    }

    const canvas = hazeCanvasRef.current;
    if (!canvas) return;

    const sceneLayout = (graphDivRef.current._fullLayout?.scene) as any;
    const glplot = sceneLayout?._scene?.glplot;
    if (!glplot) return;

    // Match overlay canvas to Plotly's GL canvas position/size.
    const glCanvas = (glplot.gl?.canvas as HTMLCanvasElement) ?? null;
    const containerEl = containerRef.current;
    if (!glCanvas || !containerEl) return;

    // Only touch canvas dims/styles when the GL canvas actually moved or resized.
    // Assigning canvas.width/height resets the WebGL backing store, and style
    // writes dirty layout (forcing reflow) — doing either every frame is pure
    // waste during rotation, when position/size are stable.
    let lastLayout: OverlayLayout | null = null;
    const syncCanvasLayout = () => {
      const layout = computeOverlayLayout(containerEl.getBoundingClientRect(), glCanvas.getBoundingClientRect());
      if (overlayLayoutEqual(layout, lastLayout)) return;
      lastLayout = layout;

      canvas.style.left = `${layout.offsetX}px`;
      canvas.style.top = `${layout.offsetY}px`;
      canvas.style.width = `${layout.cssW}px`;
      canvas.style.height = `${layout.cssH}px`;

      canvas.width = Math.round(layout.cssW);
      canvas.height = Math.round(layout.cssH);

      hazeRendererRef.current?.resize(layout.cssW, layout.cssH);
    };

    syncCanvasLayout();

    const haze = new HazeRenderer(canvas, isDark);
    hazeRendererRef.current = haze;
    haze.resize(canvas.clientWidth, canvas.clientHeight);
    haze.updateClusters(clusterDataMap, pointsRef.current.length);
    appliedClusterMapRef.current = clusterDataMap;

    // Draw one frame using the live camera — paints current sprites without
    // waiting for the next Plotly GL render. Reused by the onrender hook and by
    // the cluster-update effect below.
    const renderHazeOnce = () => {
      syncCanvasLayout();
      const cameraParams = glplot.cameraParams || glplot.camera;
      if (!cameraParams) return;
      const projection = cameraParams.projection || cameraParams._projection;
      const view = cameraParams.view || cameraParams._view;
      if (!projection || !view) return;
      hazeRendererRef.current?.render(projection, view, getModelMatrix(boundsRef.current));
    };
    renderHazeOnceRef.current = renderHazeOnce;

    // Hook into glplot's render loop for camera sync
    const originalOnRender = glplot.onrender;
    glplot.onrender = () => {
      if (originalOnRender) originalOnRender();
      renderHazeOnce();
    };

    renderHazeOnce();

    return () => {
      if (glplot) glplot.onrender = originalOnRender || null;
      haze.dispose();
      hazeRendererRef.current = null;
      renderHazeOnceRef.current = null;
      appliedClusterMapRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nebulaMode, plotReady, isDark, hasNebulaClusters]);

  // 2. Cluster-content updates: rebuild only the sprite set on the existing
  //    renderer (no GL teardown), then force a single repaint so the change is
  //    visible immediately even if the camera is idle. Skips when the lifecycle
  //    effect already applied this exact map in the same commit.
  useEffect(() => {
    if (!hazeRendererRef.current) return;
    if (appliedClusterMapRef.current === clusterDataMap) return;
    hazeRendererRef.current.updateClusters(clusterDataMap, pointsRef.current.length);
    appliedClusterMapRef.current = clusterDataMap;
    renderHazeOnceRef.current?.();
  }, [clusterDataMap]);

  return (
    <div ref={containerRef} className={className ?? 'h-full w-full'} style={{ position: 'relative' }}
      onContextMenu={(e) => {
        if (onPointContextMenu && hoveredPointRef.current) {
          e.preventDefault();
          onPointContextMenu(hoveredPointRef.current, e.nativeEvent);
        }
      }}
    >
      <div ref={graphDivRef as React.RefObject<HTMLDivElement>} style={{ width: '100%', height: '100%' }} />
      {nebulaMode && (
        <canvas
          ref={hazeCanvasRef}
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            pointerEvents: 'none',
            zIndex: 5,
            mixBlendMode: isDark ? 'screen' : 'multiply',
          }}
        />
      )}
      {hasAnyLabels && (
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