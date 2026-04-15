'use client';

import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';
import type { PlotData, Layout, Config, PlotMouseEvent, PlotRelayoutEvent } from 'plotly.js';
import type { Point3D, HighlightMap, ColorScaleType, NestedColorMap } from '../../lib/types/types';
import { useTheme } from 'next-themes';
import { buildCategoryColorMap, getCategoryLabel, getSequentialScale, getDivergingScale, getMonochromeScale, desaturateHex, type SequentialScaleName, type DivergingScaleName } from '../../lib/utils/categoryColors';
import { isCrameriScale, getCrameriPlotlyScale } from '../../lib/colorMaps/crameriScales';
import { calculateMarkerStyle, calculateLuminosity, calculateHighlightScale, calculateSimilarityColors } from '../../lib/utils/plotUtils';
import { useContainerDimensions } from '../../lib/hooks/useContainerDimensions';
import { useZoomLimit } from '../../lib/hooks/useZoomLimit';
import { FrostedTooltip, type TooltipData } from './FrostedTooltip';
import { easeInOutCubic, lerp, cartesianToSpherical, sphericalToCartesian, getZoomLevel, formatHoverText } from '../utils/rendeding';
import { groupPointsByCluster, type ClusterData } from '../../lib/utils/clusterGeometry';
import { HazeRenderer } from '../../lib/utils/hazeRenderer';
import { computeMVP, buildDataToSceneMatrix, projectToScreen } from '../utils/labelPlacement';
import { CollisionGrid, type BoundingBox } from '../../lib/utils/collisionGrid';

// Hoisted identity matrix for haze renderer model matrix fallback (avoids per-frame allocation)
const GL_IDENTITY_MATRIX = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);

type PlotlyData = Partial<PlotData>;

interface ScatterPlot3DProps {
  points: Point3D[];
  colorBy?: 'category' | 'none';
  categoryField?: string | null;  // Field to color by (used for both categorical AND numeric)
  categoryValues?: string[];
  colorScaleType?: ColorScaleType;
  monochromeColor?: string;
  sequentialScaleName?: SequentialScaleName;
  divergingScaleName?: DivergingScaleName;
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
  /** 0-1, opacity for muted points (default 0.15) */
  mutedPointOpacity?: number;
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
  nebulaMode = false,
  showClusterLabels = false,
  combinedMutedIndices,
  hideFilteredPoints = false,
  mutedPointOpacity,
}: ScatterPlot3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
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
  const gridColor = isDark ? '#334155' : '#e5e7eb';
  const sceneBg = 'rgba(0,0,0,0)';
  const paperBg = 'rgba(0,0,0,0)';

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
  const animationFrameRef = useRef<number | undefined>(undefined);
  const isAnimatingRef = useRef(false);
  const lastClickTimeRef = useRef<number>(0);
  const graphDivRef = useRef<PlotlyGraphDiv | null>(null);
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




  // --- Camera Animation Effect ---
  // Fires on renderedSelectedPoint (not selectedPoint) so highlight traces are already
  // present in plotData before the fly-to animation begins.
  useEffect(() => {
    if (!renderedSelectedPoint || !bounds || !plotReady || !graphDivRef.current) return;
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

    const dataCenterX = (bounds.minX + bounds.maxX) / 2;
    const dataCenterY = (bounds.minY + bounds.maxY) / 2;
    const dataCenterZ = (bounds.minZ + bounds.maxZ) / 2;
    const maxRange = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ) || 1;

    const targetCenterX = (renderedSelectedPoint.x - dataCenterX) / maxRange;
    const targetCenterY = (renderedSelectedPoint.y - dataCenterY) / maxRange;
    const targetCenterZ = (renderedSelectedPoint.z - dataCenterZ) / maxRange;

    // Adaptive target radius based on dataset size (similar to defaultEye calculation)
    // Small datasets (100 pts): ~0.32
    // Medium datasets (10k pts): ~0.08
    // Large datasets (100k+ pts): ~0.05 (very close)
    //const baseTargetR = 0.4;
    //const zoomRate = 0.08; // How much to zoom in per power of 10
    //const calculatedTargetR = baseTargetR - (zoomRate * Math.log10(Math.max(pointCount, 1)));
    //const targetR = Math.min(Math.max(calculatedTargetR, 0.05), 0.5);

    const targetR = 0.15

    const targetPhi = 1.3;
    const duration = 2000;

    let startEye: any, startCenter: any, startSpherical: any, targetSpherical: any, startTime: number;
    let initialized = false;

    const animate = (currentTime: number) => {
      if (!isAnimatingRef.current || !graphDivRef.current) return;

      if (!initialized) {
        const scene = graphDivRef.current._fullLayout?.scene;
        const layoutCamera = scene?.camera;
        if (layoutCamera?.eye) {
          startEye = { ...layoutCamera.eye };
          startCenter = layoutCamera.center ? { ...layoutCamera.center } : { x: 0, y: 0, z: 0 };
        } else {
          startEye = { ...currentCameraRef.current.eye };
          startCenter = { ...currentCameraRef.current.center };
        }
        // Convert eye position RELATIVE to center (not absolute from origin)
        startSpherical = cartesianToSpherical(
          startEye.x - startCenter.x,
          startEye.y - startCenter.y,
          startEye.z - startCenter.z
        );
        targetSpherical = { r: targetR, theta: startSpherical.theta + 0.5, phi: targetPhi };
        startTime = currentTime;
        initialized = true;
      }

      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = easeInOutCubic(progress);

      const newCenter = {
        x: lerp(startCenter.x, targetCenterX, ease),
        y: lerp(startCenter.y, targetCenterY, ease),
        z: lerp(startCenter.z, targetCenterZ, ease),
      };

      const curR = lerp(startSpherical.r, targetSpherical.r, ease);
      const curTheta = lerp(startSpherical.theta, targetSpherical.theta, ease);
      const curPhi = lerp(startSpherical.phi, targetSpherical.phi, ease);

      // Convert spherical to cartesian (gives position relative to center)
      const relativeEye = sphericalToCartesian(curR, curTheta, curPhi);

      // Add the interpolated center to get absolute eye position
      const newEye = {
        x: relativeEye.x + newCenter.x,
        y: relativeEye.y + newCenter.y,
        z: relativeEye.z + newCenter.z,
      };

      currentCameraRef.current = { eye: newEye, center: newCenter };
      const currentScene = graphDivRef.current._fullLayout?.scene?._scene;
      const glplot = currentScene?.glplot as any;

      if (glplot?.camera) {
        if (Array.isArray(glplot.camera.eye)) {
          glplot.camera.eye = [newEye.x, newEye.y, newEye.z];
          glplot.camera.center = [newCenter.x, newCenter.y, newCenter.z];
          glplot.camera.up = [0, 0, 1];
        } else if (glplot.camera.eye && typeof glplot.camera.eye === 'object') {
          glplot.camera.eye.x = newEye.x;
          glplot.camera.eye.y = newEye.y;
          glplot.camera.eye.z = newEye.z;
          glplot.camera.center.x = newCenter.x;
          glplot.camera.center.y = newCenter.y;
          glplot.camera.center.z = newCenter.z;
        }
        if (typeof glplot.camera.update === 'function') glplot.camera.update();
        if (typeof glplot.draw === 'function') glplot.draw();
      }

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        isAnimatingRef.current = false;
        if (plotlyLibRef.current && graphDivRef.current) {
          plotlyLibRef.current.relayout(graphDivRef.current, {
            'scene.camera': { eye: newEye, center: newCenter, up: { x: 0, y: 0, z: 1 } },
          });
        }
        // Re-render labels now that animation is done (skipped during fly-to)
        renderLabelsRef.current?.();
      }
    };

    isAnimatingRef.current = true;
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      isAnimatingRef.current = false;
    };
  }, [renderedSelectedPoint, bounds, plotReady]);

  const MAX_CAMERA_DISTANCE = 3.0;

  const handleRelayout = useCallback((e: Readonly<PlotRelayoutEvent>) => {
    if (isAnimatingRef.current) return;
    const sceneCamera = (e as any)['scene.camera'];
    if (sceneCamera) {
      if (sceneCamera.eye) currentCameraRef.current.eye = sceneCamera.eye;
      if (sceneCamera.center) currentCameraRef.current.center = sceneCamera.center;
    }
  }, []);

  const colorMap = useMemo(() => {
    return buildCategoryColorMap(categoryField, categoryValues, categoricalPalette);
  }, [categoryField, categoryValues, categoricalPalette]);

  // --- 1. OPTIMIZED DATA EXTRACTION (Raw Numbers) ---
  const numericData = useMemo(() => {
    if (colorScaleType === 'categorical' || !categoryField) return null;

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
  }, [colorScaleType, categoryField, points]);

  // --- 2. GENERATE PLOTLY NATIVE COLORSCALE ---
  // Bridge your D3/custom scale logic to a Plotly array [[0, 'hex'], [1, 'hex']]
  const plotlyColorScale = useMemo(() => {
    if (colorScaleType === 'categorical') return undefined;

    // For Crameri scales, use the pre-computed 256-step Plotly array directly
    const scaleName = colorScaleType === 'diverging' ? divergingScaleName : sequentialScaleName;
    if (scaleName && isCrameriScale(scaleName)) {
      const crameriScale = getCrameriPlotlyScale(scaleName);
      if (crameriScale) return crameriScale;
      // Fall through to D3 sampling if not loaded yet
    }

    // Request a normalized interpolator (0 to 1) from your utils
    let scaleFunc: (t: number) => string;
    if (colorScaleType === 'monochrome') {
      scaleFunc = getMonochromeScale(monochromeColor, [0, 1]);
    } else if (colorScaleType === 'diverging') {
      scaleFunc = getDivergingScale([0, 0.5, 1], divergingScaleName);
    } else {
      scaleFunc = getSequentialScale([0, 1], sequentialScaleName);
    }

    // Sample the function to create a gradient definition
    const steps = 20;
    return Array.from({ length: steps + 1 }, (_, i) => {
      const t = i / steps;
      return [t, scaleFunc(t)]; // [0.1, '#ff0000']
    });
  }, [colorScaleType, monochromeColor, sequentialScaleName, divergingScaleName]);

  const markerStyle = useMemo(() => calculateMarkerStyle(points.length), [points.length]);
  const highlightScale = useMemo(() => calculateHighlightScale(points.length), [points.length]);

  // --- OPTIMIZED TRACES ---
  const baseTraces = useMemo((): PlotlyData[] => {
    const traces: PlotlyData[] = [];

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

    // Compute data arrays from filtered points
    const allX = displayPoints.map(p => p.x);
    const allY = displayPoints.map(p => p.y);
    const allZ = displayPoints.map(p => p.z);
    const allText = displayPoints.map(formatHoverText);
    const allCustomData = displayPoints;

    if (!showOnlyHighlighted) {
      const dimOpacity = markerStyle.opacity;
      const dimSize = Math.max(markerStyle.size * 0.7, 2);

      if (numericData && plotlyColorScale) {
        // --- MODE: NATIVE COLORSCALE (GPU ACCELERATED) ---
        if (combinedMutedIndices) {
          // Split into active vs muted points
          const activeIndices: number[] = [];
          const mutedIndices: number[] = [];
          displayPoints.forEach((_, i) => {
            if (combinedMutedIndices.has(displayPoints[i].index)) {
              mutedIndices.push(i);
            } else {
              activeIndices.push(i);
            }
          });
          // Active points with normal colorscale
          if (activeIndices.length > 0) {
            traces.push({
              x: activeIndices.map(i => allX[i]),
              y: activeIndices.map(i => allY[i]),
              z: activeIndices.map(i => allZ[i]),
              mode: 'markers',
              type: 'scatter3d',
              name: 'Data',
              marker: {
                sizemode: 'diameter',
                size: dimSize,
                color: activeIndices.map(i => numericData.cleanValues[i]) as any,
                colorscale: plotlyColorScale as any,
                cmin: numericData.min,
                cmax: numericData.max,
                opacity: dimOpacity,
                showscale: false,
              },
              text: activeIndices.map(i => allText[i]),
              hoverinfo: 'none',
              customdata: activeIndices.map(i => allCustomData[i]) as any,
              showlegend: false,
            });
          }
          // Temporally-muted points grayed out
          if (mutedIndices.length > 0 && !hideFilteredPoints) {
            traces.push({
              x: mutedIndices.map(i => allX[i]),
              y: mutedIndices.map(i => allY[i]),
              z: mutedIndices.map(i => allZ[i]),
              mode: 'markers',
              type: 'scatter3d',
              name: 'Temporal (muted)',
              marker: {
                sizemode: 'diameter',
                size: dimSize,
                color: '#9ca3af',
                opacity: mutedPointOpacity ?? 0.15,
              },
              text: mutedIndices.map(i => allText[i]),
              hoverinfo: 'none',
              customdata: mutedIndices.map(i => allCustomData[i]) as any,
              showlegend: false,
            });
          }
        } else {
          traces.push({
            x: allX,
            y: allY,
            z: allZ,
            mode: 'markers',
            type: 'scatter3d',
            name: 'Data',
            marker: {
              sizemode: 'diameter',
              size: dimSize,
              // Pass raw numbers array
              color: numericData.cleanValues as any,
              // Use the sampled scale
              colorscale: plotlyColorScale as any,
              // Map min/max explicitly
              cmin: numericData.min,
              cmax: numericData.max,
              opacity: dimOpacity,
              showscale: false, // Disabled - using custom Legend component instead
            },
            text: allText,
            hoverinfo: 'none',
            customdata: allCustomData as any,
            showlegend: false,
          });
        }
      } else if (colorBy === 'category' && categoryValues.length > 0) {
        if (nestedColorMap) {
          // --- MODE: NESTED CATEGORICAL ---
          const pointsBySub: Record<string, Point3D[]> = {};
          displayPoints.forEach(point => {
            const sub = String(point.metadata?.['subtopic_label'] ?? point.metadata?.['topic_label'] ?? 'unknown');
            if (!pointsBySub[sub]) pointsBySub[sub] = [];
            pointsBySub[sub].push(point);
          });

          Object.entries(pointsBySub).forEach(([sub, subPoints]) => {
            const parentTopic = String(subPoints[0]?.metadata?.['topic_label'] ?? 'unknown');
            const isMuted = mutedCategories.includes(sub) || mutedCategories.includes(parentTopic);
            if (hideFilteredPoints && isMuted) return;

            if (combinedMutedIndices) {
              const activePoints = subPoints.filter(p => !combinedMutedIndices.has(p.index));
              const mutedPts = subPoints.filter(p => combinedMutedIndices.has(p.index));

              if (activePoints.length > 0) {
                traces.push({
                  x: activePoints.map(p => p.x),
                  y: activePoints.map(p => p.y),
                  z: activePoints.map(p => p.z),
                  mode: 'markers',
                  type: 'scatter3d',
                  name: sub,
                  marker: {
                    sizemode: 'diameter',
                    size: dimSize,
                    color: isMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[sub] || '#7f7f7f'),
                    opacity: isMuted ? (mutedPointOpacity ?? 0.15) : dimOpacity,
                  },
                  text: activePoints.map(formatHoverText),
                  hoverinfo: 'none',
                  customdata: activePoints as any,
                  showlegend: false,
                });
              }
              if (mutedPts.length > 0 && !hideFilteredPoints) {
                traces.push({
                  x: mutedPts.map(p => p.x),
                  y: mutedPts.map(p => p.y),
                  z: mutedPts.map(p => p.z),
                  mode: 'markers',
                  type: 'scatter3d',
                  name: sub,
                  marker: {
                    sizemode: 'diameter',
                    size: dimSize,
                    color: '#9ca3af',
                    opacity: mutedPointOpacity ?? 0.15,
                  },
                  text: mutedPts.map(formatHoverText),
                  hoverinfo: 'none',
                  customdata: mutedPts as any,
                  showlegend: false,
                });
              }
            } else {
              traces.push({
                x: subPoints.map(p => p.x),
                y: subPoints.map(p => p.y),
                z: subPoints.map(p => p.z),
                mode: 'markers',
                type: 'scatter3d',
                name: sub,
                marker: {
                  sizemode: 'diameter',
                  size: dimSize,
                  color: isMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[sub] || '#7f7f7f'),
                  opacity: isMuted ? (mutedPointOpacity ?? 0.15) : dimOpacity,
                },
                text: subPoints.map(formatHoverText),
                hoverinfo: 'none',
                customdata: subPoints as any,
                showlegend: false,
              });
            }
          });
        } else {
          // --- MODE: CATEGORICAL (Standard) ---
          const pointsByCategory: Record<string, Point3D[]> = {};
          displayPoints.forEach(point => {
            const raw = categoryField ? point.metadata?.[categoryField] : undefined;
            const cat = (raw !== null && raw !== undefined && raw !== '') ? String(raw) : 'unknown';
            if (!pointsByCategory[cat]) pointsByCategory[cat] = [];
            pointsByCategory[cat].push(point);
          });

          Object.entries(pointsByCategory).forEach(([cat, catPoints]) => {
            const isMuted = mutedCategories.includes(cat);
            if (hideFilteredPoints && isMuted) return;

            if (combinedMutedIndices) {
              const activePoints = catPoints.filter(p => !combinedMutedIndices.has(p.index));
              const mutedPts = catPoints.filter(p => combinedMutedIndices.has(p.index));

              if (activePoints.length > 0) {
                traces.push({
                  x: activePoints.map(p => p.x),
                  y: activePoints.map(p => p.y),
                  z: activePoints.map(p => p.z),
                  mode: 'markers',
                  type: 'scatter3d',
                  name: getCategoryLabel(categoryField, cat),
                  marker: {
                    sizemode: 'diameter',
                    size: dimSize,
                    color: isMuted ? '#9ca3af' : (colorMap[cat] || '#7f7f7f'),
                    opacity: isMuted ? (mutedPointOpacity ?? 0.15) : dimOpacity,
                  },
                  text: activePoints.map(formatHoverText),
                  hoverinfo: 'none',
                  customdata: activePoints as any,
                  showlegend: false,
                });
              }
              if (mutedPts.length > 0 && !hideFilteredPoints) {
                traces.push({
                  x: mutedPts.map(p => p.x),
                  y: mutedPts.map(p => p.y),
                  z: mutedPts.map(p => p.z),
                  mode: 'markers',
                  type: 'scatter3d',
                  name: getCategoryLabel(categoryField, cat),
                  marker: {
                    sizemode: 'diameter',
                    size: dimSize,
                    color: '#9ca3af',
                    opacity: mutedPointOpacity ?? 0.15,
                  },
                  text: mutedPts.map(formatHoverText),
                  hoverinfo: 'none',
                  customdata: mutedPts as any,
                  showlegend: false,
                });
              }
            } else {
              traces.push({
                x: catPoints.map(p => p.x),
                y: catPoints.map(p => p.y),
                z: catPoints.map(p => p.z),
                mode: 'markers',
                type: 'scatter3d',
                name: getCategoryLabel(categoryField, cat),
                marker: {
                  sizemode: 'diameter',
                  size: dimSize,
                  color: isMuted ? '#9ca3af' : (colorMap[cat] || '#7f7f7f'),
                  opacity: isMuted ? (mutedPointOpacity ?? 0.15) : dimOpacity,
                },
                text: catPoints.map(formatHoverText),
                hoverinfo: 'none',
                customdata: catPoints as any,
                showlegend: false,
              });
            }
          });
        }
      } else {
        // --- MODE: NO COLORING ---
        if (combinedMutedIndices) {
          const activeIndices: number[] = [];
          const mutedIndices: number[] = [];
          displayPoints.forEach((_, i) => {
            if (combinedMutedIndices.has(displayPoints[i].index)) {
              mutedIndices.push(i);
            } else {
              activeIndices.push(i);
            }
          });
          if (activeIndices.length > 0) {
            traces.push({
              x: activeIndices.map(i => allX[i]),
              y: activeIndices.map(i => allY[i]),
              z: activeIndices.map(i => allZ[i]),
              mode: 'markers',
              type: 'scatter3d',
              name: 'Data',
              marker: {
                sizemode: 'diameter',
                size: dimSize,
                color: '#1f77b4',
                opacity: dimOpacity,
              },
              text: activeIndices.map(i => allText[i]),
              hoverinfo: 'none',
              customdata: activeIndices.map(i => allCustomData[i]) as any,
              showlegend: false,
            });
          }
          if (mutedIndices.length > 0 && !hideFilteredPoints) {
            traces.push({
              x: mutedIndices.map(i => allX[i]),
              y: mutedIndices.map(i => allY[i]),
              z: mutedIndices.map(i => allZ[i]),
              mode: 'markers',
              type: 'scatter3d',
              name: 'Temporal (muted)',
              marker: {
                sizemode: 'diameter',
                size: dimSize,
                color: '#9ca3af',
                opacity: mutedPointOpacity ?? 0.15,
              },
              text: mutedIndices.map(i => allText[i]),
              hoverinfo: 'none',
              customdata: mutedIndices.map(i => allCustomData[i]) as any,
              showlegend: false,
            });
          }
        } else {
          traces.push({
            x: allX,
            y: allY,
            z: allZ,
            mode: 'markers',
            type: 'scatter3d',
            name: 'Data',
            marker: {
              sizemode: 'diameter',
              size: dimSize,
              color: '#1f77b4',
              opacity: dimOpacity,
            },
            text: allText,
            hoverinfo: 'none',
            customdata: allCustomData as any,
            showlegend: false,
          });
        }
      }
    }

    return traces;
  }, [
    points, markerStyle, showOnlyHighlighted,
    colorBy, isDark, categoryValues, colorMap, numericData, plotlyColorScale, categoryField,
    mutedCategories, hideUnclustered, nestedColorMap, combinedMutedIndices, hideFilteredPoints, mutedPointOpacity
  ]);

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
      const coreTexts: string[] = [];
      const coreCustomData: any[] = [];

      allOverlayPoints.forEach(point => {
        const isSelected = renderedSelectedPoint && point.index === renderedSelectedPoint.index;
        // Selected point gets max similarity (brightest glow)
        const similarity = isSelected ? 1.0 : (highlightedIndices?.get(point.index) ?? 1.0);
        const luminosity = calculateLuminosity(similarity);
        const colors = calculateSimilarityColors(similarity);

        outerSizes.push(Math.max(markerStyle.size * (isSelected ? highlightScale.selectedOuterMultiplier : highlightScale.outerMultiplier), isSelected ? 35 : 30));
        outerColors.push(colors.outerGlow);

        innerSizes.push(Math.max(markerStyle.size * (isSelected ? highlightScale.selectedInnerMultiplier : highlightScale.innerMultiplier), isSelected ? 20 : 18));
        innerColors.push(colors.glowColor);

        coreSizes.push(Math.max(markerStyle.size * (isSelected ? highlightScale.selectedCoreMultiplier : highlightScale.coreMultiplier), isSelected ? 10 : 9));
        coreColors.push(colors.coreColor);
        coreTexts.push(formatHoverText(point));
        coreCustomData.push(point);
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
        text: coreTexts, hoverinfo: 'none', customdata: coreCustomData as any, showlegend: false
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
            line: { color: isDark ? 'rgba(130, 160, 200, 0.60)' : 'rgba(100, 130, 170, 0.60)', width: 0.1 },
            hoverinfo: 'skip' as any, showlegend: false
          });
        }
      }
    }

    return traces;
  }, [renderedSelectedPoint, highlightedIndices, highlightedPoints, markerStyle, highlightScale, isDark]);

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

    // Apply same hideUnclustered filter as baseTraces, then exclude muted categories
    let displayPoints = hideUnclustered
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

    return groupPointsByCluster(displayPoints, categoryField, colorMap, nestedColorMap);
  }, [nebulaMode, showClusterLabels, points, categoryField, colorMap, nestedColorMap, hideUnclustered, mutedCategories]);

  // --- Cluster label data for canvas overlay ---
  const clusterLabelDataRef = useRef<{
    labels: { x: number; y: number; z: number; label: string; color: string }[];
  } | null>(null);

  useEffect(() => {
    if (!showClusterLabels || clusterDataMap.size === 0) {
      clusterLabelDataRef.current = null;
      return;
    }
    const labels: { x: number; y: number; z: number; label: string; color: string }[] = [];
    for (const [key, cluster] of clusterDataMap) {
      labels.push({
        x: cluster.centroid.x,
        y: cluster.centroid.y,
        z: cluster.centroid.z,
        label: key,
        color: cluster.color,
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

    const model = glplot?.model || buildDataToSceneMatrix(currentBounds);
    const mvp = computeMVP(projection, view, model);

    // Get the actual GL canvas position relative to our overlay container.
    // Plotly's GL canvas may be offset within the plot div (margins, modebar, etc.)
    const gl = glplot?.gl as WebGLRenderingContext | null;
    const glCanvas = gl?.canvas as HTMLCanvasElement | undefined;
    const containerRect = container.getBoundingClientRect();

    let glOffsetX = 0;
    let glOffsetY = 0;
    let vpW: number;
    let vpH: number;

    if (glCanvas && gl) {
      const glRect = glCanvas.getBoundingClientRect();
      // Offset of GL canvas within our overlay container (in CSS pixels)
      glOffsetX = glRect.left - containerRect.left;
      glOffsetY = glRect.top - containerRect.top;
      // Viewport size in CSS pixels (what projectToScreen maps into)
      vpW = glRect.width;
      vpH = glRect.height;
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

      // Camera distance → opacity: closer = more opaque, farther = more transparent
      const camEye = currentCameraRef.current.eye;
      const camCenter = currentCameraRef.current.center;
      const camDist = Math.sqrt(
        (camEye.x - camCenter.x) ** 2 +
        (camEye.y - camCenter.y) ** 2 +
        (camEye.z - camCenter.z) ** 2
      );
      // Map: close (≤0.3) → 1.0, far (≥2.0) → 0. Skip drawing entirely below threshold.
      const clusterOpacity = Math.max(0, Math.min(1.0, 1.0 - (camDist - 0.3) * 0.5));
      if (clusterOpacity < 0.25) {
        // Too far — don't draw cluster labels at all
      } else for (const cl of clusterData!.labels) {
        const screen = projectToScreen(cl.x, cl.y, cl.z, mvp, vpW, vpH);
        if (!screen) continue;

        const sx = screen.x + glOffsetX;
        const sy = screen.y + glOffsetY;

        // Measure text and build bounding box with 4px padding
        const textWidth = ctx.measureText(cl.label).width;
        const pad = 4;
        const realBox: BoundingBox = {
          loX: sx - textWidth / 2 - pad,
          loY: sy - clusterFontSize * 0.6 - pad,
          hiX: sx + textWidth / 2 + pad,
          hiY: sy + clusterFontSize * 0.6 + pad,
        };
        if (!grid.insert(realBox)) continue;

        // Draw cluster label with stroke outline for contrast
        ctx.globalAlpha = clusterOpacity;
        ctx.font = clusterFontStr;
        ctx.textAlign = 'center';
        ctx.strokeStyle = isDark ? 'rgba(15, 23, 42, 0.85)' : 'rgba(255, 255, 255, 0.85)';
        ctx.lineWidth = 4;
        ctx.lineJoin = 'round';
        ctx.strokeText(cl.label, sx, sy);
        ctx.fillStyle = desaturateHex(cl.color, 0.3, isDark);
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
  }, [width, height, isDark, showClusterLabels, clusterDataMap]);

  // Keep ref in sync so the camera animation can call renderLabels on completion
  renderLabelsRef.current = renderLabels;

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

    // Render once immediately, then start polling
    renderLabels();
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
      xaxis: { title: { text: '' }, backgroundcolor: sceneBg, showgrid: false, zeroline: false, showticklabels: false, showspikes: false },
      yaxis: { title: { text: '' }, backgroundcolor: sceneBg, showgrid: false, zeroline: false, showticklabels: false, showspikes: false },
      zaxis: { title: { text: '' }, backgroundcolor: sceneBg, showgrid: false, zeroline: false, showticklabels: false, showspikes: false },
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
  }), [axisColor, defaultEye, height, paperBg, sceneBg, width]);

  const config = useMemo<Partial<Config>>(() => ({ displayModeBar: true, displaylogo: false, responsive: true }), []);

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
    Plotly.react(gd, allTraces as PlotData[], layout, config);
  }, [baseTraces, layout, config, plotReady]); // eslint-disable-line react-hooks/exhaustive-deps

  // 3. Update overlay traces (highlights + selected) — single atomic redraw
  useEffect(() => {
    if (!plotReady || !graphDivRef.current || !plotlyLibRef.current) return;
    const Plotly = plotlyLibRef.current;
    const gd = graphDivRef.current;

    const oldCount = overlayTraceCountRef.current;
    const newCount = overlayTraces.length;

    if (oldCount === 0 && newCount === 0) return;

    // Splice overlay traces in place and redraw once
    const baseCount = (gd.data?.length ?? 0) - oldCount;
    gd.data?.splice(baseCount, oldCount, ...(overlayTraces as PlotData[]));
    overlayTraceCountRef.current = newCount;
    Plotly.redraw(gd);
  }, [overlayTraces, plotReady]);

  const mouseDownPosRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const handleMouseDown = (e: MouseEvent) => { mouseDownPosRef.current = { x: e.clientX, y: e.clientY }; };
    container.addEventListener('mousedown', handleMouseDown);
    return () => container.removeEventListener('mousedown', handleMouseDown);
  }, []);

  // Block scroll-zoom-out when camera is at max distance.
  // Reads live camera from Plotly's internal gl-plot3d camera (gl-vec3 array).
  const isAtZoomOutLimit3D = useCallback(() => {
    const gd = graphDivRef.current as any;
    const glplotEye = gd?._fullLayout?.scene?._scene?.glplot?.camera?.eye;
    let dist: number;
    if (glplotEye) {
      dist = Math.sqrt(glplotEye[0] ** 2 + glplotEye[1] ** 2 + glplotEye[2] ** 2);
    } else {
      dist = getZoomLevel(currentCameraRef.current.eye, currentCameraRef.current.center);
    }
    return dist >= MAX_CAMERA_DISTANCE;
  }, []);
  useZoomLimit(containerRef, isAtZoomOutLimit3D);

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
    if (!point.customdata || typeof point.customdata !== 'object') return;

    const clickedPoint = point.customdata as unknown as Point3D;
    onPointClick(clickedPoint);
  }, [onPointClick]);

  useEffect(() => {
    if (!plotReady || !graphDivRef.current) return;
    const graphDiv = graphDivRef.current as any;
    const handlePlotlyHover = (data: any) => {
      if (data.points && data.points.length > 0) {
        const pt = data.points[0];
        if (pt.customdata && typeof pt.customdata === 'object') {
          const point = pt.customdata as unknown as Point3D;
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
            // Use xaxis/yaxis pixel positions if available
            x = pt.xaxis?.l2p?.(pt.x) ?? (containerRect?.width ?? 400) / 2;
            y = pt.yaxis?.l2p?.(pt.y) ?? (containerRect?.height ?? 300) / 2;
          }

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
    const handlePlotlyUnhover = () => setTooltipData(null);
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

  useEffect(() => {
    if (!nebulaMode || !plotReady || !graphDivRef.current || clusterDataMap.size === 0) {
      if (hazeRendererRef.current) {
        hazeRendererRef.current.dispose();
        hazeRendererRef.current = null;
      }
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

    const syncCanvasLayout = () => {
      const containerRect = containerEl.getBoundingClientRect();
      const glRect = glCanvas.getBoundingClientRect();
      const cssW = glRect.width;
      const cssH = glRect.height;
      const offsetX = glRect.left - containerRect.left;
      const offsetY = glRect.top - containerRect.top;

      canvas.style.left = `${offsetX}px`;
      canvas.style.top = `${offsetY}px`;
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;

      canvas.width = Math.round(cssW);
      canvas.height = Math.round(cssH);

      if (hazeRendererRef.current) {
        hazeRendererRef.current.resize(cssW, cssH);
      }
    };

    syncCanvasLayout();

    const haze = new HazeRenderer(canvas);
    hazeRendererRef.current = haze;
    haze.resize(canvas.clientWidth, canvas.clientHeight);
    haze.updateClusters(clusterDataMap, points.length);

    // Hook into glplot's render loop for camera sync
    const originalOnRender = glplot.onrender;

    glplot.onrender = () => {
      if (originalOnRender) originalOnRender();

      syncCanvasLayout();

      const cameraParams = glplot.cameraParams || glplot.camera;
      if (!cameraParams) return;

      const projection = cameraParams.projection || cameraParams._projection;
      const view = cameraParams.view || cameraParams._view;
      const model = glplot.model || (boundsRef.current ? buildDataToSceneMatrix(boundsRef.current) : GL_IDENTITY_MATRIX);
      if (!projection || !view) return;

      haze.render(projection, view, model);
    };

    return () => {
      if (glplot) glplot.onrender = originalOnRender || null;
      haze.dispose();
      hazeRendererRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nebulaMode, plotReady, clusterDataMap]);

  return (
    <div ref={containerRef} className={className ?? 'h-full w-full'} style={{ position: 'relative' }}>
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
            mixBlendMode: 'screen',
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