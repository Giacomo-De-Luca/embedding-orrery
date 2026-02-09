'use client';

import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import type { PlotData, Layout, Config, PlotMouseEvent, PlotRelayoutEvent } from 'plotly.js';
import type { Point3D, HighlightMap, ColorScaleType, NestedColorMap } from '../../lib/types/types';
import { useTheme } from 'next-themes';
import { buildCategoryColorMap, getCategoryLabel, getSequentialScale, getDivergingScale, getMonochromeScale, desaturateHex, type SequentialScaleName, type DivergingScaleName } from '../../lib/utils/categoryColors';
import { isCrameriScale, getCrameriPlotlyScale } from '../../lib/colorMaps/crameriScales';
import { calculateMarkerStyle, calculateLuminosity, calculateHighlightScale, calculateSimilarityColors } from '../../lib/utils/plotUtils';
import { useContainerDimensions } from '../../lib/hooks/useContainerDimensions';
import { FrostedTooltip, type TooltipData } from './FrostedTooltip';
import { easeInOutCubic, lerp, cartesianToSpherical, sphericalToCartesian, getZoomLevel, getZoomMultiplier, formatHoverText } from '../utils/rendeding';
import { groupPointsByCluster, computeDensityGrid, sampleNebulaParticles, hexToRgbNormalized, type ClusterData } from '../../lib/utils/clusterGeometry';
import { NebulaRenderer } from '../../lib/utils/nebulaRenderer';
import { HazeRenderer } from '../../lib/utils/hazeRenderer';
import { computeMVP, buildDataToSceneMatrix, projectToScreen } from '../utils/labelPlacement';
import { CollisionGrid, type BoundingBox } from '../../lib/utils/collisionGrid';

// Hoisted identity matrix for WebGL nebula fallback (avoids per-frame allocation)
const GL_IDENTITY_MATRIX = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);

// Factory pattern: use pre-bundled plotly.js-dist-min to avoid glslify bundler issues
const Plot = dynamic(async () => {
  const { default: createPlotlyComponent } = await import('react-plotly.js/factory');
  const { default: PlotlyLib } = await import('plotly.js-dist-min');
  return createPlotlyComponent(PlotlyLib);
}, { ssr: false });

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
  /** Nebula cloud effect mode: 'volume' for Plotly volume traces, 'webgl' for particle sprites, 'bloom' for Three.js bloom */
  nebulaMode?: 'off' | 'volume' | 'webgl' | 'bloom';
  /** Show topic/subtopic names at cluster centroids */
  showClusterLabels?: boolean;
}

interface PlotlyGraphDiv extends HTMLDivElement {
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

export function ScatterPlot3D({
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
  nebulaMode = 'off',
  showClusterLabels = false,
}: ScatterPlot3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { width, height } = useContainerDimensions(containerRef, { width: 800, height: 600 });

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
    const zoom = Math.min(Math.max(calculatedZoom, 0.1), 2.5);

    return { x: zoom, y: zoom, z: zoom };
  }, [pointCount]);

  const currentCameraRef = useRef({ eye: defaultEye, center: defaultCenter });

  const defaultDistance = useMemo(() => {
    const { x, y, z } = defaultEye;
    return Math.sqrt(x * x + y * y + z * z);
  }, [defaultEye]);

  const labelCanvasRef = useRef<HTMLCanvasElement>(null);
  const bloomCanvasRef = useRef<HTMLCanvasElement>(null);
  const labelRenderDataRef = useRef<{
    points: { x: number; y: number; z: number; label: string; index: number }[];
    similarities: Map<number, number>;
    selectedIndex: number | null;
  } | null>(null);

  useEffect(() => {
    import('plotly.js-dist-min').then((lib) => {
      plotlyLibRef.current = lib.default;
    });
  }, []);

  const currentZoomMultiplier = useRef(getZoomMultiplier(currentCameraRef.current.eye, currentCameraRef.current.center));

  // Ref to track bounds for projection (avoids stale closure)
  const boundsRef = useRef(bounds);
  boundsRef.current = bounds;




  // --- Camera Animation Effect (Unchanged logic) ---
  useEffect(() => {
    if (!selectedPoint || !bounds || !plotReady || !graphDivRef.current) return;
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

    const dataCenterX = (bounds.minX + bounds.maxX) / 2;
    const dataCenterY = (bounds.minY + bounds.maxY) / 2;
    const dataCenterZ = (bounds.minZ + bounds.maxZ) / 2;
    const maxRange = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ) || 1;

    const targetCenterX = (selectedPoint.x - dataCenterX) / maxRange;
    const targetCenterY = (selectedPoint.y - dataCenterY) / maxRange;
    const targetCenterZ = (selectedPoint.z - dataCenterZ) / maxRange;

    // Debug: log Plotly scene internals to understand coordinate mapping
    const sceneLayout = (graphDivRef.current._fullLayout?.scene) as any;
    const glplot = sceneLayout?._scene?.glplot;
    // Try multiple paths for model matrix
    const modelPaths = {
      'glplot.model': glplot?.model,
      'glplot.cameraParams.model': glplot?.cameraParams?.model,
      'glplot.objects[0].model': glplot?.objects?.[0]?.model,
      'glplot._model': glplot?._model,
    };
    const foundModel = Object.entries(modelPaths).find(([, v]) => v != null);
    console.log('[3D Camera Debug]', JSON.stringify({
      clickedPoint: { x: selectedPoint.x, y: selectedPoint.y, z: selectedPoint.z },
      computedTarget: { x: targetCenterX, y: targetCenterY, z: targetCenterZ },
      dataBounds: bounds,
      maxRange,
      axisRanges: {
        x: sceneLayout?.xaxis?.range ? [...sceneLayout.xaxis.range] : null,
        y: sceneLayout?.yaxis?.range ? [...sceneLayout.yaxis.range] : null,
        z: sceneLayout?.zaxis?.range ? [...sceneLayout.zaxis.range] : null,
      },
      aspectratio: sceneLayout?.aspectratio,
      glplotBounds: glplot?.bounds ? [Array.from(glplot.bounds[0]), Array.from(glplot.bounds[1])] : null,
      modelMatrixPath: foundModel ? foundModel[0] : null,
      modelMatrix: foundModel ? Array.from(foundModel[1]).slice(0, 16) : null,
      glplotKeys: glplot ? Object.keys(glplot).slice(0, 20) : null,
      cameraFormat: glplot?.camera ? (Array.isArray(glplot.camera.eye) ? 'array' : typeof glplot.camera.eye) : null,
    }, null, 2));

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
      }
    };

    isAnimatingRef.current = true;
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      isAnimatingRef.current = false;
    };
  }, [selectedPoint, bounds, plotReady]);

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
    const hasHighlights = highlightedIndices && highlightedIndices.size > 0;

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
      const dimOpacity = hasHighlights ? markerStyle.opacity * 0.9 : markerStyle.opacity;
      const dimSize = hasHighlights ? Math.max(markerStyle.size * 0.6, 2) : Math.max(markerStyle.size * 0.7, 2);

      if (numericData && plotlyColorScale) {
        // --- MODE: NATIVE COLORSCALE (GPU ACCELERATED) ---
        traces.push({
          x: allX,
          y: allY,
          z: allZ,
          mode: 'markers',
          type: 'scatter3d',
          name: hasHighlights ? 'Context' : 'Data',
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
                opacity: isMuted ? 0.2 : dimOpacity,
              },
              text: subPoints.map(formatHoverText),
              hoverinfo: 'none',
              customdata: subPoints as any,
              showlegend: false,
            });
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
                opacity: isMuted ? 0.2 : dimOpacity,
              },
              text: catPoints.map(formatHoverText),
              hoverinfo: 'none',
              customdata: catPoints as any,
              showlegend: false,
            });
          });
        }
      } else {
        // --- MODE: NO COLORING ---
        traces.push({
          x: allX,
          y: allY,
          z: allZ,
          mode: 'markers',
          type: 'scatter3d',
          name: hasHighlights ? 'Context' : 'Data',
          marker: {
            sizemode: 'diameter',
            size: dimSize,
            color: hasHighlights ? '#e5a819ff' : '#1f77b4',
            opacity: dimOpacity,
          },
          text: allText,
          hoverinfo: 'none',
          customdata: allCustomData as any,
          showlegend: false,
        });
      }
    }

    // --- BLOOM / HIGHLIGHTS (Rendered on top) ---
    if (hasHighlights) {
      const highlightedPoints = points.filter(p => highlightedIndices.has(p.index));
      if (highlightedPoints.length > 0) {
        const hX = highlightedPoints.map(p => p.x);
        const hY = highlightedPoints.map(p => p.y);
        const hZ = highlightedPoints.map(p => p.z);

        const outerSizes: number[] = [];
        const outerColors: string[] = [];
        const outerOpacities: number[] = [];
        const innerSizes: number[] = [];
        const innerColors: string[] = [];
        const coreSizes: number[] = [];
        const coreColors: string[] = [];
        const coreTexts: string[] = [];
        const coreCustomData: any[] = [];

        highlightedPoints.forEach(point => {
          const similarity = highlightedIndices!.get(point.index) ?? 1.0;
          const luminosity = calculateLuminosity(similarity);
          const colors = calculateSimilarityColors(similarity);

          outerSizes.push(Math.max(markerStyle.size * highlightScale.outerMultiplier, 30));
          outerColors.push(colors.outerGlow);
          outerOpacities.push(luminosity.outer);

          innerSizes.push(Math.max(markerStyle.size * highlightScale.innerMultiplier, 18));
          innerColors.push(colors.glowColor);

          coreSizes.push(Math.max(markerStyle.size * highlightScale.coreMultiplier, 9));
          coreColors.push(colors.coreColor);
          coreTexts.push(formatHoverText(point));
          coreCustomData.push(point);
        });

        // 1. Outer Glow
        traces.push({
          x: hX, y: hY, z: hZ, mode: 'markers', type: 'scatter3d',
          marker: {
            sizemode: 'diameter', size: outerSizes, color: outerColors, opacity: 0.15, line: { width: 0 }
          },
          hoverinfo: 'skip', showlegend: false
        });

        // 2. Inner Glow
        traces.push({
          x: hX, y: hY, z: hZ, mode: 'markers', type: 'scatter3d',
          marker: {
            sizemode: 'diameter', size: innerSizes, color: innerColors, opacity: 0.3, line: { width: 0 }
          },
          hoverinfo: 'skip', showlegend: false
        });

        // 3. Core
        // Only make highlights clickable when base traces aren't shown (prevents duplicate clicks)
        traces.push({
          x: hX, y: hY, z: hZ, mode: 'markers', type: 'scatter3d',
          marker: {
            sizemode: 'diameter', size: coreSizes, color: coreColors, opacity: 1, line: { color: innerColors[0], width: 1 }
          },
          text: coreTexts,
          hoverinfo: 'none',
          customdata: showOnlyHighlighted ? (coreCustomData as any) : undefined,
          showlegend: false
        });
      }
    }

    return traces;
  }, [
    points, highlightedIndices, markerStyle, highlightScale, showOnlyHighlighted,
    colorBy, isDark, categoryValues, colorMap, numericData, plotlyColorScale, categoryField,
    mutedCategories, hideUnclustered, nestedColorMap
  ]);

  // Selected point traces and layout/config remain similar...
  const selectedTraces = useMemo((): PlotlyData[] => {
    if (!selectedPoint) return [];
    const traces: PlotlyData[] = [];
    const hasHighlights = highlightedIndices && highlightedIndices.size > 0;

    if (hasHighlights) {
      const highlightedPoints = points.filter(p => highlightedIndices.has(p.index));
      const lineX: number[] = [], lineY: number[] = [], lineZ: number[] = [];
      highlightedPoints.forEach(p => {
        if (p.index !== selectedPoint.index) {
          lineX.push(selectedPoint.x, p.x, null as any);
          lineY.push(selectedPoint.y, p.y, null as any);
          lineZ.push(selectedPoint.z, p.z, null as any);
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

    traces.push({
      x: [selectedPoint.x], y: [selectedPoint.y], z: [selectedPoint.z], mode: 'markers', type: 'scatter3d',
      hoverinfo: 'skip',
      marker: { size: Math.max(markerStyle.size * highlightScale.selectedOuterMultiplier, 12), color: 'rgba(255, 215, 140, 0.15)', opacity: 0.3, line: { width: 0 } },
      showlegend: false
    });

    traces.push({
      x: [selectedPoint.x], y: [selectedPoint.y], z: [selectedPoint.z], mode: 'markers', type: 'scatter3d',
      name: 'Selected',
      marker: { size: Math.max(markerStyle.size * highlightScale.selectedCoreMultiplier, 4), color: '#fff8e8', opacity: 1, line: { color: 'rgba(255, 200, 100, 0.6)', width: 1.5 } },
      text: [formatHoverText(selectedPoint)], hoverinfo: 'none', customdata: [selectedPoint] as any, showlegend: false
    });

    return traces;
  }, [selectedPoint, highlightedIndices, points, markerStyle, highlightScale, isDark]);

  // Populate label render data (no React state — just a ref for the canvas renderer)
  useEffect(() => {
    if (!showLabels || !highlightedIndices || highlightedIndices.size === 0) {
      labelRenderDataRef.current = null;
      return;
    }
    const highlightedPoints = points.filter(p => highlightedIndices.has(p.index));
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
  }, [showLabels, highlightedIndices, points, selectedPoint]);

  // --- NEBULA / CLUSTER: Cluster data for nebula effects and cluster labels ---
  const clusterDataMap = useMemo(() => {
    if ((nebulaMode === 'off' && !showClusterLabels) || !categoryField) return new Map<string, ClusterData>();

    // Apply same hideUnclustered filter as baseTraces
    const displayPoints = hideUnclustered
      ? points.filter(p => {
          const topicId = p.metadata?.['topic_id'];
          if (topicId === '-1' || topicId === -1) return false;
          const topicLabel = p.metadata?.['topic_label'];
          if (topicLabel === 'Unclustered' || topicLabel === 'unclustered') return false;
          return true;
        })
      : points;

    return groupPointsByCluster(displayPoints, categoryField, colorMap, nestedColorMap);
  }, [nebulaMode, showClusterLabels, points, categoryField, colorMap, nestedColorMap, hideUnclustered]);

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
      // Map: close (≤0.3) → 1.0, far (≥2.0) → 0.15
      const clusterOpacity = Math.max(0.15, Math.min(1.0, 1.0 - (camDist - 0.3) * 0.5));

      for (const cl of clusterData!.labels) {
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

          renderLabels();
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
    if (hasAnyLabels) renderLabels();
  }, [width, height, hasAnyLabels, renderLabels, showLabels, highlightedIndices, selectedPoint]);

  // --- NEBULA PLAN A: Plotly isosurface/volume traces ---
  const nebulaVolumeTraces = useMemo((): PlotlyData[] => {
    if (nebulaMode !== 'volume' || clusterDataMap.size === 0) return [];
    // Performance guard: skip if too many clusters
    if (clusterDataMap.size > 30) return [];

    const traces: PlotlyData[] = [];

    for (const [, cluster] of clusterDataMap) {
      if (cluster.points.length < 10) continue;

      const grid = computeDensityGrid(cluster, 18, 2.5);
      if (grid.maxValue === 0) continue;

      const color = cluster.color;
      const isoMin = grid.maxValue * 0.03;

      // Outer halo — very low threshold, low opacity, large coverage
      traces.push({
        type: 'isosurface' as any,
        x: grid.x,
        y: grid.y,
        z: grid.z,
        value: grid.value,
        isomin: isoMin,
        isomax: grid.maxValue * 0.4,
        surface: { count: 2, fill: 0.8 },
        caps: { x: { show: false }, y: { show: false }, z: { show: false } },
        colorscale: [[0, color], [1, color]] as any,
        showscale: false,
        opacity: 0.08,
        hoverinfo: 'skip',
        showlegend: false,
        lighting: { ambient: 1, diffuse: 0, specular: 0, roughness: 1 },
      } as any);

      // Inner core — higher threshold, more visible
      traces.push({
        type: 'isosurface' as any,
        x: grid.x,
        y: grid.y,
        z: grid.z,
        value: grid.value,
        isomin: grid.maxValue * 0.25,
        isomax: grid.maxValue,
        surface: { count: 3, fill: 0.9 },
        caps: { x: { show: false }, y: { show: false }, z: { show: false } },
        colorscale: [[0, color], [1, color]] as any,
        showscale: false,
        opacity: 0.15,
        hoverinfo: 'skip',
        showlegend: false,
        lighting: { ambient: 1, diffuse: 0, specular: 0, roughness: 1 },
      } as any);
    }

    return traces;
  }, [nebulaMode, clusterDataMap]);

  // --- NEBULA PLAN B: Scatter3d glow particle traces ---
  const nebulaGlowTraces = useMemo((): PlotlyData[] => {
    if (nebulaMode !== 'webgl' || clusterDataMap.size === 0) return [];
    if (clusterDataMap.size > 30) return [];

    const traces: PlotlyData[] = [];

    for (const [, cluster] of clusterDataMap) {
      if (cluster.points.length < 10) continue;

      const particles = sampleNebulaParticles(cluster, 200, 1.8);
      const px: number[] = [], py: number[] = [], pz: number[] = [];
      const sizes: number[] = [];

      for (let i = 0; i < particles.positions.length / 3; i++) {
        px.push(particles.positions[i * 3]);
        py.push(particles.positions[i * 3 + 1]);
        pz.push(particles.positions[i * 3 + 2]);
        sizes.push(particles.sizes[i] * 1.5);
      }

      // Outer glow layer — large, very transparent
      traces.push({
        x: px, y: py, z: pz,
        mode: 'markers', type: 'scatter3d',
        marker: {
          sizemode: 'diameter',
          size: sizes.map(s => s * 2.5),
          color: cluster.color,
          opacity: 0.04,
          line: { width: 0 },
        },
        hoverinfo: 'skip',
        showlegend: false,
      });

      // Inner glow layer — smaller, slightly more visible
      traces.push({
        x: px, y: py, z: pz,
        mode: 'markers', type: 'scatter3d',
        marker: {
          sizemode: 'diameter',
          size: sizes,
          color: cluster.color,
          opacity: 0.1,
          line: { width: 0 },
        },
        hoverinfo: 'skip',
        showlegend: false,
      });
    }

    return traces;
  }, [nebulaMode, clusterDataMap]);

  const plotData = useMemo(() => [
    ...nebulaVolumeTraces,   // Volume nebula behind points
    ...nebulaGlowTraces,     // Glow particle nebula behind points
    ...baseTraces,
    ...selectedTraces,
  ], [nebulaVolumeTraces, nebulaGlowTraces, baseTraces, selectedTraces]);

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

  }), [axisColor, gridColor, height, paperBg, sceneBg, width]);

  const config: Partial<Config> = { displayModeBar: true, displaylogo: false, responsive: true };

  const mouseDownTimeRef = useRef<number>(0);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const handleMouseDown = () => { mouseDownTimeRef.current = Date.now(); };
    container.addEventListener('mousedown', handleMouseDown);
    return () => container.removeEventListener('mousedown', handleMouseDown);
  }, []);

  const handleClick = useCallback((event: PlotMouseEvent) => {
    if (!onPointClick || !event.points || event.points.length === 0) return;
    const now = Date.now();

    // Check drag - ignore clicks that were part of a drag gesture
    if (now - mouseDownTimeRef.current > 500) return;

    const point = event.points[0];
    if (!point.customdata || typeof point.customdata !== 'object') return;

    const clickedPoint = point.customdata as unknown as Point3D;

    // Log camera for debugging
    currentZoomMultiplier.current = getZoomMultiplier(currentCameraRef.current.eye, currentCameraRef.current.center);
    console.log('Point clicked:', clickedPoint.id, 'Camera:', currentCameraRef.current.eye, 'Zoom:', currentZoomMultiplier.current);

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
    }
    return () => {
      if (typeof graphDiv.removeListener === 'function') {
        graphDiv.removeListener('plotly_hover', handlePlotlyHover);
        graphDiv.removeListener('plotly_unhover', handlePlotlyUnhover);
      }
    };
  }, [plotReady, tooltipFields]);

  // --- NEBULA PLAN B: WebGL particle sprites ---
  const nebulaRenderersRef = useRef<Map<string, NebulaRenderer>>(new Map());

  useEffect(() => {
    const renderers = nebulaRenderersRef.current;
    if (nebulaMode !== 'webgl' || !plotReady || !graphDivRef.current || clusterDataMap.size === 0) {
      // Cleanup if mode changed away from webgl
      for (const renderer of renderers.values()) {
        renderer.dispose();
      }
      renderers.clear();
      return;
    }

    const sceneLayout = (graphDivRef.current._fullLayout?.scene) as any;
    const glplot = sceneLayout?._scene?.glplot;
    if (!glplot) return;

    // Use Plotly's existing GL context directly (avoids WebGL2 compatibility issues)
    const gl = glplot.gl as WebGLRenderingContext | null;
    if (!gl) return;

    // Create/update renderers for each cluster
    const activeKeys = new Set<string>();
    for (const [key, cluster] of clusterDataMap) {
      if (cluster.points.length < 10) continue;
      activeKeys.add(key);

      let renderer = renderers.get(key);
      if (!renderer) {
        renderer = new NebulaRenderer(gl);
        renderers.set(key, renderer);
      }

      const particles = sampleNebulaParticles(cluster, 300, 1.5);
      renderer.updateParticles(particles.positions, particles.opacities, particles.sizes);
    }

    // Dispose removed clusters
    for (const [key, renderer] of renderers) {
      if (!activeKeys.has(key)) {
        renderer.dispose();
        renderers.delete(key);
      }
    }

    // Hook into glplot's render loop
    const originalOnRender = glplot.onrender;

    glplot.onrender = () => {
      if (originalOnRender) originalOnRender();

      const cameraParams = glplot.cameraParams || glplot.camera;
      if (!cameraParams) return;

      // Extract matrices
      const projection = cameraParams.projection || cameraParams._projection;
      const view = cameraParams.view || cameraParams._view;
      const model = glplot.model || (boundsRef.current ? buildDataToSceneMatrix(boundsRef.current) : GL_IDENTITY_MATRIX);

      if (!projection || !view) return;

      // Save GL state (including alpha blend factors for blendFuncSeparate)
      const prevBlend = gl.isEnabled(gl.BLEND);
      const prevDepthMask = gl.getParameter(gl.DEPTH_WRITEMASK);
      const prevBlendSrcRgb = gl.getParameter(gl.BLEND_SRC_RGB);
      const prevBlendDstRgb = gl.getParameter(gl.BLEND_DST_RGB);
      const prevBlendSrcAlpha = gl.getParameter(gl.BLEND_SRC_ALPHA);
      const prevBlendDstAlpha = gl.getParameter(gl.BLEND_DST_ALPHA);
      const prevProgram = gl.getParameter(gl.CURRENT_PROGRAM);

      // Set up additive blending
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.depthMask(false);

      // Draw each cluster's nebula
      for (const [key, renderer] of renderers) {
        const cluster = clusterDataMap.get(key);
        if (!cluster) continue;
        const rgb = hexToRgbNormalized(cluster.color);
        renderer.draw(projection, view, model, rgb);
      }

      // Restore GL state
      gl.depthMask(prevDepthMask);
      if (!prevBlend) gl.disable(gl.BLEND);
      else {
        gl.blendFuncSeparate(prevBlendSrcRgb, prevBlendDstRgb, prevBlendSrcAlpha, prevBlendDstAlpha);
      }
      if (prevProgram) gl.useProgram(prevProgram);
    };

    return () => {
      // Unhook render callback
      if (glplot) {
        glplot.onrender = originalOnRender || null;
      }
      // Dispose all renderers
      for (const renderer of renderers.values()) {
        renderer.dispose();
      }
      renderers.clear();
    };
  }, [nebulaMode, plotReady, clusterDataMap]);

  // --- NEBULA PLAN C: Haze sprites (separate overlay canvas) ---
  const hazeRendererRef = useRef<HazeRenderer | null>(null);

  useEffect(() => {
    if (nebulaMode !== 'bloom' || !plotReady || !graphDivRef.current || clusterDataMap.size === 0) {
      if (hazeRendererRef.current) {
        hazeRendererRef.current.dispose();
        hazeRendererRef.current = null;
      }
      return;
    }

    const canvas = bloomCanvasRef.current;
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
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        onClick={plotReady ? handleClick : undefined}
        onInitialized={(_figure, graphDiv) => {
          graphDivRef.current = graphDiv as PlotlyGraphDiv;
          setPlotReady(true);
        }}
        onRelayout={handleRelayout}
      />
      {nebulaMode === 'bloom' && (
        <canvas
          ref={bloomCanvasRef}
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
}