'use client';

import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import type { PlotData, Layout, Config, PlotMouseEvent, PlotRelayoutEvent } from 'plotly.js';
import type { Point3D, HighlightMap, ColorScaleType } from '../../lib/types/types';
import { useTheme } from 'next-themes';
import { buildCategoryColorMap, getCategoryLabel, getSequentialScale, getDivergingScale, getMonochromeScale, type SequentialScaleName, type DivergingScaleName } from '../../lib/utils/categoryColors';
import { calculateMarkerStyle, calculateLuminosity, calculateHighlightScale, calculateSimilarityColors } from '../../lib/utils/plotUtils';
import { useContainerDimensions } from '../../lib/hooks/useContainerDimensions';
import { FrostedTooltip, type TooltipData } from './FrostedTooltip';
import { easeInOutCubic, lerp, cartesianToSpherical, sphericalToCartesian, getZoomLevel, getZoomMultiplier, formatHoverText } from '../utils/rendeding';


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


  const defaultEye = useMemo(() => {
    const count = points.length;
    if (count === 0) return { x: 2.5, y: 2.5, z: 2.5 };

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
    const calculatedZoom = startDistance - (zoomInRate * Math.log10(count));

    // Clamp: Never go closer than 0.1 (inside the points) or further than 2.5
    const zoom = Math.min(Math.max(calculatedZoom, 0.1), 2.5);

    return { x: zoom, y: zoom, z: zoom };
  }, [points.length]);

  const currentCameraRef = useRef({ eye: defaultEye, center: defaultCenter });




  useEffect(() => {
    import('plotly.js-dist-min').then((lib) => {
      plotlyLibRef.current = lib.default;
    });
  }, []);

  const currentZoomMultiplier = useRef(getZoomMultiplier(currentCameraRef.current.eye, currentCameraRef.current.center));




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

    const targetR = 0.4;
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
        startSpherical = cartesianToSpherical(startEye.x, startEye.y, startEye.z);
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
      const newEye = sphericalToCartesian(curR, curTheta, curPhi);

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
    return buildCategoryColorMap(categoryField, categoryValues);
  }, [categoryField, categoryValues]);

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

  // Pre-calculate data arrays
  const { allX, allY, allZ, allText, allCustomData } = useMemo(() => ({
    allX: points.map(p => p.x),
    allY: points.map(p => p.y),
    allZ: points.map(p => p.z),
    allText: points.map(formatHoverText),
    allCustomData: points,
  }), [points]);

  // --- OPTIMIZED TRACES ---
  const baseTraces = useMemo((): PlotlyData[] => {
    const traces: PlotlyData[] = [];
    const hasHighlights = highlightedIndices && highlightedIndices.size > 0;

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
        // --- MODE: CATEGORICAL (Standard) ---
        const pointsByCategory: Record<string, Point3D[]> = {};
        points.forEach(point => {
          const cat = point.category || 'unknown';
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
              opacity: isMuted ? 0.2 : dimOpacity,  // Even more muted when category is toggled off
            },
            text: catPoints.map(formatHoverText),
            hoverinfo: 'none',
            customdata: catPoints as any,
            showlegend: false,
          });
        });
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
        traces.push({
          x: hX, y: hY, z: hZ, mode: 'markers', type: 'scatter3d',
          marker: {
            sizemode: 'diameter', size: coreSizes, color: coreColors, opacity: 1, line: { color: innerColors[0], width: 1 }
          },
          text: coreTexts, hoverinfo: 'none', customdata: coreCustomData as any, showlegend: false
        });
      }
    }

    return traces;
  }, [
    allX, allY, allZ, allText, allCustomData,
    points, highlightedIndices, markerStyle, highlightScale, showOnlyHighlighted,
    colorBy, isDark, categoryValues, colorMap, numericData, plotlyColorScale, categoryField,
    mutedCategories
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

  const labelTraces = useMemo((): PlotlyData[] => {
    if (!showLabels || !highlightedIndices || highlightedIndices.size === 0) return [];
    const highlightedPoints = points.filter(p => highlightedIndices.has(p.index));
    if (highlightedPoints.length === 0) return [];

    return [{
      x: highlightedPoints.map(p => p.x),
      y: highlightedPoints.map(p => p.y),
      z: highlightedPoints.map(p => p.z),
      mode: 'text' as const, type: 'scatter3d' as const,
      text: highlightedPoints.map(p => p.label || p.id),
      textposition: 'top center' as const,
      textfont: { size: 11, color: isDark ? '#e2e8f0' : '#1e293b' },
      hoverinfo: 'skip' as const, showlegend: false
    }];
  }, [showLabels, highlightedIndices, points, isDark]);

  const plotData = useMemo(() => [...baseTraces, ...selectedTraces, ...labelTraces], [baseTraces, selectedTraces, labelTraces]);

  const layout = useMemo<Partial<Layout>>(() => ({
    width, height, aspectmode: 'data', autosize: true, uirevision: 'true', hovermode: 'closest', showlegend: false,
    paper_bgcolor: paperBg, font: { color: axisColor },
    scene: {
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
    console.log('Plot click event at', now);
    // log camera for debugging
    currentZoomMultiplier.current =  getZoomMultiplier(currentCameraRef.current.eye, currentCameraRef.current.center);

    console.log('Current camera:', currentCameraRef.current.eye);
    console.log('Zoom level:', currentZoomMultiplier.current);

    // Check drag
    if (now - mouseDownTimeRef.current > 500) return;

    // Prevent double-firing (coalesce multiple events including re-render ghosts)
    if (now - lastClickTimeRef.current < 600) return;
    lastClickTimeRef.current = now;

    const point = event.points[0];
    if (!point.customdata || typeof point.customdata !== 'object') return;

    onPointClick(point.customdata as unknown as Point3D);
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
  }, [plotReady]);

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
      <FrostedTooltip data={tooltipData} />
    </div>
  );
}