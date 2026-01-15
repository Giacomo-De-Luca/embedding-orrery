'use client';

import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import type { PlotData, Layout, Config, PlotMouseEvent, PlotRelayoutEvent } from 'plotly.js';
import type { Point3D, HighlightMap } from '../../lib/types/types';
import { useTheme } from 'next-themes';
import { buildCategoryColorMap, getCategoryLabel } from '../../lib/utils/categoryColors';
import { calculateMarkerStyle, calculateLuminosity, calculateHighlightScale, calculateSimilarityColors } from '../../lib/utils/plotUtils';
import { useContainerDimensions } from '../../lib/hooks/useContainerDimensions';
import { FrostedTooltip, type TooltipData } from './FrostedTooltip';

// Factory pattern: use pre-bundled plotly.js-dist-min to avoid glslify bundler issues
// Use dynamic import to avoid "self is not defined" error in SSR
const Plot = dynamic(async () => {
  const { default: createPlotlyComponent } = await import('react-plotly.js/factory');
  const { default: PlotlyLib } = await import('plotly.js-dist-min');
  return createPlotlyComponent(PlotlyLib);
}, { ssr: false });

type PlotlyData = Partial<PlotData>;

interface ScatterPlot3DProps {
  points: Point3D[];
  colorBy?: 'category' | 'none';
  categoryField?: string | null;
  categoryValues?: string[];
  highlightedIndices?: HighlightMap;
  selectedPoint?: Point3D | null;
  onPointClick?: (point: Point3D) => void;
  className?: string;
  /** When true, only show highlighted points (hide non-highlighted) */
  showOnlyHighlighted?: boolean;
  /** When true, show text labels on highlighted points */
  showLabels?: boolean;
}

// --- Animation Helpers ---

// Smooth easing (Ease In Out Cubic) for a cinematic start and stop
const easeInOutCubic = (t: number): number => {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
};

const lerp = (start: number, end: number, t: number) => {
  return start + (end - start) * t;
};

// Convert Cartesian (x,y,z) to Spherical (radius, theta, phi) for orbiting
function cartesianToSpherical(x: number, y: number, z: number) {
  const r = Math.sqrt(x * x + y * y + z * z);
  const theta = Math.atan2(y, x);
  const phi = Math.acos(z / (r || 1));
  return { r, theta, phi };
}

// Convert back to Cartesian
function sphericalToCartesian(r: number, theta: number, phi: number) {
  return {
    x: r * Math.sin(phi) * Math.cos(theta),
    y: r * Math.sin(phi) * Math.sin(theta),
    z: r * Math.cos(phi),
  };
}

function formatHoverText(point: Point3D): string {
  const label = point.label || point.id;
  const doc = point.document || '';
  const truncatedDoc = doc.length > 100 ? doc.substring(0, 100) + '...' : doc;
  return `${label}<br>${truncatedDoc}`;
}

interface PlotlyCamera {
  eye?: { x: number; y: number; z: number };
  center?: { x: number; y: number; z: number };
  up?: { x: number; y: number; z: number };
}

interface PlotlyScene {
  camera?: PlotlyCamera;
  _scene?: {
    camera?: any; // Internal WebGL camera
    setCamera?: (camera: PlotlyCamera) => void;
    glplot?: {
      camera?: any;
      draw?: () => void;
    };
  };
}

interface PlotlyGraphDiv extends HTMLDivElement {
  _fullLayout?: {
    scene?: PlotlyScene;
  };
}

export function ScatterPlot3D({
  points,
  colorBy = 'none',
  categoryField = null,
  categoryValues = [],
  highlightedIndices,
  selectedPoint,
  onPointClick,
  className,
  showOnlyHighlighted = false,
  showLabels = false,
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

  // --- Camera Logic Starts Here ---

  // 1. Calculate Bounds (Needed for Plotly's normalized center calculation)
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

  // 2. Camera State
  // Default Zoom: r=1.6 is roughly default. r=0.9 is "more zoomed in" as requested.
  const defaultEye = { x: 0.9, y: 0.9, z: 0.9 };
  const defaultCenter = { x: 0, y: 0, z: 0 };

  // We use a ref to track the "current" camera position.
  // This allows manual user rotation to be the starting point of the next animation.
  const currentCameraRef = useRef({ eye: defaultEye, center: defaultCenter });
  const animationFrameRef = useRef<number | undefined>(undefined);
  const isAnimatingRef = useRef(false);
  const graphDivRef = useRef<PlotlyGraphDiv | null>(null);
  const [plotReady, setPlotReady] = useState(false);
  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);
  const plotlyLibRef = useRef<any>(null);

  // Load Plotly lib for direct manipulation
  useEffect(() => {
    import('plotly.js-dist-min').then((lib) => {
      plotlyLibRef.current = lib.default;
    });
  }, []);

  // 3. Animation Effect - uses direct scene camera manipulation for smooth animation
  useEffect(() => {
    if (!selectedPoint || !bounds || !plotReady || !graphDivRef.current) return;

    // Cancel any existing animation before starting new one
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    // A. Target Center: Calculate offset from data center, scaled uniformly
    const dataCenterX = (bounds.minX + bounds.maxX) / 2;
    const dataCenterY = (bounds.minY + bounds.maxY) / 2;
    const dataCenterZ = (bounds.minZ + bounds.maxZ) / 2;

    const rangeX = bounds.maxX - bounds.minX;
    const rangeY = bounds.maxY - bounds.minY;
    const rangeZ = bounds.maxZ - bounds.minZ;
    const maxRange = Math.max(rangeX, rangeY, rangeZ) || 1;

    const targetCenterX = (selectedPoint.x - dataCenterX) / maxRange;
    const targetCenterY = (selectedPoint.y - dataCenterY) / maxRange;
    const targetCenterZ = (selectedPoint.z - dataCenterZ) / maxRange;

    // C. Target Eye (Cinematic orbit) - calculate relative to start theta
    const targetR = 0.4;
    const targetPhi = 1.3;
    const duration = 2000;

    // Animation state - captured in first frame after Plotly settles
    let startEye: { x: number; y: number; z: number };
    let startCenter: { x: number; y: number; z: number };
    let startSpherical: { r: number; theta: number; phi: number };
    let targetSpherical: { r: number; theta: number; phi: number };
    let startTime: number;
    let initialized = false;

    const animate = (currentTime: number) => {
      if (!isAnimatingRef.current || !graphDivRef.current) return;

      // First frame: capture actual camera state from Plotly (after it has settled)
      if (!initialized) {
        const scene = graphDivRef.current._fullLayout?.scene;
        const layoutCamera = scene?.camera;

        // Read camera from Plotly's actual state (after any resets have happened)
        if (layoutCamera?.eye) {
          startEye = { ...layoutCamera.eye };
          startCenter = layoutCamera.center ? { ...layoutCamera.center } : { x: 0, y: 0, z: 0 };
        } else {
          // Fallback to our ref
          startEye = { ...currentCameraRef.current.eye };
          startCenter = { ...currentCameraRef.current.center };
        }

        startSpherical = cartesianToSpherical(startEye.x, startEye.y, startEye.z);
        targetSpherical = {
          r: targetR,
          theta: startSpherical.theta + 0.5, // Orbit relative to current position
          phi: targetPhi,
        };

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

      // Direct WebGL camera manipulation for smooth animation
      const currentScene = graphDivRef.current._fullLayout?.scene?._scene;
      const glplot = currentScene?.glplot as any;

      // Direct glplot.camera property assignment
      if (glplot?.camera) {
        // Try setting eye/center directly (gl-plot3d camera style)
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

        // Try calling update if it exists
        if (typeof glplot.camera.update === 'function') {
          glplot.camera.update();
        }

        // Force redraw
        if (typeof glplot.draw === 'function') {
          glplot.draw();
        }
      }

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        // Animation complete - sync final camera state to Plotly
        isAnimatingRef.current = false;

        // Use Plotly.relayout to persist the final camera position
        // This prevents Plotly from resetting on the next render
        if (plotlyLibRef.current && graphDivRef.current) {
          plotlyLibRef.current.relayout(graphDivRef.current, {
            'scene.camera': {
              eye: newEye,
              center: newCenter,
              up: { x: 0, y: 0, z: 1 },
            },
          });
        }
      }
    };

    isAnimatingRef.current = true;
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      isAnimatingRef.current = false;
    };
  }, [selectedPoint, bounds, plotReady]);

  // 4. Handle manual rotation (Relayout)
  // This ensures that if the user grabs the camera, we don't snap back abruptly later
  const handleRelayout = useCallback((e: Readonly<PlotRelayoutEvent>) => {
    // Don't update during our own animation - prevents feedback loops
    if (isAnimatingRef.current) return;

    // We only update our ref, we don't force a re-render to avoid lag
    const sceneCamera = (e as any)['scene.camera'];
    if (sceneCamera) {
      if (sceneCamera.eye) currentCameraRef.current.eye = sceneCamera.eye;
      if (sceneCamera.center) currentCameraRef.current.center = sceneCamera.center;
    }
  }, []);


  const colorMap = useMemo(() => {
    return buildCategoryColorMap(categoryField, categoryValues);
  }, [categoryField, categoryValues]);

  const markerStyle = useMemo(() => {
    return calculateMarkerStyle(points.length);
  }, [points.length]);

  const highlightScale = useMemo(() => {
    return calculateHighlightScale(points.length);
  }, [points.length]);

  // 1. PRE-CALCULATE DATA ARRAYS
  // This ensures we don't map() over 150k points when you click.
  // These arrays are created ONCE and reused until 'points' actually changes.
  const { allX, allY, allZ, allText, allCustomData } = useMemo(() => {
    return {
      allX: points.map(p => p.x),
      allY: points.map(p => p.y),
      allZ: points.map(p => p.z),
      allText: points.map(formatHoverText),
      allCustomData: points, // Pass the reference directly
    };
  }, [points]);

  // 2. OPTIMIZED BASE TRACES
  const baseTraces = useMemo((): PlotlyData[] => {
    const traces: PlotlyData[] = [];
    const hasHighlights = highlightedIndices && highlightedIndices.size > 0;

    // --- PART A: THE "BACKGROUND" (ALL POINTS) ---
    // We render ALL points here. If highlighted, we just dim this entire layer.
    // The highlighted points will be drawn AGAIN on top, covering their dim versions.

    // --- PART A: THE "BACKGROUND" (ALL POINTS) ---
    if (!showOnlyHighlighted) {
      let bgColors: any;
      let bgOpacity: number;

      if (hasHighlights) {
        // MODE: DIMMED CONTEXT (GOLD)
        // We use your original gold color here
        bgColors = '#e5a819ff'; // grey : #334155
        // We multiply your calculated opacity by 0.2 to get that "dimmed" look
        bgOpacity = markerStyle.opacity * 0.2;
      } else {
        // MODE: FULL VIEW (Standard Colors)
        bgOpacity = markerStyle.opacity;
        if (colorBy === 'category' && categoryValues.length > 0) {
          bgColors = points.map(p => colorMap[p.category || ''] || '#7f7f7f');
        } else {
          bgColors = '#1f77b4';
        }
      }

      traces.push({
        x: allX,
        y: allY,
        z: allZ,
        mode: 'markers',
        type: 'scatter3d',
        name: hasHighlights ? 'Context' : 'Data',
        marker: {
          sizemode: 'diameter',
          // Use 0.6 scale when highlighted (dimmed), 0.7 normally
          size: Math.max(markerStyle.size * (hasHighlights ? 0.6 : 0.7), 2),
          color: bgColors,
          opacity: bgOpacity,
        },
        text: allText,
        hoverinfo: 'none',
        customdata: allCustomData as any,
        showlegend: false,
      });
    }

    // --- PART B: THE "BLOOM" (HIGHLIGHTED POINTS ONLY) ---
    // These are rendered ON TOP of the background trace.
    if (hasHighlights) {
      const highlightedPoints = points.filter(p => highlightedIndices.has(p.index));

      if (highlightedPoints.length > 0) {
        // We map ONLY the ~50 highlighted points (Instant)
        const hX = highlightedPoints.map(p => p.x);
        const hY = highlightedPoints.map(p => p.y);
        const hZ = highlightedPoints.map(p => p.z);

        const outerSizes: number[] = [];
        const outerColors: string[] = [];
        const outerOpacities: number[] = [];

        const innerSizes: number[] = [];
        const innerColors: string[] = [];
        const innerOpacities: number[] = [];

        const coreSizes: number[] = [];
        const coreColors: string[] = [];
        const coreTexts: string[] = [];
        const coreCustomData: any[] = [];

        highlightedPoints.forEach(point => {
          const similarity = highlightedIndices!.get(point.index) ?? 1.0;
          const luminosity = calculateLuminosity(similarity);
          const colors = calculateSimilarityColors(similarity);

          // Prepare styles
          outerSizes.push(Math.max(markerStyle.size * highlightScale.outerMultiplier, 30));
          outerColors.push(colors.outerGlow);
          outerOpacities.push(luminosity.outer);

          innerSizes.push(Math.max(markerStyle.size * highlightScale.innerMultiplier, 18));
          innerColors.push(colors.glowColor);
          innerOpacities.push(luminosity.inner);

          coreSizes.push(Math.max(markerStyle.size * highlightScale.coreMultiplier, 9));
          coreColors.push(colors.coreColor);

          coreTexts.push(formatHoverText(point));
          coreCustomData.push(point);
        });

        // 1. Outer Glow (Transparent Halo)
        traces.push({
          x: hX, y: hY, z: hZ,
          mode: 'markers',
          type: 'scatter3d',
          marker: {
            sizemode: 'diameter',
            size: outerSizes,
            color: outerColors,
            opacity: 0.15,
            line: { width: 0 },
          },
          hoverinfo: 'skip',
          showlegend: false,
        });

        // 2. Inner Glow
        traces.push({
          x: hX, y: hY, z: hZ,
          mode: 'markers',
          type: 'scatter3d',
          marker: {
            sizemode: 'diameter',
            size: innerSizes,
            color: innerColors,
            opacity: 0.3,
            line: { width: 0 },
          },
          hoverinfo: 'skip',
          showlegend: false,
        });

        // 3. Core (Solid & Clickable - covers the dim background dot)
        traces.push({
          x: hX, y: hY, z: hZ,
          mode: 'markers',
          type: 'scatter3d',
          marker: {
            sizemode: 'diameter',
            size: coreSizes,
            color: coreColors,
            opacity: 1,
            line: { color: innerColors[0], width: 1 },
          },
          text: coreTexts,
          hoverinfo: 'none',
          customdata: coreCustomData as any,
          showlegend: false,
        });
      }
    }

    return traces;
  }, [
    // Dependencies
    allX, allY, allZ, allText, allCustomData, // Fast stable refs
    points, // Needed for color mapping logic if state changes
    highlightedIndices,
    markerStyle,
    highlightScale,
    showOnlyHighlighted,
    colorBy,
    isDark,
    categoryValues,
    colorMap
  ]);


  // Selected point traces: constellation lines + golden glow (ONLY depends on selectedPoint)
  // This is fast to recalculate since it only has a few traces
  const selectedTraces = useMemo((): PlotlyData[] => {
    if (!selectedPoint) return [];

    const traces: PlotlyData[] = [];
    const hasHighlights = highlightedIndices && highlightedIndices.size > 0;

    // Constellation Lines - from selected point to each highlighted result
    if (hasHighlights) {
      const highlightedPoints = points.filter(p => highlightedIndices.has(p.index));

      if (highlightedPoints.length > 0) {
        const lineX: number[] = [];
        const lineY: number[] = [];
        const lineZ: number[] = [];

        highlightedPoints.forEach(p => {
          // Don't draw line to itself
          if (p.index !== selectedPoint.index) {
            lineX.push(selectedPoint.x, p.x, null as any);
            lineY.push(selectedPoint.y, p.y, null as any);
            lineZ.push(selectedPoint.z, p.z, null as any);
          }
        });

        if (lineX.length > 0) {
          traces.push({
            x: lineX,
            y: lineY,
            z: lineZ,
            mode: 'lines' as const,
            type: 'scatter3d' as const,
            name: 'Connections',
            line: {
              color: isDark ? 'rgba(130, 160, 200, 0.12)' : 'rgba(100, 130, 170, 0.15)',
              width: 0.1,
            },
            hoverinfo: 'skip' as any,
            showlegend: false,
          });
        }
      }
    }

    // Golden glow for selected point (layered on top of any existing highlight)
    // Outer golden glow
    traces.push({
      x: [selectedPoint.x],
      y: [selectedPoint.y],
      z: [selectedPoint.z],
      mode: 'markers',
      type: 'scatter3d',
      hoverinfo: 'skip',
      marker: {
        size: Math.max(markerStyle.size * highlightScale.selectedOuterMultiplier, 12),
        color: 'rgba(255, 215, 140, 0.15)',
        opacity: 0.3,
        line: { width: 0 },
      },
      showlegend: false,
    });

    // Inner golden glow
    traces.push({
      x: [selectedPoint.x],
      y: [selectedPoint.y],
      z: [selectedPoint.z],
      mode: 'markers',
      type: 'scatter3d',
      hoverinfo: 'skip',
      marker: {
        size: Math.max(markerStyle.size * highlightScale.selectedInnerMultiplier, 8),
        color: 'rgba(255, 223, 160, 0.35)',
        opacity: 0.5,
        line: { width: 0 },
      },
      showlegend: false,
    });

    // Golden core
    traces.push({
      x: [selectedPoint.x],
      y: [selectedPoint.y],
      z: [selectedPoint.z],
      mode: 'markers',
      type: 'scatter3d',
      name: 'Selected',
      marker: {
        size: Math.max(markerStyle.size * highlightScale.selectedCoreMultiplier, 4),
        color: '#fff8e8',
        opacity: 1,
        line: { color: 'rgba(255, 200, 100, 0.6)', width: 1.5 },
      },
      text: [formatHoverText(selectedPoint)],
      hoverinfo: 'none',
      customdata: [selectedPoint] as any,
      showlegend: false,
    });

    return traces;
  }, [selectedPoint, highlightedIndices, points, markerStyle, highlightScale, isDark]);

  // Text labels for highlighted points (rendered last, on top)
  const labelTraces = useMemo((): PlotlyData[] => {
    if (!showLabels || !highlightedIndices || highlightedIndices.size === 0) return [];

    const highlightedPoints = points.filter(p => highlightedIndices.has(p.index));
    if (highlightedPoints.length === 0) return [];

    return [{
      x: highlightedPoints.map(p => p.x),
      y: highlightedPoints.map(p => p.y),
      z: highlightedPoints.map(p => p.z),
      mode: 'text' as const,
      type: 'scatter3d' as const,
      text: highlightedPoints.map(p => p.label || p.id),
      textposition: 'top center' as const,
      textfont: {
        size: 11,
        color: isDark ? '#e2e8f0' : '#1e293b',
      },
      hoverinfo: 'skip' as const,
      showlegend: false,
    }];
  }, [showLabels, highlightedIndices, points, isDark]);

  // Combined plot data
  const plotData = useMemo((): PlotlyData[] => {
    return [...baseTraces, ...selectedTraces, ...labelTraces];
  }, [baseTraces, selectedTraces, labelTraces]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      width,
      height,
      aspectmode: 'data',
      autosize: true,
      uirevision: 'true',
      hovermode: 'closest',
      showlegend: false,
      paper_bgcolor: paperBg,
      font: { color: axisColor },
      scene: {
        xaxis: {
          title: { text: '' },
          titlefont: { color: axisColor },
          tickfont: { color: axisColor },
          backgroundcolor: sceneBg,
          gridcolor: gridColor,
          zerolinecolor: gridColor,
          showgrid: false,
          zeroline: false,
          showspikes: false,
          showticklabels: false,
        },
        yaxis: {
          title: { text: '' },
          titlefont: { color: axisColor },
          tickfont: { color: axisColor },
          backgroundcolor: sceneBg,
          gridcolor: gridColor,
          zerolinecolor: gridColor,
          showgrid: false,
          zeroline: false,
          showspikes: false,
          showticklabels: false,
        },
        zaxis: {
          title: { text: '' },
          titlefont: { color: axisColor },
          tickfont: { color: axisColor },
          backgroundcolor: sceneBg,
          gridcolor: gridColor,
          zerolinecolor: gridColor,
          showgrid: false,
          zeroline: false,
          showspikes: false,
          showticklabels: false,
        },
      },
      margin: { l: 0, r: 0, t: 0, b: 0 },
    }),
    [axisColor, gridColor, height, paperBg, sceneBg, width]
  );

  const config: Partial<Config> = {
    displayModeBar: true,
    displaylogo: false,
    responsive: true,
    doubleClickDelay: 200,
  };

  // Track actual mouse button state to distinguish real clicks from hover events
  const mouseDownTimeRef = useRef<number>(0);

  // Listen for real mousedown events on the container
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleMouseDown = () => {
      mouseDownTimeRef.current = Date.now();
    };

    container.addEventListener('mousedown', handleMouseDown);
    return () => container.removeEventListener('mousedown', handleMouseDown);
  }, []);

  // Debounced click handler to prevent rapid-fire clicks on overlapping points
  const lastClickTimeRef = useRef<number>(0);
  const lastClickedIndexRef = useRef<number | null>(null);

  const handleClick = useCallback((event: PlotMouseEvent) => {
    if (!onPointClick || !event.points || event.points.length === 0) return;

    const now = Date.now();

    // CRITICAL: Reject if no recent mousedown (hover masquerading as click)
    // Plotly 3D can fire onClick during hover due to raycaster behavior
    if (now - mouseDownTimeRef.current > 500) {
      console.log('Rejected: hover masquerading as click (no recent mousedown)');
      return;
    }

    const point = event.points[0];
    // Skip glow layers and lines (they don't have valid customdata)
    if (!point.customdata || typeof point.customdata !== 'object') return;

    const clickedPoint = point.customdata as unknown as Point3D;

    // Debounce: ignore clicks within 300ms of the last click on the same or different point
    if (now - lastClickTimeRef.current < 300) {
      console.log('Click debounced (too fast)');
      return;
    }

    // Also ignore if clicking the same point again within 500ms
    if (clickedPoint.index === lastClickedIndexRef.current && now - lastClickTimeRef.current < 500) {
      console.log('Click debounced (same point)');
      return;
    }

    lastClickTimeRef.current = now;
    lastClickedIndexRef.current = clickedPoint.index;
    onPointClick(clickedPoint);
  }, [onPointClick]);

  // Attach hover events directly to graphDiv (vanilla Plotly pattern)
  // react-plotly.js onHover prop may not work reliably for 3D scatter plots
  useEffect(() => {
    if (!plotReady || !graphDivRef.current) return;

    const graphDiv = graphDivRef.current as any;

    const handlePlotlyHover = (data: any) => {
      if (data.points && data.points.length > 0) {
        const pt = data.points[0];

        // Only show tooltip for points with valid customdata (skip glow layers)
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

    const handlePlotlyUnhover = () => {
      setTooltipData(null);
    };

    graphDiv.on('plotly_hover', handlePlotlyHover);
    graphDiv.on('plotly_unhover', handlePlotlyUnhover);

    return () => {
    };
  }, [plotReady, containerRef]);

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