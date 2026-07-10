/**
 * React lifecycle for the 2D density overlay (see lib/utils/densityRenderer/).
 *
 * Owns the DensityRenderer instance on an overlay canvas positioned over the
 * Plotly plot area, split lifecycle-vs-content like the 3D HazeRenderer:
 * creating/disposing the GL context is keyed only on enablement, while data
 * changes just re-upload buffers, and pan/zoom just re-renders (rAF-coalesced).
 */

import { useCallback, useEffect, useMemo, useRef, type RefObject } from 'react';
import type { Point2D } from '../types/types';
import { DensityRenderer } from '../utils/densityRenderer/DensityRenderer';
import {
  approximateMaxDensity2D,
  buildColorMatrix,
  buildPositionMatrix,
  computeViewingParams,
  linearizedColorBytes,
  type ChannelAssignment,
} from '../utils/densityRenderer/densityMath';
import { plotAreaFromFullLayout, type AxisRanges } from '../../app/utils/labelPlacement2D';

const GAMMA = 2.2;

interface UseDensityOverlayArgs {
  /** densityMode && plot ready — gates the whole feature. */
  enabled: boolean;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  graphDivRef: RefObject<any>;
  /** Visibly active points (muted/filtered already excluded). */
  visiblePoints: Point2D[];
  /** Metadata field the channel assignment is keyed on (null in muted mode). */
  categoryField: string | null;
  channelAssignment: ChannelAssignment;
  /** Full-view axis ranges anchoring the zoom-in crossfade. */
  initialRanges: AxisRanges | null;
  /** Brightness multiplier (Density intensity slider). */
  intensity: number;
  isDark: boolean;
  /** Container CSS dimensions (drive resize re-renders). */
  width: number;
  height: number;
}

export function useDensityOverlay({
  enabled,
  canvasRef,
  graphDivRef,
  visiblePoints,
  categoryField,
  channelAssignment,
  initialRanges,
  intensity,
  isDark,
  width,
  height,
}: UseDensityOverlayArgs): { scheduleRender: () => void; supported: boolean } {
  const supported = useMemo(
    () => typeof window !== 'undefined' && DensityRenderer.isSupported(),
    [],
  );

  const rendererRef = useRef<DensityRenderer | null>(null);
  const maxDensityRef = useRef(0);
  const rafPendingRef = useRef(false);
  // Set when a render is requested while a frame is already queued; drives the
  // trailing render below so the final settled layout is never dropped.
  const rafDirtyRef = useRef(false);

  const colorMatrix = useMemo(
    () => buildColorMatrix(channelAssignment.channelColors, GAMMA),
    [channelAssignment],
  );

  // --- Per-frame render (reads live Plotly layout) ---
  const renderFrame = useCallback(() => {
    rafPendingRef.current = false;
    const renderer = rendererRef.current;
    const canvas = canvasRef.current;
    const fl = graphDivRef.current?._fullLayout;
    if (!renderer || !canvas || !fl) return;

    const xRange = fl.xaxis?.range;
    const yRange = fl.yaxis?.range;
    if (!xRange || !yRange) return;

    // Position the canvas over the plot area only, so the density clips to the
    // axes region automatically and the position matrix stays a plain
    // range→clip mapping.
    const area = plotAreaFromFullLayout(fl, width, height);
    const style = canvas.style;
    const left = `${area.left}px`;
    const top = `${area.top}px`;
    const w = `${area.width}px`;
    const h = `${area.height}px`;
    if (style.left !== left) style.left = left;
    if (style.top !== top) style.top = top;
    if (style.width !== w) style.width = w;
    if (style.height !== h) style.height = h;

    const dpr = window.devicePixelRatio || 1;
    renderer.setSize(area.width, area.height, dpr);

    const params = computeViewingParams({
      maxDensity: maxDensityRef.current,
      xSpan: Math.abs(xRange[1] - xRange[0]),
      ySpan: Math.abs(yRange[1] - yRange[0]),
      initialXSpan: initialRanges ? Math.abs(initialRanges.xRange[1] - initialRanges.xRange[0]) : undefined,
      initialYSpan: initialRanges ? Math.abs(initialRanges.yRange[1] - initialRanges.yRange[0]) : undefined,
      plotWidthCss: area.width,
      plotHeightCss: area.height,
      dpr,
      intensity,
    });

    renderer.render({
      positionMatrix: buildPositionMatrix({
        xRange: [xRange[0], xRange[1]],
        yRange: [yRange[0], yRange[1]],
      }),
      colorMatrix,
      ...params,
      isDark,
    });
  }, [canvasRef, graphDivRef, colorMatrix, initialRanges, intensity, isDark, width, height]);

  // Keep a stable scheduleRender identity while always invoking the latest
  // renderFrame (same anti-staleness pattern as screenshotHandlerRef).
  const renderFrameRef = useRef(renderFrame);
  renderFrameRef.current = renderFrame;

  // Runs the latest renderFrame, then — if another render was requested while
  // this frame was in flight (rafDirtyRef) — queues one more. This guarantees a
  // trailing render on a frame strictly after the settling `plotly_relayout`,
  // so the density never sticks on a stale frame when the last event of a
  // pan/zoom gesture is coalesced away. (renderFrame clears rafPendingRef.)
  const runScheduledFrame = useCallback(() => {
    renderFrameRef.current();
    if (rafDirtyRef.current) {
      rafDirtyRef.current = false;
      rafPendingRef.current = true;
      requestAnimationFrame(runScheduledFrameRef.current);
    }
  }, []);
  const runScheduledFrameRef = useRef(runScheduledFrame);
  runScheduledFrameRef.current = runScheduledFrame;

  const scheduleRender = useCallback(() => {
    if (!rendererRef.current) return;
    // A frame is already queued: mark dirty so a trailing frame runs after it
    // instead of dropping this request (which could strand a stale frame when
    // the layout keeps changing after the queued frame reads it).
    if (rafPendingRef.current) {
      rafDirtyRef.current = true;
      return;
    }
    rafPendingRef.current = true;
    requestAnimationFrame(runScheduledFrameRef.current);
  }, []);

  // --- Renderer lifecycle ---
  useEffect(() => {
    if (!enabled || !supported) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    let renderer: DensityRenderer;
    try {
      renderer = new DensityRenderer(canvas);
    } catch (err) {
      console.warn('Density overlay disabled:', err);
      return;
    }
    rendererRef.current = renderer;
    return () => {
      renderer.dispose();
      rendererRef.current = null;
      rafPendingRef.current = false;
      rafDirtyRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, supported]);

  // --- Data upload (content changes don't recreate the GL context) ---
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;

    const n = visiblePoints.length;
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = visiblePoints[i].x;
      y[i] = visiblePoints[i].y;
    }

    let category: Uint8Array | null = null;
    let colors: Uint8Array | null = null;
    if (channelAssignment.mode === 'categorical' && categoryField) {
      category = new Uint8Array(n);
      const channelOf = channelAssignment.channelOf;
      for (let i = 0; i < n; i++) {
        const raw = visiblePoints[i].metadata?.[categoryField];
        const cat = raw !== null && raw !== undefined && raw !== '' ? String(raw) : 'unknown';
        category[i] = channelOf.get(cat) ?? 0;
      }
    } else if (channelAssignment.mode === 'meanColor' && categoryField && channelAssignment.colorOfCategory) {
      // Each point carries its category's linearized color (LUT per category).
      const byteLut = new Map<string, [number, number, number]>();
      for (const [name, hex] of channelAssignment.colorOfCategory) {
        byteLut.set(name, linearizedColorBytes(hex, GAMMA));
      }
      const fallback = linearizedColorBytes('#7f7f7f', GAMMA);
      colors = new Uint8Array(n * 3);
      for (let i = 0; i < n; i++) {
        const raw = visiblePoints[i].metadata?.[categoryField];
        const cat = raw !== null && raw !== undefined && raw !== '' ? String(raw) : 'unknown';
        const rgb = byteLut.get(cat) ?? fallback;
        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      }
    }

    renderer.setData({
      x,
      y,
      category,
      colors,
      categoryCount: channelAssignment.mode === 'categorical' ? channelAssignment.channelColors.length : 1,
    });
    maxDensityRef.current = approximateMaxDensity2D(x, y);
    scheduleRender();
    // `enabled` re-runs the upload after the lifecycle effect recreates the renderer.
  }, [visiblePoints, categoryField, channelAssignment, enabled, supported, scheduleRender]);

  // --- Re-render on appearance/layout inputs the data effect doesn't cover ---
  useEffect(() => {
    scheduleRender();
  }, [isDark, width, height, colorMatrix, intensity, scheduleRender]);

  return { scheduleRender, supported };
}
