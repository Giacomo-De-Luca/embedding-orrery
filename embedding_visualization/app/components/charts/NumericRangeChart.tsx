'use client';

import { useMemo, useState, useRef, useCallback, useEffect } from 'react';
import type { HistogramBin, ColorScale, CustomNumericRange } from '@/lib/types/types';
import { colorScaleInterpolator } from '@/lib/utils/categoryColors';
import { cn } from '@/lib/utils/utils';

interface NumericRangeChartProps {
  bins: HistogramBin[];
  dataMin: number;
  dataMax: number;
  colorScale: ColorScale;
  customRange?: CustomNumericRange | null;
  onRangeChange?: (range: CustomNumericRange | null) => void;
}

type HandleType = 'min' | 'max' | 'center';

interface DragState {
  type: HandleType;
  startX: number;
}

function formatValue(value: number): string {
  if (Number.isInteger(value)) return Math.round(value).toString();
  return value.toFixed(2).replace(/\.?0+$/, '');
}

export function NumericRangeChart({
  bins,
  dataMin,
  dataMax,
  colorScale,
  customRange,
  onRangeChange,
}: NumericRangeChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Local drag values for real-time UI updates (committed on mouseup)
  const [dragMin, setDragMin] = useState<number | undefined>();
  const [dragMax, setDragMax] = useState<number | undefined>();
  const [dragCenter, setDragCenter] = useState<number | undefined>();

  // Effective values: custom → drag → data
  const effMin = dragMin ?? customRange?.min ?? dataMin;
  const effMax = dragMax ?? customRange?.max ?? dataMax;
  const effCenter = dragCenter ?? customRange?.center ?? (dataMin + dataMax) / 2;

  const hasCustom = customRange != null || isDragging;
  const dataRange = dataMax - dataMin;

  // Stable refs for drag callbacks
  const effMinRef = useRef(effMin);
  const effMaxRef = useRef(effMax);
  const effCenterRef = useRef(effCenter);
  effMinRef.current = effMin;
  effMaxRef.current = effMax;
  effCenterRef.current = effCenter;

  const onRangeChangeRef = useRef(onRangeChange);
  onRangeChangeRef.current = onRangeChange;

  // Color interpolator for bars — depend on colorScale identity + scaleName/baseColor for stability
  const scaleKey = colorScale.type === 'categorical' ? '' :
    colorScale.type === 'monochrome' ? colorScale.baseColor :
    colorScale.scaleName;
  const interpolator = useMemo(() => colorScaleInterpolator(colorScale), [scaleKey, colorScale.type]);

  // Max count for bar height normalization
  const maxCount = useMemo(() => Math.max(...bins.map(b => b.count), 1), [bins]);

  // --- Value ↔ Percentage mapping ---
  const valueToPct = useCallback((v: number) => {
    if (dataRange === 0) return 50;
    return ((v - dataMin) / dataRange) * 100;
  }, [dataMin, dataRange]);

  const clientXToValue = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || dataRange === 0) return dataMin;
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return dataMin + frac * dataRange;
  }, [dataMin, dataRange]);

  // --- Drag handlers (adapted from TemporalFilterChart) ---
  const handleMove = useCallback((clientX: number) => {
    const drag = dragRef.current;
    if (!drag) return;
    const value = clientXToValue(clientX);

    if (drag.type === 'min') {
      setDragMin(Math.min(value, effMaxRef.current - dataRange * 0.01));
    } else if (drag.type === 'max') {
      setDragMax(Math.max(value, effMinRef.current + dataRange * 0.01));
    } else if (drag.type === 'center') {
      setDragCenter(Math.max(effMinRef.current, Math.min(value, effMaxRef.current)));
    }
  }, [clientXToValue, dataRange]);

  const commitRange = useCallback(() => {
    const min = effMinRef.current;
    const max = effMaxRef.current;
    const center = effCenterRef.current;

    const next: CustomNumericRange = {};
    if (Math.abs(min - dataMin) > dataRange * 0.005) next.min = min;
    if (Math.abs(max - dataMax) > dataRange * 0.005) next.max = max;
    if (Math.abs(center - (dataMin + dataMax) / 2) > dataRange * 0.005) {
      next.center = center;
    }

    const isEmpty = next.min === undefined && next.max === undefined && next.center === undefined;
    onRangeChangeRef.current?.(isEmpty ? null : next);
  }, [dataMin, dataMax, dataRange]);

  const handleEnd = useCallback(() => {
    dragRef.current = null;
    setIsDragging(false);
    commitRange();
    setDragMin(undefined);
    setDragMax(undefined);
    setDragCenter(undefined);
  }, [commitRange]);

  // Document-level listeners
  const onMouseMove = useCallback((e: MouseEvent) => handleMove(e.clientX), [handleMove]);
  const onTouchMove = useCallback((e: TouchEvent) => {
    if (e.touches.length === 1) handleMove(e.touches[0].clientX);
  }, [handleMove]);

  const onMouseUp = useCallback(() => {
    handleEnd();
    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);
  }, [handleEnd, onMouseMove]);

  const onTouchEnd = useCallback(() => {
    handleEnd();
    document.removeEventListener('touchmove', onTouchMove);
    document.removeEventListener('touchend', onTouchEnd);
  }, [handleEnd, onTouchMove]);

  const startDrag = useCallback((type: HandleType, clientX: number) => {
    dragRef.current = { type, startX: clientX };
    setIsDragging(true);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [onMouseMove, onMouseUp]);

  const startTouchDrag = useCallback((type: HandleType, clientX: number) => {
    dragRef.current = { type, startX: clientX };
    setIsDragging(true);
    document.addEventListener('touchmove', onTouchMove, { passive: true });
    document.addEventListener('touchend', onTouchEnd);
  }, [onTouchMove, onTouchEnd]);

  // Double-click to reset
  const handleDoubleClick = useCallback(() => {
    onRangeChangeRef.current?.(null);
    setDragMin(undefined);
    setDragMax(undefined);
    setDragCenter(undefined);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.removeEventListener('touchmove', onTouchMove);
      document.removeEventListener('touchend', onTouchEnd);
    };
  }, [onMouseMove, onMouseUp, onTouchMove, onTouchEnd]);

  // --- Percentage positions ---
  const minPct = valueToPct(effMin);
  const maxPct = valueToPct(effMax);
  const centerPct = valueToPct(effCenter);

  return (
    <div className="space-y-1">
      {/* Histogram + overlay */}
      <div
        ref={containerRef}
        className="relative"
        style={isDragging ? { userSelect: 'none' } : undefined}
        onDoubleClick={handleDoubleClick}
      >
        {/* Histogram bars */}
        <div className="flex items-end gap-px h-12">
          {bins.map((bin, i) => {
            const heightPct = maxCount > 0 ? (bin.count / maxCount) * 100 : 0;
            const binMid = (bin.binStart + bin.binEnd) / 2;
            // Piecewise linear: effMin→0, effCenter→0.5, effMax→1
            // This ensures the center handle visibly shifts the color midpoint
            let t: number;
            if (effMax <= effMin) {
              t = 0.5;
            } else if (binMid <= effCenter) {
              const leftRange = effCenter - effMin;
              t = leftRange > 0 ? 0.5 * (binMid - effMin) / leftRange : 0;
            } else {
              const rightRange = effMax - effCenter;
              t = rightRange > 0 ? 0.5 + 0.5 * (binMid - effCenter) / rightRange : 1;
            }
            t = Math.max(0, Math.min(1, t));
            const barColor = interpolator ? interpolator(t) : '#888';
            const isOutside = binMid < effMin || binMid > effMax;

            return (
              <div
                key={i}
                className="flex-1 rounded-t-sm transition-opacity duration-100"
                style={{
                  height: `${heightPct}%`,
                  minHeight: bin.count > 0 ? 1 : 0,
                  backgroundColor: barColor,
                  opacity: isOutside ? 0.2 : 1,
                }}
              />
            );
          })}
        </div>

        {/* Drag overlay */}
        <div className="absolute inset-0">
          {/* Left dim overlay */}
          <div
            className="absolute inset-y-0 left-0 bg-background/50 rounded-l"
            style={{ width: `${minPct}%` }}
          />
          {/* Right dim overlay */}
          <div
            className="absolute inset-y-0 right-0 bg-background/50 rounded-r"
            style={{ width: `${100 - maxPct}%` }}
          />

          {/* Min handle */}
          <div
            className="absolute inset-y-0 w-2 -translate-x-1/2 cursor-ew-resize z-10 group"
            style={{ left: `${minPct}%` }}
            onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); startDrag('min', e.clientX); }}
            onTouchStart={(e) => { e.stopPropagation(); if (e.touches.length === 1) startTouchDrag('min', e.touches[0].clientX); }}
          >
            <div className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-foreground/60 group-hover:bg-foreground transition-colors rounded-full" />
          </div>

          {/* Max handle */}
          <div
            className="absolute inset-y-0 w-2 -translate-x-1/2 cursor-ew-resize z-10 group"
            style={{ left: `${maxPct}%` }}
            onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); startDrag('max', e.clientX); }}
            onTouchStart={(e) => { e.stopPropagation(); if (e.touches.length === 1) startTouchDrag('max', e.touches[0].clientX); }}
          >
            <div className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-foreground/60 group-hover:bg-foreground transition-colors rounded-full" />
          </div>

          {/* Center handle */}
          <div
            className="absolute inset-y-0 w-2 -translate-x-1/2 cursor-ew-resize z-10 group"
            style={{ left: `${centerPct}%` }}
            onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); startDrag('center', e.clientX); }}
            onTouchStart={(e) => { e.stopPropagation(); if (e.touches.length === 1) startTouchDrag('center', e.touches[0].clientX); }}
          >
            <div className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-foreground/40 group-hover:bg-foreground border-x border-dashed border-foreground/20 transition-colors" />
          </div>
        </div>
      </div>

      {/* Value labels */}
      <div className="flex justify-between text-xs tabular-nums">
        <span className={cn("text-muted-foreground", hasCustom && customRange?.min !== undefined && "text-primary font-medium")}>
          {formatValue(effMin)}
        </span>
        <span className={cn("text-muted-foreground", hasCustom && customRange?.center !== undefined && "text-primary font-medium")}>
          {formatValue(effCenter)}
        </span>
        <span className={cn("text-muted-foreground", hasCustom && customRange?.max !== undefined && "text-primary font-medium")}>
          {formatValue(effMax)}
        </span>
      </div>
    </div>
  );
}
