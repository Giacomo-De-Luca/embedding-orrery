'use client';

import { useMemo, useState, useRef, useCallback, useEffect, startTransition } from 'react';
import { Area, AreaChart, XAxis } from 'recharts';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/lib/ui-primitives/card';
import {
  ChartContainer,
  type ChartConfig,
} from '@/lib/ui-primitives/chart';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { buildCategoryColorMap, getCategoryLabel, getCategoryDisplayName } from '@/lib/utils/categoryColors';
import { useVisualizationStore } from '@/lib/stores/useVisualizationStore';
import { fieldToDisplayName } from '@/lib/utils/fieldAnalysis';
import type { TemporalCrossTabRow, TemporalCountRow } from '@/lib/utils/temporalAnalysis';

/** Only use stacked mode when visible categories are at or below this count */
const STACKED_THRESHOLD = 6;

/** Sentinel value for "auto-detect" in the field selector */
const AUTO_DETECT_FIELD = '__auto__';

/** Sanitize category value for use as a recharts/CSS variable key */
const sanitizeKey = (value: string) => value.replace(/[^a-zA-Z0-9_-]/g, '_');

interface TemporalFilterChartProps {
  // Shared
  temporalField: string;
  allPeriods: string[];
  onBrushChange?: (startPeriod: string, endPeriod: string, allPeriods: string[]) => void;
  brushStartIndex?: number;
  brushEndIndex?: number;
  // Standalone mode
  temporalCounts?: TemporalCountRow[];
  // Stacked mode
  categoryField?: string | null;
  categoryValues?: string[];
  categoryCounts?: Record<string, number>;
  crossTabData?: TemporalCrossTabRow[];
  categoricalPalette?: string;
  mutedCategories?: string[];
  // Field selector
  availableFields?: string[];
  temporalFieldOverride?: string | null;
  onTemporalFieldChange?: (field: string | null) => void;
}

interface DragState {
  type: 'left' | 'right' | 'middle';
  startX: number;
  startRange: [number, number];
}

export function TemporalFilterChart({
  temporalField,
  allPeriods,
  onBrushChange,
  brushStartIndex,
  brushEndIndex,
  temporalCounts,
  categoryField,
  categoryValues = [],
  categoryCounts = {},
  crossTabData = [],
  categoricalPalette,
  mutedCategories = [],
  availableFields = [],
  temporalFieldOverride,
  onTemporalFieldChange,
}: TemporalFilterChartProps) {
  // Local range state (indices into allPeriods)
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(Math.max(0, allPeriods.length - 1));
  const [isDragging, setIsDragging] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [hoveredY, setHoveredY] = useState(0);

  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const rangeStartRef = useRef(rangeStart);
  const rangeEndRef = useRef(rangeEnd);
  rangeStartRef.current = rangeStart;
  rangeEndRef.current = rangeEnd;

  const onBrushChangeRef = useRef(onBrushChange);
  onBrushChangeRef.current = onBrushChange;

  // Sync local state from parent-controlled brush indices
  useEffect(() => {
    if (brushStartIndex !== undefined) setRangeStart(brushStartIndex);
  }, [brushStartIndex]);

  useEffect(() => {
    if (brushEndIndex !== undefined) setRangeEnd(brushEndIndex);
  }, [brushEndIndex]);

  // Reset range when allPeriods changes (collection change)
  useEffect(() => {
    if (brushStartIndex === undefined) setRangeStart(0);
    if (brushEndIndex === undefined) setRangeEnd(allPeriods.length - 1);
  }, [allPeriods.length, brushStartIndex, brushEndIndex]);

  const colorOverrides = useVisualizationStore(
    (s) => categoryField ? s.categoryColorOverrides[categoryField] : undefined
  );

  const visibleCategories = useMemo(() => {
    if (!categoryField || categoryValues.length === 0) return [];
    return categoryValues.filter(c => !mutedCategories.includes(c));
  }, [categoryField, categoryValues, mutedCategories]);

  const isStackedMode = visibleCategories.length > 0
    && visibleCategories.length <= STACKED_THRESHOLD
    && crossTabData.length >= 2;

  // --- Stacked mode computations ---
  const colorMap = useMemo(
    () => isStackedMode ? buildCategoryColorMap(categoryField!, categoryValues, categoricalPalette, colorOverrides) : {},
    [isStackedMode, categoryField, categoryValues, categoricalPalette, colorOverrides]
  );

  const topCategories = useMemo(() => {
    if (!isStackedMode) return [];
    return [...visibleCategories]
      .sort((a, b) => (categoryCounts[b] ?? 0) - (categoryCounts[a] ?? 0));
  }, [isStackedMode, visibleCategories, categoryCounts]);

  const chartConfig = useMemo(() => {
    if (isStackedMode) {
      const config: ChartConfig = {};
      for (const cat of topCategories) {
        const safeKey = sanitizeKey(cat);
        config[safeKey] = {
          label: getCategoryLabel(categoryField!, cat),
          color: colorMap[cat] ?? '#7f7f7f',
        };
      }
      return config;
    }
    return {
      count: {
        label: 'Count',
        color: 'var(--chart-1)',
      },
    } satisfies ChartConfig;
  }, [isStackedMode, topCategories, categoryField, colorMap]);

  // Remap crossTabData keys to safe keys for recharts (stacked mode)
  const safeStackedData = useMemo(() => {
    if (!isStackedMode) return [];
    return crossTabData.map(row => {
      const safeRow: Record<string, string | number> = { period: row.period };
      for (const cat of topCategories) {
        safeRow[sanitizeKey(cat)] = (row[cat] as number) ?? 0;
      }
      return safeRow;
    });
  }, [isStackedMode, crossTabData, topCategories]);

  const chartData = isStackedMode ? safeStackedData : (temporalCounts ?? []);
  const temporalDisplayName = fieldToDisplayName(temporalField);
  const displayName = isStackedMode ? getCategoryDisplayName(categoryField!) : 'Items';
  const showFieldSelector = availableFields.length > 0 && !!onTemporalFieldChange;

  // --- Drag interaction ---
  const clientXToIndex = useCallback((clientX: number): number => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || allPeriods.length <= 1) return 0;
    const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return Math.round(fraction * (allPeriods.length - 1));
  }, [allPeriods.length]);

  const commitRange = useCallback((start: number, end: number) => {
    if (onBrushChangeRef.current && allPeriods.length > 0) {
      const startPeriod = allPeriods[start] ?? allPeriods[0];
      const endPeriod = allPeriods[end] ?? allPeriods[allPeriods.length - 1];
      startTransition(() => {
        onBrushChangeRef.current!(startPeriod, endPeriod, allPeriods);
      });
    }
  }, [allPeriods]);

  const handleMove = useCallback((clientX: number) => {
    const drag = dragRef.current;
    if (!drag) return;

    const maxIdx = allPeriods.length - 1;

    if (drag.type === 'left') {
      const newIdx = Math.min(clientXToIndex(clientX), rangeEndRef.current);
      setRangeStart(newIdx);
    } else if (drag.type === 'right') {
      const newIdx = Math.max(clientXToIndex(clientX), rangeStartRef.current);
      setRangeEnd(newIdx);
    } else {
      // middle: slide the window
      const delta = clientXToIndex(clientX) - clientXToIndex(drag.startX);
      const width = drag.startRange[1] - drag.startRange[0];
      let newStart = drag.startRange[0] + delta;
      let newEnd = drag.startRange[1] + delta;
      // Clamp
      if (newStart < 0) { newStart = 0; newEnd = width; }
      if (newEnd > maxIdx) { newEnd = maxIdx; newStart = maxIdx - width; }
      setRangeStart(newStart);
      setRangeEnd(newEnd);
    }
  }, [allPeriods.length, clientXToIndex]);

  const handleEnd = useCallback(() => {
    dragRef.current = null;
    setIsDragging(false);
    commitRange(rangeStartRef.current, rangeEndRef.current);
  }, [commitRange]);

  // Document-level mouse listeners
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

  const startDrag = useCallback((type: DragState['type'], clientX: number) => {
    dragRef.current = {
      type,
      startX: clientX,
      startRange: [rangeStartRef.current, rangeEndRef.current],
    };
    setIsDragging(true);
    setHoveredIndex(null);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [onMouseMove, onMouseUp]);

  const startTouchDrag = useCallback((type: DragState['type'], clientX: number) => {
    dragRef.current = {
      type,
      startX: clientX,
      startRange: [rangeStartRef.current, rangeEndRef.current],
    };
    setIsDragging(true);
    setHoveredIndex(null);
    document.addEventListener('touchmove', onTouchMove, { passive: true });
    document.addEventListener('touchend', onTouchEnd);
  }, [onTouchMove, onTouchEnd]);

  // Double-click to reset
  const handleDoubleClick = useCallback(() => {
    const maxIdx = allPeriods.length - 1;
    setRangeStart(0);
    setRangeEnd(maxIdx);
    commitRange(0, maxIdx);
  }, [allPeriods.length, commitRange]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.removeEventListener('touchmove', onTouchMove);
      document.removeEventListener('touchend', onTouchEnd);
    };
  }, [onMouseMove, onMouseUp, onTouchMove, onTouchEnd]);

  // --- Percentage computation ---
  const leftPct = allPeriods.length > 1 ? (rangeStart / (allPeriods.length - 1)) * 100 : 0;
  const rightPct = allPeriods.length > 1 ? (rangeEnd / (allPeriods.length - 1)) * 100 : 100;

  // Full range detection
  const isFullRange = rangeStart === 0 && rangeEnd === allPeriods.length - 1;

  // Range text
  const rangeText = !isFullRange && allPeriods.length > 0
    ? `${allPeriods[rangeStart]} \u2013 ${allPeriods[rangeEnd]}`
    : null;

  const hasChartData = chartData.length >= 2;

  // Show selector even without chart data so users can pick a temporal field
  if (!hasChartData && !showFieldSelector) return null;

  return (
    <Card className="border-0 shadow-none bg-transparent">
      <CardHeader className="px-0 pt-0 pb-2 space-y-2">
        <CardTitle className="text-sm">
          {hasChartData ? `${displayName} over ${temporalDisplayName}` : 'Temporal Distribution'}
        </CardTitle>
        {showFieldSelector && (
          <Select
            value={temporalFieldOverride ?? AUTO_DETECT_FIELD}
            onValueChange={(val) => onTemporalFieldChange!(val === AUTO_DETECT_FIELD ? null : val)}
          >
            <SelectTrigger size="sm" className="w-full text-xs h-7">
              <SelectValue placeholder="Select field..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={AUTO_DETECT_FIELD}>
                <span className="text-xs text-muted-foreground">Auto-detect</span>
              </SelectItem>
              <SelectSeparator />
              {availableFields.map(field => (
                <SelectItem key={field} value={field}>
                  <span className="text-xs">{fieldToDisplayName(field)}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        {hasChartData && <CardDescription className="text-xs flex items-center gap-1">
          {allPeriods.length} periods
          {rangeText && (
            <>
              {' '}&middot; {rangeText}
              <button
                type="button"
                onClick={handleDoubleClick}
                className="ml-1 text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Clear temporal filter"
              >
                &times;
              </button>
            </>
          )}
        </CardDescription>}
      </CardHeader>
      {hasChartData && <CardContent className="px-0 pb-0">
        <div
          ref={containerRef}
          className="relative"
          style={isDragging ? { userSelect: 'none' } : undefined}
          onDoubleClick={handleDoubleClick}
        >
          {/* 1. Minimal Recharts area chart */}
          <ChartContainer config={chartConfig} className="aspect-[3/1] w-full">
            <AreaChart
              data={chartData}
              margin={{ top: 4, right: 0, bottom: 20, left: 0 }}
            >
              <XAxis
                dataKey="period"
                tickLine={false}
                axisLine={false}
                tickMargin={4}
                tick={{ fontSize: 11 }}
                tickFormatter={(value) => {
                  const str = String(value);
                  return str.length > 4 ? str.slice(0, 4) : str;
                }}
              />
              {isStackedMode ? (
                <defs>
                  {topCategories.map(cat => {
                    const safeKey = sanitizeKey(cat);
                    return (
                      <linearGradient key={safeKey} id={`filter-fill-${safeKey}`} x1="0" y1="0" x2="0" y2="1">
                        <stop
                          offset="5%"
                          stopColor={`var(--color-${safeKey})`}
                          stopOpacity={0.8}
                        />
                        <stop
                          offset="95%"
                          stopColor={`var(--color-${safeKey})`}
                          stopOpacity={0.1}
                        />
                      </linearGradient>
                    );
                  })}
                </defs>
              ) : (
                <defs>
                  <linearGradient id="filter-fill-count" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-count)" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="var(--color-count)" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
              )}
              {isStackedMode
                ? topCategories.map(cat => {
                    const safeKey = sanitizeKey(cat);
                    return (
                      <Area
                        key={safeKey}
                        dataKey={safeKey}
                        type="natural"
                        fill={`url(#filter-fill-${safeKey})`}
                        fillOpacity={0.4}
                        stroke={`var(--color-${safeKey})`}
                        stackId="a"
                      />
                    );
                  })
                : (
                  <Area
                    dataKey="count"
                    type="natural"
                    fill="url(#filter-fill-count)"
                    fillOpacity={0.4}
                    stroke="var(--color-count)"
                  />
                )
              }
            </AreaChart>
          </ChartContainer>

          {/* 2. Range picker overlay */}
          <div
            className="absolute inset-0 rounded-lg border border-border/40"
            onMouseMove={(e) => {
              if (!isDragging) {
                setHoveredIndex(clientXToIndex(e.clientX));
                const rect = containerRef.current?.getBoundingClientRect();
                if (rect) setHoveredY(e.clientY - rect.top);
              }
            }}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            {/* Left dim overlay */}
            <div
              className="absolute inset-y-0 left-0 bg-background/60 rounded-l-lg"
              style={{ width: `${leftPct}%` }}
            />

            {/* Right dim overlay */}
            <div
              className="absolute inset-y-0 right-0 bg-background/60 rounded-r-lg"
              style={{ width: `${100 - rightPct}%` }}
            />

            {/* Selected region — draggable middle */}
            <div
              className={`absolute inset-y-0 bg-white/[0.04] ${isDragging && dragRef.current?.type === 'middle' ? 'cursor-grabbing' : 'cursor-grab'}`}
              style={{ left: `${leftPct}%`, width: `${rightPct - leftPct}%` }}
              onMouseDown={(e) => {
                e.preventDefault();
                startDrag('middle', e.clientX);
              }}
              onTouchStart={(e) => {
                if (e.touches.length === 1) startTouchDrag('middle', e.touches[0].clientX);
              }}
            />

            {/* Left handle bar */}
            <div
              className="absolute inset-y-0 w-1.5 -translate-x-1/2 bg-border/80 cursor-ew-resize rounded-sm hover:bg-border transition-colors"
              style={{ left: `${leftPct}%` }}
              onMouseDown={(e) => {
                e.preventDefault();
                e.stopPropagation();
                startDrag('left', e.clientX);
              }}
              onTouchStart={(e) => {
                e.stopPropagation();
                if (e.touches.length === 1) startTouchDrag('left', e.touches[0].clientX);
              }}
            />

            {/* Right handle bar */}
            <div
              className="absolute inset-y-0 w-1.5 -translate-x-1/2 bg-border/80 cursor-ew-resize rounded-sm hover:bg-border transition-colors"
              style={{ left: `${rightPct}%` }}
              onMouseDown={(e) => {
                e.preventDefault();
                e.stopPropagation();
                startDrag('right', e.clientX);
              }}
              onTouchStart={(e) => {
                e.stopPropagation();
                if (e.touches.length === 1) startTouchDrag('right', e.touches[0].clientX);
              }}
            />

            {/* Hover tooltip */}
            {hoveredIndex !== null && !isDragging && chartData[hoveredIndex] && (() => {
              const data = chartData[hoveredIndex] as Record<string, unknown>;
              const period = data.period as string;
              const tooltipPct = allPeriods.length > 1
                ? (hoveredIndex / (allPeriods.length - 1)) * 100
                : 50;

              return (
                <div
                  className="absolute pointer-events-none z-10"
                  style={{
                    left: `${tooltipPct}%`,
                    top: hoveredY,
                    transform: 'translate(-50%, -100%)',
                  }}
                >
                  <div className="rounded-lg border bg-background p-2 shadow-sm whitespace-nowrap">
                    <p className="text-xs font-medium mb-1">{period}</p>
                    {isStackedMode ? (
                      <>
                        {topCategories.map(cat => {
                          const safeKey = sanitizeKey(cat);
                          const value = (data[safeKey] as number | undefined) ?? 0;
                          const color = chartConfig[safeKey]?.color ?? '#7f7f7f';
                          return (
                            <div key={safeKey} className="flex items-center gap-1.5">
                              <span
                                className="inline-block w-2 h-2 rounded-full shrink-0"
                                style={{ backgroundColor: color }}
                              />
                              <span className="text-xs text-muted-foreground">{chartConfig[safeKey]?.label ?? cat}:</span>
                              <span className="text-xs ml-auto pl-2">{value.toLocaleString()}</span>
                            </div>
                          );
                        })}
                        <div className="border-t mt-1 pt-1 flex justify-between gap-3">
                          <span className="text-xs text-muted-foreground">Total:</span>
                          <span className="text-xs font-medium">
                            {topCategories
                              .reduce((sum, cat) => sum + ((data[sanitizeKey(cat)] as number | undefined) ?? 0), 0)
                              .toLocaleString()}
                          </span>
                        </div>
                      </>
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        Count: {((data.count as number | undefined) ?? 0).toLocaleString()}
                      </p>
                    )}
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      </CardContent>}
    </Card>
  );
}
