'use client';

import { useMemo, useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import {
  buildCategoryColorMap,
  getCategoryLabel,
  getCategoryDisplayName,
  colorScaleGradientCSS,
} from '../../lib/utils/categoryColors';
import { cn } from '@/lib/utils/utils';
import { ScrollArea, ScrollBar } from '@/lib/ui-primitives/scroll-area';
import { RotateCcw } from 'lucide-react';
import type { NestedColorMap, ColorScale, HistogramBin, CustomNumericRange } from '../../lib/types/types';
import { NumericRangeChart } from './charts/NumericRangeChart';

interface LegendProps {
  className?: string;
  categoryField?: string | null;
  categoryValues?: string[];
  categoryCounts?: Record<string, number>;
  mutedCategories?: string[];
  onCategoryToggle?: (category: string, shiftKey: boolean) => void;
  onCategoryReset?: () => void;
  colorScale?: ColorScale;
  numericRange?: { min: number; max: number };
  histogramBins?: HistogramBin[];
  customNumericRange?: CustomNumericRange | null;
  onCustomRangeChange?: (range: CustomNumericRange | null) => void;
  categoricalPalette?: string;
  nestedColorMap?: NestedColorMap | null;
  /** Pixel max-height for the scrollable content area. Overrides built-in max-h when provided. */
  maxHeight?: number;
  /** Rendered at the bottom of the Card (e.g. a drag handle). */
  dragHandle?: React.ReactNode;
}

/**
 * Format a count with thousand separators for display.
 */
function formatCount(count: number): string {
  return count.toLocaleString();
}

/**
 * Format a numeric value without thousand separators (for ranges in gradients).
 */
function formatNumericValue(value: number): string {
  // Round to reasonable precision, no locale formatting
  if (Number.isInteger(value)) {
    return Math.round(value).toString();
  }
  // For decimals, show up to 2 decimal places
  return value.toFixed(2).replace(/\.?0+$/, '');
}

/**
 * Click-to-edit label for gradient min/max/center values.
 * Shows a styled span by default; transforms to an input on click.
 */
function EditableRangeLabel({ value, dataValue, onCommit, align }: {
  value: number;
  dataValue: number;
  onCommit: (v: number) => void;
  align?: 'left' | 'center' | 'right';
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const isCustom = Math.abs(value - dataValue) > 1e-10;
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  if (!editing) {
    return (
      <span
        className={cn(
          "cursor-pointer hover:text-foreground transition-colors",
          isCustom ? "text-primary font-medium" : "text-muted-foreground",
          align === 'left' && 'text-left',
          align === 'center' && 'text-center',
          align === 'right' && 'text-right',
        )}
        onClick={() => { setDraft(formatNumericValue(value)); setEditing(true); }}
        title={isCustom ? `Data: ${formatNumericValue(dataValue)}` : 'Click to set custom value'}
      >
        {formatNumericValue(value)}
      </span>
    );
  }

  const handleCommit = () => {
    const n = parseFloat(draft);
    if (!isNaN(n) && isFinite(n)) onCommit(n);
    setEditing(false);
  };

  return (
    <input
      ref={inputRef}
      type="number"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={handleCommit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') e.currentTarget.blur();
        if (e.key === 'Escape') setEditing(false);
      }}
      className={cn(
        "w-14 text-xs bg-transparent border-b border-foreground/30 outline-none tabular-nums",
        align === 'left' && 'text-left',
        align === 'center' && 'text-center',
        align === 'right' && 'text-right',
      )}
      autoFocus
    />
  );
}

/**
 * Dynamic legend component that displays category colors with point counts.
 * Click on a category to toggle its visibility (mute/unmute).
 * For continuous scales, displays a horizontal gradient bar with min/center/max labels.
 */
export function Legend({
  className,
  categoryField,
  categoryValues,
  categoryCounts,
  mutedCategories = [],
  onCategoryToggle,
  onCategoryReset,
  colorScale = { type: 'categorical' },
  numericRange,
  histogramBins,
  customNumericRange,
  onCustomRangeChange,
  categoricalPalette,
  nestedColorMap,
  maxHeight,
  dragHandle,
}: LegendProps) {
  // Check if this is a continuous scale (sequential, diverging, or monochrome)
  const isContinuous = colorScale.type === 'sequential' || colorScale.type === 'diverging' || colorScale.type === 'monochrome';

  // Category filter state (local, resets on field change)
  const [filterText, setFilterText] = useState('');
  const [debouncedFilter] = useDebounceValue(filterText, 200);
  const filterInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { setFilterText(''); }, [categoryField]);

  const filterRow = (
    <div className="flex items-center gap-2 pointer-events-auto">
      <input
        ref={filterInputRef}
        type="text"
        value={filterText}
        onChange={(e) => setFilterText(e.target.value)}
        className={cn(
          "flex-1 min-w-0 h-5 bg-transparent text-sm text-foreground/70",
          "border-0 border-b border-muted-foreground/20 outline-none",
          "focus:border-muted-foreground/50 transition-colors"
        )}
        aria-label="Filter categories"
      />
      {filterText && (
        <button
          onClick={() => { setFilterText(''); filterInputRef.current?.focus(); }}
          className="text-muted-foreground/50 hover:text-muted-foreground transition-colors"
          aria-label="Clear filter"
        >
          <X className="h-3 w-3" />
        </button>
      )}
    </div>
  );

  // Generate gradient dynamically from the actual scale function
  const gradient = useMemo(() => colorScaleGradientCSS(colorScale), [colorScale]);

  // For continuous scales, render histogram range chart or fallback gradient bar
  if (isContinuous && numericRange) {
    const { min, max } = numericRange;

    return (
      <Card
        className={`w-fit gap-2 min-w-48 ${className ?? ''}`}
        variant="noBg"
      >
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle className="font-mono text-xs flex-1">{getCategoryDisplayName(categoryField ?? 'value')}</CardTitle>
            {customNumericRange != null && (
              <button
                onClick={() => onCustomRangeChange?.(null)}
                className="text-muted-foreground/50 hover:text-muted-foreground transition-colors pointer-events-auto"
                aria-label="Reset to data range"
                title="Reset to data range"
              >
                <RotateCcw className="h-3 w-3" />
              </button>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-1 pointer-events-auto">
          {histogramBins && histogramBins.length > 0 ? (
            <NumericRangeChart
              bins={histogramBins}
              dataMin={min}
              dataMax={max}
              colorScale={colorScale}
              customRange={customNumericRange}
              onRangeChange={onCustomRangeChange}
              isDiverging={colorScale.type === 'diverging'}
            />
          ) : (
            <>
              <div
                className="h-2 w-full rounded"
                style={{ background: gradient }}
                aria-label={`Color scale from ${min} to ${max}`}
              />
              <div className="flex justify-between text-xs text-muted-foreground tabular-nums">
                <span>{formatNumericValue(min)}</span>
                <span>{formatNumericValue((min + max) / 2)}</span>
                <span>{formatNumericValue(max)}</span>
              </div>
            </>
          )}
        </CardContent>
        {dragHandle}
      </Card>
    );
  }

  // ---- NESTED (two-level) LEGEND ----
  if (nestedColorMap && filteredHierarchy) {
    const displayTopics = Object.keys(filteredHierarchy);
    const hasNoMatches = debouncedFilter.trim() !== '' && displayTopics.length === 0;

    return (
      <Card
        className={`w-fit py-2 ${className ?? ''}`}
        variant="noBg"
      >
        <div className="px-4 pb-0.5 pointer-events-auto">{filterRow}</div>
        <ScrollArea className="overflow-y-auto pointer-events-auto" style={{ maskImage: 'linear-gradient(transparent, black 12px, black calc(100% - 12px), transparent)', WebkitMaskImage: 'linear-gradient(transparent, black 12px, black calc(100% - 12px), transparent)' }}>
        <CardContent className="space-y-0.5" style={{ maxHeight: maxHeight ?? 320 }}>
          {hasNoMatches ? (
            <div className="text-sm text-muted-foreground/60 italic px-1 py-2">No matches</div>
          ) : displayTopics.map((topic) => {
            const isTopicMuted = mutedCategories.includes(topic);
            const topicCount = nestedColorMap.topicCounts[topic];
            const subtopics = filteredHierarchy[topic];
            const isClickable = !!onCategoryToggle;

            return (
              <div key={topic}>
                {/* Topic header row */}
                <div
                  className={cn(
                    "flex items-center gap-2 py-1 px-1 rounded-md transition-all",
                    isClickable && "cursor-pointer hover:bg-accent/50",
                    isTopicMuted && "opacity-40"
                  )}
                  onClick={(e) => onCategoryToggle?.(topic, e.shiftKey)}
                  onDoubleClick={() => onCategoryReset?.()}
                  role={isClickable ? "button" : undefined}
                  tabIndex={isClickable ? 0 : undefined}
                  onKeyDown={(e) => {
                    if (isClickable && (e.key === 'Enter' || e.key === ' ')) {
                      e.preventDefault();
                      onCategoryToggle?.(topic, e.shiftKey);
                    }
                  }}
                >
                  <span
                    className="h-3 w-3 rounded-sm shrink-0 transition-colors"
                    style={{
                      backgroundColor: isTopicMuted ? '#9ca3af' : (nestedColorMap.topicColors[topic] || '#7f7f7f'),
                    }}
                    aria-hidden="true"
                  />
                  <span
                    className={cn(
                      "text-sm font-semibold flex-1 max-w-44 truncate text-foreground/70",
                      isTopicMuted && "line-through"
                    )}
                  >
                    {topic}
                  </span>
                  {topicCount !== undefined && (
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {formatCount(topicCount)}
                    </span>
                  )}
                </div>

                {/* Subtopic rows (hidden/collapsed when topic is muted) */}
                {!isTopicMuted && subtopics.map((sub) => {
                  const isSubMuted = mutedCategories.includes(sub);
                  const subCount = nestedColorMap.subtopicCounts[sub];

                  return (
                    <div
                      className={cn(
                        "flex items-center gap-2 py-0.5 px-1 pl-5 rounded-md transition-all",
                        isClickable && "cursor-pointer hover:bg-accent/50",
                        isSubMuted && "opacity-40"
                      )}
                      key={sub}
                      onClick={(e) => {
                        e.stopPropagation();
                        onCategoryToggle?.(sub, e.shiftKey);
                      }}
                      onDoubleClick={(e) => {
                        e.stopPropagation();
                        onCategoryReset?.();
                      }}
                      role={isClickable ? "button" : undefined}
                      tabIndex={isClickable ? 0 : undefined}
                      onKeyDown={(e) => {
                        if (isClickable && (e.key === 'Enter' || e.key === ' ')) {
                          e.preventDefault();
                          onCategoryToggle?.(sub, e.shiftKey);
                        }
                      }}
                    >
                      <span
                        className="h-2.5 w-2.5 rounded-full shrink-0 transition-colors"
                        style={{
                          backgroundColor: isSubMuted ? '#9ca3af' : (nestedColorMap.subtopicColors[sub] || '#7f7f7f'),
                        }}
                        aria-hidden="true"
                      />
                      <span
                        className={cn(
                          "text-xs flex-1 max-w-40 truncate text-foreground/60",
                          isSubMuted && "line-through"
                        )}
                      >
                        {sub}
                      </span>
                      {subCount !== undefined && (
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {formatCount(subCount)}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </CardContent>
        <ScrollBar className="px-0" orientation="vertical" />
        </ScrollArea>
        {dragHandle}
      </Card>
    );
  }

  // ---- FLAT (standard) LEGEND ----
  const hasNoFlatMatches = debouncedFilter.trim() !== '' && filteredValues.length === 0;

  return (
    <Card
      className={`w-fit py-2 ${className ?? ''}`}
      variant="noBg"
    >
      <div className="px-4 pb-0.5 pointer-events-auto">{filterRow}</div>
      <ScrollArea className="overflow-y-auto pointer-events-auto" style={{ maskImage: 'linear-gradient(transparent, black 12px, black calc(100% - 12px), transparent)', WebkitMaskImage: 'linear-gradient(transparent, black 12px, black calc(100% - 12px), transparent)' }}>
      <CardContent className="space-y-1" style={{ maxHeight: maxHeight ?? 256 }}>
        {hasNoFlatMatches ? (
          <div className="text-sm text-muted-foreground/60 italic px-1 py-2">No matches</div>
        ) : filteredValues.map((value) => {
          const isMuted = mutedCategories.includes(value);
          const count = categoryCounts?.[value];
          const isClickable = !!onCategoryToggle;

          return (
            <div
              className={cn(
                "flex items-center gap-2 py-1 px-1 rounded-md transition-all",
                isClickable && "cursor-pointer hover:bg-accent/50",
                isMuted && "opacity-40"
              )}
              key={value}
              onClick={(e) => onCategoryToggle?.(value, e.shiftKey)}
              onDoubleClick={() => onCategoryReset?.()}
              role={isClickable ? "button" : undefined}
              tabIndex={isClickable ? 0 : undefined}
              onKeyDown={(e) => {
                if (isClickable && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault();
                  onCategoryToggle?.(value, e.shiftKey);
                }
              }}
            >
              <span
                className="h-3 w-3 rounded-full shrink-0 transition-colors"
                style={{
                  backgroundColor: isMuted ? '#9ca3af' : (colorMap[value] || '#7f7f7f'),
                }}
                aria-hidden="true"
              />
              <span
                className={cn(
                  "text-sm flex-1 max-w-48 truncate text-foreground/70",
                  isMuted && "line-through"
                )}
              >
                {getCategoryLabel(categoryField ?? null, value)}
              </span>
              {count !== undefined && (
                <span className="text-xs text-muted-foreground tabular-nums">
                  {formatCount(count)}
                </span>
              )}
            </div>
          );
        })}
      </CardContent>
      <ScrollBar className="px-0" orientation="vertical" />
      </ScrollArea>
      {dragHandle}
    </Card>
  );
}
