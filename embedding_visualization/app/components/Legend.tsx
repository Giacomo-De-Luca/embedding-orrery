'use client';

import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import {
  buildCategoryColorMap,
  getCategoryLabel,
  getCategoryDisplayName,
  getSequentialScale,
  getDivergingScale,
  getMonochromeScale,
  generateGradientCSS,
  type SequentialScaleName,
  type DivergingScaleName,
} from '../../lib/utils/categoryColors';
import { isCrameriScale, crameriGradientCSS } from '../../lib/colorMaps/crameriScales';
import { cn } from '@/lib/utils/utils';
import { ScrollArea, ScrollBar } from '@/lib/ui-primitives/scroll-area';
import type { NestedColorMap } from '../../lib/types/types';

interface LegendProps {
  className?: string;
  categoryField?: string | null;
  categoryValues?: string[];
  categoryCounts?: Record<string, number>;
  mutedCategories?: string[];
  onCategoryToggle?: (category: string, shiftKey: boolean) => void;
  onCategoryReset?: () => void;
  colorScaleType?: 'categorical' | 'sequential' | 'diverging' | 'monochrome';
  numericRange?: { min: number; max: number };
  sequentialScaleName?: SequentialScaleName;
  divergingScaleName?: DivergingScaleName;
  monochromeColor?: string;
  categoricalPalette?: string;
  nestedColorMap?: NestedColorMap | null;
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
  colorScaleType = 'categorical',
  numericRange,
  sequentialScaleName = 'sinebow',
  divergingScaleName = 'blueGold',
  monochromeColor = '#1f77b4',
  categoricalPalette,
  nestedColorMap,
}: LegendProps) {
  // Check if this is a continuous scale (sequential, diverging, or monochrome)
  const isContinuous = colorScaleType === 'sequential' || colorScaleType === 'diverging' || colorScaleType === 'monochrome';

  // Generate gradient dynamically from the actual scale function
  const gradient = useMemo(() => {
    if (colorScaleType === 'sequential') {
      // Try Crameri gradient first (from cache, returns null if not loaded)
      if (isCrameriScale(sequentialScaleName)) {
        const crameri = crameriGradientCSS(sequentialScaleName, 20);
        if (crameri) return crameri;
      }
      return generateGradientCSS(getSequentialScale([0, 1], sequentialScaleName));
    } else if (colorScaleType === 'diverging') {
      if (isCrameriScale(divergingScaleName)) {
        const crameri = crameriGradientCSS(divergingScaleName, 20);
        if (crameri) return crameri;
      }
      return generateGradientCSS(getDivergingScale([0, 0.5, 1], divergingScaleName));
    } else if (colorScaleType === 'monochrome') {
      return generateGradientCSS(getMonochromeScale(monochromeColor, [0, 1]));
    }
    return '';
  }, [colorScaleType, sequentialScaleName, divergingScaleName, monochromeColor]);

  // For continuous scales, render a gradient bar
  if (isContinuous && numericRange) {
    const { min, max } = numericRange;
    const center = (min + max) / 2;

    return (
      <Card
        className={`w-fit gap-2 min-w-48 ${className ?? ''}`}
        variant="outline"
      >
        <CardHeader className="">
          <CardTitle className="font-mono text-xs">{getCategoryDisplayName(categoryField ?? 'value')}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {/* Gradient bar */}
          <div
            className="h-2 w-full rounded"
            style={{ background: gradient }}
            aria-label={`Color scale from ${min} to ${max}`}
          />
          {/* Labels: min, center, max */}
          <div className="flex justify-between text-xs text-muted-foreground tabular-nums">
            <span>{formatNumericValue(min)}</span>
            <span>{formatNumericValue(center)}</span>
            <span>{formatNumericValue(max)}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Default to POS legend if no category info provided
  const isPosLegend = !categoryField || categoryField === 'pos';
  const values = categoryValues || (isPosLegend ? ['n', 'v', 'a', 'r', 's', 'unknown'] : []);

  const colorMap = buildCategoryColorMap(categoryField ?? 'pos', values, categoricalPalette);

  // ---- NESTED (two-level) LEGEND ----
  if (nestedColorMap) {
    const topics = Object.keys(nestedColorMap.hierarchy);

    return (
      <Card
        className={`w-fit py-2 ${className ?? ''}`}
        variant="noBg"
      >
        <ScrollArea className="overflow-y-auto pointer-events-auto" style={{ maskImage: 'linear-gradient(transparent, black 12px, black calc(100% - 12px), transparent)', WebkitMaskImage: 'linear-gradient(transparent, black 12px, black calc(100% - 12px), transparent)' }}>
        <CardContent className="space-y-0.5 max-h-80">
          {topics.map((topic) => {
            const isTopicMuted = mutedCategories.includes(topic);
            const topicCount = nestedColorMap.topicCounts[topic];
            const subtopics = nestedColorMap.hierarchy[topic];
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
                      "text-sm font-semibold flex-1 max-w-44 truncate",
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
                          "text-xs flex-1 max-w-40 truncate",
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
      </Card>
    );
  }

  // ---- FLAT (standard) LEGEND ----
  return (
    <Card
      className={`w-fit py-2 ${className ?? ''}`}
      variant="noBg"
    >
      {/*<CardHeader className="">
        <CardTitle className="font-mono text-md">{getCategoryDisplayName(categoryField ?? 'pos')}</CardTitle>
      </CardHeader> */}


      <ScrollArea className="overflow-y-auto pointer-events-auto" style={{ maskImage: 'linear-gradient(transparent, black 12px, black calc(100% - 12px), transparent)', WebkitMaskImage: 'linear-gradient(transparent, black 12px, black calc(100% - 12px), transparent)' }}>
      <CardContent className="space-y-1 max-h-64">
        {values.map((value) => {
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
                  "text-sm flex-1 max-w-48 truncate",
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
    </Card>
  );
}
