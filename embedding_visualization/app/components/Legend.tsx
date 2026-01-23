'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import {
  buildCategoryColorMap,
  getCategoryLabel,
  getCategoryDisplayName,
} from '../../lib/utils/categoryColors';
import { cn } from '@/lib/utils/utils';
import { ScrollArea, ScrollBar } from '@/lib/ui-primitives/scroll-area';

interface LegendProps {
  className?: string;
  categoryField?: string | null;
  categoryValues?: string[];
  categoryCounts?: Record<string, number>;
  mutedCategories?: string[];
  onCategoryToggle?: (category: string) => void;
}

/**
 * Format a number with thousand separators for display.
 */
function formatCount(count: number): string {
  return count.toLocaleString();
}

/**
 * Dynamic legend component that displays category colors with point counts.
 * Click on a category to toggle its visibility (mute/unmute).
 */
export function Legend({
  className,
  categoryField,
  categoryValues,
  categoryCounts,
  mutedCategories = [],
  onCategoryToggle,
}: LegendProps) {
  // Default to POS legend if no category info provided
  const isPosLegend = !categoryField || categoryField === 'pos';
  const values = categoryValues || (isPosLegend ? ['n', 'v', 'a', 'r', 's', 'unknown'] : []);

  const colorMap = buildCategoryColorMap(categoryField ?? 'pos', values);

  return (
    <Card
      className={`w-fit ${className ?? ''}`}
      variant="outline"
    >
      <CardHeader className="">
        <CardTitle className="font-mono">{getCategoryDisplayName(categoryField ?? 'pos')}</CardTitle>
      </CardHeader>
        
        
      <ScrollArea className="overflow-y-auto pointer-events-auto">
      <CardContent className="space-y-1 max-h-64">
        {values.map((value) => {
          const isMuted = mutedCategories.includes(value);
          const count = categoryCounts?.[value];
          const isClickable = !!onCategoryToggle;

          return (
            <div
              className={cn(
                "flex items-center gap-12 py-1 px-2 -mx-2 rounded-md transition-all",
                isClickable && "cursor-pointer hover:bg-accent/50",
                isMuted && "opacity-40"
              )}
              key={value}
              onClick={() => onCategoryToggle?.(value)}
              role={isClickable ? "button" : undefined}
              tabIndex={isClickable ? 0 : undefined}
              onKeyDown={(e) => {
                if (isClickable && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault();
                  onCategoryToggle?.(value);
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
                  "text-md flex-1 truncate",
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
      <ScrollBar orientation="vertical" />
      </ScrollArea>
    </Card>
  );
}
