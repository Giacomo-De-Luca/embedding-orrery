'use client';

import { useMemo } from 'react';
import { Bar, BarChart, CartesianGrid, Rectangle, XAxis } from 'recharts';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/lib/ui-primitives/card';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from '@/lib/ui-primitives/chart';
import { buildCategoryColorMap, getCategoryLabel, getCategoryDisplayName } from '@/lib/utils/categoryColors';

const MAX_BARS = 15;

interface CategoryBarChartProps {
  categoryField: string | null;
  categoryValues: string[];
  categoryCounts: Record<string, number>;
  categoricalPalette?: string;
}

export function CategoryBarChart({
  categoryField,
  categoryValues,
  categoryCounts,
  categoricalPalette,
}: CategoryBarChartProps) {
  const colorMap = useMemo(
    () => buildCategoryColorMap(categoryField, categoryValues, categoricalPalette),
    [categoryField, categoryValues, categoricalPalette]
  );

  // Sort categories descending by count, truncate to MAX_BARS
  const sortedCategories = useMemo(() => {
    return [...categoryValues]
      .sort((a, b) => (categoryCounts[b] ?? 0) - (categoryCounts[a] ?? 0));
  }, [categoryValues, categoryCounts]);

  const displayCategories = sortedCategories.slice(0, MAX_BARS);
  const hiddenCount = sortedCategories.length - displayCategories.length;

  // Sanitize category keys for CSS variable compatibility
  // ChartConfig keys become CSS variable names (--color-KEY), so we need safe keys
  const sanitizeKey = (value: string) => value.replace(/[^a-zA-Z0-9_-]/g, '_');

  const chartData = useMemo(() => {
    return displayCategories.map(cat => ({
      category: cat,
      safeKey: sanitizeKey(cat),
      count: categoryCounts[cat] ?? 0,
      fill: `var(--color-${sanitizeKey(cat)})`,
    }));
  }, [displayCategories, categoryCounts]);

  const chartConfig = useMemo(() => {
    const config: ChartConfig = {
      count: { label: 'Points' },
    };
    for (const cat of displayCategories) {
      const safeKey = sanitizeKey(cat);
      config[safeKey] = {
        label: getCategoryLabel(categoryField, cat),
        color: colorMap[cat] ?? '#7f7f7f',
      };
    }
    return config;
  }, [displayCategories, categoryField, colorMap]);

  const totalPoints = Object.values(categoryCounts).reduce((s, c) => s + c, 0);
  const displayName = getCategoryDisplayName(categoryField);

  if (displayCategories.length === 0) return null;

  return (
    <Card className="border-0 shadow-none bg-transparent">
      <CardHeader className="px-0 pt-0 pb-2">
        <CardTitle className="text-sm">{displayName} Distribution</CardTitle>
        <CardDescription className="text-xs">
          {totalPoints.toLocaleString()} points across {sortedCategories.length} categories
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0 pb-0">
        <ChartContainer config={chartConfig} className="aspect-[4/3] w-full">
          <BarChart accessibilityLayer data={chartData}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="safeKey"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              tickFormatter={(value) => {
                const label = chartConfig[value]?.label;
                if (!label) return value;
                const str = String(label);
                return str.length > 10 ? str.slice(0, 9) + '\u2026' : str;
              }}
            />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Bar
              dataKey="count"
              strokeWidth={2}
              radius={8}
              activeBar={({ ...props }) => (
                <Rectangle
                  {...props}
                  fillOpacity={0.8}
                  stroke={props.payload.fill}
                  strokeDasharray={4}
                  strokeDashoffset={4}
                />
              )}
            />
          </BarChart>
        </ChartContainer>
      </CardContent>
      {hiddenCount > 0 && (
        <CardFooter className="px-0 pt-2 pb-0">
          <p className="text-xs text-muted-foreground">
            + {hiddenCount} more {hiddenCount === 1 ? 'category' : 'categories'}
          </p>
        </CardFooter>
      )}
    </Card>
  );
}
