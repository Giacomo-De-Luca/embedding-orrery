'use client';

import { useMemo } from 'react';
import { Area, AreaChart, CartesianGrid, XAxis } from 'recharts';
import {
  Card,
  CardContent,
  CardDescription,
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
import { fieldToDisplayName } from '@/lib/utils/fieldAnalysis';
import type { TemporalCrossTabRow } from '@/lib/utils/temporalAnalysis';

/** Max categories to show as stacked areas (top N by total count) */
const MAX_SERIES = 8;

interface TemporalChartProps {
  categoryField: string | null;
  categoryValues: string[];
  categoryCounts: Record<string, number>;
  temporalField: string;
  crossTabData: TemporalCrossTabRow[];
  categoricalPalette?: string;
}

export function TemporalChart({
  categoryField,
  categoryValues,
  categoryCounts,
  temporalField,
  crossTabData,
  categoricalPalette,
}: TemporalChartProps) {
  const colorMap = useMemo(
    () => buildCategoryColorMap(categoryField, categoryValues, categoricalPalette),
    [categoryField, categoryValues, categoricalPalette]
  );

  // Pick top N categories by count for the stacked areas
  const topCategories = useMemo(() => {
    return [...categoryValues]
      .sort((a, b) => (categoryCounts[b] ?? 0) - (categoryCounts[a] ?? 0))
      .slice(0, MAX_SERIES);
  }, [categoryValues, categoryCounts]);

  const sanitizeKey = (value: string) => value.replace(/[^a-zA-Z0-9_-]/g, '_');

  const chartConfig = useMemo(() => {
    const config: ChartConfig = {};
    for (const cat of topCategories) {
      const safeKey = sanitizeKey(cat);
      config[safeKey] = {
        label: getCategoryLabel(categoryField, cat),
        color: colorMap[cat] ?? '#7f7f7f',
      };
    }
    return config;
  }, [topCategories, categoryField, colorMap]);

  // Remap crossTabData keys to safe keys for recharts
  const safeData = useMemo(() => {
    return crossTabData.map(row => {
      const safeRow: Record<string, string | number> = { period: row.period };
      for (const cat of topCategories) {
        safeRow[sanitizeKey(cat)] = (row[cat] as number) ?? 0;
      }
      return safeRow;
    });
  }, [crossTabData, topCategories]);

  const displayName = getCategoryDisplayName(categoryField);
  const temporalDisplayName = fieldToDisplayName(temporalField);

  if (crossTabData.length < 2 || topCategories.length === 0) return null;

  return (
    <Card className="border-0 shadow-none bg-transparent">
      <CardHeader className="px-0 pt-0 pb-2">
        <CardTitle className="text-sm">{displayName} over {temporalDisplayName}</CardTitle>
        <CardDescription className="text-xs">
          {crossTabData.length} periods, top {topCategories.length} categories
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0 pb-0">
        <ChartContainer config={chartConfig} className="aspect-[4/3] w-full">
          <AreaChart
            accessibilityLayer
            data={safeData}
            margin={{ left: 12, right: 12 }}
          >
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="period"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={(value) => {
                const str = String(value);
                return str.length > 6 ? str.slice(0, 6) : str;
              }}
            />
            <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
            <defs>
              {topCategories.map(cat => {
                const safeKey = sanitizeKey(cat);
                return (
                  <linearGradient key={safeKey} id={`fill-${safeKey}`} x1="0" y1="0" x2="0" y2="1">
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
            {topCategories.map(cat => {
              const safeKey = sanitizeKey(cat);
              return (
                <Area
                  key={safeKey}
                  dataKey={safeKey}
                  type="natural"
                  fill={`url(#fill-${safeKey})`}
                  fillOpacity={0.4}
                  stroke={`var(--color-${safeKey})`}
                  stackId="a"
                />
              );
            })}
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
