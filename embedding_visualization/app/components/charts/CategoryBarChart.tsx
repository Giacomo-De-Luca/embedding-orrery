'use client';

import { useMemo, useState, useEffect } from 'react';
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
import { Input } from '@/lib/ui-primitives/input';
import { buildCategoryColorMap, getCategoryLabel, getCategoryDisplayName } from '@/lib/utils/categoryColors';
import type { ColorFieldOption } from '@/lib/utils/fieldAnalysis';

const MAX_BARS = 15;
const FOLLOW_COLOR_FIELD = '__follow__';

const sanitizeKey = (value: string) => value.replace(/[^a-zA-Z0-9_-]/g, '_');

interface CategoryBarChartProps {
  categoryField: string | null;
  categoryValues: string[];
  categoryCounts: Record<string, number>;
  categoricalPalette?: string;
  searchMatchCounts?: Record<string, number> | null;
  colorFieldOptions?: ColorFieldOption[];
  analysisField?: string | null;
  onAnalysisFieldChange?: (field: string | null) => void;
}

export function CategoryBarChart({
  categoryField,
  categoryValues,
  categoryCounts,
  categoricalPalette,
  searchMatchCounts,
  colorFieldOptions,
  analysisField,
  onAnalysisFieldChange,
}: CategoryBarChartProps) {
  const [categoryFilter, setCategoryFilter] = useState('');

  // Reset filter when field changes
  useEffect(() => {
    setCategoryFilter('');
  }, [categoryField]);

  const colorMap = useMemo(
    () => buildCategoryColorMap(categoryField, categoryValues, categoricalPalette),
    [categoryField, categoryValues, categoricalPalette]
  );

  const isSearchActive = useMemo(
    () => searchMatchCounts != null && Object.keys(searchMatchCounts).length > 0,
    [searchMatchCounts]
  );

  // Apply category name filter before sorting/slicing
  const filteredByName = useMemo(() => {
    if (!categoryFilter.trim()) return categoryValues;
    const q = categoryFilter.toLowerCase();
    return categoryValues.filter(cat => {
      const label = getCategoryLabel(categoryField, cat);
      return String(label).toLowerCase().includes(q);
    });
  }, [categoryValues, categoryFilter, categoryField]);

  // Sort categories: by search match count (descending) when search is active, else by total count
  const sortedCategories = useMemo(() => {
    return [...filteredByName].sort((a, b) => {
      if (isSearchActive) {
        return (searchMatchCounts![b] ?? 0) - (searchMatchCounts![a] ?? 0);
      }
      return (categoryCounts[b] ?? 0) - (categoryCounts[a] ?? 0);
    });
  }, [filteredByName, categoryCounts, isSearchActive, searchMatchCounts]);

  const displayCategories = useMemo(
    () => sortedCategories.slice(0, MAX_BARS),
    [sortedCategories]
  );
  const hiddenCount = sortedCategories.length - displayCategories.length;

  const chartData = useMemo(() => {
    return displayCategories.map(cat => ({
      category: cat,
      safeKey: sanitizeKey(cat),
      count: categoryCounts[cat] ?? 0,
      matches: isSearchActive ? (searchMatchCounts![cat] ?? 0) : 0,
      fill: `var(--color-${sanitizeKey(cat)})`,
    }));
  }, [displayCategories, categoryCounts, isSearchActive, searchMatchCounts]);

  const chartConfig = useMemo(() => {
    const config: ChartConfig = {
      count: { label: 'Total' },
      matches: { label: 'Matches' },
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

  // Compute search summary stats
  const totalMatches = isSearchActive
    ? Object.values(searchMatchCounts!).reduce((s, c) => s + c, 0)
    : 0;
  const matchedCategoryCount = isSearchActive
    ? Object.keys(searchMatchCounts!).filter(k => searchMatchCounts![k] > 0).length
    : 0;

  // Categorical field options for the dropdown
  const categoricalOptions = useMemo(() => {
    if (!colorFieldOptions) return [];
    return colorFieldOptions.filter(o => o.recommendedScale === 'categorical');
  }, [colorFieldOptions]);

  const showFieldSelector = categoricalOptions.length > 0 && onAnalysisFieldChange;
  const showCategoryFilter = categoryValues.length > MAX_BARS;

  // Show field selector even with no data (so user can pick a field)
  if (displayCategories.length === 0 && !showFieldSelector) return null;

  return (
    <Card className="border-0 shadow-none bg-transparent">
      <CardHeader className="px-0 pt-0 pb-2 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm shrink-0">
            {displayCategories.length > 0 ? `${displayName} Distribution` : 'Distribution'}
          </CardTitle>
        </div>
        {showFieldSelector && (
          <Select
            value={analysisField ?? FOLLOW_COLOR_FIELD}
            onValueChange={(val) => onAnalysisFieldChange!(val === FOLLOW_COLOR_FIELD ? null : val)}
          >
            <SelectTrigger size="sm" className="w-full text-xs h-7">
              <SelectValue placeholder="Select field..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={FOLLOW_COLOR_FIELD}>
                <span className="text-xs text-muted-foreground">Follow color field</span>
              </SelectItem>
              <SelectSeparator />
              {categoricalOptions.map(opt => (
                <SelectItem key={opt.field} value={opt.field}>
                  <span className="text-xs">{opt.displayName} ({opt.uniqueCount})</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        {displayCategories.length > 0 && (
          <CardDescription className="text-xs">
            {isSearchActive
              ? `${totalMatches.toLocaleString()} matches across ${matchedCategoryCount} categories`
              : `${totalPoints.toLocaleString()} points across ${sortedCategories.length} categories`
            }
          </CardDescription>
        )}
      </CardHeader>

      {showCategoryFilter && (
        <div className="px-0 pb-2">
          <Input
            placeholder="Filter categories..."
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="h-7 text-xs"
          />
        </div>
      )}

      {displayCategories.length > 0 && (
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
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const data = payload[0]?.payload;
                  if (!data) return null;
                  const label = String(chartConfig[data.safeKey]?.label ?? data.category);
                  return (
                    <div className="rounded-lg border bg-background p-2 shadow-sm">
                      <p className="text-xs font-medium mb-1">{label}</p>
                      {isSearchActive ? (
                        <>
                          <p className="text-xs text-muted-foreground">Matches: {(data.matches ?? 0).toLocaleString()}</p>
                          <p className="text-xs text-muted-foreground">Total: {(data.count ?? 0).toLocaleString()}</p>
                        </>
                      ) : (
                        <p className="text-xs text-muted-foreground">Points: {(data.count ?? 0).toLocaleString()}</p>
                      )}
                    </div>
                  );
                }}
              />
              <Bar
                dataKey={isSearchActive ? 'matches' : 'count'}
                strokeWidth={2}
                radius={4}
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
      )}
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
