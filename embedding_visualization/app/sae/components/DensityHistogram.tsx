'use client';

import { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { computeLogHistogramBins, findBinIndex } from '../utils/histogramUtils';

interface DensityHistogramProps {
  allDensities: number[];
  currentDensity: number | null;
  loading?: boolean;
}

export function DensityHistogram({ allDensities, currentDensity, loading }: DensityHistogramProps) {
  const bins = useMemo(() => {
    if (allDensities.length === 0) return [];
    return computeLogHistogramBins(allDensities, 20);
  }, [allDensities]);

  const highlightIdx = useMemo(() => {
    if (currentDensity == null || bins.length === 0) return -1;
    return findBinIndex(bins, currentDensity);
  }, [bins, currentDensity]);

  if (loading) {
    return (
      <div className="flex justify-center py-8">
        <Spinner className="h-4 w-4" />
      </div>
    );
  }

  if (bins.length === 0) {
    return <p className="text-xs text-muted-foreground text-center py-4">No density data</p>;
  }

  return (
    <div>
      <h4 className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">
        Density Distribution (log scale)
      </h4>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={bins} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <XAxis
            dataKey="label"
            tick={{ fontSize: 8 }}
            interval="preserveStartEnd"
            tickLine={false}
            axisLine={false}
            angle={-30}
            textAnchor="end"
            height={35}
          />
          <YAxis tick={{ fontSize: 9 }} tickLine={false} axisLine={false} width={30} />
          <Tooltip
            contentStyle={{ fontSize: 11, borderRadius: 6 }}
            formatter={(value: number) => [value.toLocaleString(), 'Features']}
            labelFormatter={(label: string, payload) => {
              if (payload?.[0]?.payload) {
                const b = payload[0].payload;
                return `${b.binStart.toExponential(1)} – ${b.binEnd.toExponential(1)}`;
              }
              return label;
            }}
          />
          <Bar dataKey="count" radius={[2, 2, 0, 0]}>
            {bins.map((_, i) => (
              <Cell
                key={i}
                fill={i === highlightIdx ? 'hsl(340, 70%, 55%)' : 'hsl(210, 15%, 60%)'}
                stroke={i === highlightIdx ? 'hsl(340, 70%, 40%)' : 'none'}
                strokeWidth={i === highlightIdx ? 2 : 0}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {currentDensity != null && (
        <p className="text-[10px] text-muted-foreground text-center mt-1">
          Current feature: {currentDensity < 0.001 ? currentDensity.toExponential(2) : currentDensity.toFixed(4)}
        </p>
      )}
    </div>
  );
}
