'use client';

import { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts';
import type { SaeActivation } from '@/lib/types/types';
import { computeHistogramBins } from '../utils/histogramUtils';

interface ActivationHistogramProps {
  activations: SaeActivation[];
  hoveredValue?: number | null;
}

export function ActivationHistogram({ activations, hoveredValue }: ActivationHistogramProps) {
  const bins = useMemo(() => {
    if (activations.length === 0) return [];
    const allValues: number[] = [];
    for (const act of activations) {
      for (const v of act.values) {
        if (v > 0) allValues.push(v);
      }
    }
    return computeHistogramBins(allValues, 25);
  }, [activations]);

  if (bins.length === 0) {
    return <p className="text-xs text-muted-foreground text-center py-4">No activation data</p>;
  }

  return (
    <div>
      <h4 className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">
        Activation Distribution
      </h4>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={bins} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <XAxis
            dataKey="label"
            tick={{ fontSize: 9 }}
            interval="preserveStartEnd"
            tickLine={false}
            axisLine={false}
          />
          <YAxis tick={{ fontSize: 9 }} tickLine={false} axisLine={false} width={30} />
          <Tooltip
            contentStyle={{ fontSize: 11, borderRadius: 6 }}
            formatter={(value: number) => [value.toLocaleString(), 'Count']}
            labelFormatter={(label: string, payload) => {
              if (payload?.[0]?.payload) {
                const b = payload[0].payload;
                return `${b.binStart.toFixed(2)} – ${b.binEnd.toFixed(2)}`;
              }
              return label;
            }}
          />
          <Bar dataKey="count" fill="hsl(25, 90%, 55%)" radius={[2, 2, 0, 0]} />
          {hoveredValue != null && (
            <ReferenceLine
              x={findClosestBinLabel(bins, hoveredValue)}
              stroke="hsl(0, 0%, 50%)"
              strokeDasharray="4 2"
              strokeWidth={1.5}
              label={{ value: hoveredValue.toFixed(3), position: 'top', fontSize: 9 }}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function findClosestBinLabel(bins: { binStart: number; binEnd: number; label: string }[], value: number): string {
  let closest = bins[0];
  let minDist = Infinity;
  for (const b of bins) {
    const mid = (b.binStart + b.binEnd) / 2;
    const dist = Math.abs(mid - value);
    if (dist < minDist) { minDist = dist; closest = b; }
  }
  return closest.label;
}
