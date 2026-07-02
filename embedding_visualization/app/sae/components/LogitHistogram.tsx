'use client';

import { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts';
import type { SaeLogitEntry } from '@/lib/types/types';
import { computeHistogramBins } from '../utils/histogramUtils';

interface LogitHistogramProps {
  topLogits: SaeLogitEntry[] | null;
  bottomLogits: SaeLogitEntry[] | null;
}

export function LogitHistogram({ topLogits, bottomLogits }: LogitHistogramProps) {
  const bins = useMemo(() => {
    const scores: number[] = [];
    if (topLogits) scores.push(...topLogits.map((e) => e.score));
    if (bottomLogits) scores.push(...bottomLogits.map((e) => -e.score));
    if (scores.length === 0) return [];
    return computeHistogramBins(scores, 15);
  }, [topLogits, bottomLogits]);

  if (bins.length === 0) {
    return <p className="text-xs text-muted-foreground text-center py-4">No logit data</p>;
  }

  return (
    <div>
      <h4 className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">
        Logit Score Distribution
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
          <Bar dataKey="count" radius={[2, 2, 0, 0]}>
            {bins.map((bin, i) => {
              const mid = (bin.binStart + bin.binEnd) / 2;
              return (
                <Cell
                  key={i}
                  fill={mid >= 0 ? 'hsl(210, 70%, 55%)' : 'hsl(25, 80%, 55%)'}
                />
              );
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
