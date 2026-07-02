'use client';

import type { SaeLogitEntry } from '@/lib/types/types';
import { cn } from '@/lib/utils/utils';

interface LogitBarChartProps {
  entries: SaeLogitEntry[];
  variant: 'positive' | 'negative';
  maxBars?: number;
}

/**
 * Horizontal bar chart for top/bottom logit entries.
 * Positive logits in blue, negative logits in red/orange.
 */
export function LogitBarChart({ entries, variant, maxBars = 10 }: LogitBarChartProps) {
  const visible = entries.slice(0, maxBars);
  if (visible.length === 0) return <p className="text-xs text-muted-foreground">No logits</p>;

  const maxScore = Math.max(...visible.map((e) => Math.abs(e.score)));

  const barColor = variant === 'positive'
    ? 'bg-blue-500 dark:bg-blue-400'
    : 'bg-orange-500 dark:bg-orange-400';

  return (
    <div className="space-y-1">
      {visible.map((entry, i) => {
        const width = maxScore > 0 ? (Math.abs(entry.score) / maxScore) * 100 : 0;
        return (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span className="w-24 truncate text-right font-mono text-muted-foreground" title={entry.token}>
              {entry.token}
            </span>
            <div className="flex-1 h-4 bg-muted/30 rounded-sm overflow-hidden">
              <div
                className={cn('h-full rounded-sm transition-all', barColor)}
                style={{ width: `${width}%` }}
              />
            </div>
            <span className="w-12 text-right font-mono text-muted-foreground tabular-nums">
              {entry.score.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
