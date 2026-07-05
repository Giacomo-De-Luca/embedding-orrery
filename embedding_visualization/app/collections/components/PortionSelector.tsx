'use client';

import { useEffect } from 'react';
import { Label } from '@/lib/ui-primitives/label';
import { Input } from '@/lib/ui-primitives/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import type { PortionStrategy } from '@/lib/graphql/mutations';

const ALL_STRATEGIES: PortionStrategy[] = ['ALL', 'FIRST_N', 'RANDOM_SAMPLE', 'ROW_RANGE'];

interface PortionSelectorProps {
  strategy: PortionStrategy;
  onStrategyChange: (strategy: PortionStrategy) => void;
  n: number;
  onNChange: (n: number) => void;
  start?: number;
  onStartChange?: (start: number) => void;
  end?: number;
  onEndChange?: (end: number) => void;
  seed: number;
  onSeedChange: (seed: number) => void;
  totalRows: number | null;
  availableSplits?: string[];
  /**
   * Which strategies to offer (default: all). Backends without row-range
   * support (embedLocalFile) pass a subset instead of rendering an inert
   * option.
   */
  allowedStrategies?: PortionStrategy[];
}

export function PortionSelector({
  strategy,
  onStrategyChange,
  n,
  onNChange,
  start = 0,
  onStartChange,
  end = 1000,
  onEndChange,
  seed,
  onSeedChange,
  totalRows,
  availableSplits,
  allowedStrategies = ALL_STRATEGIES,
}: PortionSelectorProps) {
  const showWarning = strategy === 'ALL' && totalRows && totalRows > 10000;

  // If the current strategy is not offered (e.g. after a prop change), fall
  // back to FIRST_N so the form never holds an unsubmittable value.
  useEffect(() => {
    if (!allowedStrategies.includes(strategy)) {
      onStrategyChange('FIRST_N');
    }
  }, [allowedStrategies, strategy, onStrategyChange]);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="portion-strategy">Dataset Portion</Label>
        <Select value={strategy} onValueChange={(v) => onStrategyChange(v as PortionStrategy)}>
          <SelectTrigger id="portion-strategy">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {allowedStrategies.includes('ALL') && (
              <SelectItem value="ALL">
                All Rows
                {availableSplits && availableSplits.length > 1 && ' (All Splits)'}
              </SelectItem>
            )}
            {allowedStrategies.includes('FIRST_N') && (
              <SelectItem value="FIRST_N">First N Rows</SelectItem>
            )}
            {allowedStrategies.includes('RANDOM_SAMPLE') && (
              <SelectItem value="RANDOM_SAMPLE">Random Sample</SelectItem>
            )}
            {allowedStrategies.includes('ROW_RANGE') && (
              <SelectItem value="ROW_RANGE">Row Range</SelectItem>
            )}
          </SelectContent>
        </Select>
        {strategy === 'ALL' && availableSplits && availableSplits.length > 1 && (
          <p className="text-xs text-muted-foreground">
            Will embed all splits ({availableSplits.join(', ')}) into a single collection with 'source_split' metadata column
          </p>
        )}
      </div>

      {strategy === 'FIRST_N' && (
        <div className="space-y-2">
          <Label htmlFor="num-rows">Number of Rows</Label>
          <Input
            id="num-rows"
            type="number"
            value={n}
            onChange={(e) => onNChange(parseInt(e.target.value) || 0)}
            min={1}
            max={totalRows || undefined}
          />
          {totalRows && (
            <p className="text-xs text-muted-foreground">
              Total available: {totalRows.toLocaleString()} rows
            </p>
          )}
        </div>
      )}

      {strategy === 'RANDOM_SAMPLE' && (
        <>
          <div className="space-y-2">
            <Label htmlFor="sample-size">Sample Size</Label>
            <Input
              id="sample-size"
              type="number"
              value={n}
              onChange={(e) => onNChange(parseInt(e.target.value) || 0)}
              min={1}
              max={totalRows || undefined}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="random-seed">Random Seed</Label>
            <Input
              id="random-seed"
              type="number"
              value={seed}
              onChange={(e) => onSeedChange(parseInt(e.target.value) || 42)}
            />
            <p className="text-xs text-muted-foreground">
              Use the same seed for reproducible sampling
            </p>
          </div>
        </>
      )}

      {strategy === 'ROW_RANGE' && (
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="range-start">Start Row</Label>
            <Input
              id="range-start"
              type="number"
              value={start}
              onChange={(e) => onStartChange?.(parseInt(e.target.value) || 0)}
              min={0}
              max={end - 1}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="range-end">End Row</Label>
            <Input
              id="range-end"
              type="number"
              value={end}
              onChange={(e) => onEndChange?.(parseInt(e.target.value) || 0)}
              min={start + 1}
              max={totalRows || undefined}
            />
          </div>
          <p className="text-xs text-muted-foreground col-span-2">
            Embedding rows {start} to {end} (inclusive)
          </p>
        </div>
      )}

      {showWarning && (
        <div className="rounded-md bg-yellow-50 dark:bg-yellow-900/20 p-3 text-sm text-yellow-800 dark:text-yellow-200">
          ⚠️ Embedding {totalRows.toLocaleString()} rows may take several minutes
        </div>
      )}
    </div>
  );
}
