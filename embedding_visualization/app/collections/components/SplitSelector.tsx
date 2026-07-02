'use client';

import { Label } from '@/lib/ui-primitives/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';

interface SplitInfo {
  name: string;
  numRows: number | null;
}

interface SplitSelectorProps {
  splits: SplitInfo[];
  selectedSplit: string;
  onSplitChange: (split: string) => void;
  disabled?: boolean;
}

export function SplitSelector({ splits, selectedSplit, onSplitChange, disabled }: SplitSelectorProps) {
  if (splits.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2">
      <Label htmlFor="split-select">Dataset Split</Label>
      <Select value={selectedSplit} onValueChange={onSplitChange} disabled={disabled}>
        <SelectTrigger id="split-select">
          <SelectValue placeholder="Select a split" />
        </SelectTrigger>
        <SelectContent>
          {splits.map((split) => (
            <SelectItem key={split.name} value={split.name}>
              {split.name}
              {split.numRows !== null && ` (${split.numRows.toLocaleString()} rows)`}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        Choose which split to embed (train, test, validation, etc.)
      </p>
    </div>
  );
}
