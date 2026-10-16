'use client';

import { Label } from '@/lib/ui-primitives/label';
import { Slider } from '@/lib/ui-primitives/slider';

interface SliderRowProps {
  label: string;
  /** Formatted current value shown at the right edge (e.g. "1.50×"). */
  valueLabel: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  /** Native tooltip on the label. */
  title?: string;
}

/** Label + value readout + slider, the Controls panel's standard row layout. */
export function SliderRow({ label, valueLabel, min, max, step, value, onChange, title }: SliderRowProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-normal" title={title}>{label}</Label>
        <span className="text-xs text-muted-foreground tabular-nums">{valueLabel}</span>
      </div>
      <Slider min={min} max={max} step={step} value={[value]} onValueChange={([v]) => onChange(v)} />
    </div>
  );
}
