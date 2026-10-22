'use client';

import { useId } from 'react';
import { Label } from '@/lib/ui-primitives/label';
import { Input } from '@/lib/ui-primitives/input';
import { Slider } from '@/lib/ui-primitives/slider';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/lib/ui-primitives/select';
import type { ReductionMethod, ReductionStrategy } from '@/lib/graphql/mutations';
import {
  REDUCTION_STRATEGY_OPTIONS,
  ReductionRules,
  type ReductionChoice,
} from '../../lib/topicConfigInput';
import { DISTANCE_THRESHOLD_RANGE } from '@/lib/utils/topicReduction';

interface ReductionStrategyFieldsProps {
  value: ReductionChoice;
  onChange: (value: ReductionChoice) => void;
  /** Density needs an HDBSCAN density tree (inline: HDBSCAN clustering; standalone: the tree query). */
  densityAvailable: boolean;
  /** Most topics the "Topics" slider offers (noise not counted). */
  maxTopics: number;
}

/**
 * Strategy, cut (fixed N or auto) and density options of a topic reduction,
 * shared by the inline reduction (TopicConfigForm) and the standalone one
 * (ReduceTopicsSection). Only options the backend accepts for the chosen
 * strategy are shown (ReductionRules); `ReductionInputs` sends the same.
 */
export function ReductionStrategyFields({ value, onChange, densityAvailable, maxTopics }: ReductionStrategyFieldsProps) {
  const id = useId();
  const update = (patch: Partial<ReductionChoice>) => onChange({ ...value, ...patch });
  const strategy = ReductionRules.effectiveStrategy(value.strategy, densityAvailable);
  const method = ReductionRules.effectiveMethod(strategy, value.method);
  const options = REDUCTION_STRATEGY_OPTIONS.filter((o) => ReductionRules.strategies(densityAvailable).includes(o.value));
  const hint = REDUCTION_STRATEGY_OPTIONS.find((o) => o.value === strategy)?.hint;
  const topicMax = Math.max(maxTopics, 2);

  return (
    <div className="space-y-3">
      <div className="space-y-2">
        <Label htmlFor={`${id}-strategy`}>Strategy</Label>
        <Select value={strategy} onValueChange={(v) => update({ strategy: v as ReductionStrategy })}>
          <SelectTrigger id={`${id}-strategy`}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map((o) => (
              <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      </div>

      {ReductionRules.supportsAuto(strategy) && (
        <div className="space-y-2">
          <Label>Cut</Label>
          <ToggleGroup
            type="single"
            variant="outline"
            value={method}
            onValueChange={(v) => { if (v) update({ method: v as ReductionMethod }); }}
          >
            <ToggleGroupItem value="fixed_n" className="text-xs">Fixed N</ToggleGroupItem>
            <ToggleGroupItem value="auto" className="text-xs">Auto</ToggleGroupItem>
          </ToggleGroup>
        </div>
      )}

      {method === 'fixed_n' && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor={`${id}-n-topics`}>Topics</Label>
            <span className="text-sm text-muted-foreground">{Math.min(value.nTopics, topicMax)}</span>
          </div>
          <Slider
            id={`${id}-n-topics`}
            min={2}
            max={topicMax}
            step={1}
            value={[Math.min(value.nTopics, topicMax)]}
            onValueChange={([v]) => update({ nTopics: v })}
          />
          <p className="text-xs text-muted-foreground">Noise is not counted.</p>
        </div>
      )}

      {method === 'auto' && ReductionRules.supportsThreshold(strategy) && (
        <div className="space-y-2">
          <Label htmlFor={`${id}-threshold`}>Distance threshold</Label>
          <Input
            id={`${id}-threshold`}
            type="number"
            min={DISTANCE_THRESHOLD_RANGE.min}
            max={DISTANCE_THRESHOLD_RANGE.max}
            step={0.05}
            value={value.distanceThreshold}
            onChange={(e) => {
              const v = Number.parseFloat(e.target.value);
              const { min, max } = DISTANCE_THRESHOLD_RANGE;
              if (Number.isFinite(v)) update({ distanceThreshold: Math.min(Math.max(v, min), max) });
            }}
            className="w-28"
          />
          <p className="text-xs text-muted-foreground">
            Merges topics while their centred cosine distance is at most this; lower keeps more topics.
          </p>
        </div>
      )}

      {ReductionRules.supportsAbsorbNoise(strategy) && (
        <div className="flex items-center gap-2">
          <Checkbox
            id={`${id}-absorb`}
            checked={value.absorbNoise}
            onCheckedChange={(checked) => update({ absorbNoise: checked === true })}
          />
          <Label htmlFor={`${id}-absorb`} className="cursor-pointer">
            Absorb noise into merged topics
          </Label>
        </div>
      )}
    </div>
  );
}
