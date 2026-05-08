'use client';

import { useState } from 'react';
import { ChevronRight, Plus, X } from 'lucide-react';
import { Badge } from '@/lib/ui-primitives/badge';
import { Button } from '@/lib/ui-primitives/button';
import { Slider } from '@/lib/ui-primitives/slider';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import { cn } from '@/lib/utils/utils';
import { parseSaeId } from '@/lib/utils/saeCollections';
import type { SaeFeature, SteeringConfig, SteeringFeature } from '@/lib/types/types';

const DEFAULT_STRENGTH = 800;
const STRENGTH_MIN = -2000;
const STRENGTH_MAX = 2000;
const STRENGTH_STEP = 50;

/** Composite identity key for a steering feature. */
export function steeringFeatureKey(f: SteeringFeature | { modelId: string; saeId: string; featureIndex: number }): string {
  return `${f.modelId}::${f.saeId}::${f.featureIndex}`;
}

interface SteeringControlsProps {
  config: SteeringConfig;
  onAddFeature: (feature: SteeringFeature) => void;
  onRemoveFeature: (key: string) => void;
  onUpdateStrength: (key: string, strength: number) => void;
  currentFeature: SaeFeature | null;
  currentModelId: string | null;
  currentSaeId: string | null;
}

export function SteeringControls({
  config,
  onAddFeature,
  onRemoveFeature,
  onUpdateStrength,
  currentFeature,
  currentModelId,
  currentSaeId,
}: SteeringControlsProps) {
  const [open, setOpen] = useState(true);
  const count = config.features.length;

  const isCurrentAlreadyAdded =
    currentFeature != null &&
    currentModelId != null &&
    currentSaeId != null &&
    config.features.some(
      (f) => steeringFeatureKey(f) === `${currentModelId}::${currentSaeId}::${currentFeature.featureIndex}`,
    );

  const canAdd = currentFeature != null && currentModelId != null && currentSaeId != null && !isCurrentAlreadyAdded;

  const handleAdd = () => {
    if (!canAdd) return;
    const parsed = parseSaeId(currentSaeId!);
    onAddFeature({
      modelId: currentModelId!,
      saeId: currentSaeId!,
      layerIndex: parsed.layerIndex,
      featureIndex: currentFeature!.featureIndex,
      strength: DEFAULT_STRENGTH,
      label: currentFeature!.label ?? undefined,
      hookType: parsed.hookType,
      width: parsed.width,
    });
  };

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="border-b border-border/30">
      <CollapsibleTrigger className="flex w-full items-center gap-2 px-4 py-2.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
        <ChevronRight
          className={cn(
            'size-3.5 transition-transform duration-200',
            open && 'rotate-90',
          )}
        />
        Steering
        {count > 0 && (
          <Badge variant="secondary" className="ml-auto text-[10px] px-1.5 py-0">
            {count}
          </Badge>
        )}
      </CollapsibleTrigger>

      <CollapsibleContent className="px-4 pb-3">
        {config.features.length === 0 && (
          <p className="mb-2 text-[11px] text-muted-foreground/60">
            No features added yet. Browse features and add them to steer the model.
          </p>
        )}

        <div className="flex flex-col gap-2">
          {config.features.map((f) => {
            const key = steeringFeatureKey(f);
            return (
              <div
                key={key}
                className="flex items-center gap-2 rounded-lg bg-muted/30 px-2.5 py-1.5"
              >
                <Badge variant="outline" className="shrink-0 text-[10px] font-mono px-1.5 py-0">
                  #{f.featureIndex}
                </Badge>

                <span className="min-w-0 flex-1 truncate text-[11px] text-muted-foreground">
                  {f.label || `Layer ${f.layerIndex}`}
                </span>

                <Slider
                  value={[f.strength]}
                  min={STRENGTH_MIN}
                  max={STRENGTH_MAX}
                  step={STRENGTH_STEP}
                  onValueChange={([v]) => onUpdateStrength(key, v)}
                  className="w-20 shrink-0"
                />

                <span className="w-12 shrink-0 text-right font-mono text-[10px] text-muted-foreground tabular-nums">
                  {f.strength}
                </span>

                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => onRemoveFeature(key)}
                  className="size-5 shrink-0 text-muted-foreground/50 hover:text-destructive"
                >
                  <X className="size-3" />
                </Button>
              </div>
            );
          })}
        </div>

        <Button
          size="sm"
          variant="outline"
          onClick={handleAdd}
          disabled={!canAdd}
          className="mt-2 h-7 w-full text-[11px]"
        >
          <Plus className="mr-1 size-3" />
          {isCurrentAlreadyAdded
            ? 'Already added'
            : currentFeature
              ? `Add #${currentFeature.featureIndex}`
              : 'Select a feature first'}
        </Button>
      </CollapsibleContent>
    </Collapsible>
  );
}
