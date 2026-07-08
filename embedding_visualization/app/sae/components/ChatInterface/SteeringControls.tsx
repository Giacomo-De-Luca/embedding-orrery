'use client';

import { useState } from 'react';
import { ChevronRight, Plus, Sparkles, X } from 'lucide-react';
import { SteeringIdenticon } from './SteeringIdenticon';
import { Badge } from '@/lib/ui-primitives/badge';
import { Button } from '@/lib/ui-primitives/button';
import { Slider } from '@/lib/ui-primitives/slider';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import { cn } from '@/lib/utils/utils';
import { activeSteeringFeatures } from '@/lib/hooks/useSteeringChat';
import { useModelIdentityStore, steeringFeatureKey } from '@/lib/stores/useModelIdentityStore';
import {
  steeringHint,
  snapStrengthToSlider,
  type StrengthBand,
} from '@/lib/utils/steeringStrengthHint';
import type { SaeFeature } from '@/lib/types/types';

const DEFAULT_STRENGTH = 800;
const STRENGTH_MIN = -2000;
const STRENGTH_MAX = 2000;
const STRENGTH_STEP = 50;

// Pre-extracted directions (refusal, poetry) are applied at coefficients
// in the unit-ish range; the SAE-feature scale is wildly excessive.
const DIRECTION_STRENGTH_MIN = -5;
const DIRECTION_STRENGTH_MAX = 5;
const DIRECTION_STRENGTH_STEP = 0.1;

// Colour the residual-norm band readout: subtle nudge → strong intervention.
const BAND_TEXT: Record<StrengthBand, string> = {
  subtle: 'text-muted-foreground',
  medium: 'text-amber-500',
  strong: 'text-rose-500',
};

interface SteeringControlsProps {
  currentFeature: SaeFeature | null;
}

export function SteeringControls({ currentFeature }: SteeringControlsProps) {
  const [open, setOpen] = useState(true);
  const config = useModelIdentityStore((s) => s.steeringConfig);
  const modelId = useModelIdentityStore((s) => s.modelId);
  const saeId = useModelIdentityStore((s) => s.saeId);
  const parsedSae = useModelIdentityStore((s) => s.parsedSae);
  const count = config.features.length;
  const activeCount = activeSteeringFeatures(config.features).length;

  const isCurrentAlreadyAdded =
    currentFeature != null &&
    modelId != null &&
    saeId != null &&
    config.features.some(
      (f) => steeringFeatureKey(f) === `${modelId}::${saeId}::${currentFeature.featureIndex}`,
    );

  const canAdd = currentFeature != null && modelId != null && saeId != null && !isCurrentAlreadyAdded;

  const handleAdd = () => {
    if (!canAdd || !parsedSae) return;
    // Start at a layer-aware "moderate nudge" strength when residual norms are
    // available (deeper layers need a larger coefficient for the same effect);
    // fall back to the flat default when the table has no data for this model.
    // suggestedStrength is strength-independent, so passing strength 0 is fine.
    const addHint = steeringHint({ modelId, layerIndex: parsedSae.layerIndex, strength: 0 });
    const defaultStrength = addHint
      ? snapStrengthToSlider(addHint.suggestedStrength, {
          min: STRENGTH_MIN,
          max: STRENGTH_MAX,
          step: STRENGTH_STEP,
        })
      : DEFAULT_STRENGTH;
    useModelIdentityStore.getState().addSteeringFeature({
      modelId: modelId!,
      saeId: saeId!,
      layerIndex: parsedSae.layerIndex,
      featureIndex: currentFeature!.featureIndex,
      strength: defaultStrength,
      label: currentFeature!.label ?? undefined,
      hookType: parsedSae.hookType,
      width: parsedSae.width,
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
          <Badge
            variant="secondary"
            className="ml-auto text-[10px] px-1.5 py-0"
            title={`${activeCount} of ${count} features actively steering (strength ≠ 0)`}
          >
            {activeCount}/{count}
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
            const isDirection = !!f.directionName;
            const sliderMin = isDirection ? DIRECTION_STRENGTH_MIN : STRENGTH_MIN;
            const sliderMax = isDirection ? DIRECTION_STRENGTH_MAX : STRENGTH_MAX;
            const sliderStep = isDirection ? DIRECTION_STRENGTH_STEP : STRENGTH_STEP;
            const strengthDisplay = isDirection ? f.strength.toFixed(1) : f.strength;
            // Advisory hint: how large the current strength is relative to the
            // residual-stream norm at the applied layer (null when the
            // residualNorms table has no data for this model/layer).
            const hint = steeringHint({
              modelId,
              layerIndex: f.layerIndex,
              strength: f.strength,
              directionName: f.directionName,
            });
            const recValue =
              hint != null
                ? snapStrengthToSlider(hint.suggestedStrength, {
                    min: sliderMin,
                    max: sliderMax,
                    step: sliderStep,
                  })
                : null;
            return (
              <div
                key={key}
                className="flex flex-col gap-1 rounded-lg bg-muted/30 px-2.5 py-1.5"
              >
                <div className="flex items-center gap-2">
                  {/* Per-feature identicon — same visual language as the chat
                      history avatars, seeded from this single feature's label. */}
                  {/* Pulses while the feature is live (strength ≠ 0) — the
                      visual cue that a preset has been dialled in. */}
                  <span className="flex size-5 shrink-0 items-center justify-center rounded-md bg-muted/60 ring-1 ring-border/40">
                    <SteeringIdenticon
                      features={[f]}
                      size={16}
                      fallback={<Sparkles className="size-3 text-muted-foreground/70" />}
                      animation="alternate-pulse"
                      active={f.strength !== 0}
                    />
                  </span>

                  <Badge
                    variant="outline"
                    className={cn(
                      'shrink-0 text-[10px] font-mono px-1.5 py-0',
                      isDirection && 'uppercase tracking-wide',
                    )}
                  >
                    {isDirection ? 'dir' : `#${f.featureIndex}`}
                  </Badge>

                  <span className="min-w-0 flex-1 truncate text-[11px] text-muted-foreground">
                    {f.label || (isDirection ? f.directionName : `Layer ${f.layerIndex}`)}
                  </span>

                  <Slider
                    value={[f.strength]}
                    min={sliderMin}
                    max={sliderMax}
                    step={sliderStep}
                    onValueChange={([v]) => useModelIdentityStore.getState().updateSteeringStrength(key, v)}
                    className="w-20 shrink-0"
                  />

                  <span className="w-12 shrink-0 text-right font-mono text-[10px] text-muted-foreground tabular-nums">
                    {strengthDisplay}
                  </span>

                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => useModelIdentityStore.getState().removeSteeringFeature(key)}
                    className="size-5 shrink-0 text-muted-foreground/50 hover:text-destructive"
                  >
                    <X className="size-3" />
                  </Button>
                </div>

                {hint && (
                  <div
                    className="flex items-center gap-1.5 pl-7 text-[10px] tabular-nums"
                    title={
                      `≈ ${(hint.rho * 100).toFixed(0)}% of the residual-stream norm at layer ${hint.layer} ` +
                      `(‖h‖≈${hint.residualNorm.toFixed(0)}, ‖v‖≈${hint.vecNorm < 10 ? hint.vecNorm.toFixed(2) : hint.vecNorm.toFixed(0)}). ` +
                      `Recommended ≈ ${isDirection ? recValue?.toFixed(1) : recValue}.`
                    }
                  >
                    <span className={cn('font-medium capitalize', BAND_TEXT[hint.band])}>
                      {hint.band}
                    </span>
                    <span className="text-muted-foreground/70">
                      ≈ {(hint.rho * 100).toFixed(0)}% of ‖resid‖
                    </span>
                    {recValue != null && recValue !== f.strength && (
                      <button
                        type="button"
                        onClick={() =>
                          useModelIdentityStore.getState().updateSteeringStrength(key, recValue)
                        }
                        className="ml-auto rounded px-1 text-muted-foreground/60 hover:bg-muted/60 hover:text-foreground"
                      >
                        use {isDirection ? recValue.toFixed(1) : recValue}
                      </button>
                    )}
                  </div>
                )}
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
