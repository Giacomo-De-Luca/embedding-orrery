'use client';

import { Sparkles } from 'lucide-react';
import type { SteeringFeature } from '@/lib/types/types';
import { SteeringIdenticon } from './SteeringIdenticon';

interface AssistantAvatarProps {
  features?: SteeringFeature[];
  /** Animate while the model is generating this turn. */
  active?: boolean;
}

export function AssistantAvatar({ features = [], active = false }: AssistantAvatarProps) {
  return (
    <div className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-muted/60 ring-1 ring-border/50">
      <SteeringIdenticon
        features={features}
        size={22}
        fallback={<Sparkles className="size-3.5 text-muted-foreground" />}
        animation="breathe"
        active={active}
        crossfadeOnChange
      />
    </div>
  );
}
