'use client';

import { motion, useReducedMotion } from 'motion/react';
import { Columns2, Sparkles } from 'lucide-react';
import { activeSteeringFeatures } from '@/lib/hooks/useSteeringChat';
import type { SteeringFeature } from '@/lib/types/types';
import { SteeringIdenticon } from './SteeringIdenticon';

interface ChatGreetingProps {
  /** THIS thread's steering features — empty for the baseline thread. */
  features: SteeringFeature[];
  isBaseline?: boolean;
  /** Renders a "Compare with baseline" pill when provided (single mode only). */
  onStartCompare?: () => void;
}

export function ChatGreeting({ features, isBaseline = false, onStartCompare }: ChatGreetingProps) {
  const reducedMotion = useReducedMotion();
  // Only features actually steering (strength ≠ 0) count — auto-loaded
  // presets sit at strength 0 and are filtered from the GraphQL payload.
  const activeCount = activeSteeringFeatures(features).length;

  return (
    // z-10: the message scroll container is a later `absolute inset-0`
    // sibling, so without it the greeting sits underneath and the compare
    // pill never receives clicks despite its pointer-events-auto.
    <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
      <div className="flex flex-col items-center px-6">
        {/* Icon — a slow hue-drift shimmer on the container (not inside
            SteeringIdenticon, whose animation path skips the fallback).
            Deliberately low-rate: motion here should not compete with the
            heading. Grays are hue-invariant, so the baseline Sparkles and
            the ring stay calm while identicon colors drift. */}
        <motion.div
          className="flex size-10 items-center justify-center rounded-xl bg-muted/60 ring-1 ring-border/50"
          initial={{ opacity: 0, y: 10 }}
          animate={{
            opacity: 1,
            y: 0,
            filter: reducedMotion
              ? 'none'
              : [
                  'hue-rotate(0deg) saturate(1)',
                  'hue-rotate(180deg) saturate(1.2)',
                  'hue-rotate(360deg) saturate(1)',
                ],
          }}
          transition={{
            duration: 0.5,
            ease: [0.22, 1, 0.36, 1],
            filter: { delay: 0.6, duration: 8, repeat: Infinity, ease: 'linear' },
          }}
        >
          <SteeringIdenticon
            features={features}
            size={40}
            fallback={<Sparkles className="size-5 text-muted-foreground" />}
            crossfadeOnChange
          />
        </motion.div>

        {/* Heading */}
        <motion.h3
          className="mt-4 text-center text-2xl font-semibold tracking-tight text-foreground"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        >
          {isBaseline ? 'Chat with Gemma' : 'Chat with Steered Gemma'}
        </motion.h3>

        {/* Subtitle */}
        <motion.p
          className="mt-3 text-center text-sm text-muted-foreground/80"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        >
          {isBaseline
            ? 'Baseline — no steering'
            : activeCount > 0
              ? `Steering with ${activeCount} active feature${activeCount > 1 ? 's' : ''}`
              : 'Add features to steer the model'}
        </motion.p>

        {/* Compare-mode affordance */}
        {onStartCompare && (
          <motion.button
            type="button"
            onClick={onStartCompare}
            className="pointer-events-auto mt-4 inline-flex items-center gap-1.5 rounded-full border border-border/50 bg-muted/40 px-3 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted/70 hover:text-foreground"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.65, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          >
            <Columns2 className="size-3" />
            Compare with baseline
          </motion.button>
        )}
      </div>
    </div>
  );
}
