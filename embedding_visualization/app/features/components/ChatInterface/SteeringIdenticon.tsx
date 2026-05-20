'use client';

import { useMemo, type ReactNode } from 'react';
import {
  AnimatePresence,
  motion,
  useReducedMotion,
  type Target,
  type Transition,
} from 'motion/react';
import { toSvg } from 'jdenticon';
import { cn } from '@/lib/utils/utils';
import type { SteeringFeature } from '@/lib/types/types';
import { steeringIdenticonHash } from '@/lib/utils/steeringIdenticon';

export type IdenticonAnimation =
  | 'none'
  // container-level (motion on the wrapper)
  | 'spin'
  | 'breathe'
  | 'pulse'
  | 'shimmer'
  // shape-level (CSS on the two jdenticon color-group paths)
  | 'counter-rotate'
  | 'assemble'
  | 'alternate-pulse';

interface SteeringIdenticonProps {
  features: SteeringFeature[];
  size: number;
  fallback: ReactNode;
  className?: string;
  /** Which animation to run while `active` (default 'none'). */
  animation?: IdenticonAnimation;
  /** Gates the continuous animation — wire to generation state. */
  active?: boolean;
  /** Slight scale/tilt on hover (default true). */
  hover?: boolean;
  /** Crossfade between identicons when the steered bundle changes. */
  crossfadeOnChange?: boolean;
}

/** motion `animate`/`transition` configs for container-level variants. */
const CONTAINER_ANIMATIONS: Record<
  'spin' | 'breathe' | 'pulse' | 'shimmer',
  { animate: Target; transition: Transition }
> = {
  spin: {
    animate: { rotate: 360 },
    transition: { duration: 6, repeat: Infinity, ease: 'linear' },
  },
  breathe: {
    animate: { scale: [1, 1.08, 1] },
    transition: { duration: 1.8, repeat: Infinity, ease: 'easeInOut' },
  },
  pulse: {
    animate: { opacity: [1, 0.55, 1] },
    transition: { duration: 1.4, repeat: Infinity, ease: 'easeInOut' },
  },
  shimmer: {
    animate: { filter: ['hue-rotate(0deg) saturate(1)', 'hue-rotate(360deg) saturate(1.4)'] },
    transition: { duration: 3, repeat: Infinity, ease: 'linear' },
  },
};

/**
 * Wrapper classes for shape-level variants. These act on the two jdenticon
 * color-group `<path>`s (cells are merged per color, so finer targeting isn't
 * possible). Defined as plain CSS (`.ji-*` + `ji-*` keyframes) in globals.css.
 */
const SHAPE_ANIMATION_CLASSES: Record<
  'counter-rotate' | 'assemble' | 'alternate-pulse',
  string
> = {
  'counter-rotate': 'ji-counter-rotate',
  assemble: 'ji-assemble',
  'alternate-pulse': 'ji-alternate-pulse',
};

function isContainerAnimation(a: IdenticonAnimation): a is 'spin' | 'breathe' | 'pulse' | 'shimmer' {
  return a === 'spin' || a === 'breathe' || a === 'pulse' || a === 'shimmer';
}

export function SteeringIdenticon({
  features,
  size,
  fallback,
  className,
  animation = 'none',
  active = false,
  hover = true,
  crossfadeOnChange = false,
}: SteeringIdenticonProps) {
  const reducedMotion = useReducedMotion();
  const hash = useMemo(() => steeringIdenticonHash(features), [features]);
  const svg = useMemo(() => {
    if (!hash) return null;
    // Tag the SVG with a `size-` class so Button's global
    // `[&_svg:not([class*='size-'])]:size-4` rule (specificity 0,2,1) excludes it
    // and stops clamping it to 16px. `size-full` then fills the sized wrapper.
    return toSvg(hash, size).replace('<svg', '<svg class="size-full"');
  }, [hash, size]);

  if (!svg) return <>{fallback}</>;

  const animateContinuous = active && !reducedMotion && animation !== 'none';
  const containerMotion =
    animateContinuous && isContainerAnimation(animation)
      ? CONTAINER_ANIMATIONS[animation]
      : null;
  const shapeClass =
    animateContinuous && !isContainerAnimation(animation)
      ? SHAPE_ANIMATION_CLASSES[animation]
      : undefined;
  const doCrossfade = crossfadeOnChange && !reducedMotion;
  const hoverEnabled = hover && !reducedMotion;

  return (
    <AnimatePresence mode="wait" initial={false}>
      {/* Outer span: one-shot crossfade on bundle change + hover gesture. The
          hover (scale/rotate) lives here, NOT on the inner span, so it never
          fights the inner continuous keyframe loop over the same property. */}
      <motion.span
        key={hash}
        className={cn('inline-block', className)}
        style={{ width: size, height: size }}
        initial={doCrossfade ? { opacity: 0, scale: 0.85 } : false}
        animate={doCrossfade ? { opacity: 1, scale: 1 } : undefined}
        exit={doCrossfade ? { opacity: 0, scale: 0.85 } : undefined}
        transition={{ duration: 0.25, ease: 'easeOut' }}
        whileHover={hoverEnabled ? { scale: 1.12, rotate: 3 } : undefined}
      >
        {/* Inner span: the continuous container animation loop (or shape class). */}
        <motion.span
          className={cn('block size-full', shapeClass)}
          animate={containerMotion?.animate}
          transition={containerMotion?.transition}
          // jdenticon SVG output is generated deterministically from the hash and
          // contains no user-controlled content, so dangerouslySetInnerHTML is safe.
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </motion.span>
    </AnimatePresence>
  );
}
