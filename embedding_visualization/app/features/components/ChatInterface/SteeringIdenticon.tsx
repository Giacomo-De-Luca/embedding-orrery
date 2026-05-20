'use client';

import { useMemo, type ReactNode } from 'react';
import { toSvg } from 'jdenticon';
import { cn } from '@/lib/utils/utils';
import type { SteeringFeature } from '@/lib/types/types';
import { steeringIdenticonHash } from '@/lib/utils/steeringIdenticon';

interface SteeringIdenticonProps {
  features: SteeringFeature[];
  size: number;
  className?: string;
  fallback: ReactNode;
}

export function SteeringIdenticon({
  features,
  size,
  className,
  fallback,
}: SteeringIdenticonProps) {
  const hash = useMemo(() => steeringIdenticonHash(features), [features]);
  const svg = useMemo(() => {
    if (!hash) return null;
    // Tag the SVG with a `size-` class so Button's global
    // `[&_svg:not([class*='size-'])]:size-4` rule (specificity 0,2,1) excludes it
    // and stops clamping it to 16px. `size-full` then fills the sized wrapper.
    return toSvg(hash, size).replace('<svg', '<svg class="size-full"');
  }, [hash, size]);

  if (!svg) return <>{fallback}</>;

  return (
    <span
      className={cn('inline-block', className)}
      style={{ width: size, height: size }}
      // jdenticon SVG output is generated deterministically from the hash and
      // contains no user-controlled content, so dangerouslySetInnerHTML is safe.
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
