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
  const svg = useMemo(() => (hash ? toSvg(hash, size) : null), [hash, size]);

  if (!svg) return <>{fallback}</>;

  return (
    <span
      // `[&>svg]:size-full` defeats Button's `[&_svg:not([class*='size-'])]:size-4`
      // global rule that would otherwise clamp our injected SVG to 16px.
      className={cn('inline-block [&>svg]:size-full', className)}
      style={{ width: size, height: size }}
      // jdenticon SVG output is generated deterministically from the hash and
      // contains no user-controlled content, so dangerouslySetInnerHTML is safe.
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
