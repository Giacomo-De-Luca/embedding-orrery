import { describe, expect, it } from 'vitest';

import { cn as shadcnCn } from '@/lib/utils';
import { cn } from '@/lib/utils/utils';

describe('cn', () => {
  it('has one implementation behind both import paths', () => {
    expect(shadcnCn).toBe(cn);
  });

  it('joins conditionally and lets the last conflicting Tailwind class win', () => {
    expect(cn('p-2', 'p-4')).toBe('p-4');
    expect(cn('text-sm', false, undefined, ['font-medium'])).toBe('text-sm font-medium');
  });
});
