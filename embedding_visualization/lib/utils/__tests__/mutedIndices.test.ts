import { describe, expect, it } from 'vitest';

import { MutedIndices } from '../mutedIndices';

describe('MutedIndices.union', () => {
  it('returns null when no set is active', () => {
    expect(MutedIndices.union([null, null])).toBeNull();
  });

  it('keeps the identity of a single active set', () => {
    const only = new Set([1, 2]);
    expect(MutedIndices.union([null, only])).toBe(only);
  });

  it('ignores empty sets', () => {
    const only = new Set([4]);
    expect(MutedIndices.union([new Set(), only])).toBe(only);
  });

  it('merges several active sets', () => {
    expect(MutedIndices.union([new Set([1]), new Set([2, 3])])).toEqual(new Set([1, 2, 3]));
  });
});
