import { describe, expect, it } from 'vitest';
import { HiddenPoints } from '../hiddenPoints';

const points = [
  { index: 0, metadata: { topic_id: -1, category: 'a' } },
  { index: 1, metadata: { topic_id: '-1' } },
  { index: 2, metadata: { topic_label: 'Unclustered', category: 'b' } },
  { index: 3, metadata: { topic_label: 'Arts' } },
  { index: 4, metadata: { topic_label: 'Arts', category: '' } },
  { index: 5, metadata: { topic_label: 'Arts', category: 'c' } },
];
const indices = (kept: typeof points) => kept.map(p => p.index);

describe('HiddenPoints.filter', () => {
  it('returns the same array when no toggle applies', () => {
    expect(HiddenPoints.filter(points, 'category', { hideUnclustered: false, hideMissing: false })).toBe(points);
    expect(HiddenPoints.filter(points, null, { hideUnclustered: false, hideMissing: true })).toBe(points);
  });

  it('hides unclustered noise by topic id or topic label', () => {
    expect(indices(HiddenPoints.filter(points, 'category', { hideUnclustered: true, hideMissing: false }))).toEqual([3, 4, 5]);
  });

  it('hides points whose colour field is absent, null or empty', () => {
    expect(indices(HiddenPoints.filter(points, 'category', { hideUnclustered: false, hideMissing: true }))).toEqual([0, 2, 5]);
  });

  it('combines both toggles', () => {
    expect(indices(HiddenPoints.filter(points, 'category', { hideUnclustered: true, hideMissing: true }))).toEqual([5]);
  });
});
