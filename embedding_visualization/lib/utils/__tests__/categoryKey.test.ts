import { describe, expect, it } from 'vitest';
import { CategoryKey } from '../categoryKey';

const { MISSING } = CategoryKey;

/** Points shaped like the `combined` collection: `category` is often absent. */
const points = [
  { index: 0, metadata: { category: 'b' } },
  { index: 1, metadata: {} },
  { index: 2, metadata: { category: null } },
  { index: 3, metadata: { category: '' } },
  { index: 4, metadata: { category: 'a' } },
  { index: 5, metadata: { category: 'b' } },
];

describe('CategoryKey.of', () => {
  it('maps absent, null and empty values to the shared Missing key', () => {
    expect(CategoryKey.of(undefined)).toBe(MISSING);
    expect(CategoryKey.of(null)).toBe(MISSING);
    expect(CategoryKey.of('')).toBe(MISSING);
  });

  it('stringifies present values, including falsy ones and a literal "unknown"', () => {
    expect(CategoryKey.of(0)).toBe('0');
    expect(CategoryKey.of(false)).toBe('false');
    expect(CategoryKey.of('unknown')).toBe('unknown');
    expect(CategoryKey.of('unknown')).not.toBe(MISSING);
  });
});

describe('CategoryKey.count', () => {
  it('counts missing values under the Missing key, listed last', () => {
    expect(CategoryKey.count(points, 'category')).toEqual({
      categoryValues: ['a', 'b', MISSING],
      categoryCounts: { a: 1, b: 2, [MISSING]: 3 },
    });
  });

  it('accounts for every point, so totals match the collection size', () => {
    const { categoryCounts } = CategoryKey.count(points, 'category');
    expect(Object.values(categoryCounts).reduce((sum, n) => sum + n, 0)).toBe(points.length);
  });

  it('omits the Missing key when every point has a value', () => {
    const complete = points.filter(p => p.metadata.category);
    expect(CategoryKey.count(complete, 'category').categoryValues).toEqual(['a', 'b']);
  });

  it('keeps the plain string order of real values (palette slots depend on it)', () => {
    const mixed = ['b', 'B', '10', '9', 'a'].map((category, index) => ({ index, metadata: { category } }));
    expect(CategoryKey.count(mixed, 'category').categoryValues).toEqual(['10', '9', 'B', 'a', 'b']);
  });

  it('skips excluded point indices', () => {
    expect(CategoryKey.count(points, 'category', new Set([1, 2, 5])).categoryCounts).toEqual({
      a: 1, b: 1, [MISSING]: 1,
    });
  });

  it('returns nothing without a field', () => {
    expect(CategoryKey.count(points, null)).toEqual({ categoryValues: [], categoryCounts: {} });
  });
});

describe('CategoryKey.nested', () => {
  it('keys the subtopic, falling back to the topic when the subtopic is missing', () => {
    expect(CategoryKey.nested({ topic_label: 'Arts', subtopic_label: 'Poetry' })).toEqual({ topic: 'Arts', subtopic: 'Poetry' });
    expect(CategoryKey.nested({ topic_label: 'Arts', subtopic_label: '' })).toEqual({ topic: 'Arts', subtopic: 'Arts' });
    expect(CategoryKey.nested({ topic_label: 'Arts' })).toEqual({ topic: 'Arts', subtopic: 'Arts' });
  });

  it('keys a point without a topic as Missing at both levels', () => {
    expect(CategoryKey.nested({})).toEqual({ topic: MISSING, subtopic: MISSING });
    expect(CategoryKey.nested(undefined)).toEqual({ topic: MISSING, subtopic: MISSING });
  });

  it('keeps a present subtopic under a Missing topic', () => {
    expect(CategoryKey.nested({ subtopic_label: 'Poetry' })).toEqual({ topic: MISSING, subtopic: 'Poetry' });
  });
});

describe('CategoryKey.missingFade', () => {
  it('scales only the Missing group by the missing-value opacity', () => {
    expect(CategoryKey.missingFade(MISSING, 0.3)).toBe(0.3);
    expect(CategoryKey.missingFade('a', 0.3)).toBe(1);
  });
});

describe('CategoryKey.missingLast', () => {
  it('orders the Missing key after every other key and keeps the rest stable', () => {
    expect(['b', MISSING, 'a'].sort(CategoryKey.missingLast)).toEqual(['b', 'a', MISSING]);
  });
});
