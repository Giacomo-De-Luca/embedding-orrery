import { describe, expect, it } from 'vitest';
import { CategorySelection } from '../categorySelection';
import { CategoryKey } from '../categoryKey';
import { buildNestedColorMap } from '../nestedColorUtils';

describe('shared legend selection', () => {
  it('isolates one category, restores all on a second click, and shift-toggles individual categories', () => {
    const all = ['A', 'B', 'C'];
    expect(CategorySelection.toggle('B', false, [], all)).toEqual(['A', 'C']);
    expect(CategorySelection.toggle('B', false, ['A', 'C'], all)).toEqual([]);
    expect(CategorySelection.toggle('A', true, [], all)).toEqual(['A']);
    expect(CategorySelection.toggle('A', true, ['A', 'B'], all)).toEqual(['B']);
  });

  it('treats Missing as a regular entry: isolating another category mutes it, and it can be isolated or toggled', () => {
    const { MISSING } = CategoryKey;
    const all = ['A', 'B', MISSING];
    expect(CategorySelection.toggle('A', false, [], all)).toEqual(['B', MISSING]);
    expect(CategorySelection.toggle(MISSING, false, [], all)).toEqual(['A', 'B']);
    expect(CategorySelection.toggle(MISSING, true, [], all)).toEqual([MISSING]);
    expect(CategorySelection.toggle(MISSING, true, [MISSING], all)).toEqual([]);
  });

  it('treats a Missing topic like any topic in nested mode', () => {
    const { MISSING } = CategoryKey;
    const nested = buildNestedColorMap([
      { index: 0, metadata: { topic_label: 'Arts', subtopic_label: 'Poetry' } },
      { index: 1, metadata: { topic_label: 'Arts', subtopic_label: 'Music' } },
      { index: 2, metadata: {} },
    ]);
    expect(new Set(CategorySelection.toggle('Arts', false, [], [], nested))).toEqual(new Set([MISSING]));
    expect(new Set(CategorySelection.toggle(MISSING, false, [], [], nested))).toEqual(new Set(['Arts', 'Music', 'Poetry']));
  });
});
