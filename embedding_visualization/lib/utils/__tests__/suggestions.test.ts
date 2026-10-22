import { describe, expect, it } from 'vitest';

import { matchesSuggestion, type SuggestionItem } from '../suggestions';

const ITEM: SuggestionItem = {
  value: 'google/embeddinggemma-2',
  label: 'EmbeddingGemma 2',
  detail: '768d · used by 2 collections',
  description: 'Text, images, audio and video in one space',
};

describe('matchesSuggestion', () => {
  it('matches the value, label, detail or description, case-insensitively', () => {
    expect(matchesSuggestion(ITEM, 'GOOGLE/')).toBe(true);
    expect(matchesSuggestion(ITEM, 'gemma 2')).toBe(true);
    expect(matchesSuggestion(ITEM, '768D')).toBe(true);
    expect(matchesSuggestion(ITEM, 'audio and video')).toBe(true);
    expect(matchesSuggestion(ITEM, 'minilm')).toBe(false);
  });

  it('matches every item on a blank query', () => {
    expect(matchesSuggestion(ITEM, '')).toBe(true);
    expect(matchesSuggestion({ value: 'STS' }, '   ')).toBe(true);
  });

  it('ignores the missing optional texts', () => {
    expect(matchesSuggestion({ value: 'Clustering' }, 'clust')).toBe(true);
    expect(matchesSuggestion({ value: 'Clustering' }, 'search')).toBe(false);
  });
});
