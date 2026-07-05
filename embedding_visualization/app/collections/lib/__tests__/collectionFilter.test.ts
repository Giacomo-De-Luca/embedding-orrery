import { describe, it, expect } from 'vitest';
import { filterCollections, providerShortLabel, formatItemCount } from '../collectionFilter';

const COLLECTIONS = [
  { name: 'emotion', embeddingModel: 'all-MiniLM-L6-v2' },
  { name: 'xkcd_hilbert_gemini', embeddingModel: 'gemini-embedding-001' },
  { name: 'wordnet', embeddingModel: null },
];

describe('filterCollections', () => {
  it('returns everything for an empty or whitespace query', () => {
    expect(filterCollections(COLLECTIONS, '')).toEqual(COLLECTIONS);
    expect(filterCollections(COLLECTIONS, '   ')).toEqual(COLLECTIONS);
  });

  it('matches on name, case-insensitively', () => {
    expect(filterCollections(COLLECTIONS, 'XKCD').map(c => c.name)).toEqual(['xkcd_hilbert_gemini']);
  });

  it('matches on embedding model', () => {
    expect(filterCollections(COLLECTIONS, 'minilm').map(c => c.name)).toEqual(['emotion']);
  });

  it('handles null models and no matches', () => {
    expect(filterCollections(COLLECTIONS, 'nonexistent')).toEqual([]);
  });
});

describe('providerShortLabel', () => {
  it('maps known providers', () => {
    expect(providerShortLabel('SENTENCE_TRANSFORMERS')).toBe('ST');
    expect(providerShortLabel('HUGGINGFACE_API')).toBe('HF API');
  });

  it('is case-insensitive and falls back to the raw value', () => {
    expect(providerShortLabel('openai')).toBe('OpenAI');
    expect(providerShortLabel('CUSTOM_THING')).toBe('CUSTOM_THING');
    expect(providerShortLabel(null)).toBeNull();
    expect(providerShortLabel(undefined)).toBeNull();
  });
});

describe('formatItemCount', () => {
  it('formats small, thousand, and million ranges', () => {
    expect(formatItemCount(954)).toBe('954');
    expect(formatItemCount(1000)).toBe('1,000');
    expect(formatItemCount(153_000)).toBe('153k');
    expect(formatItemCount(2_500_000)).toBe('2.5M');
  });
});
