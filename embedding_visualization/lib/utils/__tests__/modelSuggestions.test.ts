import { describe, expect, it } from 'vitest';

import type { EmbeddingModelCatalogEntry } from '../../graphql/mutations';
import { ModelSuggestions, SUGGESTED_GROUP, USED_GROUP } from '../modelSuggestions';

type EntryFields = Partial<EmbeddingModelCatalogEntry> &
  Pick<EmbeddingModelCatalogEntry, 'provider' | 'modelName'>;

/** A catalog entry with the backend's defaults (EmbeddingModelCatalogEntry in API/types.py). */
function entry(fields: EntryFields): EmbeddingModelCatalogEntry {
  return {
    description: null,
    modalities: ['text'],
    dimension: null,
    defaultFor: [],
    suggested: true,
    collectionCount: 0,
    ...fields,
  };
}

const MINILM = entry({
  provider: 'SENTENCE_TRANSFORMERS',
  modelName: 'all-MiniLM-L6-v2',
  description: 'Small and fast; the default text model',
  dimension: 384,
  collectionCount: 3,
});
const EG2 = entry({
  provider: 'SENTENCE_TRANSFORMERS',
  modelName: 'google/embeddinggemma-2',
  description: 'EmbeddingGemma 2: text, images, audio and video in one space',
  modalities: ['text', 'image', 'audio', 'video'],
  dimension: 768,
  defaultFor: ['multimodal'],
});
const OPENAI_SMALL = entry({ provider: 'OPENAI', modelName: 'text-embedding-3-small', dimension: 1536 });
// Used only: a local checkpoint the catalog does not list.
const CHECKPOINT = entry({
  provider: 'SENTENCE_TRANSFORMERS',
  modelName: '/models/gemma-finetune',
  suggested: false,
  collectionCount: 2,
});

const CATALOG = [
  MINILM,
  EG2,
  OPENAI_SMALL,
  CHECKPOINT,
  // The same model as MINILM in another case: listed once.
  entry({ provider: 'SENTENCE_TRANSFORMERS', modelName: 'ALL-MINILM-L6-V2', suggested: false, collectionCount: 1 }),
  // A used-only entry no collection uses any more is not a suggestion.
  entry({ provider: 'OPENAI', modelName: 'text-embedding-ada-002', suggested: false, collectionCount: 0 }),
];

const values = (groups: ReturnType<ModelSuggestions['groups']>) =>
  groups.map((group) => [group.label, group.items.map((item) => item.value)]);

describe('ModelSuggestions.groups', () => {
  const suggestions = new ModelSuggestions(CATALOG);

  it('lists catalog models as Suggested and the other used models after them', () => {
    expect(values(suggestions.groups({ provider: 'SENTENCE_TRANSFORMERS' }))).toEqual([
      [SUGGESTED_GROUP, ['all-MiniLM-L6-v2', 'google/embeddinggemma-2']],
      [USED_GROUP, ['/models/gemma-finetune']],
    ]);
  });

  it('keeps only the selected provider and drops empty groups', () => {
    expect(values(suggestions.groups({ provider: 'OPENAI' }))).toEqual([
      [SUGGESTED_GROUP, ['text-embedding-3-small']],
    ]);
    expect(suggestions.groups({ provider: 'QWEN' })).toEqual([]);
  });

  it('filters to models that embed every required modality', () => {
    expect(values(suggestions.groups({ provider: 'SENTENCE_TRANSFORMERS', requires: ['image', 'audio'] }))).toEqual([
      [SUGGESTED_GROUP, ['google/embeddinggemma-2']],
    ]);
    expect(suggestions.groups({ provider: 'SENTENCE_TRANSFORMERS', requires: ['pdf'] })).toEqual([]);
  });

  it('filters to media models when no kind is chosen yet', () => {
    expect(values(suggestions.groups({ provider: 'SENTENCE_TRANSFORMERS', anyMedia: true }))).toEqual([
      [SUGGESTED_GROUP, ['google/embeddinggemma-2']],
    ]);
  });

  it('describes each model with its description and a hint', () => {
    const [suggested, used] = suggestions.groups({ provider: 'SENTENCE_TRANSFORMERS' });
    expect(suggested.items[0]).toEqual({
      value: 'all-MiniLM-L6-v2',
      description: 'Small and fast; the default text model',
      detail: '384d · used by 3 collections',
    });
    expect(suggested.items[1].detail).toBe('text / image / audio / video · 768d');
    expect(used.items[0]).toEqual({ value: '/models/gemma-finetune', detail: 'used by 2 collections' });
  });
});

describe('ModelSuggestions.hint', () => {
  it('names a single collection in the singular and skips unknown facts', () => {
    expect(ModelSuggestions.hint(entry({ provider: 'OLLAMA', modelName: 'x', collectionCount: 1 }))).toBe(
      'used by 1 collection',
    );
    expect(ModelSuggestions.hint(entry({ provider: 'OLLAMA', modelName: 'x' }))).toBe('');
  });
});

describe('ModelSuggestions lookups', () => {
  const suggestions = new ModelSuggestions(CATALOG);

  it('finds the default model of a data type', () => {
    expect(suggestions.defaultFor('multimodal')).toBe(EG2);
    expect(suggestions.defaultFor('vector')).toBeNull();
    expect(new ModelSuggestions([]).defaultFor('multimodal')).toBeNull();
  });

  it('finds catalog models by name, case-insensitively, and never used-only ones', () => {
    expect(suggestions.find('SENTENCE_TRANSFORMERS', ' Google/EmbeddingGemma-2 ')).toBe(EG2);
    expect(suggestions.find('OPENAI', 'google/embeddinggemma-2')).toBeNull();
    // Used-only entries carry no modalities of their own (the backend checks them at load).
    expect(suggestions.find('SENTENCE_TRANSFORMERS', '/models/gemma-finetune')).toBeNull();
  });

  it('reports the modalities a catalog model lacks', () => {
    expect(suggestions.missingModalities('SENTENCE_TRANSFORMERS', 'all-MiniLM-L6-v2', ['image', 'audio'])).toEqual([
      'image',
      'audio',
    ]);
    expect(suggestions.missingModalities('SENTENCE_TRANSFORMERS', 'google/embeddinggemma-2', ['image', 'video'])).toEqual([]);
    expect(suggestions.missingModalities('SENTENCE_TRANSFORMERS', '/models/unknown', ['image'])).toEqual([]);
  });
});
