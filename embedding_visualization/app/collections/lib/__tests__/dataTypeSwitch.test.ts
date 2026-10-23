import { describe, expect, it } from 'vitest';

import type { EmbeddingModelCatalogEntry } from '@/lib/graphql/mutations';
import { ModelSuggestions } from '@/lib/utils/modelSuggestions';
import { modelOnDataTypeSwitch } from '../embeddingFormUtils';

const entry = (fields: Pick<EmbeddingModelCatalogEntry, 'modelName' | 'modalities'>): EmbeddingModelCatalogEntry => ({
  provider: 'SENTENCE_TRANSFORMERS',
  description: null,
  dimension: null,
  defaultFor: [],
  suggested: true,
  collectionCount: 0,
  ...fields,
});

const CATALOG = new ModelSuggestions([
  entry({ modelName: 'all-MiniLM-L6-v2', modalities: ['text'] }),
  entry({ modelName: 'google/embeddinggemma-2', modalities: ['text', 'image', 'audio', 'video'] }),
]);

describe('modelOnDataTypeSwitch', () => {
  it('switches to sentence-transformers with an empty name (the backend default) for media', () => {
    expect(
      modelOnDataTypeSwitch({ from: 'TEXT', to: 'MULTIMODAL', provider: 'OPENAI', modelName: 'text-embedding-3-small', catalog: CATALOG }),
    ).toEqual({ provider: 'SENTENCE_TRANSFORMERS', modelName: '' });
    expect(
      modelOnDataTypeSwitch({ from: 'TEXT', to: 'MULTIMODAL', provider: 'SENTENCE_TRANSFORMERS', modelName: 'all-MiniLM-L6-v2', catalog: CATALOG }),
    ).toEqual({ provider: 'SENTENCE_TRANSFORMERS', modelName: '' });
  });

  it('keeps a media model already typed', () => {
    expect(
      modelOnDataTypeSwitch({ from: 'TEXT', to: 'MULTIMODAL', provider: 'SENTENCE_TRANSFORMERS', modelName: 'Google/EmbeddingGemma-2', catalog: CATALOG }),
    ).toBeNull();
    // ...also when leaving media mode: EmbeddingGemma 2 embeds text too.
    expect(
      modelOnDataTypeSwitch({ from: 'MULTIMODAL', to: 'TEXT', provider: 'SENTENCE_TRANSFORMERS', modelName: 'google/embeddinggemma-2', catalog: CATALOG }),
    ).toBeNull();
  });

  it("restores the provider's default model when leaving media mode with an empty name", () => {
    expect(
      modelOnDataTypeSwitch({ from: 'MULTIMODAL', to: 'TEXT', provider: 'SENTENCE_TRANSFORMERS', modelName: ' ', catalog: CATALOG }),
    ).toEqual({ provider: 'SENTENCE_TRANSFORMERS', modelName: 'all-MiniLM-L6-v2' });
    expect(
      modelOnDataTypeSwitch({ from: 'MULTIMODAL', to: 'VECTOR', provider: 'SENTENCE_TRANSFORMERS', modelName: '', catalog: CATALOG }),
    ).toEqual({ provider: 'SENTENCE_TRANSFORMERS', modelName: 'all-MiniLM-L6-v2' });
  });

  it('changes nothing between the other data types, or without a change', () => {
    expect(
      modelOnDataTypeSwitch({ from: 'TEXT', to: 'VECTOR', provider: 'OPENAI', modelName: 'text-embedding-3-small', catalog: CATALOG }),
    ).toBeNull();
    expect(
      modelOnDataTypeSwitch({ from: 'MULTIMODAL', to: 'MULTIMODAL', provider: 'SENTENCE_TRANSFORMERS', modelName: '', catalog: CATALOG }),
    ).toBeNull();
  });
});
