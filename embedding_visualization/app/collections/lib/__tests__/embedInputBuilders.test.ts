import { describe, it, expect } from 'vitest';
import {
  mergeMetadataColumns,
  buildPortionInput,
  buildHFEmbedInput,
  buildLocalEmbedInput,
  buildReEmbedInput,
  type CommonEmbedFormValues,
} from '../embeddingFormUtils';
import type { EmbeddingModelInput } from '@/lib/graphql/mutations';

const MODEL: EmbeddingModelInput = {
  provider: 'SENTENCE_TRANSFORMERS',
  modelName: 'all-MiniLM-L6-v2',
};

function form(partial: Partial<CommonEmbedFormValues> = {}): CommonEmbedFormValues {
  return {
    collectionName: 'my_coll',
    selectedEmbeddingColumns: ['text'],
    selectedMetadataColumns: ['label'],
    textTemplate: '{text}',
    idColumn: 'auto',
    batchSize: 100,
    embeddingModel: MODEL,
    topicParams: { extractTopics: true, topicConfig: undefined },
    ...partial,
  };
}

describe('mergeMetadataColumns', () => {
  it('does not duplicate a single embedded column into metadata', () => {
    expect(mergeMetadataColumns(['text'], ['label'])).toEqual(['label']);
  });

  it('preserves multi-column embeds as metadata', () => {
    expect(mergeMetadataColumns(['title', 'body'], ['label'])).toEqual(['label', 'title', 'body']);
  });
});

describe('buildPortionInput', () => {
  const base = { numRows: 500, rangeStart: 10, rangeEnd: 90, seed: 7 };

  it('maps only strategy-relevant fields', () => {
    expect(buildPortionInput({ strategy: 'FIRST_N', ...base })).toEqual({
      strategy: 'FIRST_N', n: 500, start: undefined, end: undefined, seed: undefined,
    });
    expect(buildPortionInput({ strategy: 'RANDOM_SAMPLE', ...base })).toEqual({
      strategy: 'RANDOM_SAMPLE', n: 500, start: undefined, end: undefined, seed: 7,
    });
    expect(buildPortionInput({ strategy: 'ROW_RANGE', ...base })).toEqual({
      strategy: 'ROW_RANGE', n: undefined, start: 10, end: 90, seed: undefined,
    });
  });
});

describe('buildHFEmbedInput', () => {
  const source = {
    datasetId: 'dair-ai/emotion',
    defaultConfig: 'default',
    selectedSplit: 'train',
    allSplits: ['train', 'test'],
    portion: { strategy: 'FIRST_N' as const, numRows: 1000, seed: 42 },
  };

  it('builds a single-split input with portion', () => {
    const input = buildHFEmbedInput(form(), source);
    expect(input.split).toBe('train');
    expect(input.splits).toBeUndefined();
    expect(input.portion).toEqual({ strategy: 'FIRST_N', n: 1000, start: undefined, end: undefined, seed: undefined });
    expect(input.metadataColumns).toEqual(['label']);
    expect(input.idColumn).toBeUndefined(); // auto
    expect(input.extractTopics).toBe(true);
    expect(input.computeProjections).toBe(true);
  });

  it('embeds all splits in one pass for ALL', () => {
    const input = buildHFEmbedInput(form(), {
      ...source,
      portion: { strategy: 'ALL', numRows: 0, seed: 42 },
    });
    expect(input.splits).toEqual(['train', 'test']);
    expect(input.split).toBeUndefined();
    expect(input.portion).toEqual({ strategy: 'ALL' });
  });

  it('falls back to train when no splits are known', () => {
    const input = buildHFEmbedInput(form(), {
      ...source,
      allSplits: [],
      portion: { strategy: 'ALL', numRows: 0, seed: 42 },
    });
    expect(input.splits).toEqual(['train']);
  });

  it('passes explicit id columns and merges multi-column metadata', () => {
    const input = buildHFEmbedInput(
      form({ idColumn: 'doc_id', selectedEmbeddingColumns: ['title', 'body'] }),
      source
    );
    expect(input.idColumn).toBe('doc_id');
    expect(input.metadataColumns).toEqual(['label', 'title', 'body']);
  });
});

describe('buildLocalEmbedInput', () => {
  const source = {
    filePath: '/data/things.parquet',
    dataType: 'TEXT' as const,
    portion: { strategy: 'FIRST_N' as const, numRows: 1000, seed: 42 },
  };

  it('routes TEXT columns and model', () => {
    const input = buildLocalEmbedInput(form(), source);
    expect(input.columns).toEqual(['text']);
    expect(input.textTemplate).toBe('{text}');
    expect(input.imageColumn).toBeUndefined();
    expect(input.vectorColumn).toBeUndefined();
    expect(input.embeddingModel).toBe(MODEL);
    expect(input.nRows).toBe(1000);
    expect(input.sampleN).toBeUndefined();
  });

  it('routes VECTOR mode to vectorColumn and drops model/template', () => {
    const input = buildLocalEmbedInput(form({ selectedEmbeddingColumns: ['emb'] }), {
      ...source,
      dataType: 'VECTOR',
    });
    expect(input.vectorColumn).toBe('emb');
    expect(input.columns).toBeUndefined();
    expect(input.textTemplate).toBeUndefined();
    expect(input.embeddingModel).toBeUndefined();
  });

  it('routes IMAGE mode to imageColumn', () => {
    const input = buildLocalEmbedInput(form({ selectedEmbeddingColumns: ['img'] }), {
      ...source,
      dataType: 'IMAGE',
    });
    expect(input.imageColumn).toBe('img');
    expect(input.columns).toBeUndefined();
  });

  it('maps random sampling fields', () => {
    const input = buildLocalEmbedInput(form(), {
      ...source,
      portion: { strategy: 'RANDOM_SAMPLE', numRows: 250, seed: 7 },
    });
    expect(input.nRows).toBeUndefined();
    expect(input.sampleN).toBe(250);
    expect(input.sampleSeed).toBe(7);
  });
});

describe('buildReEmbedInput', () => {
  it('omits columns/template for the __document__ special case', () => {
    const input = buildReEmbedInput(
      { ...form({ selectedEmbeddingColumns: ['__document__'] }), embeddingModel: MODEL },
      'source_ds'
    );
    expect(input.sourceDatasetName).toBe('source_ds');
    expect(input.columns).toBeUndefined();
    expect(input.textTemplate).toBeUndefined();
  });

  it('passes explicit columns and template otherwise', () => {
    const input = buildReEmbedInput(
      { ...form({ selectedEmbeddingColumns: ['title', 'body'], textTemplate: '{title}: {body}' }), embeddingModel: MODEL },
      'source_ds'
    );
    expect(input.columns).toEqual(['title', 'body']);
    expect(input.textTemplate).toBe('{title}: {body}');
  });
});
