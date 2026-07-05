import { describe, it, expect } from 'vitest';
import { getEmbedValidationIssues, buildEmbedSummary } from '../embedValidation';

const VALID_HF = {
  source: 'hf' as const,
  datasetId: 'dair-ai/emotion',
  collectionName: 'emotion_test',
  embeddingColumns: ['text'],
};

const VALID_LOCAL = {
  source: 'local-file' as const,
  filePath: '/data/things.parquet',
  collectionName: 'things',
  embeddingColumns: ['name'],
};

describe('getEmbedValidationIssues', () => {
  it('passes a valid HF config', () => {
    expect(getEmbedValidationIssues(VALID_HF)).toEqual([]);
  });

  it('requires org/dataset form for HF ids', () => {
    expect(getEmbedValidationIssues({ ...VALID_HF, datasetId: 'emotion' })).toContain(
      'Dataset ID must be in the form org/dataset'
    );
    expect(getEmbedValidationIssues({ ...VALID_HF, datasetId: undefined })).toContain(
      'Dataset ID must be in the form org/dataset'
    );
  });

  it('requires a present, absolute file path for local files', () => {
    expect(getEmbedValidationIssues({ ...VALID_LOCAL, filePath: '' })).toContain(
      'Provide a file path'
    );
    expect(getEmbedValidationIssues({ ...VALID_LOCAL, filePath: 'data.csv' })).toContain(
      'File path must be absolute (starting with /)'
    );
  });

  it('requires at least one embedding column, with a VECTOR variant', () => {
    expect(getEmbedValidationIssues({ ...VALID_HF, embeddingColumns: [] })).toContain(
      'Select at least one embedding column'
    );
    expect(
      getEmbedValidationIssues({ ...VALID_LOCAL, embeddingColumns: [], dataType: 'VECTOR' })
    ).toContain('Select a vector column');
  });

  it('requires a non-blank collection name', () => {
    expect(getEmbedValidationIssues({ ...VALID_HF, collectionName: '  ' })).toContain(
      'Provide a collection name'
    );
  });

  it('validates row range ordering', () => {
    expect(
      getEmbedValidationIssues({
        ...VALID_HF,
        portionStrategy: 'ROW_RANGE',
        rangeStart: 100,
        rangeEnd: 100,
      })
    ).toContain('Row range start must be before end');
    expect(
      getEmbedValidationIssues({
        ...VALID_HF,
        portionStrategy: 'ROW_RANGE',
        rangeStart: 0,
        rangeEnd: 100,
      })
    ).toEqual([]);
  });

  it('accumulates multiple issues', () => {
    expect(
      getEmbedValidationIssues({
        source: 'hf',
        datasetId: 'nope',
        collectionName: '',
        embeddingColumns: [],
      })
    ).toHaveLength(3);
  });
});

describe('buildEmbedSummary', () => {
  it('describes a first-N run with model and topics', () => {
    expect(
      buildEmbedSummary({
        collectionName: 'emotion_test',
        portionStrategy: 'FIRST_N',
        numRows: 1000,
        modelName: 'sentence-transformers/all-MiniLM-L6-v2',
        enableTopics: true,
      })
    ).toBe('emotion_test · first 1,000 rows · all-MiniLM-L6-v2 · topics on');
  });

  it('describes ALL with known total and vector imports', () => {
    expect(
      buildEmbedSummary({
        collectionName: 'vecs',
        portionStrategy: 'ALL',
        totalRows: 20000,
        dataType: 'VECTOR',
      })
    ).toBe('vecs · all 20,000 rows · pre-computed vectors');
  });

  it('describes a row range and handles a blank name', () => {
    expect(
      buildEmbedSummary({
        collectionName: ' ',
        portionStrategy: 'ROW_RANGE',
        rangeStart: 10,
        rangeEnd: 500,
      })
    ).toBe('unnamed collection · rows 10–500');
  });
});
