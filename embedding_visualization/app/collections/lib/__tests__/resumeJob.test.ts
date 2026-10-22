import { describe, it, expect } from 'vitest';
import { resumePlanFor } from '../embeddingFormUtils';
import type { EmbeddingJob } from '@/lib/graphql/mutations';

function job(jobType: EmbeddingJob['jobType'], config: Record<string, unknown>): EmbeddingJob {
  return {
    collectionName: 'col',
    status: 'interrupted',
    jobType,
    itemsEmbedded: 5,
    totalExpected: 10,
    batchesCompleted: 1,
    totalBatches: 2,
    percentComplete: 50,
    source: 'src',
    columns: null,
    embeddingModel: null,
    batchSize: 100,
    startedAt: '',
    config,
  };
}

describe('resumePlanFor', () => {
  it('replays the stored GraphQL input with resume on, keeping topics and the SAE link', () => {
    const input = {
      datasetId: 'org/data',
      collectionName: 'col',
      columns: ['title'],
      extractTopics: true,
      topicConfig: { minTopicSize: 5 },
      saeLink: { saeModelId: 'm', saeId: 's' },
      resume: false,
    };

    expect(resumePlanFor(job('huggingface', { dataset_id: 'org/data', input }))).toEqual({
      kind: 'hf',
      input: { ...input, resume: true },
    });
  });

  it('replays a multimodal job with its media columns, media options and resolved model', () => {
    // The backend stores the input after filling the multimodal default model
    // (graphql_input_payload: camelCase keys, enum names, unset options null).
    const input = {
      filePath: '/data/media.parquet',
      collectionName: 'col',
      dataType: 'MULTIMODAL',
      columns: ['caption'],
      textTemplate: null,
      mediaColumns: ['img', 'clip'],
      mediaOptions: {
        imageMaxSoftTokens: null,
        videoFps: null,
        videoMaxFrames: 16,
        videoMaxSoftTokens: null,
        maxAudioSeconds: null,
      },
      vectorColumn: null,
      embeddingModel: { provider: 'SENTENCE_TRANSFORMERS', modelName: 'google/embeddinggemma-2' },
      resume: false,
    };
    expect(resumePlanFor(job('local_file', { input }))).toEqual({
      kind: 'local',
      input: { ...input, resume: true },
    });

    const hfInput = { datasetId: 'org/clips', collectionName: 'col', dataType: 'MULTIMODAL', mediaColumns: ['audio'] };
    expect(resumePlanFor(job('huggingface', { input: hfInput }))).toEqual({
      kind: 'hf',
      input: { ...hfInput, resume: true },
    });
  });

  it('drops imageColumn from inputs stored before the multimodal mode replaced it', () => {
    const input = { filePath: '/tmp/a.csv', collectionName: 'col', dataType: 'TEXT', columns: ['text'], imageColumn: null };
    const plan = resumePlanFor(job('local_file', { input }));
    expect(plan?.input).not.toHaveProperty('imageColumn');
    expect(plan?.input).toMatchObject({ dataType: 'TEXT', columns: ['text'], resume: true });
  });

  it('maps each embed job type to its mutation', () => {
    const input = { collectionName: 'col' };
    expect(resumePlanFor(job('local_file', { input }))?.kind).toBe('local');
    expect(resumePlanFor(job('re_embed', { input }))?.kind).toBe('reembed');
    expect(resumePlanFor(job('llm_labeling', { collection_name: 'col' }))).toBeNull();
  });

  it('keeps resuming jobs recorded before inputs were stored (legacy configs)', () => {
    const plan = resumePlanFor(
      job('local_file', {
        file_path: '/tmp/a.csv',
        data_type: 'text',
        columns: ['text'],
        embedding_model: { provider: 'sentence_transformers', model_name: 'all-MiniLM-L6-v2' },
        batch_size: 50,
      })
    );

    expect(plan).toMatchObject({
      kind: 'local',
      input: {
        filePath: '/tmp/a.csv',
        collectionName: 'col',
        dataType: 'TEXT',
        columns: ['text'],
        batchSize: 50,
        embeddingModel: { provider: 'SENTENCE_TRANSFORMERS', modelName: 'all-MiniLM-L6-v2' },
        resume: true,
      },
    });
  });

  it('converts a legacy HuggingFace portion back to its enum', () => {
    const plan = resumePlanFor(
      job('huggingface', { dataset_id: 'org/data', portion: { strategy: 'first_n', n: 10 } })
    );
    expect(plan).toMatchObject({ kind: 'hf', input: { portion: { strategy: 'FIRST_N', n: 10 } } });
  });
});
