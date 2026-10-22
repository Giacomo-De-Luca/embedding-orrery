import { describe, it, expect } from 'vitest';
import {
  commitMaxLengthDraft,
  describeModelLimit,
  embeddingModelInputFrom,
  parseMaxLengthDraft,
  type EmbeddingModelFields,
} from '../embeddingLimits';
import { EMBEDDING_PROVIDERS, isLocalProvider } from '@/lib/utils/embeddingProviders';

describe('parseMaxLengthDraft', () => {
  it('treats an empty field as the model limit', () => {
    expect(parseMaxLengthDraft('', 8192)).toEqual({ value: null, error: null });
    expect(parseMaxLengthDraft('   ', 8192)).toEqual({ value: null, error: null });
  });

  it('accepts positive integers up to the limit', () => {
    expect(parseMaxLengthDraft('2048', 8192)).toEqual({ value: 2048, error: null });
    expect(parseMaxLengthDraft('8192', 8192)).toEqual({ value: 8192, error: null });
  });

  it('rejects values above the model limit', () => {
    const parsed = parseMaxLengthDraft('9000', 8192);
    expect(parsed.value).toBeNull();
    expect(parsed.error).toContain('8,192');
  });

  it.each(['0', '-5', '1.5', 'abc', '12abc'])('rejects %s', (draft) => {
    const parsed = parseMaxLengthDraft(draft, 8192);
    expect(parsed.value).toBeNull();
    expect(parsed.error).not.toBeNull();
  });

  it('accepts any positive integer when the limit is unknown', () => {
    expect(parseMaxLengthDraft('100000', null)).toEqual({ value: 100000, error: null });
  });
});

describe('describeModelLimit', () => {
  it('names the limit and where it came from', () => {
    const text = describeModelLimit({ maxLength: 8192, source: 'catalog' }, false);
    expect(text).toContain('8,192 tokens');
    expect(text).toContain('catalog');
  });

  it('says when the limit is unknown', () => {
    expect(describeModelLimit({ maxLength: null, source: null }, false)).toContain('unknown');
  });

  it('says when the lookup is still running', () => {
    expect(describeModelLimit(null, true)).toContain('Looking up');
  });
});

describe('provider defaults', () => {
  it('local providers default to small batches, API providers to 100', () => {
    expect(EMBEDDING_PROVIDERS.SENTENCE_TRANSFORMERS.defaultBatchSize).toBe(32);
    expect(EMBEDDING_PROVIDERS.QWEN.defaultBatchSize).toBe(32);
    expect(EMBEDDING_PROVIDERS.BGE.defaultBatchSize).toBe(12);
    expect(EMBEDDING_PROVIDERS.GEMINI.defaultBatchSize).toBe(100);
  });

  it('only in-process providers are local', () => {
    expect(isLocalProvider('SENTENCE_TRANSFORMERS')).toBe(true);
    expect(isLocalProvider('QWEN')).toBe(true);
    expect(isLocalProvider('BGE')).toBe(true);
    expect(isLocalProvider('OLLAMA')).toBe(false);
    expect(isLocalProvider('OPENAI')).toBe(false);
  });
});

describe('embeddingModelInputFrom', () => {
  // A local checkpoint: not in the task-prefix table, so its prompt name applies.
  const fields: EmbeddingModelFields = {
    embeddingProvider: 'SENTENCE_TRANSFORMERS',
    modelName: '/models/gemma-checkpoint',
    ollamaUrl: 'http://localhost:11434',
    qwenTask: 'task',
    geminiTaskType: 'SEMANTIC_SIMILARITY',
    prompt: 'Retrieval-document',
    maxLength: 2048,
    taskPrefix: null,
  };
  const searchPrefixes = {
    documentPrefix: 'title: {title} | text: ',
    queryPrefix: 'task: search result | query: ',
  };

  it('sends the cap for local providers', () => {
    const input = embeddingModelInputFrom(fields);
    expect(input.maxLength).toBe(2048);
    expect(input.prompt).toBe('Retrieval-document');
  });

  it('task-prefix models send the document prefix as prompt and the query prefix as task', () => {
    const input = embeddingModelInputFrom({
      ...fields,
      modelName: 'google/embeddinggemma-2',
      taskPrefix: searchPrefixes,
    });
    expect(input.prompt).toBe(searchPrefixes.documentPrefix);
    expect(input.task).toBe(searchPrefixes.queryPrefix);
    expect(input.maxLength).toBe(2048);
  });

  it('Gemini Embedding 2 sends prefixes and no task type', () => {
    const input = embeddingModelInputFrom({
      ...fields,
      embeddingProvider: 'GEMINI',
      modelName: 'gemini-embedding-2',
      taskPrefix: searchPrefixes,
    });
    expect(input.taskType).toBeUndefined();
    expect(input.prompt).toBe(searchPrefixes.documentPrefix);
    expect(input.task).toBe(searchPrefixes.queryPrefix);
  });

  it('no task means neither prefix', () => {
    const input = embeddingModelInputFrom({
      ...fields,
      modelName: 'google/embeddinggemma-2',
      taskPrefix: { documentPrefix: '', queryPrefix: '' },
    });
    expect(input.prompt).toBeUndefined();
    expect(input.task).toBeUndefined();
  });

  it('sends one prompt: a prompt name or literal text as typed, or none when blank', () => {
    // A literal prompt's trailing space is part of it ("query: ").
    expect(embeddingModelInputFrom({ ...fields, prompt: 'query: ' }).prompt).toBe('query: ');
    expect(embeddingModelInputFrom({ ...fields, prompt: '   ' })).toHaveProperty('prompt', undefined);
    // Prompts are a sentence-transformers option.
    expect(embeddingModelInputFrom({ ...fields, embeddingProvider: 'OLLAMA' }).prompt).toBeUndefined();
  });

  it('omits an unset cap so the backend uses the model limit', () => {
    expect(embeddingModelInputFrom({ ...fields, maxLength: null })).not.toHaveProperty('maxLength');
  });

  it('never sends a cap for API providers', () => {
    const input = embeddingModelInputFrom({
      ...fields,
      embeddingProvider: 'GEMINI',
      modelName: 'gemini-embedding-001',
    });
    expect(input).not.toHaveProperty('maxLength');
    expect(input.taskType).toBe('SEMANTIC_SIMILARITY');
    expect(input.prompt).toBeUndefined();
  });
});

describe('commitMaxLengthDraft', () => {
  it('commits a valid draft', () => {
    expect(commitMaxLengthDraft('2048', 8192, null)).toBe(2048);
  });

  it('an empty draft means the model limit', () => {
    expect(commitMaxLengthDraft('', 8192, 2048)).toBeNull();
  });

  it('keeps the previous value when the draft is invalid', () => {
    // Typing "9000" against 8,192 must never leave "900" committed.
    expect(commitMaxLengthDraft('9000', 8192, 1024)).toBe(1024);
    expect(commitMaxLengthDraft('abc', 8192, null)).toBeNull();
  });
});
