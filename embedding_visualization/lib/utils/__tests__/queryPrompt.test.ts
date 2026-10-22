import { describe, expect, it } from 'vitest';

import { resolveQueryPrompt, textQueryUnavailableReason } from '../queryPrompt';

describe('resolveQueryPrompt', () => {
  it('leaves auto to the backend, which knows the collection', () => {
    // The backend maps prompt names (Retrieval-document -> Retrieval-query) and
    // uses a task-prefix collection's stored query prefix.
    expect(resolveQueryPrompt('auto')).toBeNull();
  });

  it('sends nothing when no prompt is chosen (Auto)', () => {
    expect(resolveQueryPrompt(null)).toBeNull();
    expect(resolveQueryPrompt(undefined)).toBeNull();
    expect(resolveQueryPrompt('')).toBeNull();
    expect(resolveQueryPrompt('   ')).toBeNull();
  });

  it('keeps a typed prompt as typed, spaces included', () => {
    // The field passes raw text: a prefix's trailing space is part of it.
    expect(resolveQueryPrompt('task: search result | query: ')).toBe('task: search result | query: ');
    expect(resolveQueryPrompt(' Retrieval-query')).toBe(' Retrieval-query');
  });

  it('passes an explicit prompt through unchanged', () => {
    expect(resolveQueryPrompt('Retrieval-query')).toBe('Retrieval-query');
    expect(resolveQueryPrompt('task: question answering | query: ')).toBe(
      'task: question answering | query: ',
    );
  });
});

describe('textQueryUnavailableReason', () => {
  it('allows text queries on collections that record their provider and model', () => {
    expect(
      textQueryUnavailableReason({ embedding_provider: 'sentence_transformers', embedding_model: 'all-MiniLM-L6-v2' }),
    ).toBeNull();
  });

  it('explains why collections without a recorded provider and model take no text query', () => {
    // e.g. SAE decoder-vector collections: the backend (chromadb_client._embedding_function_for)
    // needs both to embed the query.
    const reason = textQueryUnavailableReason({ embedding_provider: undefined, embedding_model: undefined });
    expect(reason).toMatch(/no embedding model/);
    expect(reason).toMatch(/semantic \(text-query\) search/);
    expect(textQueryUnavailableReason({ embedding_provider: 'gemini', embedding_model: '' })).toBe(reason);
    expect(textQueryUnavailableReason({ embedding_provider: null, embedding_model: 'all-MiniLM-L6-v2' })).toBe(reason);
  });

  it('blocks nothing before the collection is known', () => {
    expect(textQueryUnavailableReason(null)).toBeNull();
    expect(textQueryUnavailableReason(undefined)).toBeNull();
  });
});
