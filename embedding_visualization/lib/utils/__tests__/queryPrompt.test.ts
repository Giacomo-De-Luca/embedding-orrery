import { describe, expect, it } from 'vitest';

import { getQueryPromptName, resolveQueryPrompt } from '../queryPrompt';

describe('getQueryPromptName', () => {
  it('maps a document-time prompt to its query-time counterpart', () => {
    expect(getQueryPromptName('Retrieval-document')).toBe('Retrieval-query');
  });

  it('keeps prompts without a query counterpart', () => {
    expect(getQueryPromptName('Clustering')).toBe('Clustering');
  });

  it('returns null without a prompt', () => {
    expect(getQueryPromptName(null)).toBeNull();
    expect(getQueryPromptName(undefined)).toBeNull();
  });
});

describe('resolveQueryPrompt', () => {
  it('derives the prompt from the collection when set to auto', () => {
    expect(resolveQueryPrompt('auto', 'Retrieval-document')).toBe('Retrieval-query');
    expect(resolveQueryPrompt('auto', 'Clustering')).toBe('Clustering');
  });

  it('uses no prompt for auto when the collection has none', () => {
    expect(resolveQueryPrompt('auto', null)).toBeNull();
  });

  it('passes an explicit prompt through unchanged', () => {
    expect(resolveQueryPrompt('Instruct: find papers', 'Retrieval-document')).toBe(
      'Instruct: find papers',
    );
  });

  it('uses no prompt when none is selected', () => {
    expect(resolveQueryPrompt(null, 'Retrieval-document')).toBeNull();
    expect(resolveQueryPrompt(undefined, undefined)).toBeNull();
  });
});
