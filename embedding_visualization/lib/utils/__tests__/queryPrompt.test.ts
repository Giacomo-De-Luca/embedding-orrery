import { describe, expect, it } from 'vitest';

import { resolveQueryPrompt } from '../queryPrompt';

describe('resolveQueryPrompt', () => {
  it('leaves auto to the backend, which knows the collection', () => {
    // The backend maps prompt names (Retrieval-document -> Retrieval-query) and
    // uses a task-prefix collection's stored query prefix.
    expect(resolveQueryPrompt('auto')).toBeNull();
  });

  it('sends nothing when no prompt is chosen', () => {
    expect(resolveQueryPrompt(null)).toBeNull();
    expect(resolveQueryPrompt(undefined)).toBeNull();
    expect(resolveQueryPrompt('')).toBeNull();
  });

  it('passes an explicit prompt through unchanged', () => {
    expect(resolveQueryPrompt('Retrieval-query')).toBe('Retrieval-query');
    expect(resolveQueryPrompt('task: question answering | query: ')).toBe(
      'task: question answering | query: ',
    );
  });
});
