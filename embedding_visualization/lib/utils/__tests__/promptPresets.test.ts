import { describe, expect, it } from 'vitest';

import { resolveQueryPrompt } from '../queryPrompt';
import {
  AUTO_QUERY_PROMPT,
  COMMON_PROMPTS_GROUP,
  PROMPT_PRESETS,
  PromptPresets,
} from '../promptPresets';

describe('PromptPresets.queryGroups', () => {
  it('starts with Auto, then the shared presets with their hints', () => {
    const [auto, common] = PromptPresets.queryGroups();
    expect(auto.items).toEqual([AUTO_QUERY_PROMPT]);
    expect(AUTO_QUERY_PROMPT.value).toBe('auto');
    expect(common.label).toBe(COMMON_PROMPTS_GROUP);
    expect(common.items.map((item) => item.value)).toEqual(PROMPT_PRESETS.map((p) => p.value));
    expect(common.items.find((item) => item.value === 'Retrieval-document')?.detail).toBe('RAG storage');
  });

  it('Auto sends no prompt, so the backend picks the collection’s own', () => {
    expect(resolveQueryPrompt(AUTO_QUERY_PROMPT.value)).toBeNull();
  });
});
