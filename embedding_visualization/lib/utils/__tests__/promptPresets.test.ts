import { describe, expect, it } from 'vitest';

import { resolveQueryPrompt } from '../queryPrompt';
import {
  AUTO_QUERY_PROMPT,
  COMMON_PROMPTS_GROUP,
  MODEL_PROMPTS_GROUP,
  PROMPT_PRESETS,
  PromptPresets,
} from '../promptPresets';

describe('PromptPresets.documentGroups', () => {
  it('offers the shared presets when the model lists no prompts', () => {
    const groups = PromptPresets.documentGroups(null);
    expect(groups.map((g) => g.label)).toEqual([COMMON_PROMPTS_GROUP]);
    expect(groups[0].items.map((item) => item.value)).toEqual(PROMPT_PRESETS.map((p) => p.value));
    expect(groups[0].items.find((item) => item.value === 'Retrieval-document')?.detail).toBe('RAG storage');
    expect(PromptPresets.documentGroups([])).toEqual(groups);
  });

  it("offers the model's own prompt names when the catalog lists them, once each", () => {
    const groups = PromptPresets.documentGroups(['query', 'Retrieval-document', 'query']);
    expect(groups).toEqual([
      {
        label: MODEL_PROMPTS_GROUP,
        items: [{ value: 'query' }, { value: 'Retrieval-document', detail: 'RAG storage' }],
      },
    ]);
  });
});

describe('PromptPresets.queryGroups', () => {
  it('starts with Auto, then the shared presets', () => {
    const [auto, common] = PromptPresets.queryGroups();
    expect(auto.items).toEqual([AUTO_QUERY_PROMPT]);
    expect(AUTO_QUERY_PROMPT.value).toBe('auto');
    expect(common.label).toBe(COMMON_PROMPTS_GROUP);
    expect(common.items.map((item) => item.value)).toEqual(PROMPT_PRESETS.map((p) => p.value));
  });

  it('Auto sends no prompt, so the backend picks the collection’s own', () => {
    expect(resolveQueryPrompt(AUTO_QUERY_PROMPT.value)).toBeNull();
  });
});
