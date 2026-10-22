/**
 * Prompt suggestions shared by the prompt pickers: the embed form's document
 * prompt (Collections page) and the Explore query prompt (Search panel). Both
 * accept any text; a sentence-transformers prompt name is sent as a name, any
 * other text as a literal prompt (the backend decides, see
 * documentation/EMBEDDING_TASK_PREFIXES.md).
 */
import type { SuggestionGroup, SuggestionItem } from './suggestions';

export interface PromptPreset {
  value: string;
  hint?: string;
}

/** The one preset list (prompt names of EmbeddingGemma-style models). */
export const PROMPT_PRESETS: readonly PromptPreset[] = [
  { value: 'Retrieval-query', hint: 'RAG search' },
  { value: 'Retrieval-document', hint: 'RAG storage' },
  { value: 'STS', hint: 'Sentence similarity' },
  { value: 'Classification' },
  { value: 'Clustering' },
];

/** Explore's "Auto": no prompt is sent and the backend uses the collection's own. */
export const AUTO_QUERY_PROMPT: SuggestionItem = {
  value: 'auto',
  label: 'Auto',
  detail: "matches the collection's prompt",
};

export const COMMON_PROMPTS_GROUP = 'Common prompts';
export const MODEL_PROMPTS_GROUP = 'Prompts of this model';
const DEFAULT_GROUP = 'Default';

export class PromptPresets {
  /** A prompt name as a suggestion, with the preset's hint when it has one. */
  static item(name: string): SuggestionItem {
    const hint = PROMPT_PRESETS.find((preset) => preset.value === name)?.hint;
    return hint ? { value: name, detail: hint } : { value: name };
  }

  /**
   * Document prompts of the embed form: the model's own prompt names when the
   * catalog lists them, else the shared presets.
   */
  static documentGroups(modelPrompts?: readonly string[] | null): SuggestionGroup[] {
    const names = [...new Set(modelPrompts ?? [])];
    if (names.length > 0) {
      return [{ label: MODEL_PROMPTS_GROUP, items: names.map(PromptPresets.item) }];
    }
    return [{ label: COMMON_PROMPTS_GROUP, items: PROMPT_PRESETS.map((p) => PromptPresets.item(p.value)) }];
  }

  /** Query prompts of Explore: Auto, then the shared presets. */
  static queryGroups(): SuggestionGroup[] {
    return [{ label: DEFAULT_GROUP, items: [AUTO_QUERY_PROMPT] }, ...PromptPresets.documentGroups(null)];
  }
}
