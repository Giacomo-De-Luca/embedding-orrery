/**
 * Input-length cap of the embed form: parsing the field, describing the
 * model's limit, and building the GraphQL model input. Pure (no React).
 *
 * An empty field means "the model's own limit" (looked up by the backend's
 * `embeddingModelMaxLength` query); longer documents are truncated and the job
 * reports how many, without stopping. Only local providers take a cap.
 */
import type { EmbeddingModelInput, EmbeddingProvider, GeminiTaskType } from '@/lib/graphql/mutations';
import { isLocalProvider } from '@/lib/utils/embeddingProviders';
import type { TaskPrefixInput } from './taskPrefixes';

export interface ModelLimitInfo {
  maxLength: number | null;
  source: string | null;
}

export interface MaxLengthDraft {
  /** Committed cap; null = the model's limit. */
  value: number | null;
  error: string | null;
}

const SOURCE_LABELS: Record<string, string> = {
  catalog: 'the model catalog',
  sentence_transformers_config: "the model's sentence-transformers config",
  tokenizer: "the model's tokenizer",
  model_config: "the model's config",
};

export function parseMaxLengthDraft(draft: string, limit: number | null): MaxLengthDraft {
  const text = draft.trim();
  if (text === '') return { value: null, error: null };
  if (!/^\d+$/.test(text) || Number(text) < 1) {
    return { value: null, error: 'Enter a whole number of tokens, or leave empty for the model limit.' };
  }
  const value = Number(text);
  if (limit !== null && value > limit) {
    return { value: null, error: `Above the model limit of ${limit.toLocaleString('en-US')} tokens.` };
  }
  return { value, error: null };
}

/**
 * The cap to keep when the field loses focus: the draft if it is valid, else
 * the value from before editing. Drafts are never committed keystroke by
 * keystroke, so an invalid entry cannot leave a prefix of it ("900" of "9000")
 * behind.
 */
export function commitMaxLengthDraft(
  draft: string,
  limit: number | null,
  current: number | null,
): number | null {
  const parsed = parseMaxLengthDraft(draft, limit);
  return parsed.error === null ? parsed.value : current;
}

export function describeModelLimit(info: ModelLimitInfo | null, loading: boolean): string {
  if (loading && !info) return 'Looking up the model limit…';
  if (!info || info.maxLength === null) {
    return 'Model limit unknown: documents are embedded with the model\'s own truncation.';
  }
  const source = info.source ? ` (from ${SOURCE_LABELS[info.source] ?? info.source})` : '';
  return `Model limit: ${info.maxLength.toLocaleString('en-US')} tokens${source}. Longer documents are truncated, with a warning.`;
}

/** The form fields `embeddingModelInputFrom` reads (a subset of EmbeddingModelState). */
export interface EmbeddingModelFields {
  embeddingProvider: EmbeddingProvider;
  modelName: string;
  ollamaUrl: string;
  qwenTask: string;
  geminiTaskType: GeminiTaskType;
  /** Sentence-transformers prompt: a prompt name of the model, or literal prompt text. */
  prompt: string;
  maxLength: number | null;
  /** Set for models that read their task from a prefix (see taskPrefixes.ts). */
  taskPrefix: TaskPrefixInput | null;
}

export function embeddingModelInputFrom(fields: EmbeddingModelFields): EmbeddingModelInput {
  const provider = fields.embeddingProvider;
  const prefixes = fields.taskPrefix;
  const input: EmbeddingModelInput = prefixes
    ? {
        // Task-prefix models: the document prefix template and the query
        // prefix replace prompt names and the Gemini task type.
        provider,
        modelName: fields.modelName,
        prompt: prefixes.documentPrefix || undefined,
        task: prefixes.queryPrefix || undefined,
      }
    : {
        provider,
        modelName: fields.modelName,
        ollamaUrl: provider === 'OLLAMA' ? fields.ollamaUrl : undefined,
        task: provider === 'QWEN' ? fields.qwenTask : undefined,
        taskType: provider === 'GEMINI' ? fields.geminiTaskType : undefined,
        prompt: provider === 'SENTENCE_TRANSFORMERS' && fields.prompt.trim() ? fields.prompt : undefined,
      };
  if (isLocalProvider(provider) && fields.maxLength !== null) {
    input.maxLength = fields.maxLength;
  }
  return input;
}
