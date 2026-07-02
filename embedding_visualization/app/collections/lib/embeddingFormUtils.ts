import type {
  EmbeddingProvider,
  GeminiTaskType,
  EmbeddingModelInput,
  EmbeddingJob,
  GenerateLlmLabelsInput,
  GenerateLlmLabelsResult,
} from '@/lib/graphql/mutations';

/**
 * Update a text template when embedding columns are toggled.
 * Removes `{col}` placeholders for unchecked columns and appends for newly checked ones.
 */
export function updateTextTemplate(
  currentTemplate: string,
  previousColumns: string[],
  newColumns: string[]
): string {
  const removed = previousColumns.filter(c => !newColumns.includes(c));
  const added = newColumns.filter(c => !previousColumns.includes(c));

  let updated = currentTemplate;

  for (const col of removed) {
    updated = updated.replace(
      new RegExp(`,\\s*\\{${col}\\}|\\{${col}\\}\\s*,\\s*|\\{${col}\\}`, 'g'),
      ''
    );
  }
  updated = updated.trim();

  for (const col of added) {
    if (updated) {
      updated = `${updated}, {${col}}`;
    } else {
      updated = `{${col}}`;
    }
  }

  return updated;
}

/**
 * Transform a stored embedding model config (Python snake_case) to the TS EmbeddingModelInput interface.
 */
export function transformStoredEmbeddingModel(
  storedModel: Record<string, unknown> | undefined
): EmbeddingModelInput | undefined {
  if (!storedModel) return undefined;

  return {
    provider: (storedModel.provider as string)?.toUpperCase() as EmbeddingProvider,
    modelName: storedModel.model_name as string,
    ollamaUrl: storedModel.ollama_url as string | undefined,
    task: storedModel.task as string | undefined,
    taskType: storedModel.task_type as GeminiTaskType | undefined,
    prompt: (storedModel.prompt ?? storedModel.prompt_name) as string | undefined,
  };
}

/**
 * Resume an interrupted LLM labeling job.
 * Returns `true` if this was an LLM labeling job (handled), `false` otherwise.
 */
export async function resumeLlmLabelingJob(
  job: EmbeddingJob,
  generateLlmLabels: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>,
  callbacks: {
    setLlmResumeJobId: (id: string | null) => void;
    refreshCollections: () => Promise<void>;
  }
): Promise<boolean> {
  if (job.jobType !== 'llm_labeling') return false;

  const llmConfig = job.config as {
    collection_name?: string;
    llm_provider?: string;
    llm_model?: string;
    label_scope?: string;
  };
  const jobId = `${llmConfig.collection_name || job.collectionName}_llm_labeling`;

  callbacks.setLlmResumeJobId(jobId);
  await generateLlmLabels({
    collectionName: llmConfig.collection_name || job.collectionName,
    llmProvider: llmConfig.llm_provider || 'gemini',
    llmModel: llmConfig.llm_model || 'gemini-3-flash-preview',
    labelScope: llmConfig.label_scope || 'both',
    resume: true,
  });
  callbacks.setLlmResumeJobId(null);
  await callbacks.refreshCollections();

  return true;
}
