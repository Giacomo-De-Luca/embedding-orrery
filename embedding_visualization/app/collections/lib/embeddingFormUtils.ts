import type {
  EmbeddingProvider,
  GeminiTaskType,
  EmbeddingModelInput,
  EmbeddingJob,
  GenerateLlmLabelsInput,
  GenerateLlmLabelsResult,
  EmbedDatasetInput,
  EmbedLocalFileInput,
  ReEmbedDatasetInput,
  EmbedDatasetResult,
  DataType,
  MediaOptionsInput,
  PortionStrategy,
  PortionInput,
  SaeLinkInput,
  TopicConfigInput,
} from '@/lib/graphql/mutations';
import { JobIds } from '@/lib/utils/jobIds';
import { EMBEDDING_PROVIDERS } from '@/lib/utils/embeddingProviders';
import type { ModelSuggestions } from '@/lib/utils/modelSuggestions';

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
  const jobId = JobIds.llmLabeling(llmConfig.collection_name || job.collectionName);

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

export interface ResumeJobDeps extends SaePostEmbedDeps {
  embedHFDataset: (input: EmbedDatasetInput) => Promise<EmbedDatasetResult | null>;
  embedLocalFile: (input: EmbedLocalFileInput) => Promise<EmbedDatasetResult | null>;
  reEmbedDataset: (input: ReEmbedDatasetInput) => Promise<EmbedDatasetResult | null>;
  generateLlmLabels: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>;
  refreshCollections: () => Promise<void>;
  setLlmResumeJobId: (id: string | null) => void;
}

// ========== Data type switch ==========

export interface DataTypeSwitch {
  from: DataType;
  to: DataType;
  provider: EmbeddingProvider;
  /** The model name as typed (empty in media mode means the backend default). */
  modelName: string;
  catalog: Pick<ModelSuggestions, 'find'>;
}

/**
 * The provider and model name a data-type switch moves the embed form to, or
 * null to keep them. Images, audio and video need a sentence-transformers media
 * model: switching to MULTIMODAL keeps a media model already typed, else
 * empties the name (the backend's default, EmbeddingGemma 2). Leaving
 * MULTIMODAL with an empty name restores the provider's default model.
 */
export function modelOnDataTypeSwitch({
  from,
  to,
  provider,
  modelName,
  catalog,
}: DataTypeSwitch): { provider: EmbeddingProvider; modelName: string } | null {
  if (from === to) return null;
  if (to === 'MULTIMODAL') {
    const typed = provider === 'SENTENCE_TRANSFORMERS' ? catalog.find(provider, modelName) : null;
    return typed?.modalities.some((m) => m !== 'text')
      ? null
      : { provider: 'SENTENCE_TRANSFORMERS', modelName: '' };
  }
  if (from === 'MULTIMODAL' && !modelName.trim()) {
    return { provider, modelName: EMBEDDING_PROVIDERS[provider].defaultModel };
  }
  return null;
}

// ========== Embed input builders (shared by HF / local / re-embed flows) ==========

/**
 * Shared metadata-column rule: a single embedded column is not duplicated
 * into metadata; multi-column embeds also store the embedded columns as
 * metadata to preserve the original data.
 */
export function mergeMetadataColumns(
  embeddingColumns: string[],
  metadataColumns: string[]
): string[] {
  return embeddingColumns.length === 1
    ? metadataColumns
    : [...metadataColumns, ...embeddingColumns];
}

export interface PortionParams {
  strategy: PortionStrategy;
  numRows: number;
  rangeStart?: number;
  rangeEnd?: number;
  seed: number;
}

/** Map form portion state to the GraphQL PortionInput (strategy-relevant fields only). */
export function buildPortionInput(p: PortionParams): PortionInput {
  return {
    strategy: p.strategy,
    n: p.strategy === 'FIRST_N' || p.strategy === 'RANDOM_SAMPLE' ? p.numRows : undefined,
    start: p.strategy === 'ROW_RANGE' ? p.rangeStart : undefined,
    end: p.strategy === 'ROW_RANGE' ? p.rangeEnd : undefined,
    seed: p.strategy === 'RANDOM_SAMPLE' ? p.seed : undefined,
  };
}

/** Form values common to every embed flow. */
export interface CommonEmbedFormValues {
  collectionName: string;
  /** Text columns (MULTIMODAL: the optional text embedded with the media) */
  selectedEmbeddingColumns: string[];
  selectedMetadataColumns: string[];
  textTemplate: string;
  idColumn: string;
  batchSize: number;
  /** MULTIMODAL: the image / audio / video columns */
  selectedMediaColumns?: string[];
  /** MULTIMODAL: the changed media options (MediaOptionsBudget.request), undefined for none */
  mediaOptions?: MediaOptionsInput;
  embeddingModel?: EmbeddingModelInput;
  topicParams: { extractTopics?: boolean; topicConfig?: TopicConfigInput };
  /** SAE to link the collection to after a successful embed (see saeLinkFromParams) */
  saeLink?: SaeLinkInput | null;
}

type ColumnFields = Pick<
  EmbedLocalFileInput,
  'columns' | 'textTemplate' | 'mediaColumns' | 'mediaOptions' | 'metadataColumns'
>;

/**
 * The column fields of an embed input. TEXT embeds the text columns;
 * MULTIMODAL embeds the media columns with the optional text columns, and
 * keeps media columns out of the text and metadata columns (the backend
 * stores each media cell's path or placeholder in the item metadata itself).
 */
function columnFields(form: CommonEmbedFormValues, dataType: DataType): ColumnFields {
  if (dataType !== 'MULTIMODAL') {
    return {
      columns: form.selectedEmbeddingColumns,
      textTemplate: form.textTemplate || undefined,
      metadataColumns: mergeMetadataColumns(form.selectedEmbeddingColumns, form.selectedMetadataColumns),
    };
  }
  const media = form.selectedMediaColumns ?? [];
  const text = form.selectedEmbeddingColumns.filter((c) => !media.includes(c));
  const options = form.mediaOptions;
  return {
    columns: text.length > 0 ? text : undefined,
    textTemplate: text.length > 0 ? (form.textTemplate || undefined) : undefined,
    mediaColumns: media,
    mediaOptions: options && Object.keys(options).length > 0 ? options : undefined,
    metadataColumns: mergeMetadataColumns(text, form.selectedMetadataColumns).filter((c) => !media.includes(c)),
  };
}

export function buildHFEmbedInput(
  form: CommonEmbedFormValues,
  source: {
    datasetId: string;
    defaultConfig?: string | null;
    selectedSplit: string;
    /** All split names, used when portion strategy is ALL (embed every split) */
    allSplits: string[];
    portion: PortionParams;
    /** TEXT (default) or MULTIMODAL */
    dataType?: DataType;
  }
): EmbedDatasetInput {
  const dataType = source.dataType ?? 'TEXT';
  const common: EmbedDatasetInput = {
    datasetId: source.datasetId,
    collectionName: form.collectionName,
    config: source.defaultConfig || undefined,
    dataType,
    ...columnFields(form, dataType),
    idColumn: form.idColumn !== 'auto' ? form.idColumn : undefined,
    computeProjections: true,
    batchSize: form.batchSize,
    embeddingModel: form.embeddingModel,
    ...form.topicParams,
    saeLink: form.saeLink ?? undefined,
  };

  if (source.portion.strategy === 'ALL') {
    // Embed every split into one collection in a single backend pass. The
    // backend tags each row with `source_split` and shares one ID
    // deduplicator across splits, so nothing gets overwritten.
    return {
      ...common,
      splits: source.allSplits.length > 0 ? source.allSplits : ['train'],
      portion: { strategy: 'ALL' },
    };
  }

  return {
    ...common,
    split: source.selectedSplit,
    portion: buildPortionInput(source.portion),
  };
}

export function buildLocalEmbedInput(
  form: CommonEmbedFormValues,
  source: {
    filePath: string;
    dataType: DataType;
    portion: Pick<PortionParams, 'strategy' | 'numRows' | 'seed'>;
  }
): EmbedLocalFileInput {
  const { dataType, portion } = source;
  const isVector = dataType === 'VECTOR';
  const fields = columnFields(form, dataType);
  return {
    filePath: source.filePath,
    collectionName: form.collectionName,
    dataType,
    ...fields,
    columns: isVector ? undefined : fields.columns,
    textTemplate: isVector ? undefined : fields.textTemplate,
    vectorColumn: isVector ? form.selectedEmbeddingColumns[0] : undefined,
    idColumn: form.idColumn !== 'auto' ? form.idColumn : undefined,
    nRows: portion.strategy === 'FIRST_N' ? portion.numRows : undefined,
    sampleN: portion.strategy === 'RANDOM_SAMPLE' ? portion.numRows : undefined,
    sampleSeed: portion.strategy === 'RANDOM_SAMPLE' ? portion.seed : undefined,
    computeProjections: true,
    batchSize: form.batchSize,
    embeddingModel: isVector ? undefined : form.embeddingModel,
    ...form.topicParams,
    saeLink: form.saeLink ?? undefined,
  };
}

export function buildReEmbedInput(
  form: CommonEmbedFormValues & { embeddingModel: EmbeddingModelInput },
  sourceDatasetName: string
): ReEmbedDatasetInput {
  // If only __document__ is selected, don't pass columns (use existing document text)
  const useExistingDoc =
    form.selectedEmbeddingColumns.length === 1 &&
    form.selectedEmbeddingColumns[0] === '__document__';

  return {
    sourceDatasetName,
    collectionName: form.collectionName,
    embeddingModel: form.embeddingModel,
    columns: useExistingDoc ? undefined : form.selectedEmbeddingColumns,
    textTemplate: useExistingDoc ? undefined : (form.textTemplate || undefined),
    batchSize: form.batchSize,
    computeProjections: true,
    ...form.topicParams,
    saeLink: form.saeLink ?? undefined,
  };
}

// ========== Post-embed SAE step (link + compute activations) ==========

/**
 * Parse the `modelId::saeId` composite used by the SAE selects (same format
 * as the manage tab's SaeLinkSection) into the metadata payload for
 * updateCollectionMetadata. Returns null for empty/malformed selections.
 */
export function parseSaeSelection(
  selection: string | null | undefined
): { sae_model_id: string; sae_id: string } | null {
  if (!selection) return null;
  const sep = selection.indexOf('::');
  if (sep <= 0) return null;
  const saeId = selection.slice(sep + 2);
  if (!saeId) return null;
  return { sae_model_id: selection.slice(0, sep), sae_id: saeId };
}

export interface SaePostEmbedParams {
  enabled: boolean;
  /** `modelId::saeId` composite, or null when nothing selected */
  selection: string | null;
}

/**
 * The embed input's `saeLink` for the optional post-embed SAE step: the
 * backend links the collection to it once the embed succeeds (and keeps it in
 * the job's stored input, so a resumed run links it too). Null when the step
 * is off or nothing valid is selected.
 */
export function saeLinkFromParams(params: SaePostEmbedParams): SaeLinkInput | null {
  if (!params.enabled) return null;
  const link = parseSaeSelection(params.selection);
  return link ? { saeModelId: link.sae_model_id, saeId: link.sae_id, computeActivations: true } : null;
}

export interface SaePostEmbedDeps {
  computeDocumentActivations: (
    collectionName: string
  ) => Promise<{ error?: string | null } | null>;
}

/**
 * Optional final step of the embed flow: after an embed whose SAE link asks
 * for activations (`computeActivations`) succeeds, compute the collection's
 * document activations — the backend linked the SAE as soon as every item was
 * embedded, so the compute knows which SAE to run. Link-only links (SAE
 * feature collections from the SAE tab) never compute.
 *
 * Returns whether the activations compute was reached. A compute-side error
 * still counts as run — the embed itself succeeded and the deps surface the
 * error (toast/error state); activations can be recomputed from the manage
 * tab. A failed or cancelled embed skips the step.
 */
export async function runPostEmbedSaeStep(
  embedResult: EmbedDatasetResult | null,
  saeLink: SaeLinkInput | null | undefined,
  deps: SaePostEmbedDeps
): Promise<boolean> {
  if (!saeLink?.computeActivations || !embedResult || embedResult.error) return false;
  await deps.computeDocumentActivations(embedResult.collectionName);
  return true;
}

// ========== Resume ==========

/** The embed mutation (and its input) that resumes an interrupted job. */
export type ResumePlan =
  | { kind: 'hf'; input: EmbedDatasetInput }
  | { kind: 'local'; input: EmbedLocalFileInput }
  | { kind: 'reembed'; input: ReEmbedDatasetInput };

/**
 * How to resume an interrupted embedding job, or null for job types resumed
 * elsewhere (LLM labeling). Jobs store the GraphQL input they were started
 * with (`config.input`, see BaseConfig.to_job_config on the backend), which is
 * replayed with `resume: true` — columns, topic settings and the SAE link
 * included. Jobs recorded before inputs were stored fall back to unpacking
 * the snake_case config summary.
 */
export function resumePlanFor(job: EmbeddingJob): ResumePlan | null {
  const config = job.config as Record<string, unknown>;
  const stored = config.input as Record<string, unknown> | null | undefined;

  switch (job.jobType) {
    case 'huggingface':
      return {
        kind: 'hf',
        input: { ...(stored ? (stored as unknown as EmbedDatasetInput) : legacyHFInput(job, config)), resume: true },
      };
    case 'local_file':
      return {
        kind: 'local',
        input: { ...(stored ? storedLocalInput(stored) : legacyLocalInput(job, config)), resume: true },
      };
    case 're_embed':
      return {
        kind: 'reembed',
        input: { ...(stored ? (stored as unknown as ReEmbedDatasetInput) : legacyReEmbedInput(job, config)), resume: true },
      };
    default:
      return null;
  }
}

/**
 * A stored local-file input as the current schema takes it: inputs stored
 * before the Image / Audio / Video / Mixed mode carry `imageColumn` (null for
 * text jobs), a field the schema no longer has.
 */
function storedLocalInput(stored: Record<string, unknown>): EmbedLocalFileInput {
  const { imageColumn: _removed, ...input } = stored;
  void _removed;
  return input as unknown as EmbedLocalFileInput;
}

function legacyHFInput(job: EmbeddingJob, config: Record<string, unknown>): EmbedDatasetInput {
  const storedPortion = config.portion as Record<string, unknown> | undefined;
  return {
    datasetId: config.dataset_id as string,
    collectionName: job.collectionName,
    config: config.config as string | undefined,
    split: config.split as string | undefined,
    splits: config.splits as string[] | undefined,
    columns: config.columns as string[] | undefined,
    textTemplate: config.text_template as string | undefined,
    idColumn: config.id_column as string | undefined,
    metadataColumns: config.metadata_columns as string[] | undefined,
    portion: storedPortion
      ? {
          strategy: (storedPortion.strategy as string)?.toUpperCase() as PortionStrategy,
          n: storedPortion.n as number | undefined,
          start: storedPortion.start as number | undefined,
          end: storedPortion.end as number | undefined,
          seed: storedPortion.seed as number | undefined,
        }
      : undefined,
    computeProjections: true,
    batchSize: config.batch_size as number | undefined,
    embeddingModel: transformStoredEmbeddingModel(
      config.embedding_model as Record<string, unknown> | undefined
    ),
  };
}

function legacyLocalInput(job: EmbeddingJob, config: Record<string, unknown>): EmbedLocalFileInput {
  return {
    filePath: config.file_path as string,
    collectionName: job.collectionName,
    dataType: (config.data_type as string | undefined)?.toUpperCase() as DataType | undefined,
    columns: config.columns as string[] | undefined,
    textTemplate: config.text_template as string | undefined,
    vectorColumn: config.vector_column as string | undefined,
    idColumn: config.id_column as string | undefined,
    metadataColumns: config.metadata_columns as string[] | undefined,
    nRows: config.n_rows as number | undefined,
    sampleN: config.sample_n as number | undefined,
    sampleSeed: config.sample_seed as number | undefined,
    computeProjections: true,
    batchSize: config.batch_size as number | undefined,
    embeddingModel: transformStoredEmbeddingModel(
      config.embedding_model as Record<string, unknown> | undefined
    ),
  };
}

function legacyReEmbedInput(job: EmbeddingJob, config: Record<string, unknown>): ReEmbedDatasetInput {
  // Legacy re-embed jobs stored only source/provider/model (no columns or
  // template), so they resume on the source dataset's document text.
  return {
    sourceDatasetName: config.source_dataset_name as string,
    collectionName: job.collectionName,
    embeddingModel: transformStoredEmbeddingModel({
      provider: config.embedding_provider,
      model_name: config.embedding_model,
    })!,
    computeProjections: true,
  };
}

/**
 * Resume any interrupted job: LLM labeling through its own mutation, embed
 * jobs by replaying their plan — then, when the stored input linked an SAE and
 * the resumed embed succeeded, compute the document activations it never got.
 */
export async function resumeJob(job: EmbeddingJob, deps: ResumeJobDeps): Promise<void> {
  const handled = await resumeLlmLabelingJob(job, deps.generateLlmLabels, {
    setLlmResumeJobId: deps.setLlmResumeJobId,
    refreshCollections: deps.refreshCollections,
  });
  if (handled) return;

  const plan = resumePlanFor(job);
  if (!plan) return;

  const result =
    plan.kind === 'hf'
      ? await deps.embedHFDataset(plan.input)
      : plan.kind === 'local'
        ? await deps.embedLocalFile(plan.input)
        : await deps.reEmbedDataset(plan.input);
  await deps.refreshCollections();

  if (await runPostEmbedSaeStep(result, plan.input.saeLink, deps)) {
    await deps.refreshCollections();
  }
}
