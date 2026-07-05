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
  PortionStrategy,
  PortionInput,
  TopicConfigInput,
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

export interface ResumeJobDeps {
  embedHFDataset: (input: EmbedDatasetInput) => Promise<EmbedDatasetResult | null>;
  embedLocalFile: (input: EmbedLocalFileInput) => Promise<EmbedDatasetResult | null>;
  generateLlmLabels: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>;
  refreshCollections: () => Promise<void>;
  setLlmResumeJobId: (id: string | null) => void;
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
  selectedEmbeddingColumns: string[];
  selectedMetadataColumns: string[];
  textTemplate: string;
  idColumn: string;
  batchSize: number;
  embeddingModel?: EmbeddingModelInput;
  topicParams: { extractTopics?: boolean; topicConfig?: TopicConfigInput };
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
  }
): EmbedDatasetInput {
  const common: EmbedDatasetInput = {
    datasetId: source.datasetId,
    collectionName: form.collectionName,
    config: source.defaultConfig || undefined,
    columns: form.selectedEmbeddingColumns,
    textTemplate: form.textTemplate || undefined,
    idColumn: form.idColumn !== 'auto' ? form.idColumn : undefined,
    metadataColumns: mergeMetadataColumns(form.selectedEmbeddingColumns, form.selectedMetadataColumns),
    computeProjections: true,
    batchSize: form.batchSize,
    embeddingModel: form.embeddingModel,
    ...form.topicParams,
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
  return {
    filePath: source.filePath,
    collectionName: form.collectionName,
    dataType,
    columns: dataType === 'TEXT' ? form.selectedEmbeddingColumns : undefined,
    textTemplate: dataType === 'TEXT' ? (form.textTemplate || undefined) : undefined,
    imageColumn: dataType === 'IMAGE' ? form.selectedEmbeddingColumns[0] : undefined,
    vectorColumn: dataType === 'VECTOR' ? form.selectedEmbeddingColumns[0] : undefined,
    idColumn: form.idColumn !== 'auto' ? form.idColumn : undefined,
    metadataColumns: mergeMetadataColumns(form.selectedEmbeddingColumns, form.selectedMetadataColumns),
    nRows: portion.strategy === 'FIRST_N' ? portion.numRows : undefined,
    sampleN: portion.strategy === 'RANDOM_SAMPLE' ? portion.numRows : undefined,
    sampleSeed: portion.strategy === 'RANDOM_SAMPLE' ? portion.seed : undefined,
    computeProjections: true,
    batchSize: form.batchSize,
    embeddingModel: dataType === 'TEXT' ? form.embeddingModel : undefined,
    ...form.topicParams,
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
  };
}

/**
 * Resume any interrupted job, dispatching on its jobType. Unpacks the stored
 * backend config (snake_case) back into the matching embed mutation input
 * with `resume: true`.
 */
export async function resumeJob(job: EmbeddingJob, deps: ResumeJobDeps): Promise<void> {
  const handled = await resumeLlmLabelingJob(job, deps.generateLlmLabels, {
    setLlmResumeJobId: deps.setLlmResumeJobId,
    refreshCollections: deps.refreshCollections,
  });
  if (handled) return;

  const config = job.config as Record<string, unknown>;

  if (job.jobType === 'local_file') {
    const dataTypeValue = config.data_type as string | undefined;

    await deps.embedLocalFile({
      filePath: config.file_path as string,
      collectionName: job.collectionName,
      dataType: dataTypeValue?.toUpperCase() as DataType | undefined,
      columns: config.columns as string[] | undefined,
      textTemplate: config.text_template as string | undefined,
      imageColumn: config.image_column as string | undefined,
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
      resume: true,
    });
  } else {
    const storedPortion = config.portion as Record<string, unknown> | undefined;
    const portion = storedPortion ? {
      strategy: (storedPortion.strategy as string)?.toUpperCase() as PortionStrategy,
      n: storedPortion.n as number | undefined,
      start: storedPortion.start as number | undefined,
      end: storedPortion.end as number | undefined,
      seed: storedPortion.seed as number | undefined,
    } : undefined;

    await deps.embedHFDataset({
      datasetId: config.dataset_id as string,
      collectionName: job.collectionName,
      config: config.config as string | undefined,
      split: config.split as string | undefined,
      splits: config.splits as string[] | undefined,
      columns: config.columns as string[] | undefined,
      textTemplate: config.text_template as string | undefined,
      idColumn: config.id_column as string | undefined,
      metadataColumns: config.metadata_columns as string[] | undefined,
      portion,
      computeProjections: true,
      batchSize: config.batch_size as number | undefined,
      embeddingModel: transformStoredEmbeddingModel(
        config.embedding_model as Record<string, unknown> | undefined
      ),
      resume: true,
    });
  }

  await deps.refreshCollections();
}
