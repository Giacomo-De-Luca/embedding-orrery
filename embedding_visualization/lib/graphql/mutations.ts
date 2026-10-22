/**
 * GraphQL mutations and queries for dataset embedding
 */

import { gql } from '@apollo/client';

import type { ProbeInfo } from '../utils/probeFields';
import type { HookType } from '../utils/saeCollections';
import { PROBE_INFO_FIELDS } from './queries';

// ========== HuggingFace Dataset Queries ==========

/**
 * Get information about a HuggingFace dataset (configs, splits, features)
 */
export const GET_HF_DATASET_INFO = gql`
  query GetHFDatasetInfo($datasetId: String!) {
    huggingfaceDatasetInfo(datasetId: $datasetId) {
      datasetId
      description
      license
      defaultConfig
      error
      configs {
        name
        splits {
          name
          numRows
          numBytes
        }
        features {
          name
          dtype
          description
        }
      }
    }
  }
`;

/**
 * Get preview rows from a HuggingFace dataset
 */
export const GET_HF_DATASET_PREVIEW = gql`
  query GetHFDatasetPreview(
    $datasetId: String!
    $config: String
    $split: String! = "train"
    $nRows: Int! = 5
  ) {
    huggingfaceDatasetPreview(
      datasetId: $datasetId
      config: $config
      split: $split
      nRows: $nRows
    ) {
      datasetId
      config
      split
      columns
      rows
      totalRows
      error
    }
  }
`;

// ========== Local File Queries ==========

/**
 * Get information about a local file
 */
export const GET_LOCAL_FILE_INFO = gql`
  query GetLocalFileInfo($filePath: String!) {
    localFileInfo(filePath: $filePath) {
      filePath
      fileType
      columns
      numRows
      fileSizeBytes
      error
    }
  }
`;

/**
 * Get preview rows from a local file
 */
export const GET_LOCAL_FILE_PREVIEW = gql`
  query GetLocalFilePreview($filePath: String!, $nRows: Int! = 5) {
    localFilePreview(filePath: $filePath, nRows: $nRows) {
      filePath
      columns
      rows
      totalRows
      error
      columnKinds
    }
  }
`;

// ========== Embedding Mutations ==========

/**
 * Embed a HuggingFace dataset into ChromaDB
 */
export const EMBED_HUGGINGFACE_DATASET = gql`
  mutation EmbedHuggingfaceDataset($input: EmbedDatasetInput!) {
    embedHuggingfaceDataset(input: $input) {
      collectionName
      totalEmbedded
      embeddingDim
      device
      durationSeconds
      projectionsComputed
      error
      embeddingProvider
      embeddingModel
      warnings
    }
  }
`;

/**
 * Embed a local file into ChromaDB
 */
export const EMBED_LOCAL_FILE = gql`
  mutation EmbedLocalFile($input: EmbedLocalFileInput!) {
    embedLocalFile(input: $input) {
      collectionName
      totalEmbedded
      embeddingDim
      device
      durationSeconds
      projectionsComputed
      error
      embeddingProvider
      embeddingModel
      warnings
    }
  }
`;

/**
 * Re-embed an existing dataset with a different embedding model
 */
export const RE_EMBED_DATASET = gql`
  mutation ReEmbedDataset($input: ReEmbedDatasetInput!) {
    reEmbedDataset(input: $input) {
      collectionName
      totalEmbedded
      embeddingDim
      device
      durationSeconds
      projectionsComputed
      error
      embeddingProvider
      embeddingModel
      warnings
    }
  }
`;

/**
 * Cancel a running job by its progress job id (embedding, LLM labeling,
 * SAE document activations). Resolves false when no such job is running.
 */
export const CANCEL_JOB = gql`
  mutation CancelJob($jobId: String!) {
    cancelJob(jobId: $jobId)
  }
`;

/**
 * Remove an interrupted job record from the job state
 */
export const REMOVE_EMBEDDING_JOB = gql`
  mutation RemoveEmbeddingJob($collectionName: String!) {
    removeEmbeddingJob(collectionName: $collectionName)
  }
`;

/**
 * Delete a collection from ChromaDB
 */
export const DELETE_COLLECTION = gql`
  mutation DeleteCollection($collectionName: String!) {
    deleteCollection(collectionName: $collectionName)
  }
`;

/**
 * Update collection metadata
 */
export const UPDATE_COLLECTION_METADATA = gql`
  mutation UpdateCollectionMetadata($collectionName: String!, $metadata: JSON!) {
    updateCollectionMetadata(collectionName: $collectionName, metadata: $metadata) {
      name
      metadata
      error
    }
  }
`;

// ========== TypeScript Types ==========

export interface HFSplitInfo {
  name: string;
  numRows: number | null;
  numBytes: number | null;
}

export interface HFFeatureInfo {
  name: string;
  /** Value dtype ("string", "int64", ...), "class_label", "sequence", or a media kind: "image" / "audio" / "video" */
  dtype: string;
  description: string | null;
}

export interface HFConfigInfo {
  name: string;
  splits: HFSplitInfo[];
  features: HFFeatureInfo[];
}

export interface HFDatasetInfo {
  datasetId: string;
  description: string | null;
  license: string | null;
  defaultConfig: string | null;
  error: string | null;
  configs: HFConfigInfo[];
}

export interface HFDatasetPreview {
  datasetId: string;
  config: string | null;
  split: string;
  columns: string[];
  rows: Record<string, unknown>[];
  totalRows: number | null;
  error: string | null;
}

export interface LocalFileInfo {
  filePath: string;
  fileType: string;
  columns: string[];
  numRows: number;
  fileSizeBytes: number;
  error: string | null;
}

export interface LocalFilePreview {
  filePath: string;
  columns: string[];
  rows: Record<string, unknown>[];
  totalRows: number;
  error: string | null;
  /** Media columns of the preview: {column: "image" | "audio" | "video"} */
  columnKinds?: Record<string, string> | null;
}

export type PortionStrategy = 'FIRST_N' | 'RANDOM_SAMPLE' | 'ROW_RANGE' | 'ALL';

export interface PortionInput {
  strategy: PortionStrategy;
  n?: number;
  start?: number;
  end?: number;
  seed?: number;
}

/**
 * Embedding model provider.
 * - SENTENCE_TRANSFORMERS: Local models via sentence-transformers (no API key)
 * - OPENAI: OpenAI API (requires CHROMA_OPENAI_API_KEY env var)
 * - COHERE: Cohere API (requires CHROMA_COHERE_API_KEY env var)
 * - OLLAMA: Local Ollama server (no API key)
 * - HUGGINGFACE_API: HuggingFace Inference API (requires CHROMA_HUGGINGFACE_API_KEY env var)
 */
export type EmbeddingProvider =
  | 'SENTENCE_TRANSFORMERS'
  | 'OPENAI'
  | 'COHERE'
  | 'OLLAMA'
  | 'GEMINI'
  | 'BGE'
  | 'QWEN'
  | 'HUGGINGFACE_API';

/**
 * Gemini task types for embedding optimization.
 */
export type GeminiTaskType =
  | 'SEMANTIC_SIMILARITY'
  | 'CLASSIFICATION'
  | 'CLUSTERING'
  | 'RETRIEVAL_DOCUMENT'
  | 'RETRIEVAL_QUERY'
  | 'CODE_RETRIEVAL_QUERY'
  | 'QUESTION_ANSWERING'
  | 'FACT_VERIFICATION';

/**
 * Embedding model configuration.
 * Model names are free-form strings - any valid model for the provider works.
 *
 * Examples:
 * - SentenceTransformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2", "google/embeddinggemma-2" (text, images, audio, video)
 * - OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
 * - Cohere: "embed-english-v3.0", "embed-multilingual-v3.0"
 * - Ollama: "nomic-embed-text", "mxbai-embed-large"
 * - HuggingFace API: "sentence-transformers/all-MiniLM-L6-v2"
 * - QWEN: "Qwen/Qwen3-Embedding-0.6B" (supports task instruction for queries)
 * - Gemini: "gemini-embedding-001" (supports taskType for embedding optimization)
 */
export interface EmbeddingModelInput {
  provider: EmbeddingProvider;
  modelName: string;
  ollamaUrl?: string; // Ollama: server URL (default: http://localhost:11434)
  task?: string; // Query time only: QWEN's instruction, or a task-prefix model's query prefix
  taskType?: GeminiTaskType; // Gemini 001: Embedding optimization type (Gemini 2 takes none)
  prompt?: string; // ST prompt name or custom string; a task-prefix model's document prefix template
  maxLength?: number; // Local providers: input-length cap in tokens (omit = the model's limit)
}

/** A model suggestion for the embed forms (`embeddingModelCatalog` query; lib/utils/modelSuggestions.ts). */
export interface EmbeddingModelCatalogEntry {
  provider: EmbeddingProvider;
  modelName: string;
  description: string | null;
  /** What the model embeds: "text", "image", "audio", "video" */
  modalities: string[];
  /** Its sentence-transformers prompt names */
  prompts: string[];
  dimension: number | null;
  /** Data types whose default model this is ("multimodal") */
  defaultFor: string[];
  /** Listed in the backend catalog (false: only used by existing collections, e.g. local checkpoints) */
  suggested: boolean;
  collectionCount: number;
}

/** Media options of a multimodal embed (config/media_embedding.json defaults). */
export interface MediaOptions {
  imageMaxSoftTokens: number;
  videoFps: number;
  videoMaxFrames: number;
  videoMaxSoftTokens: number;
  maxAudioSeconds: number;
}

/** Per-job overrides of the media options; omitted (or null) values keep the defaults. */
export type MediaOptionsInput = { [K in keyof MediaOptions]?: MediaOptions[K] | null };

/** Defaults, limits and token constants of the media options (`mediaEmbeddingSettings` query). */
export interface MediaEmbeddingSettings {
  defaults: MediaOptions;
  /** Allowed soft-token budgets per image or video frame */
  softTokenBudgets: number[];
  videoFpsLimit: number;
  videoFramesLimit: number;
  audioSecondsLimit: number;
  /** One input's context window, shared by every modality */
  contextTokens: number;
  /** Allowance for the text, task prompt and special tokens */
  reservedTokens: number;
  audioTokensPerSecond: number;
  /** The multimodal default model */
  defaultModel: string;
}

export interface EmbedDatasetInput {
  datasetId: string;
  collectionName: string;
  config?: string;
  split?: string;
  /** Embed several splits into one collection in a single pass (adds `source_split` metadata). Takes precedence over `split`. */
  splits?: string[];
  /** TEXT or MULTIMODAL (VECTOR is local files only) */
  dataType?: DataType;
  /** Text columns (MULTIMODAL: optional, embedded with the media) */
  columns?: string[];
  /** MULTIMODAL: the Image / Audio / Video features to embed */
  mediaColumns?: string[];
  mediaOptions?: MediaOptionsInput;
  textTemplate?: string;
  idColumn?: string;
  portion?: PortionInput;
  metadataColumns?: string[];
  computeProjections?: boolean;
  batchSize?: number;
  /** MULTIMODAL: a sentence-transformers model; an empty name uses the backend's default (EmbeddingGemma 2) */
  embeddingModel?: EmbeddingModelInput;
  resume?: boolean; // Resume an interrupted job instead of starting fresh
  extractTopics?: boolean;
  topicConfig?: TopicConfigInput;
  /** Link the collection to this SAE once the run succeeds */
  saeLink?: SaeLinkInput | null;
}

/** SAE whose document activations a collection uses (set after a successful embed) */
export interface SaeLinkInput {
  saeModelId: string;
  saeId: string;
  /** Follow a successful embed with computeDocumentActivations (false: link only,
   * e.g. SAE feature collections). Stored with the job, so resume does the same. */
  computeActivations?: boolean;
}

/** MULTIMODAL: images, audio, video and rows mixing them with text ("Image / Audio / Video / Mixed"). */
export type DataType = 'TEXT' | 'MULTIMODAL' | 'VECTOR';

export interface EmbedLocalFileInput {
  filePath: string;
  collectionName: string;
  dataType?: DataType;
  /** Text columns (MULTIMODAL: optional, embedded with the media) */
  columns?: string[];
  textTemplate?: string;
  /** MULTIMODAL: the image / audio / video columns to embed */
  mediaColumns?: string[];
  mediaOptions?: MediaOptionsInput;
  vectorColumn?: string;
  idColumn?: string;
  metadataColumns?: string[];
  nRows?: number;
  sampleN?: number;
  sampleSeed?: number;
  computeProjections?: boolean;
  batchSize?: number;
  /** MULTIMODAL: a sentence-transformers model; an empty name uses the backend's default (EmbeddingGemma 2) */
  embeddingModel?: EmbeddingModelInput;
  resume?: boolean; // Resume an interrupted job instead of starting fresh
  extractTopics?: boolean;
  topicConfig?: TopicConfigInput;
  /** Link the collection to this SAE once the run succeeds */
  saeLink?: SaeLinkInput | null;
}

export interface ReEmbedDatasetInput {
  sourceDatasetName: string;
  collectionName: string;
  embeddingModel: EmbeddingModelInput;
  columns?: string[];       // Metadata fields to compose text from (omit = use existing document)
  textTemplate?: string;    // Template e.g. "{title}: {text}" (omit = concatenate columns)
  batchSize?: number;
  resume?: boolean;
  computeProjections?: boolean;
  extractTopics?: boolean;
  topicConfig?: TopicConfigInput;
  /** Link the collection to this SAE once the run succeeds */
  saeLink?: SaeLinkInput | null;
}

export interface EmbedDatasetResult {
  collectionName: string;
  totalEmbedded: number;
  embeddingDim: number;
  device: string;
  durationSeconds: number;
  projectionsComputed: boolean;
  error: string | null;
  embeddingProvider: string | null;
  embeddingModel: string | null;
  /** Non-cancelling issues, e.g. documents truncated at the length cap or media rows that could not be decoded */
  warnings?: string[];
}

export interface UpdateCollectionMetadataResult {
  name: string;
  metadata: Record<string, unknown>;
  error: string | null;
}

// ========== Job Types ==========

export type JobStatus = 'running' | 'interrupted' | 'completed';

export interface EmbeddingJob {
  collectionName: string;
  status: JobStatus;
  jobType: 'huggingface' | 'local_file' | 're_embed' | 'llm_labeling';
  itemsEmbedded: number;
  totalExpected: number;
  batchesCompleted: number;
  totalBatches: number;
  percentComplete: number;
  source: string;
  columns: string[] | null;
  embeddingModel: string | null;
  batchSize: number;
  startedAt: string;
  config: Record<string, unknown>;
}

export interface JobProgress {
  jobId: string;
  status: 'running' | 'completed' | 'failed';
  itemsProcessed: number;
  totalItems: number;
  currentBatch: number;
  totalBatches: number;
  error: string | null;
  message: string | null;
  /** Non-cancelling issue (e.g. documents truncated at the length cap); stays shown */
  warning?: string | null;
}

// ========== Topic Extraction Types ==========

/** Linkages of a topic merge tree (`topicMergeTree` query, tree strategies). */
export type TopicTreeLinkage = 'average' | 'ward' | 'density';

/** Topic reduction strategies; the legacy ones are kept only for comparison. */
export type ReductionStrategy = TopicTreeLinkage | 'legacy_ctfidf' | 'legacy_semantic';

export type ReductionMethod = 'fixed_n' | 'auto';

export interface TopicReductionInput {
  enabled: boolean;
  method: ReductionMethod;
  nTopics?: number;     // Topics to keep, noise not counted; required for "fixed_n"
  strategy?: ReductionStrategy | null; // null = the legacy strategy useCtfidf picks
  useCtfidf?: boolean;  // Legacy only, ignored when strategy is set
  distanceThreshold?: number; // "auto" + "average" only (backend default 0.7)
  absorbNoise?: boolean;      // "density" only
}

export interface TopicConfigInput {
  minTopicSize?: number;
  nKeywords?: number;
  useLlmLabels?: boolean;
  llmProvider?: string;
  llmModel?: string;
  projectionType?: string;
  clusteringMethod?: string;  // "hdbscan" | "kmeans" | "gmm" | "spectral"
  nClusters?: number;         // Required for kmeans, gmm, spectral
  clusterOn?: string;         // "projection" | "cluster_umap" | "embedding"
  clusterNComponents?: number;  // BERTopic UMAP dims (cluster_umap only)
  clusterMinDist?: number;      // BERTopic UMAP min_dist (cluster_umap only)
  clusterNNeighbors?: number;   // BERTopic UMAP n_neighbors (cluster_umap only)
  reduction?: TopicReductionInput;
}

// Import from canonical location and re-export to avoid duplication
import type { TopicInfo } from '../types/types';
export type { TopicKeyword, TopicInfo } from '../types/types';

export interface ExtractTopicsResult {
  collectionName: string;
  numTopics: number;
  numNoisePoints: number;
  topics: TopicInfo[];
  durationSeconds: number;
  error: string | null;
  numTopicsBeforeReduction: number | null;
  reductionApplied: boolean;
  // Stored quality scores keyed by level: { topic: {...}, subtopic: {...} }.
  // Only requested by GET_COLLECTION_TOPICS (absent on the extract mutation).
  qualityMetrics?: Record<string, Record<string, unknown>> | null;
}

// ========== Topic Quality Scoring ==========

export interface EvaluateTopicsInput {
  collectionName: string;
  level?: string;      // "topic" (default) | "subtopic"
  metrics?: string[];  // subset of dbcv/silhouette/diversity/coherence_cv/coherence_umass; omit = all
  sampleSize?: number; // silhouette subsample cap
}

export interface EvaluateTopicsResult {
  collectionName: string;
  level: string;
  metrics: Record<string, unknown> | null;
  durationSeconds: number;
  error: string | null;
}

/** Standalone reduction; always cuts the extraction's base topics (reversible). */
export interface ReduceTopicsInput {
  collectionName: string;
  method: ReductionMethod;
  nTopics?: number;       // Topics to keep, noise not counted; required for "fixed_n"
  strategy?: ReductionStrategy | null; // null = the legacy strategy useCtfidf picks
  useCtfidf?: boolean;    // Legacy only, ignored when strategy is set
  distanceThreshold?: number; // "auto" + "average" only (backend default 0.7)
  absorbNoise?: boolean;      // "density" only
  regenerateLabels: boolean;  // LLM-label merged topics (unmerged ones keep their label)
  llmProvider?: string;
  llmModel?: string;
}

export interface ReduceTopicsResult {
  collectionName: string;
  numTopicsBefore: number;
  numTopicsAfter: number;
  topics: TopicInfo[];
  topicMappings: Record<string, number>;
  durationSeconds: number;
  error: string | null;
  /** Noise documents moved into topics (density with absorbNoise). */
  absorbedNoise: number;
}

// ========== Topic Merge Trees (topicMergeTree query) ==========

/** A base topic of a merge tree (an original cluster of the extraction). */
export interface TopicTreeLeaf {
  topicId: number;
  label: string;
  size: number; // documents
}

/** One merge in scipy's linkage convention: indices below the leaf count are leaves, k + i is merge i. */
export interface TopicTreeMerge {
  left: number;
  right: number;
  height: number;
  size: number; // documents under the merged node
}

/** `topicMergeTree` result; parsed by `TopicTreeView.fromGraphQL` (lib/utils/topicTreeCut.ts). */
export interface TopicMergeTreeData {
  extractionId: string;
  linkage: TopicTreeLinkage;
  /** "topic": base ids in items' topic_id; "subtopic" (a reduced extraction): in subtopic_id. */
  baseLevel: 'topic' | 'subtopic';
  leaves: TopicTreeLeaf[];
  merges: TopicTreeMerge[];
  noiseCount: number;
  availableLinkages: TopicTreeLinkage[];
  /** Density with includeNoise: noise item ids and the merge row that absorbs each. */
  noiseItemIds: string[] | null;
  noiseMerges: number[] | null;
}

export interface GenerateLlmLabelsInput {
  collectionName: string;
  llmProvider: string;
  llmModel: string;
  labelScope: string;  // "both" | "topics_only" | "subtopics_only"
  resume: boolean;
}

export interface GenerateLlmLabelsResult {
  collectionName: string;
  topicsLabeled: number;
  subtopicsLabeled: number;
  totalTopics: number;
  totalSubtopics: number;
  durationSeconds: number;
  error: string | null;
}

// ========== Topic Label Renaming ==========

export const RENAME_TOPIC_LABEL = gql`
  mutation RenameTopicLabel($input: RenameTopicLabelInput!) {
    renameTopicLabel(input: $input) {
      collectionName
      topicId
      newLabel
      error
    }
  }
`;

export const REGENERATE_TOPIC_LABEL = gql`
  mutation RegenerateTopicLabel($input: RenameTopicLabelInput!) {
    regenerateTopicLabel(input: $input) {
      collectionName
      topicId
      newLabel
      error
    }
  }
`;

export interface RenameTopicLabelInput {
  collectionName: string;
  topicId: number;
  newLabel: string;
  isSubtopic?: boolean;
}

export interface RenameTopicLabelResult {
  collectionName: string;
  topicId: number;
  newLabel: string;
  error: string | null;
}

// ========== SAE Ingestion ==========

export const INGEST_SAE_FEATURES = gql`
  mutation IngestSaeFeatures($input: IngestSaeFeaturesInput!) {
    ingestSaeFeatures(input: $input) {
      modelId
      saeId
      recordsInserted
      durationSeconds
      error
    }
  }
`;

export const INGEST_SAE_ACTIVATIONS = gql`
  mutation IngestSaeActivations($input: IngestSaeActivationsInput!) {
    ingestSaeActivations(input: $input) {
      modelId
      saeId
      recordsInserted
      durationSeconds
      error
    }
  }
`;

export interface IngestSaeResult {
  modelId: string;
  saeId: string;
  recordsInserted: number;
  durationSeconds: number;
  error: string | null;
}

// ========== SAE Pipeline (On-Demand Provision) ==========

export const PREPARE_SAE_DATA = gql`
  mutation PrepareSaeData($input: PrepareSaeInput!) {
    prepareSaeData(input: $input) {
      modelId
      saeId
      featuresParquet
      activationsJsonl
      featuresInserted
      activationsInserted
      durationSeconds
      status
      error
      collectionName
      collectionItems
    }
  }
`;

export type SaeCollectionMode = 'DECODER_VECTORS' | 'LABEL_EMBEDDINGS';

export interface PrepareSaeInput {
  layer: number;
  width?: string;
  hookType?: string;
  modelSize?: string;
  variant?: string;
  /** SAE family: 'gemma' (Neuronpedia download) or 'qwen' (extract-only, label-free) */
  family?: 'gemma' | 'qwen';
  skipDownload?: boolean;
  includeActivations?: boolean;
  createCollection?: boolean;
  collectionMode?: SaeCollectionMode;
  embeddingModel?: EmbeddingModelInput;
  extractTopics?: boolean;
  topicConfig?: TopicConfigInput;
  deleteSourceFiles?: boolean;
}

export interface PrepareSaeResult {
  modelId: string;
  saeId: string;
  featuresParquet: string | null;
  activationsJsonl: string | null;
  featuresInserted: number;
  activationsInserted: number;
  durationSeconds: number;
  status: string; // "completed" | "already_downloaded" | "failed"
  error: string | null;
  collectionName: string | null;
  collectionItems: number;
}

export const UPDATE_SAE_FEATURE_LABEL = gql`
  mutation UpdateSaeFeatureLabel(
    $modelId: String!
    $saeId: String!
    $featureIndex: Int!
    $label: String!
  ) {
    updateSaeFeatureLabel(
      modelId: $modelId
      saeId: $saeId
      featureIndex: $featureIndex
      label: $label
    )
  }
`;

export const DELETE_SAE_DATA = gql`
  mutation DeleteSaeData($modelId: String!, $saeId: String!) {
    deleteSaeData(modelId: $modelId, saeId: $saeId)
  }
`;

// ========== Model Lifecycle ==========

export const LOAD_MODEL = gql`
  mutation LoadModel($checkpoint: String) {
    loadModel(checkpoint: $checkpoint) {
      loaded
      modelName
      device
    }
  }
`;

export const UNLOAD_MODEL = gql`
  mutation UnloadModel {
    unloadModel {
      loaded
      modelName
      device
    }
  }
`;

// ========== SAE Document Activations ==========

export const COMPUTE_DOCUMENT_ACTIVATIONS = gql`
  mutation ComputeDocumentActivations($input: ComputeDocumentActivationsInput!) {
    computeDocumentActivations(input: $input) {
      collectionName
      itemsProcessed
      totalItems
      durationSeconds
      error
    }
  }
`;

export interface ComputeDocumentActivationsResult {
  collectionName: string;
  itemsProcessed: number;
  totalItems: number;
  durationSeconds: number;
  error: string | null;
}

// ========== Embedding-space probes ==========

export const TRAIN_PROBE = gql`
  ${PROBE_INFO_FIELDS}
  mutation TrainProbe($input: TrainProbeInput!) {
    trainProbe(input: $input) {
      collectionName
      probe {
        ...ProbeInfoFields
      }
      durationSeconds
      error
    }
  }
`;

export const DELETE_PROBE = gql`
  mutation DeleteProbe($collectionName: String!, $targetField: String!, $kind: String!) {
    deleteProbe(collectionName: $collectionName, targetField: $targetField, kind: $kind)
  }
`;

export interface TrainProbeResult {
  collectionName: string;
  probe: ProbeInfo | null;
  durationSeconds: number;
  error: string | null;
}

export interface DocumentActivationResult {
  itemId: string;
  document: string | null;
  metadata: Record<string, unknown> | null;
  score: number;
  matchingFeatures: number;
  rowIndex: number | null;
}

export interface MatchedFeatureInfo {
  featureIndex: number;
  label: string | null;
  density: number | null;
  modelId: string;
  saeId: string;
}

export interface DocumentActivationSearchResponse {
  results: DocumentActivationResult[];
  totalResults: number;
  matchedFeatureCount: number;
  matchedFeatures: MatchedFeatureInfo[] | null;
  error: string | null;
}

// ========== SAE Prompt Highlight ==========

export interface PromptHighlightFeature {
  featureIndex: number;
  activation: number;
}

export interface PromptHighlightResult {
  features: PromptHighlightFeature[];
  error: string | null;
}

export const RUN_PROMPT_HIGHLIGHT = gql`
  mutation RunPromptHighlight($input: RunPromptHighlightInput!) {
    runPromptHighlight(input: $input) {
      features {
        featureIndex
        activation
      }
      error
    }
  }
`;

// ========== SAE Prompt Activations (per-token) ==========

export interface ActiveFeatureResult {
  index: number;
  activation: number;
  label: string;
  density: number | null;
}

export interface TokenFeaturesResult {
  token: string;
  position: number;
  features: ActiveFeatureResult[];
}

export interface LayerActivationsResult {
  layer: number;
  width: string;
  hookType: HookType;
  tokens: TokenFeaturesResult[];
}

export interface PromptActivationsResult {
  prompt: string;
  tokenStrings: string[];
  layers: LayerActivationsResult[];
  error: string | null;
}

export const RUN_PROMPT_ACTIVATIONS = gql`
  mutation RunPromptActivations($input: RunPromptActivationsInput!) {
    runPromptActivations(input: $input) {
      prompt
      tokenStrings
      layers {
        layer
        width
        hookType
        tokens {
          token
          position
          features {
            index
            activation
            label
            density
          }
        }
      }
      error
    }
  }
`;

// ========== Chat History ==========

export const CREATE_CHAT_SESSION = gql`
  mutation CreateChatSession($input: CreateChatSessionInput!) {
    createChatSession(input: $input) {
      id
      title
      config
      createdAt
      updatedAt
    }
  }
`;

export const SAVE_CHAT_MESSAGE = gql`
  mutation SaveChatMessage($input: SaveChatMessageInput!) {
    saveChatMessage(input: $input) {
      id
      sessionId
      role
      content
      parts
      steeringSnapshot
      createdAt
    }
  }
`;

export const DELETE_CHAT_SESSION = gql`
  mutation DeleteChatSession($id: String!) {
    deleteChatSession(id: $id)
  }
`;
