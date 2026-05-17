/**
 * GraphQL mutations and queries for dataset embedding
 */

import { gql } from '@apollo/client';

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
    }
  }
`;

/**
 * Cancel a running embedding job
 */
export const CANCEL_EMBEDDING_JOB = gql`
  mutation CancelEmbeddingJob($collectionName: String!) {
    cancelEmbeddingJob(collectionName: $collectionName)
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
 * - SentenceTransformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"
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
  task?: string; // QWEN: Query instruction prefix (used at query time only)
  taskType?: GeminiTaskType; // Gemini: Embedding optimization type
  prompt?: string; // SentenceTransformers: Can be predefined name (e.g., "Retrieval-query") or custom string
}

export interface EmbedDatasetInput {
  datasetId: string;
  collectionName: string;
  config?: string;
  split?: string;
  columns?: string[];
  textTemplate?: string;
  idColumn?: string;
  portion?: PortionInput;
  metadataColumns?: string[];
  computeProjections?: boolean;
  batchSize?: number;
  embeddingModel?: EmbeddingModelInput;
  resume?: boolean; // Resume an interrupted job instead of starting fresh
  extractTopics?: boolean;
  topicConfig?: TopicConfigInput;
}

export type DataType = 'TEXT' | 'IMAGE' | 'VECTOR';

export interface EmbedLocalFileInput {
  filePath: string;
  collectionName: string;
  dataType?: DataType;
  columns?: string[];
  textTemplate?: string;
  imageColumn?: string;
  vectorColumn?: string;
  idColumn?: string;
  metadataColumns?: string[];
  nRows?: number;
  sampleN?: number;
  sampleSeed?: number;
  computeProjections?: boolean;
  batchSize?: number;
  embeddingModel?: EmbeddingModelInput;
  resume?: boolean; // Resume an interrupted job instead of starting fresh
  extractTopics?: boolean;
  topicConfig?: TopicConfigInput;
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
  jobType: 'huggingface' | 'local_file' | 'llm_labeling';
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
}

// ========== Topic Extraction Types ==========

export interface TopicReductionInput {
  enabled: boolean;
  method: string;       // "auto" or "fixed_n"
  nTopics?: number;     // Required when method="fixed_n"
  useCtfidf: boolean;   // true=c-TF-IDF (fast), false=semantic (better)
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
}

export interface ReduceTopicsInput {
  collectionName: string;
  method: string;         // "auto" or "fixed_n"
  nTopics?: number;       // Required when method="fixed_n"
  useCtfidf: boolean;     // true=c-TF-IDF, false=semantic
  regenerateLabels: boolean;
  llmProvider: string;
  llmModel: string;
}

export interface ReduceTopicsResult {
  collectionName: string;
  numTopicsBefore: number;
  numTopicsAfter: number;
  topics: TopicInfo[];
  topicMappings: Record<string, number>;
  durationSeconds: number;
  error: string | null;
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
  skipDownload?: boolean;
  includeActivations?: boolean;
  createCollection?: boolean;
  collectionMode?: SaeCollectionMode;
  extractTopics?: boolean;
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
      createdAt
    }
  }
`;

export const DELETE_CHAT_SESSION = gql`
  mutation DeleteChatSession($id: String!) {
    deleteChatSession(id: $id)
  }
`;
