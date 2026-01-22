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
  embeddingModel?: EmbeddingModelInput;
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
  embeddingModel?: EmbeddingModelInput;
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
