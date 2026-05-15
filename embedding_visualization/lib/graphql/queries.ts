/**
 * GraphQL queries for embedding visualization
 */

import { gql } from '@apollo/client';

/**
 * Query to get all available collections
 */
export const GET_COLLECTIONS = gql`
  query GetCollections {
    collections {
      name
      count
      metadata
    }
  }
`;

/**
 * Query to get complete collection data with projections.
 * Supports selective projection loading — only requested types are parsed by the backend.
 * Non-requested projections return null.
 */
export const GET_COLLECTION_DATA = gql`
  query GetCollectionData($name: String!, $projectionTypes: [String!]) {
    collection(name: $name, projectionTypes: $projectionTypes) {
      ids
      documents
      itemMetadata
      availableFields
      pca2d
      pca3d
      umap2d
      umap3d
      metadata {
        totalItems
        embeddingDim
        timestamp
        pca2dVariance
        pca3dVariance
        sourceDataset
        sourceSplit
        sourceFile
        hasProjections
        embeddingProvider
        embeddingModel
        embeddingPrompt
        fieldAnalysis
        saeModelId
        saeId
      }
    }
  }
`;

/**
 * Full-text search across document content and/or metadata fields.
 * Uses ChromaDB native where_document for documents, Python filter for metadata.
 */
export const TEXT_SEARCH = gql`
  query TextSearch(
    $collectionName: String!
    $query: String!
    $fields: [String!]
    $mode: TextSearchMode
    $caseSensitive: Boolean
    $filters: [FilterInput!]
  ) {
    textSearch(
      collectionName: $collectionName
      query: $query
      fields: $fields
      mode: $mode
      caseSensitive: $caseSensitive
      filters: $filters
    ) {
      matches {
        id
        matchedField
        snippet
      }
      totalMatches
    }
  }
`;

/**
 * Query to find semantically similar items using a text query (embeds the query)
 */
export const SEMANTIC_SEARCH = gql`
  query SemanticSearch(
    $collectionName: String!
    $query: String!
    $nResults: Int = 10
    $similarityMeasure: SimilarityMeasure = COSINE
    $queryPrompt: String
    $filters: [FilterInput!]
  ) {
    semanticSearch(
      collectionName: $collectionName
      query: $query
      nResults: $nResults
      similarityMeasure: $similarityMeasure
      queryPrompt: $queryPrompt
      filters: $filters
    ) {
      id
      document
      metadata
      similarity
      distance
    }
  }
`;

/**
 * Query to find semantically similar items using an existing item's embedding (faster)
 */
export const SEMANTIC_SEARCH_BY_ID = gql`
  query SemanticSearchById(
    $collectionName: String!
    $itemId: String!
    $nResults: Int = 10
    $similarityMeasure: SimilarityMeasure = COSINE
    $filters: [FilterInput!]
  ) {
    semanticSearchById(
      collectionName: $collectionName
      itemId: $itemId
      nResults: $nResults
      similarityMeasure: $similarityMeasure
      filters: $filters
    ) {
      id
      document
      metadata
      similarity
      distance
    }
  }
`;

/**
 * Query to get a preview of collection items (first N items)
 */
export const GET_COLLECTION_PREVIEW = gql`
  query GetCollectionPreview(
    $collectionName: String!
    $limit: Int = 5
  ) {
    embeddings(
      collectionName: $collectionName
      limit: $limit
      offset: 0
      includeEmbeddings: false
      includeDocuments: true
      includeMetadata: true
    ) {
      id
      document
      metadata
    }
  }
`;

/**
 * Query to get embedding jobs with their progress and status
 */
export const GET_EMBEDDING_JOBS = gql`
  query GetEmbeddingJobs($status: String) {
    embeddingJobs(status: $status) {
      collectionName
      status
      jobType
      itemsEmbedded
      totalExpected
      batchesCompleted
      totalBatches
      percentComplete
      source
      columns
      embeddingModel
      batchSize
      startedAt
      config
    }
  }
`;

/**
 * Subscription to receive real-time progress updates for an embedding job
 */
export const EMBEDDING_PROGRESS_SUBSCRIPTION = gql`
  subscription EmbeddingProgress($jobId: String!) {
    embeddingProgress(jobId: $jobId) {
      jobId
      status
      itemsProcessed
      totalItems
      currentBatch
      totalBatches
      error
      message
    }
  }
`;

/**
 * Mutation to extract topics from an existing collection
 */
export const EXTRACT_TOPICS = gql`
  mutation ExtractTopics($collectionName: String!, $config: TopicConfigInput) {
    extractTopics(input: { collectionName: $collectionName, config: $config }) {
      collectionName
      numTopics
      numNoisePoints
      numTopicsBeforeReduction
      reductionApplied
      topics {
        topicId
        keywords {
          word
          score
        }
        label
        count
        subtopics
      }
      durationSeconds
      error
    }
  }
`;

/**
 * Mutation to reduce (merge) topics on an existing collection
 */
export const REDUCE_TOPICS = gql`
  mutation ReduceTopics($input: ReduceTopicsInput!) {
    reduceTopics(input: $input) {
      collectionName
      numTopicsBefore
      numTopicsAfter
      topics {
        topicId
        keywords {
          word
          score
        }
        label
        count
        subtopics
      }
      topicMappings
      durationSeconds
      error
    }
  }
`;

/**
 * Mutation to generate LLM labels for existing topics
 */
export const GENERATE_LLM_LABELS = gql`
  mutation GenerateLlmLabels($input: GenerateLlmLabelsInput!) {
    generateLlmLabels(input: $input) {
      collectionName
      topicsLabeled
      subtopicsLabeled
      totalTopics
      totalSubtopics
      durationSeconds
      error
    }
  }
`;

/**
 * Subscription to receive real-time progress updates for topic extraction
 */
export const TOPIC_EXTRACTION_PROGRESS_SUBSCRIPTION = gql`
  subscription TopicExtractionProgress($jobId: String!) {
    embeddingProgress(jobId: $jobId) {
      jobId
      status
      itemsProcessed
      totalItems
      currentBatch
      totalBatches
      error
      message
    }
  }
`;

// ========== Collection Topics Query ==========

/**
 * Get previously-extracted topics for a collection (loads from DB without re-extracting)
 */
export const GET_COLLECTION_TOPICS = gql`
  query GetCollectionTopics($collectionName: String!) {
    collectionTopics(collectionName: $collectionName) {
      collectionName
      numTopics
      numNoisePoints
      durationSeconds
      topics {
        topicId
        keywords {
          word
          score
        }
        label
        count
        subtopics
      }
      numTopicsBeforeReduction
      reductionApplied
    }
  }
`;

// ========== SAE (Sparse Autoencoder) Queries ==========

/**
 * List available SAE model/layer combinations with feature and activation counts
 */
export const GET_SAE_MODELS = gql`
  query GetSaeModels {
    saeModels {
      modelId
      saeId
      featureCount
      activationCount
    }
  }
`;

/**
 * Get all density values for a model/SAE pair (for histogram)
 */
export const GET_SAE_FEATURE_DENSITIES = gql`
  query GetSaeFeatureDensities($modelId: String!, $saeId: String!) {
    saeFeatureDensities(modelId: $modelId, saeId: $saeId)
  }
`;

/**
 * Get activations grouped by quantile bins (for polysemanticity analysis)
 */
export const GET_SAE_ACTIVATIONS_BY_QUANTILE = gql`
  query GetSaeActivationsByQuantile(
    $modelId: String!
    $saeId: String!
    $featureIndex: Int!
    $nQuantiles: Int = 5
    $perQuantileLimit: Int = 5
  ) {
    saeActivationsByQuantile(
      modelId: $modelId
      saeId: $saeId
      featureIndex: $featureIndex
      nQuantiles: $nQuantiles
      perQuantileLimit: $perQuantileLimit
    ) {
      quantile
      binMin
      binMax
      activations {
        id
        tokens
        values
        maxValue
        maxValueTokenIndex
      }
    }
  }
`;

/**
 * Get a single SAE feature with metadata and logits
 */
export const GET_SAE_FEATURE = gql`
  query GetSaeFeature($modelId: String!, $saeId: String!, $featureIndex: Int!) {
    saeFeature(modelId: $modelId, saeId: $saeId, featureIndex: $featureIndex) {
      modelId
      saeId
      featureIndex
      density
      label
      topLogits {
        token
        score
      }
      bottomLogits {
        token
        score
      }
    }
  }
`;

/**
 * Get top activations for a feature, ordered by max activation value
 */
export const GET_SAE_ACTIVATIONS = gql`
  query GetSaeActivations($modelId: String!, $saeId: String!, $featureIndex: Int!, $limit: Int = 20) {
    saeActivations(modelId: $modelId, saeId: $saeId, featureIndex: $featureIndex, limit: $limit) {
      id
      tokens
      values
      maxValue
      maxValueTokenIndex
    }
  }
`;

/**
 * Search SAE features by label text and/or density range
 */
export const SEARCH_SAE_FEATURES = gql`
  query SearchSaeFeatures(
    $modelId: String
    $saeId: String
    $saeIds: [String!]
    $query: String
    $minDensity: Float
    $maxDensity: Float
    $limit: Int = 50
    $offset: Int = 0
  ) {
    saeFeatureSearch(
      modelId: $modelId
      saeId: $saeId
      saeIds: $saeIds
      query: $query
      minDensity: $minDensity
      maxDensity: $maxDensity
      limit: $limit
      offset: $offset
    ) {
      feature {
        modelId
        saeId
        featureIndex
        density
        label
        topLogits {
          token
          score
        }
        bottomLogits {
          token
          score
        }
      }
      activationCount
    }
  }
`;

// ========== Streaming Chat Generation ==========

/**
 * Subscribe to streaming token generation for steered chat.
 * Backend: subscriptions.py → generate_stream
 */
export const GENERATE_STREAM = gql`
  subscription GenerateStream($input: GenerateStreamInput!) {
    generateStream(input: $input) {
      streamId
      tokenIndex
      tokenId
      text
      done
      error
    }
  }
`;

// ========== Model Lifecycle ==========

export const MODEL_STATUS = gql`
  query ModelStatus {
    modelStatus {
      loaded
      modelName
      device
    }
  }
`;
