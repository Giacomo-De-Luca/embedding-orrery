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
 * Query to get complete collection data with projections
 */
export const GET_COLLECTION = gql`
  query GetCollection($name: String!) {
    collection(name: $name) {
      words
      definitions
      pos
      pca2d
      pca3d
      umap2d
      umap3d
      metadata {
        totalWords
        embeddingDim
        embeddingProvider
        embeddingModel
        timestamp
        pca2dVariance
        pca3dVariance
      }
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
  ) {
    semanticSearch(
      collectionName: $collectionName
      query: $query
      nResults: $nResults
      similarityMeasure: $similarityMeasure
      queryPrompt: $queryPrompt
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
  ) {
    semanticSearchById(
      collectionName: $collectionName
      itemId: $itemId
      nResults: $nResults
      similarityMeasure: $similarityMeasure
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
      topics {
        topicId
        keywords {
          word
          score
        }
        label
        count
      }
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
