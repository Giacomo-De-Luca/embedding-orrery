# Topic Reduction Frontend Integration Guide

## Overview

The backend now supports **topic reduction** functionality that allows users to merge similar topics after extraction. This guide explains how to integrate this feature into the frontend.

## Backend Changes Summary

### New GraphQL Types

**TopicReductionInput** (nested in TopicConfigInput):
```graphql
input TopicReductionInput {
  enabled: Boolean!  # Default: false
  method: String!  # "auto" or "fixed_n"
  nTopics: Int  # Required when method="fixed_n"
  useCtfidf: Boolean!  # Default: true (false for semantic)
}
```

**Extended TopicConfigInput**:
```graphql
input TopicConfigInput {
  minTopicSize: Int
  nKeywords: Int
  useLlmLabels: Boolean
  llmProvider: String
  llmModel: String
  projectionType: String

  # NEW: Reduction config
  reduction: TopicReductionInput
}
```

**Extended ExtractTopicsResult**:
```graphql
type ExtractTopicsResult {
  collectionName: String!
  numTopics: Int!
  numNoisePoints: Int!
  topics: [TopicInfo!]!
  durationSeconds: Float!
  error: String

  # NEW: Reduction tracking
  numTopicsBeforeReduction: Int
  reductionApplied: Boolean!
}
```

**New Standalone Mutation**:
```graphql
input ReduceTopicsInput {
  collectionName: String!
  method: String!  # "auto" or "fixed_n"
  nTopics: Int  # Required when method="fixed_n"
  useCtfidf: Boolean!  # Default: true

  # Re-labeling after reduction
  regenerateLabels: Boolean!
  llmProvider: String!
  llmModel: String!
}

type ReduceTopicsResult {
  collectionName: String!
  numTopicsBefore: Int!
  numTopicsAfter: Int!
  topics: [TopicInfo!]!
  topicMappings: JSON!  # {old_id: new_id}
  durationSeconds: Float!
  error: String
}

mutation reduceTopics(input: ReduceTopicsInput!): ReduceTopicsResult!
```

## Frontend Integration Tasks

### 1. Update GraphQL Queries/Mutations

**File:** `embedding_visualization/lib/graphql/queries.ts`

Add the new mutation:
```typescript
export const REDUCE_TOPICS = gql`
  mutation ReduceTopics($input: ReduceTopicsInput!) {
    reduceTopics(input: $input) {
      collectionName
      numTopicsBefore
      numTopicsAfter
      topics {
        topicId
        label
        keywords {
          word
          score
        }
        count
      }
      topicMappings
      durationSeconds
      error
    }
  }
`;
```

Update the `EXTRACT_TOPICS` mutation to include reduction fields:
```typescript
export const EXTRACT_TOPICS = gql`
  mutation ExtractTopics($input: ExtractTopicsInput!) {
    extractTopics(input: $input) {
      collectionName
      numTopics
      numNoisePoints
      # NEW fields:
      numTopicsBeforeReduction
      reductionApplied
      topics {
        topicId
        label
        keywords { word score }
        count
      }
      durationSeconds
      error
    }
  }
`;
```

### 2. Extend TopicExtractionCard UI

**File:** `embedding_visualization/app/test-embed/components/TopicExtractionCard.tsx`

Add reduction configuration UI in the extraction form:

```typescript
interface TopicReductionConfig {
  enabled: boolean;
  method: 'auto' | 'fixed_n';
  nTopics?: number;
  useCtfidf: boolean;
}

// Add to component state
const [reductionConfig, setReductionConfig] = useState<TopicReductionConfig>({
  enabled: false,
  method: 'auto',
  nTopics: undefined,
  useCtfidf: true
});

// Add UI section in the form (after LLM configuration)
<div className="space-y-3">
  <label className="flex items-center gap-2">
    <input
      type="checkbox"
      checked={reductionConfig.enabled}
      onChange={(e) => setReductionConfig({
        ...reductionConfig,
        enabled: e.target.checked
      })}
    />
    <span className="text-sm font-medium">Enable Topic Reduction</span>
  </label>

  {reductionConfig.enabled && (
    <div className="ml-6 space-y-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
      <div>
        <label className="text-sm font-medium">Reduction Method</label>
        <select
          value={reductionConfig.method}
          onChange={(e) => setReductionConfig({
            ...reductionConfig,
            method: e.target.value as 'auto' | 'fixed_n',
            nTopics: e.target.value === 'auto' ? undefined : reductionConfig.nTopics
          })}
          className="mt-1 block w-full rounded-md border p-2"
        >
          <option value="auto">Auto (HDBSCAN)</option>
          <option value="fixed_n">Fixed Target Count</option>
        </select>
      </div>

      {reductionConfig.method === 'fixed_n' && (
        <div>
          <label className="text-sm font-medium">Target Number of Topics</label>
          <input
            type="number"
            min="2"
            value={reductionConfig.nTopics || ''}
            onChange={(e) => setReductionConfig({
              ...reductionConfig,
              nTopics: parseInt(e.target.value)
            })}
            placeholder="e.g., 5"
            className="mt-1 block w-full rounded-md border p-2"
          />
        </div>
      )}

      <div>
        <label className="text-sm font-medium">Similarity Method</label>
        <select
          value={reductionConfig.useCtfidf ? 'ctfidf' : 'semantic'}
          onChange={(e) => setReductionConfig({
            ...reductionConfig,
            useCtfidf: e.target.value === 'ctfidf'
          })}
          className="mt-1 block w-full rounded-md border p-2"
        >
          <option value="ctfidf">c-TF-IDF (Fast, <2s)</option>
          <option value="semantic">Semantic Embeddings (Better Quality, 5-20s)</option>
        </select>
      </div>

      <p className="text-xs text-gray-600 dark:text-gray-400">
        {reductionConfig.method === 'auto'
          ? 'Automatically merge similar topics using HDBSCAN'
          : `Reduce to exactly ${reductionConfig.nTopics || 'N'} topics using hierarchical clustering`
        }
      </p>
    </div>
  )}
</div>
```

Update the mutation call to include reduction config:
```typescript
const handleExtractTopics = async () => {
  const input = {
    collectionName: selectedCollection,
    config: {
      minTopicSize,
      nKeywords,
      useLlmLabels,
      llmProvider,
      llmModel,
      projectionType,
      // NEW: Include reduction config
      reduction: reductionConfig.enabled ? {
        enabled: true,
        method: reductionConfig.method,
        nTopics: reductionConfig.nTopics,
        useCtfidf: reductionConfig.useCtfidf
      } : undefined
    }
  };

  const result = await extractTopics({ variables: { input } });

  // Show reduction results if applied
  if (result.data?.extractTopics.reductionApplied) {
    console.log('Topics reduced from',
      result.data.extractTopics.numTopicsBeforeReduction,
      'to',
      result.data.extractTopics.numTopics
    );
  }
};
```

### 3. Create Standalone Reduction Component (Optional)

**File:** `embedding_visualization/app/test-embed/components/TopicReductionPanel.tsx`

Create a new panel for post-processing topic reduction:

```typescript
'use client';

import { useState } from 'react';
import { useMutation } from '@apollo/client';
import { REDUCE_TOPICS } from '@/lib/graphql/queries';

interface TopicReductionPanelProps {
  collectionName: string;
  currentTopicCount: number;
}

export function TopicReductionPanel({
  collectionName,
  currentTopicCount
}: TopicReductionPanelProps) {
  const [method, setMethod] = useState<'auto' | 'fixed_n'>('auto');
  const [nTopics, setNTopics] = useState<number | undefined>(undefined);
  const [useCtfidf, setUseCtfidf] = useState(true);
  const [regenerateLabels, setRegenerateLabels] = useState(false);

  const [reduceTopics, { loading, error, data }] = useMutation(REDUCE_TOPICS);

  const handleReduce = async () => {
    const input = {
      collectionName,
      method,
      nTopics: method === 'fixed_n' ? nTopics : undefined,
      useCtfidf,
      regenerateLabels,
      llmProvider: 'gemini',
      llmModel: 'gemini-3-flash-preview'
    };

    await reduceTopics({ variables: { input } });
  };

  return (
    <div className="p-6 bg-white dark:bg-gray-900 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Reduce Topics</h3>

      <div className="space-y-4">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Current topics: <span className="font-semibold">{currentTopicCount}</span>
          </p>
        </div>

        <div>
          <label className="text-sm font-medium">Method</label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as 'auto' | 'fixed_n')}
            className="mt-1 block w-full rounded-md border p-2"
          >
            <option value="auto">Auto (HDBSCAN)</option>
            <option value="fixed_n">Fixed Target</option>
          </select>
        </div>

        {method === 'fixed_n' && (
          <div>
            <label className="text-sm font-medium">Target Topics</label>
            <input
              type="number"
              min="2"
              max={currentTopicCount}
              value={nTopics || ''}
              onChange={(e) => setNTopics(parseInt(e.target.value))}
              className="mt-1 block w-full rounded-md border p-2"
            />
          </div>
        )}

        <div>
          <label className="text-sm font-medium">Similarity Method</label>
          <select
            value={useCtfidf ? 'ctfidf' : 'semantic'}
            onChange={(e) => setUseCtfidf(e.target.value === 'ctfidf')}
            className="mt-1 block w-full rounded-md border p-2"
          >
            <option value="ctfidf">c-TF-IDF (Fast)</option>
            <option value="semantic">Semantic (Better)</option>
          </select>
        </div>

        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={regenerateLabels}
            onChange={(e) => setRegenerateLabels(e.target.checked)}
          />
          <span className="text-sm">Regenerate labels with LLM</span>
        </label>

        <button
          onClick={handleReduce}
          disabled={loading || (method === 'fixed_n' && !nTopics)}
          className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400"
        >
          {loading ? 'Reducing...' : 'Reduce Topics'}
        </button>

        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded text-sm text-red-600 dark:text-red-400">
            Error: {error.message}
          </div>
        )}

        {data?.reduceTopics && (
          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
            <p className="text-sm font-semibold text-green-700 dark:text-green-400">
              Success! Reduced from {data.reduceTopics.numTopicsBefore} to{' '}
              {data.reduceTopics.numTopicsAfter} topics in{' '}
              {data.reduceTopics.durationSeconds.toFixed(1)}s
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
```

### 4. Display Reduction Info in Collection Metadata

**File:** `embedding_visualization/app/components/CollectionInfo.tsx` (or similar)

Show reduction metadata when displaying collection info:

```typescript
interface CollectionMetadata {
  has_topics: boolean;
  topic_count: number;
  reduction_applied?: boolean;
  num_topics_before_reduction?: number;
  reduction_method?: string;
  reduction_target?: number;
}

// In the component:
{metadata.reduction_applied && (
  <div className="flex items-center gap-2 text-sm">
    <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded">
      Reduced
    </span>
    <span className="text-gray-600 dark:text-gray-400">
      {metadata.num_topics_before_reduction} → {metadata.topic_count} topics
      {metadata.reduction_method === 'auto' ? ' (auto)' : ` (target: ${metadata.reduction_target})`}
    </span>
  </div>
)}
```

### 5. Update TypeScript Types

**File:** `embedding_visualization/lib/types/graphql.ts` (or similar)

Add TypeScript types for the new GraphQL schema:

```typescript
export interface TopicReductionInput {
  enabled: boolean;
  method: 'auto' | 'fixed_n';
  nTopics?: number;
  useCtfidf: boolean;
}

export interface TopicConfigInput {
  minTopicSize?: number;
  nKeywords?: number;
  useLlmLabels?: boolean;
  llmProvider?: string;
  llmModel?: string;
  projectionType?: string;
  reduction?: TopicReductionInput;
}

export interface ExtractTopicsResult {
  collectionName: string;
  numTopics: number;
  numNoisePoints: number;
  topics: TopicInfo[];
  durationSeconds: number;
  error?: string;
  numTopicsBeforeReduction?: number;
  reductionApplied: boolean;
}

export interface ReduceTopicsInput {
  collectionName: string;
  method: 'auto' | 'fixed_n';
  nTopics?: number;
  useCtfidf: boolean;
  regenerateLabels: boolean;
  llmProvider: string;
  llmModel: string;
}

export interface ReduceTopicsResult {
  collectionName: string;
  numTopicsBefore: number;
  numTopicsAfter: number;
  topics: TopicInfo[];
  topicMappings: Record<number, number>;
  durationSeconds: number;
  error?: string;
}
```

## Usage Examples

### Example 1: Inline Reduction During Extraction

User workflow:
1. Go to "Extract Topics" section
2. Select collection
3. Configure extraction settings
4. **Enable reduction** checkbox
5. Choose method (auto or fixed-N)
6. If fixed-N, enter target count (e.g., 5)
7. Choose similarity method (c-TF-IDF or semantic)
8. Click "Extract Topics"

Result: Topics are extracted and immediately reduced in one operation.

### Example 2: Standalone Reduction (Post-Processing)

User workflow:
1. Collection already has topics extracted
2. Go to "Reduce Topics" panel
3. Select collection from dropdown
4. Choose reduction method
5. Configure settings
6. Click "Reduce Topics"

Result: Existing topics are merged, metadata updated, frontend refreshes.

## Performance Considerations

### Loading States

Both extraction and reduction can take time. Show appropriate loading states:

```typescript
{loading && (
  <div className="flex items-center gap-2">
    <Spinner />
    <span>
      {reductionConfig.enabled
        ? 'Extracting and reducing topics...'
        : 'Extracting topics...'}
    </span>
  </div>
)}
```

### Progress Updates

If using WebSocket subscriptions for progress, update the subscription to handle reduction messages:

```typescript
subscription {
  embeddingProgress(jobId: $jobId) {
    status
    message  # Can be "Reducing topics..." during reduction step
    itemsProcessed
    totalItems
    currentBatch
    totalBatches
  }
}
```

## Validation

Add client-side validation:

```typescript
const validateReductionConfig = (config: TopicReductionConfig, currentTopics: number) => {
  if (!config.enabled) return null;

  if (config.method === 'fixed_n') {
    if (!config.nTopics) {
      return 'Target number of topics is required for fixed-N method';
    }
    if (config.nTopics < 2) {
      return 'Target must be at least 2 topics';
    }
    if (config.nTopics >= currentTopics) {
      return `Target (${config.nTopics}) must be less than current topics (${currentTopics})`;
    }
  }

  return null;
};
```

## Testing Checklist

- [ ] Inline reduction during topic extraction works
- [ ] Standalone reduction mutation works
- [ ] Auto method merges similar topics
- [ ] Fixed-N method reduces to exact count
- [ ] c-TF-IDF similarity is fast (<2s for 50 topics)
- [ ] Semantic similarity is slower but works (5-20s)
- [ ] Noise cluster (-1) is never merged
- [ ] LLM re-labeling after reduction works
- [ ] Reduction metadata displayed correctly
- [ ] Error handling for invalid inputs
- [ ] Loading states shown during operation
- [ ] Results displayed clearly (before/after counts)
- [ ] Frontend refreshes after reduction

## Backend API Reference

### Inline Reduction

```graphql
mutation {
  extractTopics(input: {
    collectionName: "my_collection"
    config: {
      minTopicSize: 10
      reduction: {
        enabled: true
        method: "fixed_n"
        nTopics: 5
        useCtfidf: true
      }
    }
  }) {
    numTopics
    numTopicsBeforeReduction
    reductionApplied
  }
}
```

### Standalone Reduction

```graphql
mutation {
  reduceTopics(input: {
    collectionName: "my_collection"
    method: "auto"
    useCtfidf: false
    regenerateLabels: true
    llmProvider: "gemini"
    llmModel: "gemini-3-flash-preview"
  }) {
    numTopicsBefore
    numTopicsAfter
    durationSeconds
  }
}
```

## Performance Metrics

| Operation | Topics | Method | Time | Notes |
|-----------|--------|--------|------|-------|
| c-TF-IDF | 50→10 | fixed_n | <2s | Recommended default |
| c-TF-IDF | 100→20 | fixed_n | <10s | Fast for large sets |
| Semantic | 50→10 | fixed_n | 5-10s | Better quality |
| Semantic | 100→20 | fixed_n | 15-30s | High quality, slower |
| c-TF-IDF | 50→auto | auto | <2s | Automatic merging |
| Semantic | 50→auto | auto | 5-10s | Best quality auto |

## Questions?

For backend implementation details, see:
- [topic_reducer.py](interpretability_backend/backend/topic_extraction/topic_reducer.py)
- [topic_extraction_service.py](interpretability_backend/backend/services/topic_extraction_service.py)
- [CLAUDE.md](CLAUDE.md) - Topic Reduction section
