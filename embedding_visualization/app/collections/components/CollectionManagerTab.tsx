'use client';

import { Card, CardContent } from '@/lib/ui-primitives/card';
import { FolderOpen } from 'lucide-react';
import type { UpdateCollectionMetadataResult, TopicConfigInput, ExtractTopicsResult, ReduceTopicsInput, ReduceTopicsResult, GenerateLlmLabelsInput, GenerateLlmLabelsResult, ComputeDocumentActivationsResult } from '@/lib/graphql/mutations';
import { CollectionListPane } from './manage/CollectionListPane';
import { CollectionDetailPane } from './manage/CollectionDetailPane';

export interface CollectionInfo {
  name: string;
  numItems: number;
  embeddingProvider?: string | null;
  embeddingModel?: string | null;
  metadata?: Record<string, unknown>;
}

interface CollectionManagerTabProps {
  collections: CollectionInfo[];
  collectionsLoading: boolean;
  /** Controlled selection (owned by the page, synced to ?collection=) */
  selectedCollection: string | null;
  onSelectCollection: (name: string | null) => void;
  refreshCollections: () => Promise<void>;
  deleteCollection: (name: string) => Promise<boolean>;
  updateCollectionMetadata: (
    collectionName: string,
    metadata: Record<string, unknown>
  ) => Promise<UpdateCollectionMetadataResult | null>;
  onCollectionDeleted?: () => void;
  extractTopics: (collectionName: string, config?: TopicConfigInput) => Promise<ExtractTopicsResult | null>;
  topicsLoading: boolean;
  lastTopicsResult: ExtractTopicsResult | null;
  error: string | null;
  clearError: () => void;
  // Topic reduction
  reduceTopics: (input: ReduceTopicsInput) => Promise<ReduceTopicsResult | null>;
  reduceTopicsLoading: boolean;
  lastReduceResult: ReduceTopicsResult | null;
  // LLM label generation
  generateLlmLabels: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>;
  llmLabelsLoading: boolean;
  lastLlmLabelsResult: GenerateLlmLabelsResult | null;
  // Topic label renaming
  renameTopicLabel: (collectionName: string, topicId: number, newLabel: string, isSubtopic?: boolean) => Promise<{ error?: string | null } | null>;
  regenerateTopicLabel: (collectionName: string, topicId: number, llmConfig?: string) => Promise<{ error?: string | null; newLabel?: string } | null>;
  // Load previously-extracted topics
  fetchCollectionTopics: (collectionName: string) => Promise<ExtractTopicsResult | null>;
  // Document activations (batch SAE inference)
  computeDocumentActivations?: (collectionName: string) => Promise<ComputeDocumentActivationsResult | null>;
  docActivationsLoading?: boolean;
  lastDocActivationsResult?: ComputeDocumentActivationsResult | null;
}

/**
 * Manage tab shell: master–detail layout. Searchable collection list on the
 * left; the selected collection's detail (preview, metadata, SAE, topics) on
 * the right. Selection state lives on the page and round-trips through the
 * URL.
 */
export function CollectionManagerTab({
  collections,
  collectionsLoading,
  selectedCollection,
  onSelectCollection,
  refreshCollections,
  deleteCollection,
  updateCollectionMetadata,
  onCollectionDeleted,
  extractTopics,
  topicsLoading,
  lastTopicsResult,
  error,
  clearError,
  reduceTopics,
  reduceTopicsLoading,
  lastReduceResult,
  generateLlmLabels,
  llmLabelsLoading,
  lastLlmLabelsResult,
  renameTopicLabel,
  regenerateTopicLabel,
  fetchCollectionTopics,
  computeDocumentActivations,
  docActivationsLoading,
  lastDocActivationsResult,
}: CollectionManagerTabProps) {
  const selectedCollectionInfo = collections.find(c => c.name === selectedCollection);

  return (
    <div className="grid grid-cols-1 md:grid-cols-[280px_minmax(0,1fr)] gap-6 items-start">
      <CollectionListPane
        collections={collections}
        collectionsLoading={collectionsLoading}
        selectedCollection={selectedCollection}
        onSelectCollection={onSelectCollection}
        onRefresh={refreshCollections}
      />

      {selectedCollectionInfo ? (
        <CollectionDetailPane
          key={selectedCollectionInfo.name}
          collection={selectedCollectionInfo}
          refreshCollections={refreshCollections}
          deleteCollection={deleteCollection}
          updateCollectionMetadata={updateCollectionMetadata}
          onDeleted={() => {
            onSelectCollection(null);
            onCollectionDeleted?.();
          }}
          extractTopics={extractTopics}
          topicsLoading={topicsLoading}
          lastTopicsResult={lastTopicsResult}
          error={error}
          clearError={clearError}
          reduceTopics={reduceTopics}
          reduceTopicsLoading={reduceTopicsLoading}
          lastReduceResult={lastReduceResult}
          generateLlmLabels={generateLlmLabels}
          llmLabelsLoading={llmLabelsLoading}
          lastLlmLabelsResult={lastLlmLabelsResult}
          renameTopicLabel={renameTopicLabel}
          regenerateTopicLabel={regenerateTopicLabel}
          fetchCollectionTopics={fetchCollectionTopics}
          computeDocumentActivations={computeDocumentActivations}
          docActivationsLoading={docActivationsLoading}
          lastDocActivationsResult={lastDocActivationsResult}
        />
      ) : (
        <Card className="border-dashed">
          <CardContent className="py-16 flex flex-col items-center gap-3 text-muted-foreground">
            <FolderOpen className="h-8 w-8" />
            <p className="text-sm">Select a collection to view and edit its details</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
