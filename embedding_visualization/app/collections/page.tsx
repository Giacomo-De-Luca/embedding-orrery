'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@apollo/client/react';
import { useEmbedDataset } from '@/lib/hooks/useEmbedDataset';
import { GET_COLLECTIONS } from '@/lib/graphql/queries';
import { PageNav } from '@/app/components/PageNav';

// Import tab components
import { DataSourceTabs, type DataSourceTab } from './components/DataSourceTabs';
import { HuggingFaceTab } from './components/HuggingFaceTab';
import { LocalFileTab } from './components/LocalFileTab';
import { CollectionManagerTab, type CollectionInfo } from './components/CollectionManagerTab';
import { SaeTab } from './components/SaeTab';

interface GraphQLCollection {
  name: string;
  count: number;
  metadata: Record<string, unknown> | null;
}

interface CollectionsData {
  collections: GraphQLCollection[];
}

/**
 * Dataset Embedding Page
 *
 * Features:
 * - HuggingFace datasets (Tab 1)
 * - Local file upload (Tab 2)
 * - Collection management: view, edit metadata, delete (Tab 3)
 */
export default function TestEmbedPage() {
  const [activeTab, setActiveTab] = useState<DataSourceTab>('huggingface');

  // Query collections for the manager tab
  const { data: collectionsData, loading: collectionsLoading, refetch: refetchCollections } = 
    useQuery<CollectionsData>(GET_COLLECTIONS);

  // Hook for all embedding operations
  const {
    fetchHFDatasetInfo,
    fetchHFDatasetPreview,
    fetchLocalFileInfo,
    fetchLocalFilePreview,
    embedHFDataset,
    embedLocalFile,
    reEmbedDataset,
    deleteCollection,
    updateCollectionMetadata,
    refreshCollections,
    datasetInfo,
    datasetPreview,
    localFileInfo,
    localFilePreview,
    infoLoading,
    previewLoading,
    embedLoading,
    error,
    clearError,
    lastEmbedResult,
    activeJobCollectionName,
    extractTopics,
    topicsLoading,
    lastTopicsResult,
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
    cancelEmbeddingJob,
    cancelJobLoading,
    removeEmbeddingJob,
  } = useEmbedDataset();

  // Transform collections data for the manager tab
  const collections: CollectionInfo[] = collectionsData?.collections.map(col => ({
    name: col.name,
    numItems: col.count,
    embeddingProvider: col.metadata?.embedding_provider as string | null,
    embeddingModel: col.metadata?.embedding_model as string | null,
    metadata: col.metadata || undefined,
  })) || [];

  // Clear errors and results when switching tabs
  useEffect(() => {
    clearError();
  }, [activeTab, clearError]);

  // Wrapper for refreshCollections that also refetches the collections query
  const handleRefreshCollections = async () => {
    await refreshCollections();
    await refetchCollections();
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      {/* Page navigation */}
      <div className="mb-4 flex">
        <PageNav variant="solid" />
      </div>
      <h1 className="text-3xl font-bold mb-2">Dataset Embedding</h1>
      <p className="text-muted-foreground mb-6">
        Embed HuggingFace datasets or local files into ChromaDB collections for semantic search
      </p>

      {/* Tab Selector */}
      <div className="mb-6">
        <DataSourceTabs activeTab={activeTab} onTabChange={setActiveTab} />
      </div>

      {/* Tab Content */}
      {activeTab === 'huggingface' && (
        <HuggingFaceTab
          fetchHFDatasetInfo={fetchHFDatasetInfo}
          fetchHFDatasetPreview={fetchHFDatasetPreview}
          embedHFDataset={embedHFDataset}
          refreshCollections={handleRefreshCollections}
          datasetInfo={datasetInfo}
          datasetPreview={datasetPreview}
          infoLoading={infoLoading}
          previewLoading={previewLoading}
          embedLoading={embedLoading}
          error={error}
          clearError={clearError}
          lastEmbedResult={lastEmbedResult}
          activeJobCollectionName={activeJobCollectionName}
          generateLlmLabels={generateLlmLabels}
          cancelEmbeddingJob={cancelEmbeddingJob}
          cancelJobLoading={cancelJobLoading}
          removeEmbeddingJob={removeEmbeddingJob}
        />
      )}

      {activeTab === 'local' && (
        <LocalFileTab
          fetchLocalFileInfo={fetchLocalFileInfo}
          fetchLocalFilePreview={fetchLocalFilePreview}
          embedLocalFile={embedLocalFile}
          reEmbedDataset={reEmbedDataset}
          collections={collections}
          refreshCollections={handleRefreshCollections}
          localFileInfo={localFileInfo}
          localFilePreview={localFilePreview}
          infoLoading={infoLoading}
          previewLoading={previewLoading}
          embedLoading={embedLoading}
          error={error}
          clearError={clearError}
          lastEmbedResult={lastEmbedResult}
          activeJobCollectionName={activeJobCollectionName}
          generateLlmLabels={generateLlmLabels}
          cancelEmbeddingJob={cancelEmbeddingJob}
          cancelJobLoading={cancelJobLoading}
          removeEmbeddingJob={removeEmbeddingJob}
        />
      )}

      {activeTab === 'manage' && (
        <CollectionManagerTab
          collections={collections}
          collectionsLoading={collectionsLoading}
          refreshCollections={handleRefreshCollections}
          deleteCollection={deleteCollection}
          updateCollectionMetadata={updateCollectionMetadata}
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
      )}

      {activeTab === 'sae' && <SaeTab />}
    </div>
  );
}
