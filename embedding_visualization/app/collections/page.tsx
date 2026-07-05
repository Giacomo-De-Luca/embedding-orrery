'use client';

import { Suspense, useEffect, useRef, useState } from 'react';
import { useQuery } from '@apollo/client/react';
import { useEmbedDataset } from '@/lib/hooks/useEmbedDataset';
import { GET_COLLECTIONS } from '@/lib/graphql/queries';
import type { EmbeddingJob } from '@/lib/graphql/mutations';
import { PageNav } from '@/app/components/PageNav';
import { TabsContent } from '@/lib/ui-primitives/tabs';

// Import tab components
import { DataSourceTabs } from './components/DataSourceTabs';
import { HuggingFaceTab } from './components/HuggingFaceTab';
import { LocalFileTab } from './components/LocalFileTab';
import { CollectionManagerTab, type CollectionInfo } from './components/CollectionManagerTab';
import { SaeTab } from './components/SaeTab';
import { ActiveJobsStrip } from './components/ActiveJobsStrip';
import { JobProgressDock } from './components/JobProgressDock';
import { useCollectionsUrlState, type DataSourceTab } from './lib/urlState';
import { resumeJob } from './lib/embeddingFormUtils';

interface GraphQLCollection {
  name: string;
  count: number;
  metadata: Record<string, unknown> | null;
}

interface CollectionsData {
  collections: GraphQLCollection[];
}

/**
 * Collections Page
 *
 * Features:
 * - HuggingFace datasets, local file upload, collection management, SAE data prep
 * - Tabs stay mounted once visited so in-progress form state survives switches
 * - Active tab and manage-tab selection round-trip through the URL
 *   (?tab=, ?collection=)
 * - Page-global jobs strip + non-blocking progress dock
 */
export default function CollectionsPage() {
  return (
    <Suspense
      fallback={
        <div className="container mx-auto p-6 max-w-6xl text-muted-foreground">
          Loading...
        </div>
      }
    >
      <CollectionsPageContent />
    </Suspense>
  );
}

function CollectionsPageContent() {
  const { tab: activeTab, setTab: setActiveTab, managedCollection, setManagedCollection } =
    useCollectionsUrlState();

  // Tabs render on first visit and then stay mounted (hidden) so tab-local
  // form state survives switching. Idempotent ref mutation during render.
  const visitedTabs = useRef(new Set<DataSourceTab>());
  visitedTabs.current.add(activeTab);

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
    lastEmbedSource,
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

  // Jobs are page-global: one strip lists running/interrupted jobs on every
  // tab, and resume works regardless of which tab created the job.
  const [llmResumeJobId, setLlmResumeJobId] = useState<string | null>(null);

  const handleResumeJob = (job: EmbeddingJob) =>
    resumeJob(job, {
      embedHFDataset,
      embedLocalFile,
      generateLlmLabels,
      refreshCollections: handleRefreshCollections,
      setLlmResumeJobId,
    });

  // Once visited, a tab stays mounted; the inactive ones are display:none
  // (the data-[state=inactive]:hidden class is load-bearing — Radix does not
  // hide forceMounted content by itself).
  const renderTab = (tab: DataSourceTab, content: React.ReactNode) =>
    visitedTabs.current.has(tab) ? (
      <TabsContent value={tab} forceMount className="mt-4 data-[state=inactive]:hidden">
        {content}
      </TabsContent>
    ) : null;

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      {/* Page navigation */}
      <div className="mb-4 flex">
        <PageNav variant="solid" />
      </div>
      <h1 className="text-3xl font-bold mb-2">Collections</h1>
      <p className="text-muted-foreground mb-6">
        Import and embed datasets, then manage your embedding collections
      </p>

      <DataSourceTabs activeTab={activeTab} onTabChange={setActiveTab}>
        {/* Page-global jobs strip: visible on every tab, driven by polled
            server state so jobs survive page reloads. Jobs currently shown in
            the progress dock are hidden here to avoid duplication. */}
        <div className="mt-4 empty:hidden">
          <ActiveJobsStrip
            onResumeJob={handleResumeJob}
            onCancelJob={(job) => cancelEmbeddingJob(job.collectionName)}
            onRemoveJob={(job) => removeEmbeddingJob(job.collectionName)}
            hideJobIds={[activeJobCollectionName, llmResumeJobId]}
          />
        </div>

        {renderTab('huggingface',
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
            lastEmbedSource={lastEmbedSource}
          />
        )}

        {renderTab('local',
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
            lastEmbedSource={lastEmbedSource}
          />
        )}

        {renderTab('manage',
          <CollectionManagerTab
            collections={collections}
            collectionsLoading={collectionsLoading}
            selectedCollection={managedCollection}
            onSelectCollection={setManagedCollection}
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

        {renderTab('sae', <SaeTab />)}
      </DataSourceTabs>

      {/* Non-blocking progress dock for client-initiated jobs: the page stays
          usable while an embed or LLM-labeling run is in flight. */}
      {embedLoading && activeJobCollectionName && (
        <JobProgressDock
          jobId={activeJobCollectionName}
          onCancel={() => cancelEmbeddingJob(activeJobCollectionName)}
          cancelLoading={cancelJobLoading}
        />
      )}
      {llmResumeJobId && (
        <JobProgressDock
          jobId={llmResumeJobId}
          title="Generating LLM Labels"
          subtitle="Each topic is labeled individually via LLM API calls."
          itemsLabel="topics"
        />
      )}
    </div>
  );
}
