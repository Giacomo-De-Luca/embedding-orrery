'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/lib/ui-primitives/button';
import { Input } from '@/lib/ui-primitives/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Label } from '@/lib/ui-primitives/label';
import { Separator } from '@/lib/ui-primitives/separator';
import type { PortionStrategy, EmbedDatasetInput, HFDatasetInfo, HFDatasetPreview, EmbedDatasetResult, EmbeddingJob, GenerateLlmLabelsInput, GenerateLlmLabelsResult } from '@/lib/graphql/mutations';

import { SplitSelector } from './SplitSelector';
import { PortionSelector } from './PortionSelector';
import { DatasetInfoDisplay } from './DatasetInfoDisplay';
import { ColumnSelector } from './ColumnSelector';
import { EmbeddingModelForm } from './EmbeddingModelForm';
import { EmbedResultCard } from './EmbedResultCard';
import { ErrorCard } from './ErrorCard';
import { EmbedProgressSection } from './EmbedProgressSection';
import { useEmbeddingModelState } from '../lib/useEmbeddingModelState';
import { updateTextTemplate, transformStoredEmbeddingModel, resumeLlmLabelingJob } from '../lib/embeddingFormUtils';

interface HuggingFaceTabProps {
  fetchHFDatasetInfo: (datasetId: string) => Promise<HFDatasetInfo | null>;
  fetchHFDatasetPreview: (
    datasetId: string,
    config?: string,
    split?: string,
    nRows?: number
  ) => Promise<HFDatasetPreview | null>;
  embedHFDataset: (input: EmbedDatasetInput) => Promise<EmbedDatasetResult | null>;
  refreshCollections: () => Promise<void>;
  datasetInfo: HFDatasetInfo | null;
  datasetPreview: HFDatasetPreview | null;
  infoLoading: boolean;
  previewLoading: boolean;
  embedLoading: boolean;
  error: string | null;
  clearError: () => void;
  lastEmbedResult: EmbedDatasetResult | null;
  activeJobCollectionName?: string | null;
  generateLlmLabels?: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>;
  cancelEmbeddingJob?: (collectionName: string) => Promise<boolean>;
  cancelJobLoading?: boolean;
  removeEmbeddingJob?: (collectionName: string) => Promise<boolean>;
}

export function HuggingFaceTab({
  fetchHFDatasetInfo,
  fetchHFDatasetPreview,
  embedHFDataset,
  refreshCollections,
  datasetInfo,
  datasetPreview,
  infoLoading,
  previewLoading,
  embedLoading,
  error,
  clearError,
  lastEmbedResult,
  activeJobCollectionName,
  generateLlmLabels,
  cancelEmbeddingJob,
  cancelJobLoading,
  removeEmbeddingJob,
}: HuggingFaceTabProps) {
  const model = useEmbeddingModelState();
  const [llmResumeJobId, setLlmResumeJobId] = useState<string | null>(null);

  // HuggingFace specific state
  const [datasetId, setDatasetId] = useState('dair-ai/emotion');
  const [selectedSplit, setSelectedSplit] = useState('train');
  const [collectionName, setCollectionName] = useState('');

  // Column configuration
  const [selectedEmbeddingColumns, setSelectedEmbeddingColumns] = useState<string[]>([]);
  const [selectedMetadataColumns, setSelectedMetadataColumns] = useState<string[]>([]);
  const [textTemplate, setTextTemplate] = useState('');
  const [idColumn, setIdColumn] = useState('auto');

  // Portion configuration
  const [portionStrategy, setPortionStrategy] = useState<PortionStrategy>('FIRST_N');
  const [numRows, setNumRows] = useState(1000);
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(1000);
  const [randomSeed, setRandomSeed] = useState(42);

  const handleEmbeddingColumnsChange = (cols: string[]) => {
    setSelectedEmbeddingColumns(cols);
    setTextTemplate(updateTextTemplate(textTemplate, selectedEmbeddingColumns, cols));
  };

  // Reset columns when dataset changes
  useEffect(() => {
    setSelectedEmbeddingColumns([]);
    setSelectedMetadataColumns([]);
    setTextTemplate('');
    setIdColumn('auto');
  }, [datasetId]);

  const autoConfigureColumns = (features: Array<{ name: string; dtype: string }>) => {
    const textCols = features
      .filter(f => f.dtype === 'string' || f.dtype === 'str')
      .map(f => f.name);

    const embeddingCol = textCols.length > 0 ? textCols[0] : features[0]?.name;

    if (embeddingCol) {
      setSelectedEmbeddingColumns([embeddingCol]);
      setTextTemplate(`{${embeddingCol}}`);
    }

    setSelectedMetadataColumns(
      features.map(f => f.name).filter(name => name !== embeddingCol)
    );

    const idNames = ['id', 'index', 'idx', '_id', 'row_id', 'item_id', 'doc_id'];
    const idMatch = features.find(f =>
      idNames.some(name => f.name.toLowerCase() === name)
    );
    if (idMatch) {
      setIdColumn(idMatch.name);
    }

    if (datasetId && !collectionName) {
      const suggestedName = datasetId.split('/').pop()?.replace(/[^a-zA-Z0-9_-]/g, '_') || 'dataset';
      setCollectionName(suggestedName);
    }
  };

  const handleFetchInfoAndPreview = async () => {
    clearError();

    if (!datasetId.includes('/')) {
      alert('Dataset ID format should be: org/dataset');
      return;
    }

    const info = await fetchHFDatasetInfo(datasetId);
    if (!info || info.error) return;

    const config = info.defaultConfig ?? undefined;
    await fetchHFDatasetPreview(datasetId, config, selectedSplit, 5);

    if (info.configs[0]?.features) {
      autoConfigureColumns(info.configs[0].features);
    }
  };

  const handleEmbed = async () => {
    clearError();

    if (selectedEmbeddingColumns.length === 0) {
      alert('Please select at least one embedding column');
      return;
    }
    if (!collectionName) {
      alert('Please provide a collection name');
      return;
    }

    const metadataColumns = selectedEmbeddingColumns.length === 1
      ? selectedMetadataColumns
      : [...selectedMetadataColumns, ...selectedEmbeddingColumns];

    const embeddingModel = model.buildEmbeddingModelInput();
    const topicParams = model.getTopicParams();

    const commonInput = {
      datasetId,
      collectionName,
      config: datasetInfo?.defaultConfig || undefined,
      columns: selectedEmbeddingColumns,
      textTemplate: textTemplate || undefined,
      idColumn: idColumn !== 'auto' ? idColumn : undefined,
      metadataColumns,
      computeProjections: true,
      batchSize: model.batchSize,
      embeddingModel,
      ...topicParams,
    };

    if (portionStrategy === 'ALL') {
      // Embed every split into one collection in a single backend pass. The
      // backend tags each row with `source_split` and shares one ID
      // deduplicator across splits, so nothing gets overwritten.
      const allSplits = datasetInfo?.configs[0]?.splits.map(s => s.name) || ['train'];
      await embedHFDataset({
        ...commonInput,
        splits: allSplits,
        portion: { strategy: 'ALL' },
      });
    } else {
      await embedHFDataset({
        ...commonInput,
        split: selectedSplit,
        portion: {
          strategy: portionStrategy,
          n: portionStrategy === 'FIRST_N' || portionStrategy === 'RANDOM_SAMPLE' ? numRows : undefined,
          start: portionStrategy === 'ROW_RANGE' ? rangeStart : undefined,
          end: portionStrategy === 'ROW_RANGE' ? rangeEnd : undefined,
          seed: portionStrategy === 'RANDOM_SAMPLE' ? randomSeed : undefined,
        },
      });
    }

    await refreshCollections();
  };

  const handleResumeJob = async (job: EmbeddingJob) => {
    if (generateLlmLabels) {
      const handled = await resumeLlmLabelingJob(job, generateLlmLabels, {
        setLlmResumeJobId,
        refreshCollections,
      });
      if (handled) return;
    }

    const config = job.config as Record<string, unknown>;

    const storedPortion = config.portion as Record<string, unknown> | undefined;
    const portion = storedPortion ? {
      strategy: (storedPortion.strategy as string)?.toUpperCase() as PortionStrategy,
      n: storedPortion.n as number | undefined,
      start: storedPortion.start as number | undefined,
      end: storedPortion.end as number | undefined,
      seed: storedPortion.seed as number | undefined,
    } : undefined;

    await embedHFDataset({
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
    await refreshCollections();
  };

  const isLoading = infoLoading || previewLoading;
  const isDataLoaded = Boolean(datasetInfo);
  const columns = datasetInfo?.configs[0]?.features.map(f => ({ name: f.name, dtype: f.dtype })) || [];
  const splits = datasetInfo?.configs[0]?.splits || [];
  const availableSplits = datasetInfo?.configs[0]?.splits.map(s => s.name) || [];
  const totalRows = datasetInfo?.configs[0]?.splits.find(s => s.name === selectedSplit)?.numRows;

  return (
    <div className="space-y-6">
      {/* Data Source Card */}
      <Card>
        <CardHeader>
          <CardTitle>HuggingFace Dataset</CardTitle>
          <CardDescription>
            Enter a HuggingFace dataset ID (e.g., dair-ai/emotion, ag_news, imdb)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="dataset-id">Dataset ID</Label>
              <Input
                id="dataset-id"
                value={datasetId}
                onChange={(e) => setDatasetId(e.target.value)}
                placeholder="e.g., dair-ai/emotion"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="collection-name">Collection Name</Label>
              <Input
                id="collection-name"
                value={collectionName}
                onChange={(e) => setCollectionName(e.target.value)}
                placeholder="e.g., emotion_test"
              />
            </div>
          </div>

          <Button
            onClick={handleFetchInfoAndPreview}
            disabled={isLoading || !datasetId}
            className="w-full md:w-auto"
          >
            {isLoading ? <Spinner className="mr-2 h-4 w-4" /> : null}
            Fetch Dataset Info & Preview
          </Button>
        </CardContent>
      </Card>

      {error && <ErrorCard error={error} onDismiss={clearError} />}

      {/* Dataset Info & Preview */}
      {isDataLoaded && (
        <Card>
          <CardHeader>
            <CardTitle>Dataset Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {splits.length > 0 && (
              <SplitSelector
                splits={splits}
                selectedSplit={selectedSplit}
                onSplitChange={setSelectedSplit}
              />
            )}
            <Separator />
            <DatasetInfoDisplay
              type="huggingface"
              info={datasetInfo}
              preview={datasetPreview}
            />
          </CardContent>
        </Card>
      )}

      {/* Column Configuration */}
      {isDataLoaded && columns.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Column Configuration</CardTitle>
            <CardDescription>
              Select columns for embedding and configure the text template
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ColumnSelector
              columns={columns}
              selectedEmbeddingColumns={selectedEmbeddingColumns}
              selectedMetadataColumns={selectedMetadataColumns}
              onEmbeddingColumnsChange={handleEmbeddingColumnsChange}
              onMetadataColumnsChange={setSelectedMetadataColumns}
              textTemplate={textTemplate}
              onTemplateChange={setTextTemplate}
              idColumn={idColumn}
              onIdColumnChange={setIdColumn}
            />
          </CardContent>
        </Card>
      )}

      {/* Portion Configuration */}
      {isDataLoaded && (
        <Card>
          <CardHeader>
            <CardTitle>Dataset Portion</CardTitle>
            <CardDescription>
              Choose which portion of the dataset to embed
            </CardDescription>
          </CardHeader>
          <CardContent>
            <PortionSelector
              strategy={portionStrategy}
              onStrategyChange={setPortionStrategy}
              n={numRows}
              onNChange={setNumRows}
              start={rangeStart}
              onStartChange={setRangeStart}
              end={rangeEnd}
              onEndChange={setRangeEnd}
              seed={randomSeed}
              onSeedChange={setRandomSeed}
              totalRows={totalRows || null}
              availableSplits={availableSplits}
            />
          </CardContent>
        </Card>
      )}

      {/* Embedding Model Configuration */}
      {isDataLoaded && (
        <EmbeddingModelForm
          model={model}
          showEmbedButton
          onEmbed={handleEmbed}
          embedLoading={embedLoading}
          embedDisabled={selectedEmbeddingColumns.length === 0}
          embedButtonText="Embed Dataset"
          idPrefix="hf-"
        />
      )}

      {lastEmbedResult && <EmbedResultCard result={lastEmbedResult} />}

      <EmbedProgressSection
        embedLoading={embedLoading}
        activeJobCollectionName={activeJobCollectionName}
        llmResumeJobId={llmResumeJobId}
        onResumeJob={handleResumeJob}
        onCancelActiveJob={activeJobCollectionName && cancelEmbeddingJob
          ? () => cancelEmbeddingJob(activeJobCollectionName)
          : undefined}
        cancelLoading={cancelJobLoading}
        onRemoveJob={removeEmbeddingJob
          ? (job) => removeEmbeddingJob(job.collectionName)
          : undefined}
      />
    </div>
  );
}
