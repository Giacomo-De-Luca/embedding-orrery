'use client';

import { useState, useEffect, useMemo, useRef } from 'react';
import { Button } from '@/lib/ui-primitives/button';
import { Input } from '@/lib/ui-primitives/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Label } from '@/lib/ui-primitives/label';
import { Separator } from '@/lib/ui-primitives/separator';
import type { EmbedDatasetInput, HFDatasetInfo, HFDatasetPreview, EmbedDatasetResult } from '@/lib/graphql/mutations';
import type { EmbedSource } from '@/lib/hooks/useEmbedDataset';

import { SplitSelector } from './SplitSelector';
import { PortionSelector } from './PortionSelector';
import { DatasetInfoDisplay } from './DatasetInfoDisplay';
import { ColumnSelector } from './ColumnSelector';
import { EmbeddingModelForm } from './EmbeddingModelForm';
import { EmbedResultCard } from './EmbedResultCard';
import { ErrorCard } from './ErrorCard';
import { EmbedFooterBar } from './EmbedFooterBar';
import { useEmbedFormState } from '../lib/useEmbedFormState';
import { buildHFEmbedInput } from '../lib/embeddingFormUtils';
import { getEmbedValidationIssues, buildEmbedSummary } from '../lib/embedValidation';

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
  lastEmbedSource?: EmbedSource | null;
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
  lastEmbedSource,
}: HuggingFaceTabProps) {
  // HuggingFace specific state
  const [datasetId, setDatasetId] = useState('dair-ai/emotion');
  const [selectedSplit, setSelectedSplit] = useState('train');

  // Shared embed-form state (columns, template, portion, model config);
  // column config resets when the dataset id changes
  const form = useEmbedFormState(datasetId);
  const {
    model,
    collectionName, setCollectionName,
    selectedEmbeddingColumns, setSelectedEmbeddingColumns,
    selectedMetadataColumns, setSelectedMetadataColumns,
    textTemplate, setTextTemplate,
    idColumn, setIdColumn,
    handleEmbeddingColumnsChange,
    portionStrategy, setPortionStrategy,
    numRows, setNumRows,
    rangeStart, setRangeStart,
    rangeEnd, setRangeEnd,
    randomSeed, setRandomSeed,
  } = form;

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

    // Button is disabled for malformed ids; guard anyway
    if (!datasetId.includes('/')) return;

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

    // CTA is disabled while issues exist; guard anyway
    if (validationIssues.length > 0) return;

    await embedHFDataset(buildHFEmbedInput(form.commonValues(), {
      datasetId,
      defaultConfig: datasetInfo?.defaultConfig,
      selectedSplit,
      allSplits: datasetInfo?.configs[0]?.splits.map(s => s.name) || [],
      portion: { strategy: portionStrategy, numRows, rangeStart, rangeEnd, seed: randomSeed },
    }));

    await refreshCollections();
  };

  const isLoading = infoLoading || previewLoading;
  const isDataLoaded = Boolean(datasetInfo);
  const columns = datasetInfo?.configs[0]?.features.map(f => ({ name: f.name, dtype: f.dtype })) || [];
  const splits = datasetInfo?.configs[0]?.splits || [];
  const availableSplits = datasetInfo?.configs[0]?.splits.map(s => s.name) || [];
  const totalRows = datasetInfo?.configs[0]?.splits.find(s => s.name === selectedSplit)?.numRows;

  const datasetIdInvalid = datasetId.length > 0 && !datasetId.includes('/');

  const validationIssues = useMemo(() => getEmbedValidationIssues({
    source: 'hf',
    datasetId,
    collectionName,
    embeddingColumns: selectedEmbeddingColumns,
    portionStrategy,
    rangeStart,
    rangeEnd,
  }), [datasetId, collectionName, selectedEmbeddingColumns, portionStrategy, rangeStart, rangeEnd]);

  const embedSummary = buildEmbedSummary({
    collectionName,
    portionStrategy,
    numRows,
    rangeStart,
    rangeEnd,
    totalRows: totalRows ?? null,
    modelName: model.modelName,
    enableTopics: model.enableTopics,
  });

  // Scroll the newly revealed configuration into view after fetching info
  const infoCardRef = useRef<HTMLDivElement>(null);
  const wasLoadedRef = useRef(false);
  useEffect(() => {
    if (isDataLoaded && !wasLoadedRef.current) {
      infoCardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    wasLoadedRef.current = isDataLoaded;
  }, [isDataLoaded]);

  return (
    <div className="space-y-6">
      {/* Data Source Card */}
      <Card>
        <CardHeader>
          <CardTitle>1 · Data Source</CardTitle>
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
              {datasetIdInvalid && (
                <p className="text-xs text-destructive">
                  Dataset ID must be in the form org/dataset
                </p>
              )}
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
            disabled={isLoading || !datasetId.includes('/')}
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
        <Card ref={infoCardRef}>
          <CardHeader>
            <CardTitle>2 · Dataset Information</CardTitle>
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
            <CardTitle>3 · Column Configuration</CardTitle>
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
            <CardTitle>4 · Dataset Portion</CardTitle>
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
          title="5 · Embedding Model"
          idPrefix="hf-"
        />
      )}

      {lastEmbedResult && lastEmbedSource === 'hf' && <EmbedResultCard result={lastEmbedResult} />}

      {/* Sticky CTA with config recap + inline validation */}
      {isDataLoaded && (
        <EmbedFooterBar
          summary={embedSummary}
          ctaLabel="Embed Dataset"
          onSubmit={handleEmbed}
          loading={embedLoading}
          issues={validationIssues}
        />
      )}
    </div>
  );
}
