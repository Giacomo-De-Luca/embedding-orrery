'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useLazyQuery } from '@apollo/client/react';
import { Button } from '@/lib/ui-primitives/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Label } from '@/lib/ui-primitives/label';
import { Input } from '@/lib/ui-primitives/input';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { TopicConfigForm } from './TopicConfigForm';
import type { DataType, EmbedLocalFileInput, ReEmbedDatasetInput, LocalFileInfo, LocalFilePreview, EmbedDatasetResult } from '@/lib/graphql/mutations';
import type { EmbedSource } from '@/lib/hooks/useEmbedDataset';
import type { CollectionInfo } from './CollectionManagerTab';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/lib/ui-primitives/select';
import { GET_COLLECTION_PREVIEW } from '@/lib/graphql/queries';

import { FileUploadZone } from './FileUploadZone';
import { DataTypeSelector } from './DataTypeSelector';
import { PortionSelector } from './PortionSelector';
import { DatasetInfoDisplay } from './DatasetInfoDisplay';
import { ColumnSelector } from './ColumnSelector';
import { EmbeddingModelForm } from './EmbeddingModelForm';
import { EmbedResultCard } from './EmbedResultCard';
import { ErrorCard } from './ErrorCard';
import { EmbedFooterBar } from './EmbedFooterBar';
import { useEmbedFormState } from '../lib/useEmbedFormState';
import { buildLocalEmbedInput, buildReEmbedInput } from '../lib/embeddingFormUtils';
import { getEmbedValidationIssues, buildEmbedSummary } from '../lib/embedValidation';

interface LocalFileTabProps {
  fetchLocalFileInfo: (filePath: string) => Promise<LocalFileInfo | null>;
  fetchLocalFilePreview: (filePath: string, nRows?: number) => Promise<LocalFilePreview | null>;
  embedLocalFile: (input: EmbedLocalFileInput) => Promise<EmbedDatasetResult | null>;
  reEmbedDataset: (input: ReEmbedDatasetInput) => Promise<EmbedDatasetResult | null>;
  collections: CollectionInfo[];
  refreshCollections: () => Promise<void>;
  localFileInfo: LocalFileInfo | null;
  localFilePreview: LocalFilePreview | null;
  infoLoading: boolean;
  previewLoading: boolean;
  embedLoading: boolean;
  error: string | null;
  clearError: () => void;
  lastEmbedResult: EmbedDatasetResult | null;
  lastEmbedSource?: EmbedSource | null;
}

export function LocalFileTab({
  fetchLocalFileInfo,
  fetchLocalFilePreview,
  embedLocalFile,
  reEmbedDataset,
  collections,
  refreshCollections,
  localFileInfo,
  localFilePreview,
  infoLoading,
  previewLoading,
  embedLoading,
  error,
  clearError,
  lastEmbedResult,
  lastEmbedSource,
}: LocalFileTabProps) {
  const model = useEmbeddingModelState();

  // Source selection: file path OR existing dataset
  const [sourceDataset, setSourceDataset] = useState<string | null>(null);
  const [datasetColumns, setDatasetColumns] = useState<string[]>([]);

  // Local file specific state
  const [filePath, setFilePath] = useState('');
  const [dataType, setDataType] = useState<DataType>('TEXT');

  // Shared embed-form state (columns, template, portion, model config);
  // column config resets when the file path changes
  const form = useEmbedFormState(filePath);
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
    randomSeed, setRandomSeed,
  } = form;

  // Lazy query to fetch dataset preview for column detection
  const [fetchPreview] = useLazyQuery<{ embeddings: Array<{ id: string; document: string | null; metadata: Record<string, unknown> | null }> }>(GET_COLLECTION_PREVIEW);

  const fetchDatasetColumns = useCallback(async (datasetName: string) => {
    const { data } = await fetchPreview({
      variables: { collectionName: datasetName, limit: 10 },
    });
    if (data?.embeddings && data.embeddings.length > 0) {
      // Extract unique metadata keys from preview items
      const keys = new Set<string>();
      for (const item of data.embeddings) {
        if (item.metadata) {
          Object.keys(item.metadata).forEach(k => keys.add(k));
        }
      }
      // Add __document__ as a virtual column option
      const cols = ['__document__', ...Array.from(keys).sort()];
      setDatasetColumns(cols);
      // Auto-select __document__ by default
      setSelectedEmbeddingColumns(['__document__']);
      setSelectedMetadataColumns([]);
    }
  }, [fetchPreview, setSelectedEmbeddingColumns, setSelectedMetadataColumns]);

  const autoConfigureColumns = (columns: string[]) => {
    if (columns.length === 0) return;

    let embeddingCol: string;

    if (dataType === 'VECTOR') {
      const vectorNames = ['embedding', 'embeddings', 'vector', 'vectors', 'emb'];
      const match = columns.find(col =>
        vectorNames.some(name => col.toLowerCase().includes(name))
      );
      embeddingCol = match || columns[0];
      setSelectedEmbeddingColumns([embeddingCol]);
      setTextTemplate('');
    } else {
      embeddingCol = columns[0];
      setSelectedEmbeddingColumns([embeddingCol]);
      setTextTemplate(`{${embeddingCol}}`);
    }

    setSelectedMetadataColumns(columns.filter(col => col !== embeddingCol));

    const idNames = ['id', 'index', 'idx', '_id', 'row_id', 'item_id', 'feature_id', 'doc_id'];
    const idMatch = columns.find(col =>
      idNames.some(name => col.toLowerCase() === name)
    );
    if (idMatch) {
      setIdColumn(idMatch);
    }

    if (filePath && !collectionName) {
      const filename = filePath.split('/').pop()?.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_') || 'local_data';
      setCollectionName(filename);
    }
  };

  const handleFetchInfoAndPreview = async () => {
    clearError();

    // Button is disabled for invalid paths (FileUploadZone shows the reason)
    if (!filePath.startsWith('/')) return;

    const info = await fetchLocalFileInfo(filePath);
    if (!info || info.error) return;

    await fetchLocalFilePreview(filePath, 5);
    autoConfigureColumns(info.columns);
  };

  const handleEmbed = async () => {
    clearError();

    // CTA is disabled while issues exist; guard anyway
    if (fileValidationIssues.length > 0) return;

    await embedLocalFile(buildLocalEmbedInput(form.commonValues(), {
      filePath,
      dataType,
      portion: { strategy: portionStrategy, numRows, seed: randomSeed },
    }));

    await refreshCollections();
  };

  const handleSourceDatasetChange = (value: string) => {
    if (value === '__none__') {
      setSourceDataset(null);
      setDatasetColumns([]);
      return;
    }
    setSourceDataset(value);
    setFilePath(''); // Clear file path when selecting a dataset
    // Auto-suggest collection name
    const shortModel = model.modelName.split('/').pop()?.replace(/[^a-zA-Z0-9_-]/g, '_') || 'model';
    setCollectionName(`${value}_${shortModel}`);
    // Fetch columns from dataset preview
    fetchDatasetColumns(value);
  };

  const handleReEmbed = async () => {
    clearError();
    if (!sourceDataset) return;
    // CTA is disabled while issues exist; guard anyway
    if (reEmbedValidationIssues.length > 0) return;

    await reEmbedDataset(buildReEmbedInput(
      { ...form.commonValues(), embeddingModel: model.buildEmbeddingModelInput() },
      sourceDataset
    ));

    await refreshCollections();
  };

  const isLoading = infoLoading || previewLoading;
  const isDataLoaded = Boolean(localFileInfo) && !sourceDataset;
  const columns = localFileInfo?.columns.map(name => ({ name, dtype: 'unknown' })) || [];
  const totalRows = localFileInfo?.numRows;
  const isVectorMode = dataType === 'VECTOR';
  const sourceDatasetInfo = sourceDataset ? collections.find(c => c.name === sourceDataset) : null;

  const fileValidationIssues = useMemo(() => getEmbedValidationIssues({
    source: 'local-file',
    filePath,
    collectionName,
    embeddingColumns: selectedEmbeddingColumns,
    dataType,
    portionStrategy,
  }), [filePath, collectionName, selectedEmbeddingColumns, dataType, portionStrategy]);

  const reEmbedValidationIssues = useMemo(() => getEmbedValidationIssues({
    source: 'reembed',
    collectionName,
    embeddingColumns: selectedEmbeddingColumns,
  }), [collectionName, selectedEmbeddingColumns]);

  const fileEmbedSummary = buildEmbedSummary({
    collectionName,
    portionStrategy,
    numRows,
    totalRows: totalRows ?? null,
    modelName: model.modelName,
    enableTopics: model.enableTopics,
    dataType,
  });

  const reEmbedSummary = buildEmbedSummary({
    collectionName,
    portionStrategy: 'ALL',
    totalRows: sourceDatasetInfo?.numItems ?? null,
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
            Upload or provide the path to a local data file — or re-embed an existing dataset
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!sourceDataset && (
            <>
              <FileUploadZone
                filePath={filePath}
                onFilePathChange={(path) => { setFilePath(path); setSourceDataset(null); }}
                disabled={isLoading}
              />
            </>
          )}

          {/* From Existing Dataset selector */}
          {collections.length > 0 && (
            <div className="space-y-2">
              <Label>Or re-embed from existing dataset:</Label>
              <Select
                value={sourceDataset || '__none__'}
                onValueChange={handleSourceDatasetChange}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a dataset..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">None (use file above)</SelectItem>
                  {collections.map(col => (
                    <SelectItem key={col.name} value={col.name}>
                      {col.name} ({col.numItems} items)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {sourceDatasetInfo && (
                <p className="text-xs text-muted-foreground">
                  Source: {sourceDatasetInfo.numItems} items
                  {sourceDatasetInfo.embeddingModel && ` · Currently embedded with ${sourceDatasetInfo.embeddingModel}`}
                </p>
              )}
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="local-collection-name">Collection Name</Label>
            <Input
              id="local-collection-name"
              value={collectionName}
              onChange={(e) => setCollectionName(e.target.value)}
              placeholder="e.g., my_data"
            />
          </div>

          {!sourceDataset && (
            <>
              <DataTypeSelector
                dataType={dataType}
                onDataTypeChange={setDataType}
                disabled={isLoading}
              />

              <Button
                onClick={handleFetchInfoAndPreview}
                disabled={isLoading || !filePath}
                className="w-full md:w-auto"
              >
                {isLoading ? <Spinner className="mr-2 h-4 w-4" /> : null}
                Fetch File Info & Preview
              </Button>
            </>
          )}
        </CardContent>
      </Card>

      {error && <ErrorCard error={error} onDismiss={clearError} />}

      {/* File Info & Preview */}
      {isDataLoaded && (
        <Card ref={infoCardRef}>
          <CardHeader>
            <CardTitle>2 · File Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <DatasetInfoDisplay
              type="local"
              info={localFileInfo}
              preview={localFilePreview}
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
              {isVectorMode
                ? 'Select the vector column and metadata fields'
                : 'Select columns for embedding and configure the text template'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ColumnSelector
              columns={columns}
              dataType={dataType}
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
              {isVectorMode
                ? 'Choose which portion of the file to import'
                : 'Choose which portion of the file to embed'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* embedLocalFile has no row-range input, so ROW_RANGE is not offered */}
            <PortionSelector
              strategy={portionStrategy}
              onStrategyChange={setPortionStrategy}
              n={numRows}
              onNChange={setNumRows}
              seed={randomSeed}
              onSeedChange={setRandomSeed}
              totalRows={totalRows || null}
              availableSplits={[]}
              allowedStrategies={['ALL', 'FIRST_N', 'RANDOM_SAMPLE']}
            />
          </CardContent>
        </Card>
      )}

      {/* Embedding Model Configuration (only for TEXT) */}
      {isDataLoaded && dataType === 'TEXT' && (
        <EmbeddingModelForm
          model={model}
          title="5 · Embedding Model"
          showTopics={false}
          idPrefix="local-"
        />
      )}

      {/* Topic Extraction (available for all data types) */}
      {isDataLoaded && (
        <Card>
          <CardContent className="pt-6 space-y-3">
            <div className="flex items-center gap-2">
              <Checkbox
                id="local-enable-topics"
                checked={model.enableTopics}
                onCheckedChange={(checked) => model.setEnableTopics(checked === true)}
              />
              <Label htmlFor="local-enable-topics" className="cursor-pointer">
                {isVectorMode ? 'Extract topics after import' : 'Extract topics after embedding'}
              </Label>
            </div>
            {model.enableTopics && (
              <TopicConfigForm value={model.topicConfig} onChange={model.setTopicConfig} />
            )}
          </CardContent>
        </Card>
      )}

      {/* Sticky CTA with config recap + inline validation */}
      {isDataLoaded && !embedLoading && (
        <EmbedFooterBar
          summary={fileEmbedSummary}
          ctaLabel={isVectorMode ? 'Import Vectors' : 'Embed File'}
          onSubmit={handleEmbed}
          loading={embedLoading}
          issues={fileValidationIssues}
        />
      )}

      {/* Re-embed from existing dataset: column selection + model + button */}
      {sourceDataset && datasetColumns.length > 0 && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Column Configuration</CardTitle>
              <CardDescription>
                Select which fields to embed. Use &quot;__document__&quot; for the original embedded text,
                or select metadata fields to compose new text.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ColumnSelector
                columns={datasetColumns.map(name => ({ name, dtype: 'unknown' }))}
                dataType="TEXT"
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

          <EmbeddingModelForm
            model={model}
            showTopics={false}
            idPrefix="reembed-"
          />

          <Card>
            <CardContent className="pt-6 space-y-3">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="reembed-enable-topics"
                  checked={model.enableTopics}
                  onCheckedChange={(checked) => model.setEnableTopics(checked === true)}
                />
                <Label htmlFor="reembed-enable-topics" className="cursor-pointer">
                  Extract topics after embedding
                </Label>
              </div>
              {model.enableTopics && (
                <TopicConfigForm value={model.topicConfig} onChange={model.setTopicConfig} />
              )}
            </CardContent>
          </Card>

          {!embedLoading && (
            <EmbedFooterBar
              summary={reEmbedSummary}
              ctaLabel="Re-embed Dataset"
              onSubmit={handleReEmbed}
              loading={embedLoading}
              issues={reEmbedValidationIssues}
            />
          )}
        </>
      )}

      {lastEmbedResult && (lastEmbedSource === 'local' || lastEmbedSource === 'reembed') && (
        <EmbedResultCard result={lastEmbedResult} isImportMode={isVectorMode} />
      )}
    </div>
  );
}
