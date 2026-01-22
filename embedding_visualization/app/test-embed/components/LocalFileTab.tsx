'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/lib/ui-primitives/button';
import { Input } from '@/lib/ui-primitives/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Spinner } from '@/lib/ui-primitives/spinner';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { Label } from '@/lib/ui-primitives/label';
import { Separator } from '@/lib/ui-primitives/separator';
import type { EmbeddingProvider, DataType, PortionStrategy, LocalFileInfo, LocalFilePreview, EmbedDatasetResult, GeminiTaskType } from '@/lib/graphql/mutations';

import { FileUploadZone } from './FileUploadZone';
import { DataTypeSelector } from './DataTypeSelector';
import { PortionSelector } from './PortionSelector';
import { DatasetInfoDisplay } from './DatasetInfoDisplay';
import { ColumnSelector } from './ColumnSelector';
import { EMBEDDING_PROVIDERS } from '@/lib/utils/embeddingProviders';

interface LocalFileTabProps {
  fetchLocalFileInfo: (filePath: string) => Promise<LocalFileInfo | null>;
  fetchLocalFilePreview: (filePath: string, nRows?: number) => Promise<LocalFilePreview | null>;
  embedLocalFile: (input: {
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
    embeddingModel?: { provider: EmbeddingProvider; modelName: string; ollamaUrl?: string; task?: string; taskType?: GeminiTaskType };
  }) => Promise<EmbedDatasetResult | null>;
  refreshCollections: () => Promise<void>;
  localFileInfo: LocalFileInfo | null;
  localFilePreview: LocalFilePreview | null;
  infoLoading: boolean;
  previewLoading: boolean;
  embedLoading: boolean;
  error: string | null;
  clearError: () => void;
  lastEmbedResult: EmbedDatasetResult | null;
}

export function LocalFileTab({
  fetchLocalFileInfo,
  fetchLocalFilePreview,
  embedLocalFile,
  refreshCollections,
  localFileInfo,
  localFilePreview,
  infoLoading,
  previewLoading,
  embedLoading,
  error,
  clearError,
  lastEmbedResult,
}: LocalFileTabProps) {
  // Local file specific state
  const [filePath, setFilePath] = useState('');
  const [dataType, setDataType] = useState<DataType>('TEXT');
  const [collectionName, setCollectionName] = useState('');

  // Column configuration
  const [selectedEmbeddingColumns, setSelectedEmbeddingColumns] = useState<string[]>([]);
  const [selectedMetadataColumns, setSelectedMetadataColumns] = useState<string[]>([]);
  const [textTemplate, setTextTemplate] = useState('');
  const [idColumn, setIdColumn] = useState('auto');

  // Portion configuration
  const [portionStrategy, setPortionStrategy] = useState<PortionStrategy>('FIRST_N');
  const [numRows, setNumRows] = useState(1000);
  const [randomSeed, setRandomSeed] = useState(42);

  // Embedding model state
  const [embeddingProvider, setEmbeddingProvider] = useState<EmbeddingProvider>('SENTENCE_TRANSFORMERS');
  const [modelName, setModelName] = useState(EMBEDDING_PROVIDERS.SENTENCE_TRANSFORMERS.defaultModel);
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
  const [qwenTask, setQwenTask] = useState('Given a web search query, retrieve relevant passages that answer the query');
  const [geminiTaskType, setGeminiTaskType] = useState<GeminiTaskType>('SEMANTIC_SIMILARITY');

  // Reset columns when file changes
  useEffect(() => {
    setSelectedEmbeddingColumns([]);
    setSelectedMetadataColumns([]);
    setTextTemplate('');
    setIdColumn('auto');
  }, [filePath]);

  const autoConfigureColumns = (columns: string[]) => {
    if (columns.length > 0) {
      setSelectedEmbeddingColumns([columns[0]]);
      setTextTemplate(`{${columns[0]}}`);
    }

    if (filePath && !collectionName) {
      const filename = filePath.split('/').pop()?.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_') || 'local_data';
      setCollectionName(filename);
    }
  };

  const handleFetchInfoAndPreview = async () => {
    clearError();

    if (!filePath) {
      alert('Please provide a file path');
      return;
    }
    if (!filePath.startsWith('/')) {
      alert('File path must be absolute (starting with /)');
      return;
    }

    const info = await fetchLocalFileInfo(filePath);
    if (!info || info.error) return;

    await fetchLocalFilePreview(filePath, 5);
    autoConfigureColumns(info.columns);
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

    await embedLocalFile({
      filePath,
      collectionName,
      dataType,
      columns: dataType === 'TEXT' ? selectedEmbeddingColumns : undefined,
      textTemplate: dataType === 'TEXT' ? (textTemplate || undefined) : undefined,
      imageColumn: dataType === 'IMAGE' ? selectedEmbeddingColumns[0] : undefined,
      vectorColumn: dataType === 'VECTOR' ? selectedEmbeddingColumns[0] : undefined,
      idColumn: idColumn !== 'auto' ? idColumn : undefined,
      metadataColumns,
      nRows: portionStrategy === 'FIRST_N' ? numRows : undefined,
      sampleN: portionStrategy === 'RANDOM_SAMPLE' ? numRows : undefined,
      sampleSeed: portionStrategy === 'RANDOM_SAMPLE' ? randomSeed : undefined,
      computeProjections: true,
      embeddingModel: dataType === 'TEXT' ? {
        provider: embeddingProvider,
        modelName,
        ollamaUrl: embeddingProvider === 'OLLAMA' ? ollamaUrl : undefined,
        task: embeddingProvider === 'QWEN' ? qwenTask : undefined,
        taskType: embeddingProvider === 'GEMINI' ? geminiTaskType : undefined,
      } : undefined,
    });

    await refreshCollections();
  };

  const handleProviderChange = (provider: EmbeddingProvider) => {
    setEmbeddingProvider(provider);
    setModelName(EMBEDDING_PROVIDERS[provider].defaultModel);
  };

  const isLoading = infoLoading || previewLoading;
  const isDataLoaded = Boolean(localFileInfo);
  const columns = localFileInfo?.columns.map(name => ({ name, dtype: 'unknown' })) || [];
  const totalRows = localFileInfo?.numRows;

  return (
    <div className="space-y-6">
      {/* Data Source Card */}
      <Card>
        <CardHeader>
          <CardTitle>Local File</CardTitle>
          <CardDescription>
            Upload or provide the path to a local data file
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileUploadZone
            filePath={filePath}
            onFilePathChange={setFilePath}
            disabled={isLoading}
          />
          <div className="space-y-2">
            <Label htmlFor="local-collection-name">Collection Name</Label>
            <Input
              id="local-collection-name"
              value={collectionName}
              onChange={(e) => setCollectionName(e.target.value)}
              placeholder="e.g., my_data"
            />
          </div>
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
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <div className="text-destructive">
              <strong>Error:</strong> {error}
            </div>
            <Button variant="outline" size="sm" onClick={clearError} className="mt-2">
              Dismiss
            </Button>
          </CardContent>
        </Card>
      )}

      {/* File Info & Preview */}
      {isDataLoaded && (
        <Card>
          <CardHeader>
            <CardTitle>File Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Separator />
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
              onEmbeddingColumnsChange={setSelectedEmbeddingColumns}
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
              Choose which portion of the file to embed
            </CardDescription>
          </CardHeader>
          <CardContent>
            <PortionSelector
              strategy={portionStrategy}
              onStrategyChange={setPortionStrategy}
              n={numRows}
              onNChange={setNumRows}
              start={0}
              onStartChange={() => {}}
              end={1000}
              onEndChange={() => {}}
              seed={randomSeed}
              onSeedChange={setRandomSeed}
              totalRows={totalRows || null}
              availableSplits={[]}
            />
          </CardContent>
        </Card>
      )}

      {/* Embedding Model Configuration (only for TEXT) */}
      {isDataLoaded && dataType === 'TEXT' && (
        <Card>
          <CardHeader>
            <CardTitle>Embedding Model</CardTitle>
            <CardDescription>
              Choose the embedding model to use
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="provider">Provider</Label>
                <Select
                  value={embeddingProvider}
                  onValueChange={(v) => handleProviderChange(v as EmbeddingProvider)}
                >
                  <SelectTrigger id="provider">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {(Object.keys(EMBEDDING_PROVIDERS) as Array<keyof typeof EMBEDDING_PROVIDERS>).map((provider) => (
                      <SelectItem key={provider} value={provider}>
                        {provider}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {EMBEDDING_PROVIDERS[embeddingProvider].description}
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="model-name">Model Name</Label>
                <Input
                  id="model-name"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder={EMBEDDING_PROVIDERS[embeddingProvider].defaultModel}
                />
              </div>
              {embeddingProvider === 'OLLAMA' && (
                <div className="space-y-2">
                  <Label htmlFor="ollama-url">Ollama URL</Label>
                  <Input
                    id="ollama-url"
                    value={ollamaUrl}
                    onChange={(e) => setOllamaUrl(e.target.value)}
                    placeholder="http://localhost:11434"
                  />
                </div>
              )}
              {embeddingProvider === 'QWEN' && (
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="qwen-task">Query Task Instruction</Label>
                  <Input
                    id="qwen-task"
                    value={qwenTask}
                    onChange={(e) => setQwenTask(e.target.value)}
                    placeholder="Given a web search query, retrieve relevant passages that answer the query"
                  />
                  <p className="text-xs text-muted-foreground">
                    Instruction prefix added to queries during semantic search (not used during document embedding)
                  </p>
                </div>
              )}
              {embeddingProvider === 'GEMINI' && (
                <div className="space-y-2">
                  <Label htmlFor="gemini-task-type">Task Type</Label>
                  <Select
                    value={geminiTaskType}
                    onValueChange={(v) => setGeminiTaskType(v as GeminiTaskType)}
                  >
                    <SelectTrigger id="gemini-task-type">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="SEMANTIC_SIMILARITY">Semantic Similarity</SelectItem>
                      <SelectItem value="CLASSIFICATION">Classification</SelectItem>
                      <SelectItem value="CLUSTERING">Clustering</SelectItem>
                      <SelectItem value="RETRIEVAL_DOCUMENT">Retrieval (Document)</SelectItem>
                      <SelectItem value="RETRIEVAL_QUERY">Retrieval (Query)</SelectItem>
                      <SelectItem value="CODE_RETRIEVAL_QUERY">Code Retrieval</SelectItem>
                      <SelectItem value="QUESTION_ANSWERING">Question Answering</SelectItem>
                      <SelectItem value="FACT_VERIFICATION">Fact Verification</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Optimizes embeddings for the selected task type
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Embed Button */}
      {isDataLoaded && (
        <Button
          onClick={handleEmbed}
          disabled={embedLoading || selectedEmbeddingColumns.length === 0}
          size="lg"
          className="w-full md:w-auto"
        >
          {embedLoading ? <Spinner className="mr-2 h-4 w-4" /> : null}
          Embed File
        </Button>
      )}

      {/* Embed Result */}
      {lastEmbedResult && (
        <Card className={lastEmbedResult.error ? 'border-destructive' : 'border-green-500'}>
          <CardHeader>
            <CardTitle>
              {lastEmbedResult.error ? '❌ Embedding Failed' : '✅ Embedding Complete!'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {lastEmbedResult.error ? (
              <p className="text-destructive">{lastEmbedResult.error}</p>
            ) : (
              <div className="space-y-2">
                <p><strong>Collection:</strong> {lastEmbedResult.collectionName}</p>
                <p><strong>Total Embedded:</strong> {lastEmbedResult.totalEmbedded.toLocaleString()}</p>
                <p><strong>Embedding Dim:</strong> {lastEmbedResult.embeddingDim}</p>
                <p><strong>Device:</strong> {lastEmbedResult.device}</p>
                <p><strong>Duration:</strong> {lastEmbedResult.durationSeconds.toFixed(2)}s</p>
                <p><strong>Projections:</strong> {lastEmbedResult.projectionsComputed ? '✓ Computed' : 'Not computed'}</p>
                {lastEmbedResult.embeddingProvider && (
                  <p><strong>Model:</strong> {lastEmbedResult.embeddingProvider} / {lastEmbedResult.embeddingModel}</p>
                )}
                <div className="mt-4">
                  <a
                    href={`/?collection=${lastEmbedResult.collectionName}`}
                    className="text-blue-500 hover:underline font-medium"
                  >
                    View in Visualization →
                  </a>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Loading Overlay */}
      {embedLoading && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50">
          <Card className="w-96">
            <CardContent className="pt-6 text-center">
              <Spinner className="h-8 w-8 mx-auto mb-4" />
              <p className="text-lg font-medium">Embedding file...</p>
              <p className="text-sm text-muted-foreground mt-2">
                This may take several minutes for large files.
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
