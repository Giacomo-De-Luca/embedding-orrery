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
import type { EmbeddingProvider, PortionStrategy, HFDatasetInfo, HFDatasetPreview, EmbedDatasetResult, GeminiTaskType, EmbeddingJob, TopicConfigInput, GenerateLlmLabelsInput, GenerateLlmLabelsResult } from '@/lib/graphql/mutations';
import { Checkbox } from '@/lib/ui-primitives/checkbox';

import { SplitSelector } from './SplitSelector';
import { TopicConfigForm, DEFAULT_TOPIC_CONFIG, toTopicConfigInput, type TopicConfigState } from './TopicConfigForm';
import { PortionSelector } from './PortionSelector';
import { DatasetInfoDisplay } from './DatasetInfoDisplay';
import { ColumnSelector } from './ColumnSelector';
import { ProgressModal } from './EmbeddingProgressModal';
import { JobsPanel } from './JobsPanel';
import { EMBEDDING_PROVIDERS } from '@/lib/utils/embeddingProviders';

interface HuggingFaceTabProps {
  fetchHFDatasetInfo: (datasetId: string) => Promise<HFDatasetInfo | null>;
  fetchHFDatasetPreview: (
    datasetId: string,
    config?: string,
    split?: string,
    nRows?: number
  ) => Promise<HFDatasetPreview | null>;
  embedHFDataset: (input: {
    datasetId: string;
    collectionName: string;
    config?: string;
    split?: string;
    columns?: string[];
    textTemplate?: string;
    idColumn?: string;
    metadataColumns?: string[];
    portion?: { strategy: PortionStrategy; n?: number; start?: number; end?: number; seed?: number };
    computeProjections?: boolean;
    batchSize?: number;
    embeddingModel?: { provider: EmbeddingProvider; modelName: string; ollamaUrl?: string; task?: string; taskType?: GeminiTaskType; prompt?: string };
    resume?: boolean;
    extractTopics?: boolean;
    topicConfig?: TopicConfigInput;
  }) => Promise<EmbedDatasetResult | null>;
  refreshCollections: () => Promise<void>;
  datasetInfo: HFDatasetInfo | null;
  datasetPreview: HFDatasetPreview | null;
  infoLoading: boolean;
  previewLoading: boolean;
  embedLoading: boolean;
  error: string | null;
  clearError: () => void;
  lastEmbedResult: EmbedDatasetResult | null;
  /** Collection name of currently running job for progress tracking */
  activeJobCollectionName?: string | null;
  /** LLM label generation for resuming interrupted LLM labeling jobs */
  generateLlmLabels?: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>;
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
}: HuggingFaceTabProps) {
  // LLM labeling resume state
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
  const [batchSize, setBatchSize] = useState(100);

  // Wrapper for setting columns that also updates template
  const handleEmbeddingColumnsChange = (cols: string[]) => {
    setSelectedEmbeddingColumns(cols);

    // Determine added/removed columns
    const removed = selectedEmbeddingColumns.filter(c => !cols.includes(c));
    const added = cols.filter(c => !selectedEmbeddingColumns.includes(c));

    let updated = textTemplate;

    // Remove placeholders for unchecked columns
    for (const col of removed) {
      // Remove {col} and any surrounding ", " separator
      updated = updated.replace(new RegExp(`,\\s*\\{${col}\\}|\\{${col}\\}\\s*,\\s*|\\{${col}\\}`, 'g'), '');
    }
    updated = updated.trim();

    // Append placeholders for newly checked columns
    for (const col of added) {
      if (updated) {
        updated = `${updated}, {${col}}`;
      } else {
        updated = `{${col}}`;
      }
    }

    setTextTemplate(updated);
  };

  // Portion configuration
  const [portionStrategy, setPortionStrategy] = useState<PortionStrategy>('FIRST_N');
  const [numRows, setNumRows] = useState(1000);
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(1000);
  const [randomSeed, setRandomSeed] = useState(42);

  // Embedding model state
  const [embeddingProvider, setEmbeddingProvider] = useState<EmbeddingProvider>('SENTENCE_TRANSFORMERS');
  const [modelName, setModelName] = useState(EMBEDDING_PROVIDERS.SENTENCE_TRANSFORMERS.defaultModel);
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
  const [qwenTask, setQwenTask] = useState('Given a web search query, retrieve relevant passages that answer the query');
  const [geminiTaskType, setGeminiTaskType] = useState<GeminiTaskType>('SEMANTIC_SIMILARITY');
  // SentenceTransformers prompt support (for models like Gemma Embedding)
  const [promptName, setPromptName] = useState<string | null>(null);
  const [customPrompt, setCustomPrompt] = useState('');

  // Topic extraction
  const [enableTopics, setEnableTopics] = useState(false);
  const [topicConfig, setTopicConfig] = useState<TopicConfigState>(DEFAULT_TOPIC_CONFIG);

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

    // Default all other columns to metadata (excluding the embedding column)
    setSelectedMetadataColumns(
      features.map(f => f.name).filter(name => name !== embeddingCol)
    );

    // Auto-detect ID column
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

    if (portionStrategy === 'ALL') {
      const splits = datasetInfo?.configs[0]?.splits.map(s => s.name) || ['train'];

      if (splits.length > 1) {
        for (const split of splits) {
          const result = await embedHFDataset({
            datasetId,
            collectionName,
            config: datasetInfo?.defaultConfig || undefined,
            split,
            columns: selectedEmbeddingColumns,
            textTemplate: textTemplate || undefined,
            idColumn: idColumn !== 'auto' ? idColumn : undefined,
            metadataColumns,
            portion: { strategy: 'ALL' },
            computeProjections: true,
            batchSize,
            embeddingModel: {
              provider: embeddingProvider,
              modelName,
              ollamaUrl: embeddingProvider === 'OLLAMA' ? ollamaUrl : undefined,
              task: embeddingProvider === 'QWEN' ? qwenTask : undefined,
              taskType: embeddingProvider === 'GEMINI' ? geminiTaskType : undefined,
              prompt: embeddingProvider === 'SENTENCE_TRANSFORMERS' ? (customPrompt || promptName || undefined) : undefined,
            },
            extractTopics: enableTopics || undefined,
            topicConfig: enableTopics ? toTopicConfigInput(topicConfig) : undefined,
          });

          if (result?.error) {
            console.error(`Error embedding split ${split}:`, result.error);
            break;
          }
        }
      } else {
        await embedHFDataset({
          datasetId,
          collectionName,
          config: datasetInfo?.defaultConfig || undefined,
          split: splits[0],
          columns: selectedEmbeddingColumns,
          textTemplate: textTemplate || undefined,
          idColumn: idColumn !== 'auto' ? idColumn : undefined,
          metadataColumns,
          portion: { strategy: 'ALL' },
          computeProjections: true,
          batchSize,
          embeddingModel: {
            provider: embeddingProvider,
            modelName,
            ollamaUrl: embeddingProvider === 'OLLAMA' ? ollamaUrl : undefined,
            task: embeddingProvider === 'QWEN' ? qwenTask : undefined,
            taskType: embeddingProvider === 'GEMINI' ? geminiTaskType : undefined,
            prompt: embeddingProvider === 'SENTENCE_TRANSFORMERS' ? (customPrompt || promptName || undefined) : undefined,
          },
          extractTopics: enableTopics || undefined,
          topicConfig: enableTopics ? toTopicConfigInput(topicConfig) : undefined,
        });
      }
    } else {
      await embedHFDataset({
        datasetId,
        collectionName,
        config: datasetInfo?.defaultConfig || undefined,
        split: selectedSplit,
        columns: selectedEmbeddingColumns,
        textTemplate: textTemplate || undefined,
        idColumn: idColumn !== 'auto' ? idColumn : undefined,
        metadataColumns,
        portion: {
          strategy: portionStrategy,
          n: portionStrategy === 'FIRST_N' || portionStrategy === 'RANDOM_SAMPLE' ? numRows : undefined,
          start: portionStrategy === 'ROW_RANGE' ? rangeStart : undefined,
          end: portionStrategy === 'ROW_RANGE' ? rangeEnd : undefined,
          seed: portionStrategy === 'RANDOM_SAMPLE' ? randomSeed : undefined,
        },
        computeProjections: true,
        batchSize,
        embeddingModel: {
          provider: embeddingProvider,
          modelName,
          ollamaUrl: embeddingProvider === 'OLLAMA' ? ollamaUrl : undefined,
          task: embeddingProvider === 'QWEN' ? qwenTask : undefined,
          taskType: embeddingProvider === 'GEMINI' ? geminiTaskType : undefined,
          prompt: embeddingProvider === 'SENTENCE_TRANSFORMERS' ? (customPrompt || promptName || undefined) : undefined,
        },
        extractTopics: enableTopics || undefined,
        topicConfig: enableTopics ? toTopicConfigInput(topicConfig) : undefined,
      });
    }

    await refreshCollections();
  };

  const handleProviderChange = (provider: EmbeddingProvider) => {
    setEmbeddingProvider(provider);
    setModelName(EMBEDDING_PROVIDERS[provider].defaultModel);
  };

  const handleResumeJob = async (job: EmbeddingJob) => {
    // Handle LLM labeling jobs separately
    if (job.jobType === 'llm_labeling' && generateLlmLabels) {
      const llmConfig = job.config as { collection_name?: string; llm_provider?: string; llm_model?: string; label_scope?: string };
      const jobId = `${llmConfig.collection_name || job.collectionName}_llm_labeling`;
      setLlmResumeJobId(jobId);
      await generateLlmLabels({
        collectionName: llmConfig.collection_name || job.collectionName,
        llmProvider: llmConfig.llm_provider || 'gemini',
        llmModel: llmConfig.llm_model || 'gemini-3-flash-preview',
        labelScope: llmConfig.label_scope || 'both',
        resume: true,
      });
      setLlmResumeJobId(null);
      await refreshCollections();
      return;
    }

    // Resume embedding with the stored configuration
    // Config is stored with Python snake_case - need to transform for GraphQL
    const config = job.config as Record<string, unknown>;

    // Transform portion from snake_case strategy value to uppercase enum
    const storedPortion = config.portion as Record<string, unknown> | undefined;
    const portion = storedPortion ? {
      strategy: (storedPortion.strategy as string)?.toUpperCase() as PortionStrategy,
      n: storedPortion.n as number | undefined,
      start: storedPortion.start as number | undefined,
      end: storedPortion.end as number | undefined,
      seed: storedPortion.seed as number | undefined,
    } : undefined;

    // Transform embedding_model from snake_case to camelCase
    const storedModel = config.embedding_model as Record<string, unknown> | undefined;
    const embeddingModel = storedModel ? {
      provider: (storedModel.provider as string)?.toUpperCase() as EmbeddingProvider,
      modelName: storedModel.model_name as string,
      ollamaUrl: storedModel.ollama_url as string | undefined,
      task: storedModel.task as string | undefined,
      taskType: storedModel.task_type as GeminiTaskType | undefined,
      prompt: (storedModel.prompt ?? storedModel.prompt_name) as string | undefined,
    } : undefined;

    await embedHFDataset({
      datasetId: config.dataset_id as string,
      collectionName: job.collectionName,
      config: config.config as string | undefined,
      split: config.split as string | undefined,
      columns: config.columns as string[] | undefined,
      textTemplate: config.text_template as string | undefined,
      idColumn: config.id_column as string | undefined,
      metadataColumns: config.metadata_columns as string[] | undefined,
      portion,
      computeProjections: true,
      batchSize: config.batch_size as number | undefined,
      embeddingModel,
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
                <Label htmlFor="batch-size">Batch Size</Label>
                <Input
                  id="batch-size"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  placeholder={batchSize.toString()}
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
              {embeddingProvider === 'SENTENCE_TRANSFORMERS' && (
                <>
                  <div className="space-y-2">
                    <Label htmlFor="prompt-name">Prompt Name</Label>
                    <Select
                      value={promptName ?? 'none'}
                      onValueChange={(v) => setPromptName(v === 'none' ? null : v)}
                    >
                      <SelectTrigger id="prompt-name">
                        <SelectValue placeholder="None" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">None</SelectItem>
                        <SelectItem value="Retrieval-document">Retrieval-document (RAG storage)</SelectItem>
                        <SelectItem value="Retrieval-query">Retrieval-query (RAG search)</SelectItem>
                        <SelectItem value="STS">STS (Sentence similarity)</SelectItem>
                        <SelectItem value="Classification">Classification</SelectItem>
                        <SelectItem value="Clustering">Clustering</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Task-specific prompt for models like Gemma Embedding
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="custom-prompt">Custom Prompt (Advanced)</Label>
                    <Input
                      id="custom-prompt"
                      value={customPrompt}
                      onChange={(e) => setCustomPrompt(e.target.value)}
                      placeholder="Leave empty to use prompt name"
                    />
                  </div>
                </>
              )}
            </div>

            {/* Topic Extraction Toggle */}
            <Separator />
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="hf-enable-topics"
                  checked={enableTopics}
                  onCheckedChange={(checked) => setEnableTopics(checked === true)}
                />
                <Label htmlFor="hf-enable-topics" className="cursor-pointer">Extract topics after embedding</Label>
              </div>
              {enableTopics && (
                <TopicConfigForm value={topicConfig} onChange={setTopicConfig} />
              )}
            </div>

            <Separator />

            {/* Embed Button */}
            {isDataLoaded && (
              <Button
                onClick={handleEmbed}
                disabled={embedLoading || selectedEmbeddingColumns.length === 0}
                size="lg"
                className="w-full md:w-auto"
              >
                {embedLoading ? <Spinner className="mr-2 h-4 w-4" /> : null}
                Embed Dataset
              </Button>
            )}
          </CardContent>
        </Card>
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

      {/* Progress Modal - shown during embedding */}
      {embedLoading && activeJobCollectionName && (
        <ProgressModal
          jobId={activeJobCollectionName}
        />
      )}

      {/* Progress Modal - shown during LLM labeling resume */}
      {llmResumeJobId && (
        <ProgressModal
          jobId={llmResumeJobId}
          title="Generating LLM Labels"
          subtitle="Each topic is labeled individually via LLM API calls."
          itemsLabel="topics"
        />
      )}

      {/* Interrupted Jobs Panel */}
      {!embedLoading && (
        <JobsPanel
          statusFilter="interrupted"
          onResumeJob={handleResumeJob}
        />
      )}
    </div>
  );
}
