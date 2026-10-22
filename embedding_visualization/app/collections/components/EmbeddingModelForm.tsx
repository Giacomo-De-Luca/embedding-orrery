'use client';

import { useState } from 'react';
import { Input } from '@/lib/ui-primitives/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { Label } from '@/lib/ui-primitives/label';
import { Separator } from '@/lib/ui-primitives/separator';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import type { EmbeddingProvider, GeminiTaskType } from '@/lib/graphql/mutations';
import { EMBEDDING_PROVIDERS, isLocalProvider } from '@/lib/utils/embeddingProviders';
import { TopicConfigForm } from './TopicConfigForm';
import type { EmbeddingModelState } from '../lib/useEmbeddingModelState';
import { commitMaxLengthDraft, describeModelLimit, parseMaxLengthDraft } from '../lib/embeddingLimits';
import { useModelMaxLength } from '../lib/useModelMaxLength';
import { NO_TASK } from '../lib/taskPrefixes';
import { TaskPrefixField } from './TaskPrefixField';

interface EmbeddingModelFormProps {
  model: EmbeddingModelState;
  showTopics?: boolean;
  idPrefix?: string;
  title?: string;
  /**
   * Render the task prefix here, with these columns for its placeholders, in
   * forms without a column section (SAE label maps). Otherwise the Column
   * Configuration section holds it.
   */
  taskPrefixColumns?: string[];
}

export function EmbeddingModelForm({
  model,
  showTopics = true,
  idPrefix = '',
  title = 'Embedding Model',
  taskPrefixColumns,
}: EmbeddingModelFormProps) {
  // Draft so the field can be temporarily empty/invalid while typing; the
  // committed batchSize only updates on valid positive integers and the
  // field snaps back to the committed value on blur.
  const [batchSizeDraft, setBatchSizeDraft] = useState<string | null>(null);
  // Length cap: the draft is committed on blur only (valid, or the previous
  // value is kept); empty commits null (= the model limit).
  const [maxLengthDraft, setMaxLengthDraft] = useState<string | null>(null);
  const isLocal = isLocalProvider(model.embeddingProvider);
  const modelLimit = useModelMaxLength(model.embeddingProvider, model.modelName);
  const limit = modelLimit.limit?.maxLength ?? null;
  const maxLengthError = maxLengthDraft !== null
    ? parseMaxLengthDraft(maxLengthDraft, limit).error
    : model.maxLength !== null && limit !== null && model.maxLength > limit
      ? parseMaxLengthDraft(String(model.maxLength), limit).error
      : null;
  // Task-prefix models take neither prompt names nor the Gemini task type.
  const prefixInfo = model.taskPrefixInfo;
  const prefixValue = model.taskPrefix;
  const taskLabel = prefixValue?.taskId === NO_TASK
    ? 'None'
    : prefixInfo?.tasks.find((t) => t.id === prefixValue?.taskId)?.label;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>
          Choose the embedding model to use
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor={`${idPrefix}provider`}>Provider</Label>
            <Select
              value={model.embeddingProvider}
              onValueChange={(v) => model.handleProviderChange(v as EmbeddingProvider)}
            >
              <SelectTrigger id={`${idPrefix}provider`}>
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
              {EMBEDDING_PROVIDERS[model.embeddingProvider].description}
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor={`${idPrefix}model-name`}>Model Name</Label>
            <Input
              id={`${idPrefix}model-name`}
              value={model.modelName}
              onChange={(e) => model.setModelName(e.target.value)}
              placeholder={EMBEDDING_PROVIDERS[model.embeddingProvider].defaultModel}
            />
            <Label htmlFor={`${idPrefix}batch-size`}>Batch Size</Label>
            <Input
              id={`${idPrefix}batch-size`}
              type="number"
              min={1}
              value={batchSizeDraft ?? model.batchSize}
              onChange={(e) => {
                setBatchSizeDraft(e.target.value);
                const parsed = parseInt(e.target.value, 10);
                if (!Number.isNaN(parsed) && parsed > 0) model.setBatchSize(parsed);
              }}
              onBlur={() => setBatchSizeDraft(null)}
              placeholder={model.batchSize.toString()}
            />
            <p className="text-xs text-muted-foreground">
              {isLocal
                ? 'Texts per forward pass. Lower it for long documents: memory grows with batch × length.'
                : 'Texts per API request.'}
            </p>
            {isLocal && (
              <>
                <Label htmlFor={`${idPrefix}max-length`}>Max Length (tokens)</Label>
                <Input
                  id={`${idPrefix}max-length`}
                  type="number"
                  min={1}
                  max={limit ?? undefined}
                  value={maxLengthDraft ?? (model.maxLength === null ? '' : model.maxLength.toString())}
                  onChange={(e) => setMaxLengthDraft(e.target.value)}
                  onBlur={() => {
                    if (maxLengthDraft !== null) {
                      model.setMaxLength(commitMaxLengthDraft(maxLengthDraft, limit, model.maxLength));
                    }
                    setMaxLengthDraft(null);
                  }}
                  placeholder={limit !== null ? limit.toString() : 'Model default'}
                  aria-invalid={maxLengthError ? true : undefined}
                />
                <p className="text-xs text-muted-foreground">
                  {describeModelLimit(modelLimit.limit, modelLimit.loading)}
                </p>
                {maxLengthError && <p className="text-xs text-destructive">{maxLengthError}</p>}
              </>
            )}
          </div>
          {model.embeddingProvider === 'OLLAMA' && (
            <div className="space-y-2">
              <Label htmlFor={`${idPrefix}ollama-url`}>Ollama URL</Label>
              <Input
                id={`${idPrefix}ollama-url`}
                value={model.ollamaUrl}
                onChange={(e) => model.setOllamaUrl(e.target.value)}
                placeholder="http://localhost:11434"
              />
            </div>
          )}
          {model.embeddingProvider === 'QWEN' && (
            <div className="space-y-2 md:col-span-2">
              <Label htmlFor={`${idPrefix}qwen-task`}>Query Task Instruction</Label>
              <Input
                id={`${idPrefix}qwen-task`}
                value={model.qwenTask}
                onChange={(e) => model.setQwenTask(e.target.value)}
                placeholder="Given a web search query, retrieve relevant passages that answer the query"
              />
              <p className="text-xs text-muted-foreground">
                Instruction prefix added to queries during semantic search (not used during document embedding)
              </p>
            </div>
          )}
          {prefixInfo && prefixValue && (
            <div className="space-y-2 md:col-span-2">
              {taskPrefixColumns ? (
                <>
                  <Label>Task Prefix</Label>
                  <TaskPrefixField
                    info={prefixInfo}
                    value={prefixValue}
                    onChange={model.setTaskPrefix}
                    columns={taskPrefixColumns}
                    idPrefix={idPrefix}
                  />
                </>
              ) : (
                <p className="text-xs text-muted-foreground">
                  Task prefix: <strong>{taskLabel}</strong>, set under Column Configuration.
                </p>
              )}
            </div>
          )}
          {model.embeddingProvider === 'GEMINI' && !prefixInfo && (
            <div className="space-y-2">
              <Label htmlFor={`${idPrefix}gemini-task-type`}>Task Type</Label>
              <Select
                value={model.geminiTaskType}
                onValueChange={(v) => model.setGeminiTaskType(v as GeminiTaskType)}
              >
                <SelectTrigger id={`${idPrefix}gemini-task-type`}>
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
          {model.embeddingProvider === 'SENTENCE_TRANSFORMERS' && !prefixInfo && (
            <>
              <div className="space-y-2">
                <Label htmlFor={`${idPrefix}prompt-name`}>Prompt Name</Label>
                <Select
                  value={model.promptName ?? 'none'}
                  onValueChange={(v) => model.setPromptName(v === 'none' ? null : v)}
                >
                  <SelectTrigger id={`${idPrefix}prompt-name`}>
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
                <Label htmlFor={`${idPrefix}custom-prompt`}>Custom Prompt (Advanced)</Label>
                <Input
                  id={`${idPrefix}custom-prompt`}
                  value={model.customPrompt}
                  onChange={(e) => model.setCustomPrompt(e.target.value)}
                  placeholder="Leave empty to use prompt name"
                />
              </div>
            </>
          )}
        </div>

        {/* Topic Extraction Toggle */}
        {showTopics && (
          <>
            <Separator />
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Checkbox
                  id={`${idPrefix}enable-topics`}
                  checked={model.enableTopics}
                  onCheckedChange={(checked) => model.setEnableTopics(checked === true)}
                />
                <Label htmlFor={`${idPrefix}enable-topics`} className="cursor-pointer">Extract topics after embedding</Label>
              </div>
              {model.enableTopics && (
                <TopicConfigForm value={model.topicConfig} onChange={model.setTopicConfig} />
              )}
            </div>
          </>
        )}

      </CardContent>
    </Card>
  );
}
