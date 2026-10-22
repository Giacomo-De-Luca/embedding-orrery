'use client';

import { useMemo, useState } from 'react';
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
import { PromptCombobox } from '@/lib/ui-primitives/prompt-combobox';
import type { EmbeddingProvider, GeminiTaskType } from '@/lib/graphql/mutations';
import { EMBEDDING_PROVIDERS, isLocalProvider } from '@/lib/utils/embeddingProviders';
import { useEmbeddingModelCatalog } from '@/lib/hooks/useEmbeddingModelCatalog';
import { PromptPresets } from '@/lib/utils/promptPresets';
import { TopicConfigForm } from './TopicConfigForm';
import { ModelNameCombobox } from './ModelNameCombobox';
import type { MediaKind } from '../lib/mediaColumns';
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
  /**
   * Image / Audio / Video / Mixed mode: the provider is locked to
   * sentence-transformers, an empty model name means `defaultModel` (the
   * backend fills it in), and suggestions are media models that embed `kinds`.
   */
  media?: { kinds: MediaKind[]; defaultModel: string | null } | null;
}

const MEDIA_PROVIDER: EmbeddingProvider = 'SENTENCE_TRANSFORMERS';

export function EmbeddingModelForm({
  model,
  showTopics = true,
  idPrefix = '',
  title = 'Embedding Model',
  taskPrefixColumns,
  media = null,
}: EmbeddingModelFormProps) {
  // Draft so the field can be temporarily empty/invalid while typing; the
  // committed batchSize only updates on valid positive integers and the
  // field snaps back to the committed value on blur.
  const [batchSizeDraft, setBatchSizeDraft] = useState<string | null>(null);
  // Length cap: the draft is committed on blur only (valid, or the previous
  // value is kept); empty commits null (= the model limit).
  const [maxLengthDraft, setMaxLengthDraft] = useState<string | null>(null);
  const isLocal = isLocalProvider(model.embeddingProvider);
  const modelLimit = useModelMaxLength(model.embeddingProvider, model.effectiveModelName);
  const { suggestions } = useEmbeddingModelCatalog();
  // The prompt names the catalog lists for the model; the shared presets otherwise.
  const promptGroups = useMemo(
    () => PromptPresets.documentGroups(suggestions.find(model.embeddingProvider, model.effectiveModelName)?.prompts),
    [suggestions, model.embeddingProvider, model.effectiveModelName],
  );
  const modelPlaceholder = media
    ? (media.defaultModel ?? 'Backend default')
    : EMBEDDING_PROVIDERS[model.embeddingProvider].defaultModel;
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
              disabled={media !== null}
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
              {media
                ? 'Images, audio and video need a sentence-transformers media model.'
                : EMBEDDING_PROVIDERS[model.embeddingProvider].description}
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor={`${idPrefix}model-name`}>Model Name</Label>
            <ModelNameCombobox
              id={`${idPrefix}model-name`}
              provider={media ? MEDIA_PROVIDER : model.embeddingProvider}
              value={model.modelName}
              onChange={model.setModelName}
              placeholder={modelPlaceholder}
              requires={media?.kinds}
              anyMedia={media !== null}
            />
            {media && !model.modelName.trim() && (
              <p className="text-xs text-muted-foreground">
                Empty uses the default{media.defaultModel ? `, ${media.defaultModel}` : ''}.
              </p>
            )}
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
              {media
                ? "Rows per batch, capped by the backend's media settings (images, audio and video are decoded per batch)."
                : isLocal
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
            <div className="space-y-2">
              <Label htmlFor={`${idPrefix}prompt`}>Prompt</Label>
              <PromptCombobox
                id={`${idPrefix}prompt`}
                value={model.prompt}
                onChange={model.setPrompt}
                groups={promptGroups}
                placeholder="None (pick a prompt name or type a prompt)"
              />
              <p className="text-xs text-muted-foreground">
                A prompt name of the model (e.g. Retrieval-document) or literal text placed before each
                document. Queries use the matching query prompt.
              </p>
            </div>
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
