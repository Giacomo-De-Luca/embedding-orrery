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
import type { EmbeddingProvider, GeminiTaskType } from '@/lib/graphql/mutations';
import { EMBEDDING_PROVIDERS, isLocalProvider } from '@/lib/utils/embeddingProviders';
import { ModelNameCombobox } from './ModelNameCombobox';
import type { MediaKind } from '../lib/mediaColumns';
import type { EmbeddingModelState } from '../lib/useEmbeddingModelState';
import {
  commitMaxLengthDraft,
  describeModelLimit,
  modelTaskKind,
  parseMaxLengthDraft,
  type ModelTaskKind,
} from '../lib/embeddingLimits';
import { useModelMaxLength } from '../lib/useModelMaxLength';
import type { TaskPrefixRow } from '../lib/taskPrefixes';
import { TaskPrefixField } from './TaskPrefixField';

interface EmbeddingModelFormProps {
  model: EmbeddingModelState;
  idPrefix?: string;
  title?: string;
  /**
   * The row the task prefix (EmbeddingGemma, Gemini Embedding 2) is written
   * for: the source's columns and the text template (`form.taskPrefixRow`),
   * or STATIC_TASK_PREFIX_ROW in forms without a column section (SAE label
   * maps). Validate against the same row's columns.
   */
  taskPrefixRow: TaskPrefixRow;
  /**
   * Image / Audio / Video / Mixed mode: the provider is locked to
   * sentence-transformers, an empty model name means `defaultModel` (the
   * backend fills it in), and suggestions are media models that embed `kinds`.
   */
  media?: { kinds: MediaKind[]; defaultModel: string | null } | null;
}

const MEDIA_PROVIDER: EmbeddingProvider = 'SENTENCE_TRANSFORMERS';

/** What the "Task prompt" section says each kind of task control does. */
const TASK_DESCRIPTIONS: Record<ModelTaskKind, string> = {
  prefix: 'The model reads its task from a text prefix on documents and queries.',
  taskType: 'Gemini tunes the embeddings for the chosen task.',
  instruction: 'The model puts a one-sentence task instruction before every query; documents get none.',
};

export function EmbeddingModelForm({
  model,
  idPrefix = '',
  title = 'Embedding Model',
  taskPrefixRow,
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
  const modelPlaceholder = media
    ? (media.defaultModel ?? 'Backend default')
    : EMBEDDING_PROVIDERS[model.embeddingProvider].defaultModel;
  const limit = modelLimit.limit?.maxLength ?? null;
  const maxLengthError = maxLengthDraft !== null
    ? parseMaxLengthDraft(maxLengthDraft, limit).error
    : model.maxLength !== null && limit !== null && model.maxLength > limit
      ? parseMaxLengthDraft(String(model.maxLength), limit).error
      : null;
  // The one task control the model reads, if any (embeddingModelInputFrom sends the same).
  const prefixInfo = model.taskPrefixInfo;
  const prefixValue = model.taskPrefix;
  const taskKind = modelTaskKind(model.embeddingProvider, prefixInfo !== null);
  const defaultInstruction = EMBEDDING_PROVIDERS[model.embeddingProvider].queryInstruction;

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
          </div>
          {taskKind && (
            <div className="space-y-3 rounded-md border p-3 md:col-span-2">
              <div>
                <Label className="text-base">Task prompt</Label>
                <p className="text-xs text-muted-foreground mt-1">{TASK_DESCRIPTIONS[taskKind]}</p>
              </div>
              {taskKind === 'prefix' && prefixInfo && prefixValue && (
                <>
                  <TaskPrefixField
                    info={prefixInfo}
                    value={prefixValue}
                    onChange={model.setTaskPrefix}
                    row={taskPrefixRow}
                    idPrefix={idPrefix}
                  />
                  {media && (
                    <p className="text-xs text-muted-foreground">
                      The document prefix applies to a row&apos;s text only; images, audio and video get none.
                    </p>
                  )}
                </>
              )}
              {taskKind === 'taskType' && (
                <div className="space-y-2 md:w-1/2 md:pr-2">
                  <Label htmlFor={`${idPrefix}gemini-task-type`}>Task type</Label>
                  <Select
                    value={model.geminiTaskType}
                    onValueChange={(v) => model.setGeminiTaskType(v as GeminiTaskType)}
                  >
                    <SelectTrigger id={`${idPrefix}gemini-task-type`} className="w-full">
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
                </div>
              )}
              {taskKind === 'instruction' && (
                <div className="space-y-2">
                  <Label htmlFor={`${idPrefix}qwen-task`}>Query instruction</Label>
                  <Input
                    id={`${idPrefix}qwen-task`}
                    value={model.qwenTask}
                    onChange={(e) => model.setQwenTask(e.target.value)}
                    placeholder={defaultInstruction}
                  />
                  <p className="text-xs text-muted-foreground">
                    Empty uses the default: the instruction the model was trained with. Without one,
                    retrieval loses accuracy (about 1 to 5% for Qwen3-Embedding).
                  </p>
                </div>
              )}
            </div>
          )}
          <div className="space-y-2">
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
          </div>
          {isLocal && (
            <div className="space-y-2">
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
            </div>
          )}
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
        </div>
      </CardContent>
    </Card>
  );
}
