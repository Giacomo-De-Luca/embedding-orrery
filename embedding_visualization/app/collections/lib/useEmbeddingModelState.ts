'use client';

import { useState, useCallback, useMemo } from 'react';
import type { EmbeddingProvider, GeminiTaskType, EmbeddingModelInput, TopicConfigInput } from '@/lib/graphql/mutations';
import type { TopicConfigState } from '../components/TopicConfigForm';
import { DEFAULT_TOPIC_CONFIG, toTopicConfigInput } from '../components/TopicConfigForm';
import { EMBEDDING_PROVIDERS } from '@/lib/utils/embeddingProviders';
import type { SaePostEmbedParams } from './embeddingFormUtils';
import { embeddingModelInputFrom } from './embeddingLimits';
import {
  effectiveTaskPrefix,
  taskPrefixInput,
  taskPrefixesFor,
  type TaskPrefixInfo,
  type TaskPrefixValue,
} from './taskPrefixes';
import { useTaskPrefixTable } from './useTaskPrefixTable';

export interface EmbeddingModelState {
  // Model config
  embeddingProvider: EmbeddingProvider;
  /** The model name as typed (may be empty when the form has a fallback model). */
  modelName: string;
  /** The model the job runs: the typed name, else the fallback (the multimodal default). */
  effectiveModelName: string;
  ollamaUrl: string;
  qwenTask: string;
  geminiTaskType: GeminiTaskType;
  /** Sentence-transformers prompt: a prompt name of the model, or literal prompt text ('' = none). */
  prompt: string;
  batchSize: number;
  /** Input-length cap in tokens; null = the model's own limit (local providers only). */
  maxLength: number | null;
  /** Tasks of a model that reads its task from a prefix (EmbeddingGemma, Gemini Embedding 2); null otherwise. */
  taskPrefixInfo: TaskPrefixInfo | null;
  /** The task and document prefix in effect (the default task until one is picked); null without taskPrefixInfo. */
  taskPrefix: TaskPrefixValue | null;
  /** The task-prefix table is still loading. */
  taskPrefixesLoading: boolean;

  // Setters
  setModelName: (m: string) => void;
  setOllamaUrl: (u: string) => void;
  setQwenTask: (t: string) => void;
  setGeminiTaskType: (t: GeminiTaskType) => void;
  setPrompt: (p: string) => void;
  setBatchSize: (b: number) => void;
  setMaxLength: (m: number | null) => void;
  setTaskPrefix: (v: TaskPrefixValue) => void;

  // Topic extraction
  enableTopics: boolean;
  setEnableTopics: (e: boolean) => void;
  topicConfig: TopicConfigState;
  setTopicConfig: (c: TopicConfigState) => void;

  // Post-embed SAE step (link + compute document activations)
  enableSaeActivations: boolean;
  setEnableSaeActivations: (e: boolean) => void;
  saeSelection: string | null;
  setSaeSelection: (s: string | null) => void;

  // Derived helpers
  /** Switch provider; the model name becomes `modelName`, else the provider's default. */
  handleProviderChange: (provider: EmbeddingProvider, modelName?: string) => void;
  buildEmbeddingModelInput: () => EmbeddingModelInput;
  getTopicParams: () => { extractTopics?: boolean; topicConfig?: TopicConfigInput };
  getSaePostParams: () => SaePostEmbedParams;
}

export interface EmbeddingModelStateOptions {
  /**
   * The model an empty name stands for (the backend fills it in, e.g. the
   * multimodal default): its task prefixes and limits apply.
   */
  fallbackModelName?: string | null;
}

export function useEmbeddingModelState(options: EmbeddingModelStateOptions = {}): EmbeddingModelState {
  const [embeddingProvider, setEmbeddingProvider] = useState<EmbeddingProvider>('SENTENCE_TRANSFORMERS');
  const [modelName, setModelNameState] = useState(EMBEDDING_PROVIDERS.SENTENCE_TRANSFORMERS.defaultModel);
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
  const [qwenTask, setQwenTask] = useState('Given a web search query, retrieve relevant passages that answer the query');
  const [geminiTaskType, setGeminiTaskType] = useState<GeminiTaskType>('SEMANTIC_SIMILARITY');
  const [prompt, setPrompt] = useState('');
  const [batchSize, setBatchSize] = useState(EMBEDDING_PROVIDERS.SENTENCE_TRANSFORMERS.defaultBatchSize);
  const [maxLength, setMaxLength] = useState<number | null>(null);
  // null until a task is picked: the model's default task applies.
  const [taskPrefixChoice, setTaskPrefix] = useState<TaskPrefixValue | null>(null);
  const effectiveModelName = modelName.trim() || options.fallbackModelName || '';
  const prefixTable = useTaskPrefixTable();
  const taskPrefixInfo = useMemo(
    () => taskPrefixesFor(prefixTable.table, embeddingProvider, effectiveModelName),
    [prefixTable.table, embeddingProvider, effectiveModelName],
  );

  const [enableTopics, setEnableTopics] = useState(false);
  const [topicConfig, setTopicConfig] = useState<TopicConfigState>(DEFAULT_TOPIC_CONFIG);

  const [enableSaeActivations, setEnableSaeActivations] = useState(false);
  const [saeSelection, setSaeSelection] = useState<string | null>(null);

  // A cap typed for one model may exceed another's limit: a new model starts
  // from its own limit again.
  const setModelName = useCallback((name: string) => {
    setModelNameState(name);
    setMaxLength(null);
  }, []);

  const handleProviderChange = useCallback((provider: EmbeddingProvider, name?: string) => {
    setEmbeddingProvider(provider);
    setModelName(name ?? EMBEDDING_PROVIDERS[provider].defaultModel);
    setBatchSize(EMBEDDING_PROVIDERS[provider].defaultBatchSize);
  }, [setModelName]);

  const buildEmbeddingModelInput = useCallback((): EmbeddingModelInput => embeddingModelInputFrom({
    embeddingProvider,
    modelName,
    ollamaUrl,
    qwenTask,
    geminiTaskType,
    prompt,
    maxLength,
    taskPrefix: taskPrefixInput(taskPrefixInfo, taskPrefixChoice),
  }), [embeddingProvider, modelName, ollamaUrl, qwenTask, geminiTaskType, prompt, maxLength, taskPrefixInfo, taskPrefixChoice]);

  const getTopicParams = useCallback(() => ({
    extractTopics: enableTopics || undefined,
    topicConfig: enableTopics ? toTopicConfigInput(topicConfig) : undefined,
  }), [enableTopics, topicConfig]);

  const getSaePostParams = useCallback((): SaePostEmbedParams => ({
    enabled: enableSaeActivations,
    selection: saeSelection,
  }), [enableSaeActivations, saeSelection]);

  return {
    embeddingProvider,
    modelName,
    effectiveModelName,
    ollamaUrl,
    qwenTask,
    geminiTaskType,
    prompt,
    batchSize,
    maxLength,
    taskPrefixInfo,
    taskPrefix: effectiveTaskPrefix(taskPrefixInfo, taskPrefixChoice),
    taskPrefixesLoading: prefixTable.loading,

    setModelName,
    setOllamaUrl,
    setQwenTask,
    setGeminiTaskType,
    setPrompt,
    setBatchSize,
    setMaxLength,
    setTaskPrefix,

    enableTopics,
    setEnableTopics,
    topicConfig,
    setTopicConfig,

    enableSaeActivations,
    setEnableSaeActivations,
    saeSelection,
    setSaeSelection,

    handleProviderChange,
    buildEmbeddingModelInput,
    getTopicParams,
    getSaePostParams,
  };
}
