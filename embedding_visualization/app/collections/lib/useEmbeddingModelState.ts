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
  modelName: string;
  ollamaUrl: string;
  qwenTask: string;
  geminiTaskType: GeminiTaskType;
  promptName: string | null;
  customPrompt: string;
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
  setPromptName: (p: string | null) => void;
  setCustomPrompt: (p: string) => void;
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
  handleProviderChange: (provider: EmbeddingProvider) => void;
  buildEmbeddingModelInput: () => EmbeddingModelInput;
  getTopicParams: () => { extractTopics?: boolean; topicConfig?: TopicConfigInput };
  getSaePostParams: () => SaePostEmbedParams;
}

export function useEmbeddingModelState(): EmbeddingModelState {
  const [embeddingProvider, setEmbeddingProvider] = useState<EmbeddingProvider>('SENTENCE_TRANSFORMERS');
  const [modelName, setModelNameState] = useState(EMBEDDING_PROVIDERS.SENTENCE_TRANSFORMERS.defaultModel);
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
  const [qwenTask, setQwenTask] = useState('Given a web search query, retrieve relevant passages that answer the query');
  const [geminiTaskType, setGeminiTaskType] = useState<GeminiTaskType>('SEMANTIC_SIMILARITY');
  const [promptName, setPromptName] = useState<string | null>(null);
  const [customPrompt, setCustomPrompt] = useState('');
  const [batchSize, setBatchSize] = useState(EMBEDDING_PROVIDERS.SENTENCE_TRANSFORMERS.defaultBatchSize);
  const [maxLength, setMaxLength] = useState<number | null>(null);
  // null until a task is picked: the model's default task applies.
  const [taskPrefixChoice, setTaskPrefix] = useState<TaskPrefixValue | null>(null);
  const prefixTable = useTaskPrefixTable();
  const taskPrefixInfo = useMemo(
    () => taskPrefixesFor(prefixTable.table, embeddingProvider, modelName),
    [prefixTable.table, embeddingProvider, modelName],
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

  const handleProviderChange = useCallback((provider: EmbeddingProvider) => {
    setEmbeddingProvider(provider);
    setModelName(EMBEDDING_PROVIDERS[provider].defaultModel);
    setBatchSize(EMBEDDING_PROVIDERS[provider].defaultBatchSize);
  }, [setModelName]);

  const buildEmbeddingModelInput = useCallback((): EmbeddingModelInput => embeddingModelInputFrom({
    embeddingProvider,
    modelName,
    ollamaUrl,
    qwenTask,
    geminiTaskType,
    promptName,
    customPrompt,
    maxLength,
    taskPrefix: taskPrefixInput(taskPrefixInfo, taskPrefixChoice),
  }), [embeddingProvider, modelName, ollamaUrl, qwenTask, geminiTaskType, customPrompt, promptName, maxLength, taskPrefixInfo, taskPrefixChoice]);

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
    ollamaUrl,
    qwenTask,
    geminiTaskType,
    promptName,
    customPrompt,
    batchSize,
    maxLength,
    taskPrefixInfo,
    taskPrefix: effectiveTaskPrefix(taskPrefixInfo, taskPrefixChoice),
    taskPrefixesLoading: prefixTable.loading,

    setModelName,
    setOllamaUrl,
    setQwenTask,
    setGeminiTaskType,
    setPromptName,
    setCustomPrompt,
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
