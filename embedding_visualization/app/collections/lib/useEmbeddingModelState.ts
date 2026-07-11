'use client';

import { useState, useCallback } from 'react';
import type { EmbeddingProvider, GeminiTaskType, EmbeddingModelInput, TopicConfigInput } from '@/lib/graphql/mutations';
import type { TopicConfigState } from '../components/TopicConfigForm';
import { DEFAULT_TOPIC_CONFIG, toTopicConfigInput } from '../components/TopicConfigForm';
import { EMBEDDING_PROVIDERS } from '@/lib/utils/embeddingProviders';
import type { SaePostEmbedParams } from './embeddingFormUtils';

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

  // Setters
  setModelName: (m: string) => void;
  setOllamaUrl: (u: string) => void;
  setQwenTask: (t: string) => void;
  setGeminiTaskType: (t: GeminiTaskType) => void;
  setPromptName: (p: string | null) => void;
  setCustomPrompt: (p: string) => void;
  setBatchSize: (b: number) => void;

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
  const [modelName, setModelName] = useState(EMBEDDING_PROVIDERS.SENTENCE_TRANSFORMERS.defaultModel);
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
  const [qwenTask, setQwenTask] = useState('Given a web search query, retrieve relevant passages that answer the query');
  const [geminiTaskType, setGeminiTaskType] = useState<GeminiTaskType>('SEMANTIC_SIMILARITY');
  const [promptName, setPromptName] = useState<string | null>(null);
  const [customPrompt, setCustomPrompt] = useState('');
  const [batchSize, setBatchSize] = useState(100);

  const [enableTopics, setEnableTopics] = useState(false);
  const [topicConfig, setTopicConfig] = useState<TopicConfigState>(DEFAULT_TOPIC_CONFIG);

  const [enableSaeActivations, setEnableSaeActivations] = useState(false);
  const [saeSelection, setSaeSelection] = useState<string | null>(null);

  const handleProviderChange = useCallback((provider: EmbeddingProvider) => {
    setEmbeddingProvider(provider);
    setModelName(EMBEDDING_PROVIDERS[provider].defaultModel);
  }, []);

  const buildEmbeddingModelInput = useCallback((): EmbeddingModelInput => ({
    provider: embeddingProvider,
    modelName,
    ollamaUrl: embeddingProvider === 'OLLAMA' ? ollamaUrl : undefined,
    task: embeddingProvider === 'QWEN' ? qwenTask : undefined,
    taskType: embeddingProvider === 'GEMINI' ? geminiTaskType : undefined,
    prompt: embeddingProvider === 'SENTENCE_TRANSFORMERS'
      ? (customPrompt || promptName || undefined)
      : undefined,
  }), [embeddingProvider, modelName, ollamaUrl, qwenTask, geminiTaskType, customPrompt, promptName]);

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

    setModelName,
    setOllamaUrl,
    setQwenTask,
    setGeminiTaskType,
    setPromptName,
    setCustomPrompt,
    setBatchSize,

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
