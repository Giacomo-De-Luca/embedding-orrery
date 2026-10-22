'use client';

import { useQuery } from '@apollo/client/react';
import { EMBEDDING_MODEL_MAX_LENGTH } from '@/lib/graphql/queries';
import type { EmbeddingProvider } from '@/lib/graphql/mutations';
import { isLocalProvider } from '@/lib/utils/embeddingProviders';
import { useDebounceValue } from '@/lib/hooks/use-debounce-value';
import type { ModelLimitInfo } from './embeddingLimits';

interface MaxLengthData {
  embeddingModelMaxLength: ModelLimitInfo;
}

export interface ModelMaxLengthState {
  limit: ModelLimitInfo | null;
  loading: boolean;
}

interface ModelKey {
  provider: EmbeddingProvider;
  name: string;
}

const sameModel = (a: ModelKey, b: ModelKey) => a.provider === b.provider && a.name === b.name;

/**
 * The input-length limit of the model typed in the embed form. Provider and
 * name are debounced together, so a provider switch never queries the new
 * provider with the old name; until the pair settles the limit reads as
 * loading. Skipped for remote API providers, which take no cap.
 */
export function useModelMaxLength(provider: EmbeddingProvider, modelName: string): ModelMaxLengthState {
  const current: ModelKey = { provider, name: modelName.trim() };
  const [debounced] = useDebounceValue<ModelKey>(current, 400, { equalityFn: sameModel });
  const { data, loading } = useQuery<MaxLengthData>(EMBEDDING_MODEL_MAX_LENGTH, {
    variables: { provider: debounced.provider, modelName: debounced.name },
    skip: !isLocalProvider(debounced.provider) || debounced.name === '',
  });
  if (!isLocalProvider(provider) || current.name === '') return { limit: null, loading: false };
  if (!sameModel(debounced, current)) return { limit: null, loading: true };
  return { limit: data?.embeddingModelMaxLength ?? null, loading };
}
