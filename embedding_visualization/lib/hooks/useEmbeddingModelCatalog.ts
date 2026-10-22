'use client';

import { useMemo } from 'react';
import { useQuery } from '@apollo/client/react';
import { GET_EMBEDDING_MODEL_CATALOG } from '../graphql/queries';
import type { EmbeddingModelCatalogEntry } from '../graphql/mutations';
import { ModelSuggestions } from '../utils/modelSuggestions';

interface CatalogData {
  embeddingModelCatalog: EmbeddingModelCatalogEntry[];
}

export interface EmbeddingModelCatalogState {
  /** Suggestions and lookups over the catalog (empty until it loads, or when it fails). */
  suggestions: ModelSuggestions;
  loading: boolean;
  /** Why the query failed (null otherwise); `refetch` retries it. */
  error: string | null;
  refetch: () => void;
}

/**
 * The embed forms' model catalog, fetched once per page (cache-first) and
 * shared by every form that asks for it. A failed fetch leaves it empty: the
 * model field still accepts any name.
 */
export function useEmbeddingModelCatalog(): EmbeddingModelCatalogState {
  const { data, loading, error, refetch } = useQuery<CatalogData>(GET_EMBEDDING_MODEL_CATALOG, {
    fetchPolicy: 'cache-first',
    notifyOnNetworkStatusChange: true,
  });
  const entries = data?.embeddingModelCatalog;
  const suggestions = useMemo(() => new ModelSuggestions(entries ?? []), [entries]);
  return {
    suggestions,
    loading,
    error: error && !loading ? error.message : null,
    refetch: () => void refetch(),
  };
}
