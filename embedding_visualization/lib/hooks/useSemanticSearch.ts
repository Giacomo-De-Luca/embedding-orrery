'use client';

import { useCallback, useRef } from 'react';
import { useLazyQuery } from '@apollo/client/react';
import { SEMANTIC_SEARCH, SEMANTIC_SEARCH_BY_ID } from '../graphql/queries';
import type { SemanticSearchResult, DistanceMetric, FilterInput } from '../types/types';

interface SemanticSearchData {
  semanticSearch: SemanticSearchResult[];
}

interface SemanticSearchByIdData {
  semanticSearchById: SemanticSearchResult[];
}

/**
 * Hook for performing semantic similarity search on embeddings
 */
export function useSemanticSearch(collectionName: string | null) {
  const [searchSimilar, { data: dataByQuery, loading: loadingByQuery, error: errorByQuery }] =
    useLazyQuery<SemanticSearchData>(SEMANTIC_SEARCH, { fetchPolicy: 'no-cache' });

  const [searchSimilarById, { data: dataById, loading: loadingById, error: errorById }] =
    useLazyQuery<SemanticSearchByIdData>(SEMANTIC_SEARCH_BY_ID, { fetchPolicy: 'no-cache' });

  // Shared abort controller: any new search (by-query or by-ID) cancels the previous in-flight request
  const abortControllerRef = useRef<AbortController | null>(null);

  /**
   * Find items semantically similar to the query text (embeds the query)
   */
  const findSimilarByQuery = useCallback(
    async (
      query: string,
      nResults: number = 10,
      similarityMeasure: DistanceMetric = 'COSINE',
      queryPrompt?: string | null,
      filters?: FilterInput[]
    ): Promise<SemanticSearchResult[] | null> => {
      if (!collectionName) {
        console.warn('Cannot search: no collection selected');
        return null;
      }

      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        console.log(`Searching for items similar to query: "${query}" (metric: ${similarityMeasure}${queryPrompt ? `, prompt: ${queryPrompt}` : ''}${filters?.length ? `, filters: ${filters.length}` : ''})`);

        const result = await searchSimilar({
          variables: {
            collectionName,
            query,
            nResults,
            similarityMeasure,
            queryPrompt: queryPrompt || undefined,
            filters: filters?.length ? filters : undefined,
          },
          context: { fetchOptions: { signal: controller.signal } },
        });

        if (result.data?.semanticSearch) {
          const results = result.data.semanticSearch;
          console.log(`Found ${results.length} similar items to "${query}"`);
          return results;
        }

        return null;
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return null;
        console.error('Error finding similar items:', err);
        throw err;
      }
    },
    [collectionName, searchSimilar]
  );

  /**
   * Find items semantically similar to an existing item (uses item's embedding, faster)
   */
  const findSimilarById = useCallback(
    async (
      itemId: string,
      nResults: number = 10,
      similarityMeasure: DistanceMetric = 'COSINE',
      filters?: FilterInput[]
    ): Promise<SemanticSearchResult[] | null> => {
      if (!collectionName) {
        console.warn('Cannot search: no collection selected');
        return null;
      }

      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        console.log(`Searching for items similar to: "${itemId}" (by ID, metric: ${similarityMeasure}${filters?.length ? `, filters: ${filters.length}` : ''})`);

        const result = await searchSimilarById({
          variables: {
            collectionName,
            itemId,
            nResults,
            similarityMeasure,
            filters: filters?.length ? filters : undefined,
          },
          context: { fetchOptions: { signal: controller.signal } },
        });

        if (result.data?.semanticSearchById) {
          const results = result.data.semanticSearchById;
          console.log(`Found ${results.length} similar items to "${itemId}"`);
          return results;
        }

        return null;
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return null;
        console.error('Error finding similar items by ID:', err);
        throw err;
      }
    },
    [collectionName, searchSimilarById]
  );

  return {
    findSimilarByQuery,
    findSimilarById,
    // Legacy support - defaults to query-based search
    findSimilar: findSimilarByQuery,
    results: dataByQuery?.semanticSearch ?? dataById?.semanticSearchById ?? null,
    loading: loadingByQuery || loadingById,
    error: errorByQuery || errorById || null,
  };
}
