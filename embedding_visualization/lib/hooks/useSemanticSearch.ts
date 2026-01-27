'use client';

import { useCallback } from 'react';
import { useLazyQuery } from '@apollo/client/react';
import { SEMANTIC_SEARCH, SEMANTIC_SEARCH_BY_ID } from '../graphql/queries';
import type { SemanticSearchResult, DistanceMetric } from '../types/types';

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
    useLazyQuery<SemanticSearchData>(SEMANTIC_SEARCH);

  const [searchSimilarById, { data: dataById, loading: loadingById, error: errorById }] =
    useLazyQuery<SemanticSearchByIdData>(SEMANTIC_SEARCH_BY_ID);

  /**
   * Find items semantically similar to the query text (embeds the query)
   */
  const findSimilarByQuery = useCallback(
    async (
      query: string,
      nResults: number = 10,
      similarityMeasure: DistanceMetric = 'COSINE',
      queryPromptName?: string | null
    ): Promise<SemanticSearchResult[] | null> => {
      if (!collectionName) {
        console.warn('Cannot search: no collection selected');
        return null;
      }

      try {
        console.log(`Searching for items similar to query: "${query}" (metric: ${similarityMeasure}${queryPromptName ? `, prompt: ${queryPromptName}` : ''})`);

        const result = await searchSimilar({
          variables: {
            collectionName,
            query,
            nResults,
            similarityMeasure,
            queryPromptName: queryPromptName || undefined,
          },
        });

        if (result.data?.semanticSearch) {
          const results = result.data.semanticSearch;
          console.log(`Found ${results.length} similar items to "${query}"`);
          return results;
        }

        return null;
      } catch (err) {
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
      similarityMeasure: DistanceMetric = 'COSINE'
    ): Promise<SemanticSearchResult[] | null> => {
      if (!collectionName) {
        console.warn('Cannot search: no collection selected');
        return null;
      }

      try {
        console.log(`Searching for items similar to: "${itemId}" (by ID, metric: ${similarityMeasure})`);

        const result = await searchSimilarById({
          variables: {
            collectionName,
            itemId,
            nResults,
            similarityMeasure,
          },
        });

        if (result.data?.semanticSearchById) {
          const results = result.data.semanticSearchById;
          console.log(`Found ${results.length} similar items to "${itemId}"`);
          return results;
        }

        return null;
      } catch (err) {
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
