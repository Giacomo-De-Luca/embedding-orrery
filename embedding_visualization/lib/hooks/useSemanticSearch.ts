'use client';

import { useCallback, useMemo, useState } from 'react';
import { useApolloClient } from '@apollo/client/react';
import { createSemanticSearchClient } from '../utils/semanticSearchClient';
import type { SemanticSearchResult, DistanceMetric, FilterInput } from '../types/types';

/**
 * Hook for performing semantic similarity search on embeddings.
 *
 * Backed by `createSemanticSearchClient` (one-shot client.query calls with
 * abort-previous semantics — see that module for why lazy queries and query
 * deduplication are deliberately avoided here).
 *
 * Return contract of findSimilarByQuery/findSimilarById:
 * - `SemanticSearchResult[]` — search completed ([] = genuinely no matches)
 * - `null` — superseded by a newer search (aborted); callers should leave
 *   their current results untouched
 * - throws — real network/GraphQL error
 */
export function useSemanticSearch(collectionName: string | null) {
  const client = useApolloClient();
  const searchClient = useMemo(() => createSemanticSearchClient(client), [client]);

  const [inFlight, setInFlight] = useState(0);

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

      console.log(`Searching for items similar to query: "${query}" (metric: ${similarityMeasure}${queryPrompt ? `, prompt: ${queryPrompt}` : ''}${filters?.length ? `, filters: ${filters.length}` : ''})`);
      setInFlight((n) => n + 1);
      try {
        const results = await searchClient.searchByQuery({
          collectionName,
          query,
          nResults,
          similarityMeasure,
          queryPrompt,
          filters,
        });
        if (results) console.log(`Found ${results.length} similar items to "${query}"`);
        return results;
      } catch (err) {
        console.error('Error finding similar items:', err);
        throw err;
      } finally {
        setInFlight((n) => n - 1);
      }
    },
    [collectionName, searchClient]
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

      console.log(`Searching for items similar to: "${itemId}" (by ID, metric: ${similarityMeasure}${filters?.length ? `, filters: ${filters.length}` : ''})`);
      setInFlight((n) => n + 1);
      try {
        const results = await searchClient.searchById({
          collectionName,
          itemId,
          nResults,
          similarityMeasure,
          filters,
        });
        if (results) console.log(`Found ${results.length} similar items to "${itemId}"`);
        return results;
      } catch (err) {
        console.error('Error finding similar items by ID:', err);
        throw err;
      } finally {
        setInFlight((n) => n - 1);
      }
    },
    [collectionName, searchClient]
  );

  return {
    findSimilarByQuery,
    findSimilarById,
    loading: inFlight > 0,
  };
}
