import type { ApolloClient, DocumentNode } from '@apollo/client';
import { SEMANTIC_SEARCH, SEMANTIC_SEARCH_BY_ID } from '../graphql/queries';
import type { SemanticSearchResult, DistanceMetric, FilterInput } from '../types/types';

interface SemanticSearchData {
  semanticSearch: SemanticSearchResult[];
}

interface SemanticSearchByIdData {
  semanticSearchById: SemanticSearchResult[];
}

export interface SearchByQueryParams {
  collectionName: string;
  query: string;
  nResults: number;
  similarityMeasure: DistanceMetric;
  queryPrompt?: string | null;
  filters?: FilterInput[];
}

export interface SearchByIdParams {
  collectionName: string;
  itemId: string;
  nResults: number;
  similarityMeasure: DistanceMetric;
  filters?: FilterInput[];
}

export interface SemanticSearchClient {
  /** Resolves to results ([] = genuinely no matches), or null when superseded
   * by a newer search (aborted). Throws on real network/GraphQL errors. */
  searchByQuery(params: SearchByQueryParams): Promise<SemanticSearchResult[] | null>;
  searchById(params: SearchByIdParams): Promise<SemanticSearchResult[] | null>;
}

/** True for fetch-abort rejections, raw or wrapped by the Apollo link chain. */
export function isAbortError(err: unknown): boolean {
  if (err instanceof DOMException && err.name === 'AbortError') return true;
  if (typeof err !== 'object' || err === null) return false;
  const { cause, networkError } = err as { cause?: unknown; networkError?: unknown };
  return (
    (cause instanceof DOMException && cause.name === 'AbortError') ||
    (networkError instanceof DOMException && networkError.name === 'AbortError')
  );
}

/**
 * Imperative semantic-search executor. Each new search aborts the previous
 * in-flight one (of either kind); display correctness is guaranteed by the
 * caller's request-id guard, the abort just frees the connection/backend.
 *
 * Uses one-shot `client.query` calls (not a shared lazy-query ObservableQuery)
 * and disables query deduplication: an identical query arriving right after
 * the abort below would otherwise join the in-flight operation it just killed
 * and inherit its AbortError — surfacing as "clicked a point, search silently
 * returned nothing" (gl3d can emit several click events for one physical
 * click, one per GL frame of the press).
 */
export function createSemanticSearchClient(client: ApolloClient): SemanticSearchClient {
  let abortController: AbortController | null = null;

  async function run<TData>(
    query: DocumentNode,
    variables: Record<string, unknown>,
    pick: (data: TData | undefined) => SemanticSearchResult[] | undefined,
  ): Promise<SemanticSearchResult[] | null> {
    abortController?.abort();
    const controller = new AbortController();
    abortController = controller;
    try {
      const result = await client.query<TData>({
        query,
        variables,
        fetchPolicy: 'no-cache',
        context: {
          fetchOptions: { signal: controller.signal },
          queryDeduplication: false,
        },
      });
      return pick(result.data) ?? null;
    } catch (err) {
      // Structural check first — if our own signal fired, this call was
      // superseded regardless of how the link chain wrapped the error.
      if (controller.signal.aborted || isAbortError(err)) return null;
      throw err;
    }
  }

  return {
    searchByQuery({ collectionName, query, nResults, similarityMeasure, queryPrompt, filters }) {
      return run<SemanticSearchData>(
        SEMANTIC_SEARCH,
        {
          collectionName,
          query,
          nResults,
          similarityMeasure,
          queryPrompt: queryPrompt || undefined,
          filters: filters?.length ? filters : undefined,
        },
        (data) => data?.semanticSearch,
      );
    },
    searchById({ collectionName, itemId, nResults, similarityMeasure, filters }) {
      return run<SemanticSearchByIdData>(
        SEMANTIC_SEARCH_BY_ID,
        {
          collectionName,
          itemId,
          nResults,
          similarityMeasure,
          filters: filters?.length ? filters : undefined,
        },
        (data) => data?.semanticSearchById,
      );
    },
  };
}
