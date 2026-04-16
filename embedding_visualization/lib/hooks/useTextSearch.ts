'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLazyQuery } from '@apollo/client/react';
import { TEXT_SEARCH } from '../graphql/queries';
import type { TextSearchConfig, TextSearchMatch } from '../types/types';

interface TextSearchData {
  textSearch: {
    matches: TextSearchMatch[];
    totalMatches: number;
  };
}

interface UseTextSearchResult {
  highlightedIndices: Set<number> | undefined;
  matches: TextSearchMatch[];
  loading: boolean;
  totalMatches: number;
}

/**
 * Server-side text search hook.
 *
 * Fires a GraphQL `textSearch` query whenever the search parameters change,
 * then maps the returned IDs to point array indices so the existing muting
 * pipeline can consume them without modification.
 */
export function useTextSearch(
  collectionName: string | null,
  searchQuery: string | undefined,
  config: TextSearchConfig | undefined,
  ids: string[] | undefined,
): UseTextSearchResult {
  const [executeSearch] = useLazyQuery<TextSearchData>(TEXT_SEARCH, {
    fetchPolicy: 'no-cache',
  });

  const abortRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);

  const [matches, setMatches] = useState<TextSearchMatch[]>([]);
  const [totalMatches, setTotalMatches] = useState(0);
  const [loading, setLoading] = useState(false);

  // Build an id → index lookup once when the collection data changes.
  const idToIndex = useMemo(() => {
    if (!ids) return null;
    const map = new Map<string, number>();
    for (let i = 0; i < ids.length; i++) {
      map.set(ids[i], i);
    }
    return map;
  }, [ids]);

  // Derive the highlighted index set from the latest matches.
  const highlightedIndices = useMemo(() => {
    if (matches.length === 0 || !idToIndex) return undefined;
    const set = new Set<number>();
    for (const m of matches) {
      const idx = idToIndex.get(m.id);
      if (idx !== undefined) set.add(idx);
    }
    return set.size > 0 ? set : undefined;
  }, [matches, idToIndex]);

  const doSearch = useCallback(async () => {
    const query = searchQuery?.trim();
    if (!collectionName || !query) {
      setMatches([]);
      setTotalMatches(0);
      setLoading(false);
      return;
    }

    // Cancel any in-flight request.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const thisRequest = ++requestIdRef.current;
    setLoading(true);

    try {
      const result = await executeSearch({
        variables: {
          collectionName,
          query,
          fields: config?.fields ?? undefined,
          mode: config?.mode ?? 'CONTAINS',
          caseSensitive: config?.caseSensitive ?? false,
        },
        context: { fetchOptions: { signal: controller.signal } },
      });

      // Guard against stale responses.
      if (thisRequest !== requestIdRef.current) return;

      const data = result.data?.textSearch;
      setMatches(data?.matches ?? []);
      setTotalMatches(data?.totalMatches ?? 0);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      if (thisRequest !== requestIdRef.current) return;
      console.error('Text search error:', err);
      setMatches([]);
      setTotalMatches(0);
    } finally {
      if (thisRequest === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, [collectionName, searchQuery, config?.fields, config?.mode, config?.caseSensitive, executeSearch]);

  // Trigger search whenever parameters change.
  useEffect(() => {
    doSearch();
    return () => { abortRef.current?.abort(); };
  }, [doSearch]);

  return { highlightedIndices, matches, loading, totalMatches };
}
