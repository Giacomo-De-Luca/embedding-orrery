'use client';

import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useApolloClient } from '@apollo/client/react';
import type { TopicInfo, FilterInput } from '../types/types';
import { TOPIC_SEMANTIC_SEARCH } from '../graphql/queries';
import { LatestRequestGate } from '../utils/latestRequestGate';
import { resolveQueryPrompt } from '../utils/queryPrompt';
import {
  buildTopicFilters,
  toTopicSearchResults,
  type TopicSearchResult,
  type TopicSimilarity,
} from '../utils/topicSearchResults';

export type { TopicSearchResult } from '../utils/topicSearchResults';

export type TopicSearchMode = 'direct' | 'semantic';

/** Topics requested per semantic topic search. */
const TOPIC_SEARCH_LIMIT = 20;

export interface UseTopicSearchReturn {
  // Search mode
  mode: TopicSearchMode;
  setMode: (mode: TopicSearchMode) => void;
  // Direct search
  directQuery: string;
  setDirectQuery: (q: string) => void;
  filteredTopics: TopicInfo[];
  // Semantic search
  semanticQuery: string;
  setSemanticQuery: (q: string) => void;
  searchTopicsBySimilarity: () => Promise<void>;
  /** null until a search has completed for the current collection. */
  semanticResults: TopicSearchResult[] | null;
  semanticLoading: boolean;
  semanticError: string | null;
  // Selection
  selectedTopicIds: Set<number>;
  toggleTopic: (topicId: number) => void;
  selectAll: () => void;
  clearAll: () => void;
  // Outputs for integration
  topicFilters: FilterInput[] | undefined;
}

/**
 * Hook encapsulating topic search: direct filtering, similarity-based ranking
 * (query embedding vs. stored topic centroids, server-side), selection state,
 * and filter generation. Selection and results reset when the collection changes.
 */
export function useTopicSearch(
  topics: TopicInfo[] | undefined,
  collectionName: string | null,
  queryPromptName?: string | null,
  embeddingPromptName?: string | null,
): UseTopicSearchReturn {
  const client = useApolloClient();
  const [mode, setMode] = useState<TopicSearchMode>('direct');
  const [directQuery, setDirectQuery] = useState('');
  const [semanticQuery, setSemanticQuery] = useState('');
  const [selectedTopicIds, setSelectedTopicIds] = useState<Set<number>>(new Set());
  const [semanticResults, setSemanticResults] = useState<TopicSearchResult[] | null>(null);
  const [semanticLoading, setSemanticLoading] = useState(false);
  const [semanticError, setSemanticError] = useState<string | null>(null);
  const requestGate = useRef(new LatestRequestGate());

  // Topic ids are per collection: a selection carried over would scope searches
  // in the next collection to unrelated topics.
  useEffect(() => {
    requestGate.current.invalidate();
    setSelectedTopicIds(new Set());
    setSemanticResults(null);
    setSemanticError(null);
    setSemanticLoading(false);
  }, [collectionName]);

  // Direct search: filter topics by label + keywords (case-insensitive substring)
  const filteredTopics = useMemo(() => {
    if (!topics) return [];
    if (!directQuery.trim()) return topics;
    const q = directQuery.toLowerCase();
    return topics.filter(t => {
      if (t.label?.toLowerCase().includes(q)) return true;
      if (t.keywords.some(k => k.word.toLowerCase().includes(q))) return true;
      return false;
    });
  }, [topics, directQuery]);

  // Similarity search: rank topics by query-centroid similarity on the backend
  const searchTopicsBySimilarity = useCallback(async () => {
    if (!collectionName || !semanticQuery.trim() || !topics?.length) return;
    const token = requestGate.current.begin();
    setSemanticLoading(true);
    setSemanticError(null);
    try {
      const { data } = await client.query<{ topicSemanticSearch: TopicSimilarity[] }>({
        query: TOPIC_SEMANTIC_SEARCH,
        variables: {
          collectionName,
          query: semanticQuery,
          limit: TOPIC_SEARCH_LIMIT,
          queryPrompt: resolveQueryPrompt(queryPromptName, embeddingPromptName),
        },
        fetchPolicy: 'no-cache',
      });
      if (!requestGate.current.isLatest(token)) return; // superseded or collection changed
      setSemanticResults(toTopicSearchResults(data?.topicSemanticSearch ?? [], topics));
    } catch (err) {
      if (!requestGate.current.isLatest(token)) return;
      console.error('Topic similarity search error:', err);
      setSemanticResults(null);
      setSemanticError(err instanceof Error ? err.message : 'Topic search failed');
    } finally {
      if (requestGate.current.isLatest(token)) setSemanticLoading(false);
    }
  }, [client, collectionName, semanticQuery, topics, queryPromptName, embeddingPromptName]);

  // Selection handlers
  const toggleTopic = useCallback((topicId: number) => {
    setSelectedTopicIds(prev => {
      const next = new Set(prev);
      if (next.has(topicId)) {
        next.delete(topicId);
      } else {
        next.add(topicId);
      }
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    if (!topics) return;
    // In direct mode, select filtered topics; in semantic mode, select result topics
    if (mode === 'direct') {
      setSelectedTopicIds(new Set(filteredTopics.map(t => t.topicId)));
    } else {
      setSelectedTopicIds(new Set((semanticResults ?? []).map(r => r.topic.topicId)));
    }
  }, [topics, mode, filteredTopics, semanticResults]);

  const clearAll = useCallback(() => {
    setSelectedTopicIds(new Set());
  }, []);

  // Build topic filters for scoped semantic search
  const topicFilters = useMemo(() => buildTopicFilters(selectedTopicIds), [selectedTopicIds]);

  return {
    mode,
    setMode,
    directQuery,
    setDirectQuery,
    filteredTopics,
    semanticQuery,
    setSemanticQuery,
    searchTopicsBySimilarity,
    semanticResults,
    semanticLoading,
    semanticError,
    selectedTopicIds,
    toggleTopic,
    selectAll,
    clearAll,
    topicFilters,
  };
}
