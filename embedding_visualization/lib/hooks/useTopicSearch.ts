'use client';

import { useState, useCallback, useMemo } from 'react';
import type { TopicInfo, FilterInput, DistanceMetric } from '../types/types';
import { useSemanticSearch } from './useSemanticSearch';

export type TopicSearchMode = 'direct' | 'semantic';

export interface TopicSearchResult {
  topic: TopicInfo;
  relevance: number;  // 0-1, average similarity of matched items
  matchCount: number; // number of semantic search results in this topic
}

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
  semanticResults: TopicSearchResult[];
  semanticLoading: boolean;
  // Selection
  selectedTopicIds: Set<number>;
  toggleTopic: (topicId: number) => void;
  selectAll: () => void;
  clearAll: () => void;
  // Outputs for integration
  topicFilters: FilterInput[] | undefined;
}

/**
 * Hook encapsulating topic search: direct filtering, similarity-based ranking,
 * selection state, highlight map generation, and filter generation.
 */
export function useTopicSearch(
  topics: TopicInfo[] | undefined,
  collectionName: string | null,
  distanceMetric: DistanceMetric = 'COSINE',
  queryPromptName?: string | null,
  embeddingPromptName?: string | null,
): UseTopicSearchReturn {
  const [mode, setMode] = useState<TopicSearchMode>('direct');
  const [directQuery, setDirectQuery] = useState('');
  const [semanticQuery, setSemanticQuery] = useState('');
  const [selectedTopicIds, setSelectedTopicIds] = useState<Set<number>>(new Set());
  const [semanticResults, setSemanticResults] = useState<TopicSearchResult[]>([]);
  const [semanticLoading, setSemanticLoading] = useState(false);

  const { findSimilarByQuery } = useSemanticSearch(collectionName);

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

  // Similarity search: find which topics are most relevant to a text query
  const searchTopicsBySimilarity = useCallback(async () => {
    if (!semanticQuery.trim() || !topics?.length) return;
    setSemanticLoading(true);
    try {
      // Resolve query prompt
      let effectivePrompt: string | null = null;
      if (queryPromptName === 'auto' && embeddingPromptName) {
        const map: Record<string, string> = { 'Retrieval-document': 'Retrieval-query' };
        effectivePrompt = map[embeddingPromptName] || embeddingPromptName;
      } else if (queryPromptName && queryPromptName !== 'auto') {
        effectivePrompt = queryPromptName;
      }

      const results = await findSimilarByQuery(semanticQuery, 100, distanceMetric, effectivePrompt);
      if (!results?.length) {
        setSemanticResults([]);
        return;
      }

      // Group results by topic_id, compute average similarity
      const topicMap = new Map<number, { totalSim: number; count: number }>();
      for (const r of results) {
        const topicId = r.metadata?.topic_id as number | undefined;
        if (topicId === undefined || topicId === null) continue;
        const existing = topicMap.get(topicId);
        if (existing) {
          existing.totalSim += r.similarity;
          existing.count += 1;
        } else {
          topicMap.set(topicId, { totalSim: r.similarity, count: 1 });
        }
      }

      // Build ranked topic results
      const topicById = new Map(topics.map(t => [t.topicId, t]));
      const ranked: TopicSearchResult[] = [];
      for (const [topicId, agg] of topicMap) {
        const topic = topicById.get(topicId);
        if (!topic) continue;
        ranked.push({
          topic,
          relevance: agg.totalSim / agg.count,
          matchCount: agg.count,
        });
      }
      ranked.sort((a, b) => b.relevance - a.relevance);
      setSemanticResults(ranked);
    } catch (err) {
      console.error('Topic similarity search error:', err);
    } finally {
      setSemanticLoading(false);
    }
  }, [semanticQuery, topics, findSimilarByQuery, distanceMetric, queryPromptName, embeddingPromptName]);

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
      const q = directQuery.toLowerCase().trim();
      const toSelect = q
        ? topics.filter(t => t.label?.toLowerCase().includes(q) || t.keywords.some(k => k.word.toLowerCase().includes(q)))
        : topics;
      setSelectedTopicIds(new Set(toSelect.map(t => t.topicId)));
    } else {
      setSelectedTopicIds(new Set(semanticResults.map(r => r.topic.topicId)));
    }
  }, [topics, mode, directQuery, semanticResults]);

  const clearAll = useCallback(() => {
    setSelectedTopicIds(new Set());
  }, []);

  // Build topic filters for scoped semantic search
  const topicFilters: FilterInput[] | undefined = useMemo(() => {
    if (selectedTopicIds.size === 0) return undefined;
    const ids = Array.from(selectedTopicIds);
    if (ids.length === 1) {
      return [{ field: 'topic_id', operator: 'EQ', value: ids[0] }];
    }
    return [{ field: 'topic_id', operator: 'IN', value: ids }];
  }, [selectedTopicIds]);

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
    selectedTopicIds,
    toggleTopic,
    selectAll,
    clearAll,
    topicFilters,
  };
}
