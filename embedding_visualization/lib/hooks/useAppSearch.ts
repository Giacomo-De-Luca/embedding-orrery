import { useState, useCallback } from 'react';
import { Point2D, Point3D, SemanticSearchResult, DistanceMetric, FilterInput } from '../types/types';
import { useSemanticSearch } from './useSemanticSearch';
import { transformSearchResults } from '../utils/data-transform';

/**
 * Map document-time prompt names to query-time equivalents.
 * For example, if documents were embedded with "Retrieval-document",
 * queries should use "Retrieval-query".
 */
const DOCUMENT_TO_QUERY_PROMPT_MAP: Record<string, string> = {
  'Retrieval-document': 'Retrieval-query',
};

/**
 * Get the appropriate query prompt name based on document prompt name.
 */
export function getQueryPromptName(documentPromptName: string | null | undefined): string | null {
  if (!documentPromptName) return null;
  return DOCUMENT_TO_QUERY_PROMPT_MAP[documentPromptName] || documentPromptName;
}

export function useAppSearch(
  selectedCollection: string | null,
  colorByField: string | null,
  distanceMetric: DistanceMetric = 'COSINE',
  queryPromptName?: string | null,  // null=none, 'auto'=auto-detect, or explicit value
  embeddingPromptName?: string | null,  // From collection metadata, for auto-detect
  topicFilters?: FilterInput[]  // Optional topic filters to scope semantic search
) {
  const [selectedPoint, setSelectedPoint] = useState<Point2D | Point3D | null>(null);
  const [semanticSearchResults, setSemanticSearchResults] = useState<SemanticSearchResult[] | null>(null);
  const [searchQueryLabel, setSearchQueryLabel] = useState<string | null>(null);
  const [searchType, setSearchType] = useState<'text' | 'point' | null>(null);

  const { findSimilarByQuery, findSimilarById, loading } = useSemanticSearch(selectedCollection);

  const handleSemanticSearch = useCallback(async (query: string) => {
    // Resolve effective prompt name
    let effectivePromptName: string | null = null;
    if (queryPromptName === 'auto' && embeddingPromptName) {
      effectivePromptName = getQueryPromptName(embeddingPromptName);
    } else if (queryPromptName && queryPromptName !== 'auto') {
      effectivePromptName = queryPromptName;
    }

    console.log('Search triggered:', query, 'metric:', distanceMetric, effectivePromptName ? `prompt: ${effectivePromptName}` : '');
    setSearchType('text');
    try {
      const results = await findSimilarByQuery(query, 20, distanceMetric, effectivePromptName, topicFilters);
      setSemanticSearchResults(transformSearchResults(results, colorByField));
      setSearchQueryLabel(query);
    } catch (error) {
      console.error('Search error:', error);
    }
  }, [findSimilarByQuery, colorByField, distanceMetric, queryPromptName, embeddingPromptName, topicFilters]);

  const handlePointClick = useCallback(async (point: Point2D | Point3D) => {
    console.log('Point clicked:', point.label, 'metric:', distanceMetric);
    setSearchType('point');
    try {
      const results = await findSimilarById(point.id, 20, distanceMetric, topicFilters);
      setSemanticSearchResults(transformSearchResults(results, colorByField));
      setSearchQueryLabel(point.label || point.id);
      // Set selected point AFTER semantic search completes
      // This ensures camera animation and highlights appear together
      setSelectedPoint(point);
    } catch (error) {
      console.error('Point click search error:', error);
    }
  }, [findSimilarById, colorByField, distanceMetric, topicFilters]);

  // Reset state when collection changes
  const resetSearch = useCallback(() => {
    setSemanticSearchResults(null);
    setSelectedPoint(null);
    setSearchType(null);
  }, []);

  return {
    selectedPoint,
    setSelectedPoint,
    semanticSearchResults,
    setSemanticSearchResults,
    searchQueryLabel,
    searchType,
    handleSemanticSearch,
    handlePointClick,
    searchLoading: loading,
    resetSearch
  };
}
