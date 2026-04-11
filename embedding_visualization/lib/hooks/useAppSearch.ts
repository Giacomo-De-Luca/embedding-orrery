import { useState, useCallback, useMemo } from 'react';
import { Point2D, Point3D, SemanticSearchResult, DistanceMetric, FilterInput, TemporalRange } from '../types/types';
import { useSemanticSearch } from './useSemanticSearch';
import { transformSearchResults } from '../utils/data-transform';
import { buildTemporalFilterInputs } from '../utils/temporalFilters';

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
  topicFilters?: FilterInput[],  // Optional topic filters to scope semantic search
  temporalRange?: TemporalRange | null,  // Optional temporal range to scope semantic search
) {
  const [selectedPoint, setSelectedPoint] = useState<Point2D | Point3D | null>(null);
  const [semanticSearchResults, setSemanticSearchResults] = useState<SemanticSearchResult[] | null>(null);
  const [searchQueryLabel, setSearchQueryLabel] = useState<string | null>(null);
  const [searchType, setSearchType] = useState<'text' | 'point' | null>(null);

  const { findSimilarByQuery, findSimilarById, loading } = useSemanticSearch(selectedCollection);

  // Build combined filters from topic selection + temporal range
  const allFilters = useMemo(() => {
    if (!temporalRange) return topicFilters;
    const filters: FilterInput[] = topicFilters ? [...topicFilters] : [];
    filters.push(...buildTemporalFilterInputs(temporalRange));
    return filters;
  }, [topicFilters, temporalRange]);

  const handleSemanticSearch = useCallback(async (query: string) => {
    // Resolve effective prompt name
    let effectivePromptName: string | null = null;
    if (queryPromptName === 'auto' && embeddingPromptName) {
      effectivePromptName = getQueryPromptName(embeddingPromptName);
    } else if (queryPromptName && queryPromptName !== 'auto') {
      effectivePromptName = queryPromptName;
    }

    console.log('Search triggered:', query, 'metric:', distanceMetric, effectivePromptName ? `prompt: ${effectivePromptName}` : '');
    try {
      const results = await findSimilarByQuery(query, 20, distanceMetric, effectivePromptName, allFilters);
      setSemanticSearchResults(transformSearchResults(results, colorByField));
      setSearchQueryLabel(query);
      setSearchType('text');
    } catch (error) {
      console.error('Search error:', error);
    }
  }, [findSimilarByQuery, colorByField, distanceMetric, queryPromptName, embeddingPromptName, allFilters]);

  const handlePointClick = useCallback(async (point: Point2D | Point3D) => {
    console.log('Point clicked:', point.label, 'metric:', distanceMetric);
    try {
      const results = await findSimilarById(point.id, 20, distanceMetric, allFilters);
      setSemanticSearchResults(transformSearchResults(results, colorByField));
      setSearchQueryLabel(point.label || point.id);
      // Set selected point AFTER semantic search completes
      // This ensures camera animation and highlights appear together
      setSelectedPoint(point);
      setSearchType('point');
    } catch (error) {
      console.error('Point click search error:', error);
    }
  }, [findSimilarById, colorByField, distanceMetric, allFilters]);

  // Reset state when collection changes
  const resetSearch = useCallback(() => {
    setSemanticSearchResults(null);
    setSelectedPoint(null);
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
