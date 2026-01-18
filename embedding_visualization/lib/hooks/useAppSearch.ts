import { useState, useCallback } from 'react';
import { Point2D, Point3D, SemanticSearchResult, DistanceMetric } from '../types/types';
import { useSemanticSearch } from './useSemanticSearch';
import { transformSearchResults } from '../utils/data-transform';

export function useAppSearch(
  selectedCollection: string | null,
  colorByField: string | null,
  distanceMetric: DistanceMetric = 'COSINE'
) {
  const [selectedPoint, setSelectedPoint] = useState<Point2D | Point3D | null>(null);
  const [semanticSearchResults, setSemanticSearchResults] = useState<SemanticSearchResult[] | null>(null);
  const [searchQueryLabel, setSearchQueryLabel] = useState<string | null>(null);
  const [searchType, setSearchType] = useState<'text' | 'point' | null>(null);

  const { findSimilarByQuery, findSimilarById, loading } = useSemanticSearch(selectedCollection);

  const handleSemanticSearch = useCallback(async (query: string) => {
    console.log('Search triggered:', query, 'metric:', distanceMetric);
    setSearchType('text');
    try {
      const results = await findSimilarByQuery(query, 20, distanceMetric);
      setSemanticSearchResults(transformSearchResults(results, colorByField));
      setSearchQueryLabel(query);
    } catch (error) {
      console.error('Search error:', error);
    }
  }, [findSimilarByQuery, colorByField, distanceMetric]);

  const handlePointClick = useCallback(async (point: Point2D | Point3D) => {
    console.log('Point clicked:', point.label, 'metric:', distanceMetric);
    setSearchType('point');
    try {
      const results = await findSimilarById(point.id, 20, distanceMetric);
      setSemanticSearchResults(transformSearchResults(results, colorByField));
      setSearchQueryLabel(point.label || point.id);
      // Set selected point AFTER semantic search completes
      // This ensures camera animation and highlights appear together
      setSelectedPoint(point);
    } catch (error) {
      console.error('Point click search error:', error);
    }
  }, [findSimilarById, colorByField, distanceMetric]);

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
