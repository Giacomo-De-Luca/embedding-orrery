import { useMemo } from 'react';
import type { EmbeddingData, SemanticSearchResult, HighlightMap } from '../types/types';

/**
 * Combines highlighted indices from semantic search and topic selection.
 *
 * Returns a Map where:
 * - Keys: point indices to highlight
 * - Values: similarity scores (0-1)
 *   - Selected point: 1.0 (it's the center of semantic search)
 *   - Semantic search results: actual similarity score
 *   - Topic-selected points: 1.0 (membership match)
 */
export function useHighlightedIndices(
  semanticSearchResults: SemanticSearchResult[] | null,
  data: EmbeddingData | null,
  selectedPointIndex?: number,
  topicHighlightMap?: HighlightMap
): HighlightMap | undefined {
  return useMemo(() => {
    const highlightMap = new Map<number, number>();

    // Add selected point with similarity 1.0 (it's the center of semantic search)
    // This ensures the clicked point is always highlighted and can have lines drawn to it
    if (selectedPointIndex !== undefined && semanticSearchResults && semanticSearchResults.length > 0) {
      highlightMap.set(selectedPointIndex, 1.0);
    }

    // Add semantic search highlights (with actual similarity scores)
    if (semanticSearchResults && semanticSearchResults.length > 0 && data) {
      // Create map of id → similarity for fast lookup
      const idToSimilarity = new Map(
        semanticSearchResults.map(r => [r.id, r.similarity])
      );

      // Find indices and add to map
      data.ids.forEach((id, index) => {
        if (idToSimilarity.has(id)) {
          const similarity = idToSimilarity.get(id)!;
          // If already present from selected point, keep max similarity
          const currentSimilarity = highlightMap.get(index);
          if (currentSimilarity === undefined || similarity > currentSimilarity) {
            highlightMap.set(index, similarity);
          }
        }
      });
    }

    // Merge topic highlights (max-similarity logic)
    if (topicHighlightMap && topicHighlightMap.size > 0) {
      topicHighlightMap.forEach((score, index) => {
        const current = highlightMap.get(index);
        if (current === undefined || score > current) {
          highlightMap.set(index, score);
        }
      });
    }

    // Return undefined if empty for backward compatibility
    return highlightMap.size > 0 ? highlightMap : undefined;
  }, [semanticSearchResults, data, selectedPointIndex, topicHighlightMap]);
}
