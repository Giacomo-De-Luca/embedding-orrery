import { useMemo } from 'react';
import type { EmbeddingData, SemanticSearchResult, HighlightMap } from '../types/types';

/**
 * Combines highlighted indices from semantic search and text search.
 *
 * Returns a Map where:
 * - Keys: point indices to highlight (glow overlay)
 * - Values: similarity scores (0-1)
 *   - Semantic search results: actual similarity score
 *   - Text search results: fixed 0.5 (distinct blue tone)
 *
 * Topic selection is handled by the muting system (effectiveMutedCategories),
 * not by glow overlays. Mixing thousands of topic points into the glow pipeline
 * causes Plotly freezes when clicking points after cluster zoom.
 *
 * The selected point is NOT included here — it has its own dedicated overlay
 * traces in ScatterPlot3D. Keeping it out prevents premature recomputation
 * when selectedPoint changes (before search results arrive).
 */
export function useHighlightedIndices(
  semanticSearchResults: SemanticSearchResult[] | null,
  data: EmbeddingData | null,
  textSearchHighlights?: Set<number> | null,
): HighlightMap | undefined {
  // Build id→index lookup once when data loads — O(n) one-time cost
  const idToIndex = useMemo(() => {
    if (!data) return null;
    const map = new Map<string, number>();
    for (let i = 0; i < data.ids.length; i++) {
      map.set(data.ids[i], i);
    }
    return map;
  }, [data]);

  return useMemo(() => {
    const highlightMap = new Map<number, number>();

    // O(k) lookup via pre-built idToIndex — only iterates ~20 search results, not all data
    if (semanticSearchResults && semanticSearchResults.length > 0 && idToIndex) {
      for (const r of semanticSearchResults) {
        const index = idToIndex.get(r.id);
        if (index !== undefined) {
          const current = highlightMap.get(index);
          if (current === undefined || r.similarity > current) {
            highlightMap.set(index, r.similarity);
          }
        }
      }
    }

    // Merge text search highlights (fixed 0.5 similarity = blue tone).
    // Semantic search results take priority (max-similarity logic preserves them).
    if (textSearchHighlights && textSearchHighlights.size > 0) {
      const TEXT_SEARCH_SIMILARITY = 0.5;
      for (const index of textSearchHighlights) {
        if (!highlightMap.has(index)) {
          highlightMap.set(index, TEXT_SEARCH_SIMILARITY);
        }
      }
    }

    // Return undefined if empty for backward compatibility
    return highlightMap.size > 0 ? highlightMap : undefined;
  }, [semanticSearchResults, idToIndex, textSearchHighlights]);
}
