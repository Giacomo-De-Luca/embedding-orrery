import { useMemo } from 'react';
import type { Point2D, Point3D } from '../types/types';

interface CategoryData {
  categoryValues: string[];
  categoryCounts: Record<string, number>;
}

/**
 * Computes category values and counts from points.
 * Used by Legend to display category labels with point counts.
 *
 * @param points - Array of 2D or 3D points
 * @param colorByField - The metadata field used for categorization
 * @returns Object with categoryValues (sorted unique values) and categoryCounts (value → count map)
 */
export function useCategoryData(
  points: Point2D[] | Point3D[],
  colorByField: string | null | undefined
): CategoryData {
  return useMemo(() => {
    if (!colorByField || points.length === 0) {
      return { categoryValues: [], categoryCounts: {} };
    }

    const counts: Record<string, number> = {};

    for (const point of points) {
      const value = point.metadata?.[colorByField];
      if (value !== null && value !== undefined && value !== '') {
        const key = String(value);
        counts[key] = (counts[key] ?? 0) + 1;
      }
    }

    const categoryValues = Object.keys(counts).sort();

    return { categoryValues, categoryCounts: counts };
  }, [points, colorByField]);
}
