import { useMemo } from 'react';
import type { MetadataPoint } from '../types/types';
import { CategoryKey, type CategoryData } from '../utils/categoryKey';

/**
 * Computes category values and counts from points.
 * Used by Legend to display category labels with point counts.
 *
 * @param points - Array of 2D or 3D points
 * @param colorByField - The metadata field used for categorization
 * @returns Object with categoryValues (sorted unique values, with the shared
 *   Missing key last when some points lack a value) and categoryCounts (value → count map)
 */
export function useCategoryData(
  points: MetadataPoint[],
  colorByField: string | null | undefined
): CategoryData {
  return useMemo(() => CategoryKey.count(points, colorByField), [points, colorByField]);
}
