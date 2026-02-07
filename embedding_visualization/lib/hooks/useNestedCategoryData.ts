import { useMemo } from 'react';
import type { Point2D, Point3D, NestedColorMap } from '../types/types';
import { isNestedColorAvailable, buildNestedColorMap } from '../utils/nestedColorUtils';

interface NestedCategoryResult {
  available: boolean;
  nestedColorMap: NestedColorMap | null;
}

/**
 * Memoized hook for nested topic/subtopic color mapping.
 * Returns availability flag and the color map (only computed when mode is active).
 */
export function useNestedCategoryData(
  points: (Point2D | Point3D)[],
  colorByField: string | null | undefined,
  nestedColorMode: boolean | undefined,
  palette?: string
): NestedCategoryResult {
  const available = useMemo(
    () => isNestedColorAvailable(points, colorByField),
    [points, colorByField]
  );

  const nestedColorMap = useMemo(() => {
    if (!available || !nestedColorMode) return null;
    return buildNestedColorMap(points, palette);
  }, [available, nestedColorMode, points, palette]);

  return { available, nestedColorMap };
}
