import { useMemo } from 'react';
import type { Point2D, Point3D } from '../types/types';
import {
  detectTemporalField,
  computeTemporalCrossTab,
  type TemporalCrossTabRow,
} from '../utils/temporalAnalysis';

interface TemporalData {
  temporalField: string | null;
  crossTabData: TemporalCrossTabRow[];
}

/**
 * Detects temporal fields and computes cross-tabulation data for the TemporalChart.
 * Only runs when a categorical colorByField is active.
 */
export function useTemporalData(
  points: (Point2D | Point3D)[],
  categoryField: string | null | undefined,
  categoryValues: string[],
  availableFields: string[]
): TemporalData {
  // Extract itemMetadata from points for field analysis
  const itemMetadata = useMemo(() => {
    return points.map(p => p.metadata ?? {}) as Record<string, unknown>[];
  }, [points]);

  const temporalField = useMemo(() => {
    if (!categoryField || points.length === 0 || availableFields.length === 0) return null;
    return detectTemporalField(availableFields, itemMetadata);
  }, [categoryField, points.length, availableFields, itemMetadata]);

  const crossTabData = useMemo(() => {
    if (!temporalField || !categoryField || categoryValues.length === 0) return [];
    return computeTemporalCrossTab(points, categoryField, temporalField, categoryValues);
  }, [points, categoryField, temporalField, categoryValues]);

  return { temporalField, crossTabData };
}
