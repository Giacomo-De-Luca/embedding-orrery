import { useMemo } from 'react';
import type { Point2D, Point3D } from '../types/types';
import {
  detectTemporalField,
  computeTemporalCrossTab,
  computeTemporalCounts,
  type TemporalCrossTabRow,
  type TemporalCountRow,
} from '../utils/temporalAnalysis';

interface TemporalData {
  temporalField: string | null;
  crossTabData: TemporalCrossTabRow[];
  temporalCounts: TemporalCountRow[];
  allPeriods: string[];
}

/**
 * Detects temporal fields and computes cross-tabulation data for the TemporalChart.
 * Detects temporal field regardless of whether a category field is set.
 * Returns temporalCounts (standalone mode) and crossTabData (stacked mode).
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
    if (points.length === 0 || availableFields.length === 0) return null;
    return detectTemporalField(availableFields, itemMetadata);
  }, [points.length, availableFields, itemMetadata]);

  const crossTabData = useMemo(() => {
    if (!temporalField || !categoryField || categoryValues.length === 0) return [];
    return computeTemporalCrossTab(points, categoryField, temporalField, categoryValues);
  }, [points, categoryField, temporalField, categoryValues]);

  const temporalCounts = useMemo(() => {
    if (!temporalField) return [];
    return computeTemporalCounts(points, temporalField);
  }, [points, temporalField]);

  const allPeriods = useMemo(() => {
    if (!temporalField) return [];
    return temporalCounts.map(r => r.period);
  }, [temporalField, temporalCounts]);

  return { temporalField, crossTabData, temporalCounts, allPeriods };
}
