import { useMemo } from 'react';
import type { Point2D, Point3D } from '../types/types';
import {
  detectTemporalFields,
  computeTemporalCrossTab,
  computeTemporalCounts,
  type TemporalCrossTabRow,
  type TemporalCountRow,
} from '../utils/temporalAnalysis';

interface TemporalData {
  temporalField: string | null;
  /** All detected temporal field candidates, ordered by priority. */
  temporalFieldCandidates: string[];
  crossTabData: TemporalCrossTabRow[];
  temporalCounts: TemporalCountRow[];
  allPeriods: string[];
}

/**
 * Detects temporal fields and computes cross-tabulation data for the TemporalChart.
 * Detects temporal field regardless of whether a category field is set.
 * Returns temporalCounts (standalone mode) and crossTabData (stacked mode).
 *
 * @param temporalFieldOverride - When set, uses this field instead of auto-detection.
 */
export function useTemporalData(
  points: (Point2D | Point3D)[],
  categoryField: string | null | undefined,
  categoryValues: string[],
  availableFields: string[],
  temporalFieldOverride?: string | null
): TemporalData {
  // Extract itemMetadata from points for field analysis
  const itemMetadata = useMemo(() => {
    return points.map(p => p.metadata ?? {}) as Record<string, unknown>[];
  }, [points]);

  const temporalFieldCandidates = useMemo(() => {
    if (points.length === 0 || availableFields.length === 0) return [];
    return detectTemporalFields(availableFields, itemMetadata);
  }, [points.length, availableFields, itemMetadata]);

  const temporalField = temporalFieldOverride ?? temporalFieldCandidates[0] ?? null;

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

  return { temporalField, temporalFieldCandidates, crossTabData, temporalCounts, allPeriods };
}
