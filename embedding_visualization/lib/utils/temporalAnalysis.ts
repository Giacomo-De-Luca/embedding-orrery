/**
 * Temporal field detection and cross-tabulation utilities.
 * Used by the TemporalChart to detect year/date fields and cross-tabulate with categories.
 */

import type { Point2D, Point3D } from '../types/types';
import { analyzeField, type FieldAnalysisResult } from './fieldAnalysis';

export interface TemporalCrossTabRow {
  period: string;
  [category: string]: string | number;
}

export interface TemporalCountRow {
  period: string;
  count: number;
}

/** Field names that strongly suggest temporal data */
const TEMPORAL_NAME_PATTERNS = /^(year|date|time|month|period|decade|quarter|created_at|updated_at|timestamp|published)/i;

/**
 * Check if a field is likely temporal based on name heuristics and value analysis.
 */
export function isTemporalField(fieldName: string, analysis: FieldAnalysisResult): boolean {
  // Name-based heuristic
  if (TEMPORAL_NAME_PATTERNS.test(fieldName)) {
    return analysis.uniqueCount >= 3;
  }

  // Numeric field with year-like values
  if (
    analysis.isNumeric &&
    analysis.min !== undefined &&
    analysis.max !== undefined &&
    analysis.min >= 1800 &&
    analysis.max <= 2100 &&
    analysis.uniqueCount >= 3
  ) {
    return true;
  }

  return false;
}

/**
 * Find the best temporal field from available fields.
 * Returns null if no temporal field is detected.
 */
export function detectTemporalField(
  availableFields: string[],
  itemMetadata: Record<string, unknown>[]
): string | null {
  if (itemMetadata.length === 0) return null;

  // Priority 1: Fields with temporal names
  for (const field of availableFields) {
    if (TEMPORAL_NAME_PATTERNS.test(field)) {
      const analysis = analyzeField(field, itemMetadata, 200);
      if (analysis.uniqueCount >= 3 && analysis.uniqueCount < 200) {
        return field;
      }
    }
  }

  // Priority 2: Numeric fields with year-like ranges
  for (const field of availableFields) {
    const analysis = analyzeField(field, itemMetadata, 200);
    if (isTemporalField(field, analysis) && analysis.uniqueCount < 200) {
      return field;
    }
  }

  return null;
}

/**
 * Sort period strings: numerically if possible, otherwise lexicographically.
 */
export function sortPeriods(periods: string[]): string[] {
  return [...periods].sort((a, b) => {
    const numA = parseFloat(a);
    const numB = parseFloat(b);
    if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
    return a.localeCompare(b);
  });
}

/**
 * Compute simple period -> count for standalone temporal chart (no category breakdown).
 */
export function computeTemporalCounts(
  points: (Point2D | Point3D)[],
  temporalField: string
): TemporalCountRow[] {
  const counts = new Map<string, number>();

  for (const point of points) {
    const period = point.metadata?.[temporalField];
    if (period === null || period === undefined || period === '') continue;
    const key = String(period);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  const sortedPeriods = sortPeriods(Array.from(counts.keys()));
  return sortedPeriods.map(period => ({ period, count: counts.get(period)! }));
}

/**
 * Cross-tabulate category values across temporal periods.
 * Produces recharts-compatible data: [{ period: "2020", "Topic A": 12, "Topic B": 8 }, ...]
 */
export function computeTemporalCrossTab(
  points: (Point2D | Point3D)[],
  categoryField: string,
  temporalField: string,
  categoryValues: string[]
): TemporalCrossTabRow[] {
  // Build a map: period -> category -> count
  const crossTab = new Map<string, Record<string, number>>();

  for (const point of points) {
    const period = point.metadata?.[temporalField];
    const category = point.metadata?.[categoryField];
    if (period === null || period === undefined || period === '') continue;
    if (category === null || category === undefined || category === '') continue;

    const periodKey = String(period);
    const categoryKey = String(category);

    if (!crossTab.has(periodKey)) {
      crossTab.set(periodKey, {});
    }
    const counts = crossTab.get(periodKey)!;
    counts[categoryKey] = (counts[categoryKey] ?? 0) + 1;
  }

  // Convert to recharts format, sorted by period
  const periods = sortPeriods(Array.from(crossTab.keys()));

  return periods.map(period => {
    const counts = crossTab.get(period)!;
    const row: TemporalCrossTabRow = { period };
    for (const cat of categoryValues) {
      row[cat] = counts[cat] ?? 0;
    }
    return row;
  });
}
