/**
 * Shared temporal range filtering utilities.
 * Used by useAppSearch (backend filters) and page.tsx (client-side text search filtering).
 */

import type { FilterInput, TemporalRange } from '../types/types';

/**
 * Convert a TemporalRange into GTE/LTE FilterInput[] for backend queries.
 */
export function buildTemporalFilterInputs(range: TemporalRange): FilterInput[] {
  const startNum = Number(range.startPeriod);
  const endNum = Number(range.endPeriod);
  const isNumeric = !isNaN(startNum) && !isNaN(endNum);
  return [
    {
      field: range.field,
      operator: 'GTE',
      value: isNumeric ? startNum : range.startPeriod,
    },
    {
      field: range.field,
      operator: 'LTE',
      value: isNumeric ? endNum : range.endPeriod,
    },
  ];
}

/**
 * Check if a single point's metadata falls within the temporal range.
 * Uses the allPeriods index approach for correctness with any period type.
 */
export function isInTemporalRange(
  metadata: Record<string, unknown>,
  range: TemporalRange,
): boolean {
  const startIdx = range.allPeriods.indexOf(range.startPeriod);
  const endIdx = range.allPeriods.indexOf(range.endPeriod);
  if (startIdx < 0 || endIdx < 0) return true; // invalid range, don't filter
  const val = String(metadata[range.field] ?? '');
  const periodIdx = range.allPeriods.indexOf(val);
  if (periodIdx < 0) return false; // unknown period, filter out
  return periodIdx >= startIdx && periodIdx <= endIdx;
}
