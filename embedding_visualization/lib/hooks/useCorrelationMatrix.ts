import { useDeferredValue, useMemo } from 'react';
import type { MetadataPoint } from '../types/types';
import { FieldCorrelation, type CorrelationMatrix, type CorrelationMethod } from '../utils/fieldCorrelation';

/**
 * Pairwise correlations between `fields` over the points left visible by the
 * filters; null while `enabled` is false or fewer than two fields are chosen.
 * Columns, sort orders and pair results are cached per point set and filter,
 * so toggling a field or switching back to a method only computes new pairs.
 */
export function useCorrelationMatrix(
  points: MetadataPoint[],
  fields: readonly string[],
  method: CorrelationMethod,
  mutedIndices: Set<number> | null | undefined,
  enabled: boolean,
): CorrelationMatrix | null {
  const deferredMuted = useDeferredValue(mutedIndices);
  const columns = useMemo(() => FieldCorrelation.columns(points, deferredMuted), [points, deferredMuted]);
  return useMemo(
    () => (enabled && fields.length >= 2 ? columns.matrix(fields, method) : null),
    [enabled, columns, fields, method],
  );
}
