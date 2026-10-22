'use client';

import { useMemo } from 'react';

import { useVisualizationStore } from '../stores/useVisualizationStore';
import {
  buildAliasedCategoryColorMap,
  buildCategoryColorMap,
  buildPinnedCategoryColorMap,
} from '../utils/categoryColors';

/**
 * Color map for a categorical field's values, resolved from the store's
 * per-field color overrides, colour pins (a topic-tree preview group takes its
 * representative topic's colour) and color aliases (a probe's predicted-class
 * field borrows its target field's colors). Every categorical view — scatter
 * plots, legend, analytics charts — goes through this so they always agree.
 */
export function useCategoryColorMap(
  field: string | null,
  values: string[],
  palette?: string,
): Record<string, string> {
  const overrides = useVisualizationStore((s) =>
    field ? s.categoryColorOverrides[field] : undefined,
  );
  const pin = useVisualizationStore((s) => (field ? s.categoryColorPins[field] : undefined));
  const alias = useVisualizationStore((s) => (field ? s.categoryColorAliases[field] : undefined));
  const sourceField = pin?.sourceField ?? alias?.sourceField;
  const sourceOverrides = useVisualizationStore((s) =>
    sourceField ? s.categoryColorOverrides[sourceField] : undefined,
  );
  return useMemo(() => {
    if (pin) return buildPinnedCategoryColorMap(field, pin, values, palette, sourceOverrides, overrides);
    if (alias) return buildAliasedCategoryColorMap(alias, values, palette, sourceOverrides, overrides);
    return buildCategoryColorMap(field, values, palette, overrides);
  }, [field, values, palette, overrides, pin, alias, sourceOverrides]);
}
