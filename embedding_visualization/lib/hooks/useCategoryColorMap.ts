'use client';

import { useMemo } from 'react';

import { useVisualizationStore } from '../stores/useVisualizationStore';
import { buildAliasedCategoryColorMap, buildCategoryColorMap } from '../utils/categoryColors';

/**
 * Color map for a categorical field's values, resolved from the store's
 * per-field color overrides and color aliases (a probe's predicted-class
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
  const alias = useVisualizationStore((s) => (field ? s.categoryColorAliases[field] : undefined));
  const sourceOverrides = useVisualizationStore((s) =>
    alias ? s.categoryColorOverrides[alias.sourceField] : undefined,
  );
  return useMemo(
    () =>
      alias
        ? buildAliasedCategoryColorMap(alias, values, palette, sourceOverrides, overrides)
        : buildCategoryColorMap(field, values, palette, overrides),
    [field, values, palette, overrides, alias, sourceOverrides],
  );
}
