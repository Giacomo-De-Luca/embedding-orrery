/**
 * Pure view-model builder for the CategoryBarList component.
 *
 * Turns category values/counts (plus optional filtered counts from
 * text-search/temporal muting) into ordered, normalized row data.
 * Kept free of React and DOM concerns so it is unit-testable under
 * the node vitest environment.
 */

import { sortPeriods } from './temporalAnalysis';

export type CategorySortMode = 'count' | 'rate' | 'natural';

export interface CategoryRow {
  /** Raw category value — key for color map and onCategoryToggle. */
  value: string;
  /** Resolved display label. */
  label: string;
  /** Total point count (0 when the value is missing from categoryCounts). */
  count: number;
  /** Count surviving active filters; null when no filter is active. */
  filteredCount: number | null;
  /** count / totalCount over all categories (0 when total is 0). */
  pctOfTotal: number;
  /** filteredCount / count (0 for zero-count rows); null when no filter. */
  matchRate: number | null;
  /** count / max displayed count — bar track width in [0, 1]. */
  trackFraction: number;
  /** filteredCount / max displayed count (<= trackFraction); null when no filter. */
  fillFraction: number | null;
}

export interface CategoryRowSummary {
  /** Sum of counts over ALL category values (unaffected by the name filter). */
  totalCount: number;
  /** Sum of filtered counts over category values only; null when no filter. */
  totalFiltered: number | null;
  /** Number of categories with filteredCount > 0; null when no filter. */
  matchedCategoryCount: number | null;
  /** Number of rows remaining after the name filter. */
  visibleCategoryCount: number;
}

export interface BuildCategoryRowsInput {
  categoryValues: string[];
  categoryCounts: Record<string, number>;
  filteredCounts?: Record<string, number> | null;
  sortMode: CategorySortMode;
  nameFilter?: string;
  getLabel?: (value: string) => string;
}

/**
 * A filter is active when filteredCounts is a non-empty object — an empty
 * object (e.g. every point muted) intentionally reads as inactive so the
 * chart falls back to showing totals.
 */
export function isCategoryFilterActive(
  filteredCounts: Record<string, number> | null | undefined
): filteredCounts is Record<string, number> {
  return filteredCounts != null && Object.keys(filteredCounts).length > 0;
}

/** 'rate' sorting is only meaningful while a filter is active; fall back to 'count'. */
export function resolveSortMode(
  sortMode: CategorySortMode,
  filterActive: boolean
): CategorySortMode {
  return sortMode === 'rate' && !filterActive ? 'count' : sortMode;
}

export function buildCategoryRows(input: BuildCategoryRowsInput): {
  rows: CategoryRow[];
  summary: CategoryRowSummary;
} {
  const { categoryValues, categoryCounts, filteredCounts, sortMode, nameFilter, getLabel } = input;
  const filterActive = isCategoryFilterActive(filteredCounts);
  const resolveLabel = getLabel ?? ((value: string) => value);

  const rawRows = categoryValues.map(value => ({
    value,
    label: resolveLabel(value),
    count: categoryCounts[value] ?? 0,
    filteredCount: filterActive ? (filteredCounts[value] ?? 0) : null,
  }));

  // Summary is computed before the name filter; filtered totals only consider
  // categoryValues keys, so stray keys (e.g. upstream-stripped noise) are ignored.
  const totalCount = rawRows.reduce((sum, r) => sum + r.count, 0);
  const totalFiltered = filterActive
    ? rawRows.reduce((sum, r) => sum + (r.filteredCount ?? 0), 0)
    : null;
  const matchedCategoryCount = filterActive
    ? rawRows.filter(r => (r.filteredCount ?? 0) > 0).length
    : null;

  const query = nameFilter?.trim().toLowerCase() ?? '';
  const visible = query
    ? rawRows.filter(r => r.label.toLowerCase().includes(query))
    : rawRows;

  // Natural rank of visible labels (numeric-aware); duplicate labels share the
  // rank of their first sorted occurrence and resolve via stable sort.
  const naturalRank = new Map<string, number>();
  sortPeriods(visible.map(r => r.label)).forEach((label, index) => {
    if (!naturalRank.has(label)) naturalRank.set(label, index);
  });
  const rankOf = (row: { label: string }) => naturalRank.get(row.label) ?? 0;
  const rateOf = (row: { count: number; filteredCount: number | null }) =>
    row.count > 0 ? (row.filteredCount ?? 0) / row.count : 0;

  const effectiveSort = resolveSortMode(sortMode, filterActive);

  const sorted = [...visible].sort((a, b) => {
    if (effectiveSort === 'natural') {
      return rankOf(a) - rankOf(b);
    }
    if (effectiveSort === 'rate') {
      return (
        rateOf(b) - rateOf(a) ||
        (b.filteredCount ?? 0) - (a.filteredCount ?? 0) ||
        rankOf(a) - rankOf(b)
      );
    }
    if (filterActive) {
      return (
        (b.filteredCount ?? 0) - (a.filteredCount ?? 0) ||
        b.count - a.count ||
        rankOf(a) - rankOf(b)
      );
    }
    return b.count - a.count || rankOf(a) - rankOf(b);
  });

  // Fractions are normalized to the visible max so bars use the full width.
  const maxDisplayedCount = sorted.reduce((max, r) => Math.max(max, r.count), 0);

  const rows: CategoryRow[] = sorted.map(r => ({
    value: r.value,
    label: r.label,
    count: r.count,
    filteredCount: r.filteredCount,
    pctOfTotal: totalCount > 0 ? r.count / totalCount : 0,
    matchRate: filterActive ? rateOf(r) : null,
    trackFraction: maxDisplayedCount > 0 ? r.count / maxDisplayedCount : 0,
    fillFraction: filterActive
      ? maxDisplayedCount > 0
        ? (r.filteredCount ?? 0) / maxDisplayedCount
        : 0
      : null,
  }));

  return {
    rows,
    summary: { totalCount, totalFiltered, matchedCategoryCount, visibleCategoryCount: rows.length },
  };
}
