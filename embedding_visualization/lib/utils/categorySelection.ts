import type { NestedColorMap } from '../types/types';

/** Legend click semantics shared by the Legend and the Analytics list. */
export class CategorySelection {
  static toggle(category: string, shift: boolean, muted: string[], values: string[], nested?: NestedColorMap | null): string[] {
    const children = nested?.hierarchy[category] ?? [];
    const all = nested ? [...Object.keys(nested.hierarchy), ...Object.values(nested.hierarchy).flat()] : values;
    const related = [category, ...children];
    if (shift) return muted.includes(category) ? muted.filter(c => !related.includes(c)) : [...new Set([...muted, ...related])];
    const parent = nested && Object.entries(nested.hierarchy).find(([, subs]) => subs.includes(category))?.[0];
    const keep = new Set(parent ? [...related, parent] : related);
    const selected = all.filter(c => !muted.includes(c));
    return selected.length > 0 && selected.every(c => keep.has(c)) ? [] : all.filter(c => !keep.has(c));
  }
}
