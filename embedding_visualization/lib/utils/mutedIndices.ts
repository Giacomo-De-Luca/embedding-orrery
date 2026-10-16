/**
 * Index-based point muting (points drawn grey/faded rather than in their
 * colour). Each source — temporal window, text search — produces a
 * `Set<number>` of point indices or null; the scatter plots receive their
 * union. (Topic and legend selections mute *categories* instead.)
 */
export class MutedIndices {
  /** Union of the active sets; a single active set is returned as-is (memo identity). */
  static union(sets: ReadonlyArray<Set<number> | null>): Set<number> | null {
    const active = sets.filter((s): s is Set<number> => s !== null && s.size > 0);
    if (active.length === 0) return null;
    if (active.length === 1) return active[0];
    const merged = new Set<number>();
    for (const set of active) for (const i of set) merged.add(i);
    return merged;
  }
}
