/**
 * URL query-string merge for the Explore page's view state.
 *
 * The page owns a small set of params (collection + colour scheme, plus the
 * demo `preset` id). Everything else in the query string — one-shot demo
 * params excepted — belongs to someone else (future features, hand-added
 * params) and must survive the page's `router.replace` sync untouched.
 */

/** Params the Explore page's URL-sync effect owns and rewrites. */
export const OWNED_VIEW_PARAMS = [
  'collection',
  'colorBy',
  'scale',
  'scaleName',
  'color',
  'palette',
  'preset',
] as const;

export type OwnedViewParamKey = (typeof OWNED_VIEW_PARAMS)[number];

/**
 * One-shot params: consumed (latched into state) during the first render and
 * always stripped from the URL by the sync effect.
 */
export const ONE_SHOT_PARAMS = ['tour', 'intro'] as const;

/**
 * Per owned key: a string sets the param, `null` deletes it, and an absent /
 * `undefined` key leaves whatever is currently in the URL untouched.
 */
export type OwnedViewParams = Partial<Record<OwnedViewParamKey, string | null>>;

/** Drop a leading `?` from a search string, if present. */
export function stripQueryPrefix(search: string): string {
  return search.startsWith('?') ? search.slice(1) : search;
}

/**
 * Merge the owned view params into an existing search string, preserving all
 * unknown params. Accepts `currentSearch` with or without the leading `?`.
 * Returns `'?...'`, or `''` when nothing remains.
 */
export function mergeViewSearch(currentSearch: string, owned: OwnedViewParams): string {
  const params = new URLSearchParams(stripQueryPrefix(currentSearch));
  for (const key of OWNED_VIEW_PARAMS) {
    const value = owned[key];
    if (value === undefined) continue;
    if (value === null) params.delete(key);
    else params.set(key, value);
  }
  for (const key of ONE_SHOT_PARAMS) params.delete(key);
  const merged = params.toString();
  return merged ? `?${merged}` : '';
}

/**
 * A `?preset=` param stays in the URL while the user is still on the preset's
 * collection (so re-shared links keep the preset's non-URL flags — nebula,
 * cluster labels, …). The moment the user navigates to a different collection
 * the preset no longer describes the view and must be dropped.
 */
export function shouldDropPreset(
  presetCollection: string | undefined,
  selectedCollection: string | null,
): boolean {
  if (!presetCollection || selectedCollection === null) return false;
  return selectedCollection !== presetCollection;
}
