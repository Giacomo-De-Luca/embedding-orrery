import { SEMANTIC_SEARCH } from '../graphql/queries';
import { IS_DEMO } from './demoMode';
import { DEMO_DEFAULT_COLLECTION } from './tourPresets';
import { stripQueryPrefix } from './urlViewParams';

/**
 * Gating and persistence for the demo onboarding surfaces (welcome dialog +
 * spotlight tour). Pure decision logic here; the Explore page latches the
 * result once at first render.
 */

/** Versioned storage keys — bump the suffix to re-show after a redesign. */
export const INTRO_STORAGE_KEY = 'orrery.demo-intro.v1';
export const TOUR_STORAGE_KEY = 'orrery.demo-tour.v1';

export type OnboardingMark = 'dismissed' | 'completed';

/** Below this viewport width the spotlight tour is too cramped — intro only. */
export const TOUR_MIN_VIEWPORT = 768;

export type OnboardingAction = 'intro' | 'tour' | null;

/** URL params whose presence means the visitor followed a deep link (no auto-intro). */
const DEEP_LINK_PARAMS = ['collection', 'colorBy', 'preset', 'tour'] as const;

/**
 * Decide what to auto-present, from first-render inputs only.
 * - `?tour=1` starts the tour in ANY build (dev testing included), downgraded
 *   to the intro on viewports too narrow for a spotlight tour.
 * - `?intro=1` reopens the welcome dialog in any build, ignoring storage.
 * - Otherwise the intro auto-opens once per browser, demo builds only, and
 *   never on top of a deep link.
 */
export function getOnboardingAction(args: {
  isDemo: boolean;
  search: string;
  introSeen: boolean;
  viewportWidth: number;
}): OnboardingAction {
  const params = new URLSearchParams(stripQueryPrefix(args.search));
  if (params.get('tour') === '1') {
    return args.viewportWidth < TOUR_MIN_VIEWPORT ? 'intro' : 'tour';
  }
  if (params.get('intro') === '1') return 'intro';
  if (!args.isDemo || args.introSeen) return null;
  if (DEEP_LINK_PARAMS.some((p) => params.has(p))) return null;
  return 'intro';
}

function safeSet(key: string, value: OnboardingMark): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Private mode / blocked storage — the dialog just re-shows next visit.
  }
}

export function readIntroSeen(): boolean {
  try {
    return window.localStorage.getItem(INTRO_STORAGE_KEY) !== null;
  } catch {
    return false;
  }
}

export function markIntro(value: OnboardingMark): void {
  safeSet(INTRO_STORAGE_KEY, value);
}

export function markTour(value: OnboardingMark): void {
  safeSet(TOUR_STORAGE_KEY, value);
}

// ---------------------------------------------------------------------------
// Search pre-warm
// ---------------------------------------------------------------------------

/** Duck-typed Apollo client — keeps this module import-light for node tests. */
interface QueryClientLike {
  query: (options: {
    query: typeof SEMANTIC_SEARCH;
    variables: Record<string, unknown>;
    fetchPolicy: 'no-cache';
  }) => Promise<unknown>;
}

let searchWarmed = false;

/** Test-only: reset the once-per-page-load warm-up latch. */
export function resetWarmEmotionSearchForTests(): void {
  searchWarmed = false;
}

/**
 * Fire-and-forget warm-up of the emotion collection's server-side embedding
 * model (MiniLM cold-starts inside the Space container on first search).
 * Demo builds only, at most once per page load; must never target the
 * Gemini-embedded collections. Called when the welcome dialog opens or the
 * tour starts, whichever comes first.
 */
export function warmEmotionSearch(client: QueryClientLike, isDemo: boolean = IS_DEMO): void {
  if (!isDemo || searchWarmed) return;
  searchWarmed = true;
  client
    .query({
      query: SEMANTIC_SEARCH,
      variables: { collectionName: DEMO_DEFAULT_COLLECTION, query: 'warm up', nResults: 1 },
      fetchPolicy: 'no-cache',
    })
    .catch(() => {
      // Warm-up is best-effort; the tour's search step tolerates a cold model.
    });
}
