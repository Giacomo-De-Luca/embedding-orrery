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
export const SAE_TOUR_STORAGE_KEY = 'orrery.demo-sae-tour.v1';
export const MOBILE_NOTICE_STORAGE_KEY = 'orrery.demo-mobile-notice.v1';

export type OnboardingMark = 'dismissed' | 'completed';

/**
 * Below this viewport width the spotlight tour is too cramped — intro only.
 * Doubles as the "phone-sized" floor for the desktop-notice card, so the two
 * gates can never disagree about what counts as a small screen.
 */
export const TOUR_MIN_VIEWPORT = 768;

export type OnboardingAction = 'intro' | 'tour' | 'sae-tour' | 'mobile-notice' | null;

/**
 * Whether to show the one-time "best viewed on desktop" card. Demo builds only
 * — a self-hoster on a phone is doing it deliberately and gets no gate.
 *
 * Exported so /sae can apply the identical rule without duplicating it (that
 * page has no `getOnboardingAction` call of its own).
 */
export function shouldShowMobileNotice(args: {
  isDemo: boolean;
  viewportWidth: number;
  mobileNoticeSeen: boolean;
}): boolean {
  return args.isDemo && args.viewportWidth < TOUR_MIN_VIEWPORT && !args.mobileNoticeSeen;
}

/** URL params whose presence means the visitor followed a deep link (no auto-intro). */
const DEEP_LINK_PARAMS = ['collection', 'colorBy', 'preset', 'tour'] as const;

/**
 * Decide what to auto-present, from first-render inputs only.
 * - The mobile notice wins over everything, so a phone visitor is told about
 *   the layout before anything else competes for the screen. It deliberately
 *   pre-empts `?tour=1` / `?intro=1` / `?tour=sae`: the tours are already
 *   downgraded below `TOUR_MIN_VIEWPORT`, and a `?tour=sae` deep link stays on
 *   Explore rather than forwarding until the notice has been dismissed once.
 * - `?tour=sae` is the "Inspect SAE" tour: its first segment runs on the
 *   Explore page's label map (the page owns the viewport downgrade), then
 *   hands off to /sae for the inspection segment.
 * - `?tour=1` starts the Explore tour in ANY build (dev testing included),
 *   downgraded to the intro on viewports too narrow for a spotlight tour.
 * - `?intro=1` reopens the welcome dialog in any build, ignoring storage.
 * - Otherwise the intro auto-opens once per browser, demo builds only, and
 *   never on top of a deep link.
 *
 * Returning a single action is what keeps the surfaces mutually exclusive —
 * the Explore page derives each dialog's initial open state from this one
 * value, so the notice and the welcome dialog can never both open.
 */
export function getOnboardingAction(args: {
  isDemo: boolean;
  search: string;
  introSeen: boolean;
  viewportWidth: number;
  mobileNoticeSeen: boolean;
}): OnboardingAction {
  if (shouldShowMobileNotice(args)) return 'mobile-notice';
  const params = new URLSearchParams(stripQueryPrefix(args.search));
  if (params.get('tour') === 'sae') return 'sae-tour';
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

export function readMobileNoticeSeen(): boolean {
  try {
    return window.localStorage.getItem(MOBILE_NOTICE_STORAGE_KEY) !== null;
  } catch {
    return false;
  }
}

export function markMobileNoticeSeen(): void {
  safeSet(MOBILE_NOTICE_STORAGE_KEY, 'dismissed');
}

/** Record a tour outcome under an arbitrary storage key (one per tour). */
export function markTourKey(key: string, value: OnboardingMark): void {
  safeSet(key, value);
}

export function markTour(value: OnboardingMark): void {
  markTourKey(TOUR_STORAGE_KEY, value);
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
