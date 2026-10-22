import { SEMANTIC_SEARCH } from '../graphql/queries';
import { MOBILE_BREAKPOINT } from '../hooks/use-mobile';
import { IS_DEMO } from './demoMode';
import { SPACE_LOCAL_SEARCH_COLLECTION } from './tourPresets';
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
export const WORDNET_TOUR_STORAGE_KEY = 'orrery.demo-wordnet-tour.v1';
export const PROBE_TOUR_STORAGE_KEY = 'orrery.demo-probe-tour.v1';
export const MOBILE_NOTICE_STORAGE_KEY = 'orrery.demo-mobile-notice.v1';

export type OnboardingMark = 'dismissed' | 'completed';

/**
 * Phone-sized viewports: below this width the Explore panels cover the map.
 * It IS the app's mobile breakpoint (`md`, which also collapses the legend),
 * so every small-screen rule shares one floor: the desktop-notice card, the
 * tours' panel-free map steps (`TourRuntime.isCompactViewport`), and the
 * desktop-only WordNet tour.
 */
export const COMPACT_VIEWPORT_WIDTH = MOBILE_BREAKPOINT;

export function isCompactWidth(viewportWidth: number): boolean {
  return viewportWidth < COMPACT_VIEWPORT_WIDTH;
}

export type TourAction = 'tour' | 'sae-tour' | 'wordnet-tour' | 'probe-tour';

export type OnboardingAction = TourAction | 'intro' | 'mobile-notice' | null;

/** `?tour=<value>` → the tour it starts. */
const TOUR_PARAM_ACTIONS: ReadonlyMap<string, TourAction> = new Map([
  ['1', 'tour'],
  ['sae', 'sae-tour'],
  ['wordnet', 'wordnet-tour'],
  ['probe', 'probe-tour'],
]);

/**
 * Tours that need a desktop: the WordNet galaxy loads 212,478 points in 3D
 * with haze, too heavy for a phone. The others run at any width.
 */
const DESKTOP_ONLY_TOURS: ReadonlySet<TourAction> = new Set(['wordnet-tour']);

export function tourFitsViewport(tour: TourAction, viewportWidth: number): boolean {
  return !(DESKTOP_ONLY_TOURS.has(tour) && isCompactWidth(viewportWidth));
}

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
  return args.isDemo && isCompactWidth(args.viewportWidth) && !args.mobileNoticeSeen;
}

/** URL params whose presence means the visitor followed a deep link (no auto-intro). */
const DEEP_LINK_PARAMS = ['collection', 'colorBy', 'preset', 'tour'] as const;

export interface OnboardingInputs {
  isDemo: boolean;
  search: string;
  introSeen: boolean;
  viewportWidth: number;
  mobileNoticeSeen: boolean;
}

/**
 * Decide what to auto-present, from first-render inputs only.
 * - A `?tour=` link starts its tour in ANY build (dev testing included) at
 *   any viewport the tour fits, ahead of the mobile notice: the link is
 *   explicit intent, and the notice waits for the next ordinary visit.
 *   `?tour=1` is the Explore tour, `?tour=sae` the "Inspect SAE" tour (its
 *   first segment runs on the Explore page's label map, then hands off to
 *   /sae), `?tour=wordnet` / `?tour=probe` the single-page WordNet-galaxy and
 *   Glasgow-probing tours.
 * - Otherwise the mobile notice comes first on a phone, so the visitor is
 *   told about the layout before anything else competes for the screen.
 * - A tour too heavy for the viewport (WordNet on a phone) downgrades to the
 *   welcome dialog, which offers the tours that do fit.
 * - `?intro=1` reopens the welcome dialog in any build, ignoring storage.
 * - Otherwise the intro auto-opens once per browser, demo builds only, and
 *   never on top of a deep link.
 *
 * Returning a single action is what keeps the surfaces mutually exclusive —
 * the Explore page derives each dialog's initial open state from this one
 * value, so the notice and the welcome dialog can never both open.
 */
export function getOnboardingAction(args: OnboardingInputs): OnboardingAction {
  const params = new URLSearchParams(stripQueryPrefix(args.search));
  const tour = TOUR_PARAM_ACTIONS.get(params.get('tour') ?? '') ?? null;
  if (tour && tourFitsViewport(tour, args.viewportWidth)) return tour;
  if (shouldShowMobileNotice(args)) return 'mobile-notice';
  if (tour) return 'intro';
  if (params.get('intro') === '1') return 'intro';
  if (!args.isDemo || args.introSeen) return null;
  if (DEEP_LINK_PARAMS.some((p) => params.has(p))) return null;
  return 'intro';
}

export interface OnboardingPlan {
  action: OnboardingAction;
  /**
   * What the mobile notice pre-empted, shown once it is dismissed: the same
   * decision with the notice marked seen. Only ever the welcome dialog (a
   * bare first visit, `?intro=1`, or a desktop-only tour link) or nothing,
   * since a tour that fits the viewport wins over the notice in the first place.
   */
  afterNotice: 'intro' | null;
}

export const NO_ONBOARDING: OnboardingPlan = { action: null, afterNotice: null };

export function planOnboarding(args: OnboardingInputs): OnboardingPlan {
  const action = getOnboardingAction(args);
  const pre =
    action === 'mobile-notice' ? getOnboardingAction({ ...args, mobileNoticeSeen: true }) : null;
  return { action, afterNotice: pre === 'intro' ? 'intro' : null };
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
      variables: { collectionName: SPACE_LOCAL_SEARCH_COLLECTION, query: 'warm up', nResults: 1 },
      fetchPolicy: 'no-cache',
    })
    .catch(() => {
      // Warm-up is best-effort; the tour's search step tolerates a cold model.
    });
}
