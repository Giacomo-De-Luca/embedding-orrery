import { TOUR_COLLECTION, TOUR_PRESET_ID, TOUR_SEARCH_QUERY, TOUR_PRESETS } from './tourPresets';

/** The finale switches to the colour manifold — the "collections are spaces" payoff. */
export const FINALE_PRESET_ID = 'xkcd-manifold';

/**
 * Demo tour step definitions as data, decoupled from the tour library.
 * `TourController` translates these into react-joyride steps; the `prepare`
 * hooks receive a `TourRuntime` implemented by the Explore page. Keeping the
 * library out of this module keeps it unit-testable in node.
 */

/**
 * data-tour anchor names → CSS selectors. Single source of truth for every
 * `data-tour` attribute set in the DOM; entries below the marker are reserved
 * for future steps (already anchored, not yet targeted).
 */
export const TOUR_ANCHORS = {
  plot: '[data-tour="plot"]',
  collectionSelector: '[data-tour="collection-selector"]',
  searchInput: '[data-tour="search-input"]',
  panelAnalytics: '[data-tour="panel-analytics"]',
  // Reserved:
  toggleControls: '[data-tour="toggle-controls"]',
  toggleSearch: '[data-tour="toggle-search"]',
  toggleAnalytics: '[data-tour="toggle-analytics"]',
  panelControls: '[data-tour="panel-controls"]',
  panelSearch: '[data-tour="panel-search"]',
} as const;

export type TourAnchor = keyof typeof TOUR_ANCHORS;

/** Imperative surface the Explore page hands to the tour's prepare hooks. */
export interface TourRuntime {
  /** Switch collection + apply a preset's flags/colour (the welcome-dialog path). */
  applyPreset: (presetId: string) => void;
  /** Run a semantic search; resolves when results have landed. */
  runSearch: (query: string) => Promise<void>;
  /** Clear any semantic-search highlight (between steps). */
  clearSearch: () => void;
  /** Isolate the first topic cluster (others mute); returns its label. */
  isolateFirstTopic: () => string | null;
  /** Drop any tour-applied topic isolation (tour-end cleanup). */
  clearTopicSelection: () => void;
  setActivePanel: (panel: 'controls' | 'search' | 'analytics' | null) => void;
  setShowLabels: (value: boolean) => void;
  getLoadedCollection: () => string | null;
  getColorByField: () => string | null;
}

export interface TourStepDefinition {
  id: string;
  anchor: TourAnchor;
  title: string;
  body: string;
  /** Let pointer events through the spotlight (rotate/zoom the plot). */
  allowInteraction?: boolean;
  placement?: 'auto' | 'center' | 'bottom' | 'left' | 'right';
  /** Ceiling for `prepare` (react-joyride `beforeTimeout`), ms. */
  prepareTimeoutMs?: number;
  /** State setup run before the step is shown; the tour waits for it. */
  prepare?: (runtime: TourRuntime) => Promise<void>;
}

/** Poll `predicate` until true or `timeoutMs`; resolves whether it held. */
export function waitFor(
  predicate: () => boolean,
  timeoutMs: number,
  intervalMs = 100,
): Promise<boolean> {
  return new Promise((resolve) => {
    const start = Date.now();
    const tick = () => {
      if (predicate()) return resolve(true);
      if (Date.now() - start >= timeoutMs) return resolve(false);
      setTimeout(tick, intervalMs);
    };
    tick();
  });
}

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * The five demo tour steps. Steps prepare their own state programmatically
 * and narrate the visible outcome — the user is never asked to operate a
 * control mid-tour. The search step targets ONLY the tour collection
 * (`TOUR_COLLECTION`, EMNLP): one Gemini embed call per tour run, a
 * deliberate cost — no other collection is ever auto-queried.
 */
export const TOUR_STEPS: TourStepDefinition[] = [
  {
    id: 'map',
    anchor: 'plot',
    title: 'A map of meaning',
    body:
      'Every point is a document, placed by a language model so that distance mirrors meaning: ' +
      'nearby points say similar things. Drag to rotate, scroll to zoom — the tour will wait.',
    allowInteraction: true,
    placement: 'center',
  },
  {
    id: 'collections',
    anchor: 'collectionSelector',
    title: 'Collections are spaces',
    body:
      'Each collection is its own embedding space — different documents, different model, ' +
      'different geometry. Switch anytime; the view adapts to whatever the collection provides.',
  },
  {
    id: 'structure',
    anchor: 'plot',
    title: 'Structure becomes visible',
    body:
      'These are 13,980 EMNLP paper abstracts in 3D UMAP, clustered into 60 research topics ' +
      'named by an LLM. Clusters are shared meaning — labels mark their centers and the haze ' +
      'traces their extent. Colors, projections, and overlays all live in the Controls panel (top-left).',
    allowInteraction: true,
    placement: 'center',
    prepareTimeoutMs: 30000,
    prepare: async (runtime) => {
      // Normally applied when the tour starts; re-applying is an idempotent
      // safety net for direct `?tour=1` entries that beat the manifest load.
      runtime.applyPreset(TOUR_PRESET_ID);
      await waitFor(
        () =>
          runtime.getLoadedCollection() === TOUR_COLLECTION &&
          runtime.getColorByField() !== null,
        25000,
      );
      // Let the plot commit a frame with the new colouring before spotlighting.
      await delay(400);
    },
  },
  {
    id: 'search',
    anchor: 'searchInput',
    title: 'Search by meaning',
    body:
      `We just searched "${TOUR_SEARCH_QUERY}" — the query is embedded into the same space, ` +
      'so matching abstracts glow by similarity even when they share no words with it. ' +
      'The Search panel adds substring and metadata filtering on top.',
    prepareTimeoutMs: 30000,
    prepare: async (runtime) => {
      // Hard guard: auto-search is allowed against the tour collection only.
      if (runtime.getLoadedCollection() !== TOUR_COLLECTION) return;
      runtime.setShowLabels(true);
      await runtime.runSearch(TOUR_SEARCH_QUERY);
    },
  },
  {
    id: 'analytics',
    anchor: 'panelAnalytics',
    title: 'Filter and dissect',
    body:
      'The Analytics panel shows distributions for any field — and we just isolated one ' +
      'research topic by clicking its row: every other cluster fades in the map behind. ' +
      'Click a row to isolate, shift-click to combine, filter by year on the timeline.',
    placement: 'right',
    prepare: async (runtime) => {
      // A fresh canvas for the isolation demo: drop the search glow first.
      runtime.clearSearch();
      runtime.isolateFirstTopic();
      runtime.setActivePanel('analytics');
      // The panel is always mounted, slid offscreen; wait out its transition
      // so the spotlight measures the on-screen position. (Headless test runs
      // have no document — treat the panel as already in place.)
      await waitFor(() => {
        if (typeof document === 'undefined') return true;
        const el = document.querySelector(TOUR_ANCHORS.panelAnalytics);
        return el !== null && el.getBoundingClientRect().left >= 0;
      }, 2000);
      await delay(150);
    },
  },
  {
    id: 'finale',
    anchor: 'plot',
    title: 'Every collection is a new space',
    body:
      'One more: 954 color names embedded as pure text — no pixels, no wavelengths — and the ' +
      'rainbow reassembles itself from language alone. That was the tour. The map is yours; ' +
      'replay it or pick another mission from the ? button up top.',
    allowInteraction: true,
    placement: 'center',
    prepareTimeoutMs: 30000,
    prepare: async (runtime) => {
      runtime.setActivePanel(null);
      runtime.applyPreset(FINALE_PRESET_ID);
      await waitFor(
        () =>
          runtime.getLoadedCollection() === TOUR_PRESETS[FINALE_PRESET_ID].collection &&
          runtime.getColorByField() !== null,
        25000,
      );
      await delay(400);
    },
  },
];
