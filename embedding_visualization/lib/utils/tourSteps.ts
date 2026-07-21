import { DEMO_DEFAULT_COLLECTION, TOUR_PRESET_ID, TOUR_SEARCH_QUERY } from './tourPresets';

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
 * control mid-tour. The search step only ever targets the demo default
 * collection (emotion), whose embedding model runs inside the Space; the
 * Gemini-embedded collections must never be auto-queried.
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
      'This is the emotion collection: 1,000 tweets in 3D UMAP, colored by emotion label. ' +
      'Clusters are shared meaning — labels mark their centers and the haze traces their extent. ' +
      'Colors, projections, and overlays all live in the Controls panel (top-left).',
    allowInteraction: true,
    placement: 'center',
    prepareTimeoutMs: 30000,
    prepare: async (runtime) => {
      runtime.applyPreset(TOUR_PRESET_ID);
      await waitFor(
        () =>
          runtime.getLoadedCollection() === DEMO_DEFAULT_COLLECTION &&
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
      'so matches glow by similarity even when they share no words with it. ' +
      'The Search panel adds substring and metadata filtering on top.',
    prepareTimeoutMs: 30000,
    prepare: async (runtime) => {
      // Only the locally-embedded demo collection may be auto-queried.
      if (runtime.getLoadedCollection() !== DEMO_DEFAULT_COLLECTION) return;
      runtime.setShowLabels(true);
      await runtime.runSearch(TOUR_SEARCH_QUERY);
    },
  },
  {
    id: 'analytics',
    anchor: 'panelAnalytics',
    title: 'Analytics, and over to you',
    body:
      'Distributions for any field — click a row to isolate a category, shift-click to combine. ' +
      "That's the tour. Explore this space, or pick another mission from the ? button up top.",
    placement: 'right',
    prepare: async (runtime) => {
      runtime.setActivePanel('analytics');
      // The panel is always mounted, slid offscreen; wait out its transition
      // so the spotlight measures the on-screen position.
      await waitFor(() => {
        const el = document.querySelector(TOUR_ANCHORS.panelAnalytics);
        return el !== null && el.getBoundingClientRect().left >= 0;
      }, 2000);
      await delay(150);
    },
  },
];
