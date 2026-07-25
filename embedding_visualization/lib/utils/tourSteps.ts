import {
  TOUR_COLLECTION,
  TOUR_PRESET_ID,
  TOUR_SEARCH_QUERY,
  TOUR_FEATURE_QUERY,
  TOUR_PRESETS,
} from './tourPresets';

/**
 * The finale switches to the emotion collection — the "collections are
 * spaces" payoff, landing on the demo default whose search model runs inside
 * the Space (visitors can keep querying without spending Gemini quota).
 */
export const FINALE_PRESET_ID = 'emotion';

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
  /**
   * Invisible 1px anchor pinned at the vertical center of the plot's right
   * edge. Whole-plot steps target it with `placement: 'left'` so their card
   * sits to the side instead of covering the very view it narrates
   * (react-joyride's `center` placement parks the tooltip mid-screen).
   */
  plotSide: '[data-tour="plot-side"]',
  collectionSelector: '[data-tour="collection-selector"]',
  searchInput: '[data-tour="search-input"]',
  panelAnalytics: '[data-tour="panel-analytics"]',
  temporalChart: '[data-tour="temporal-chart"]',
  featureSearch: '[data-tour="feature-search"]',
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
  /**
   * SAE feature-name → document search: resolve `labelQuery` against the
   * loaded collection's linked SAE (top label matches), select those
   * features, and wait for the ranked-document highlight to land. Pure
   * DuckDB on stored activations — no embedding call. Returns whether any
   * documents matched.
   */
  runFeatureSearch: (labelQuery: string) => Promise<boolean>;
  /** Clear any search highlight — semantic glow AND feature ranking. */
  clearSearch: () => void;
  /** Animate the 3D camera back to the default wide framing. */
  resetCamera: () => void;
  /** Isolate the first topic cluster (others mute); returns its label. */
  isolateFirstTopic: () => string | null;
  /** Drop any tour-applied topic isolation (tour-end cleanup). */
  clearTopicSelection: () => void;
  /**
   * Filter to a fractional window of the detected temporal field's periods
   * (e.g. `(0, 1/3)` = the earliest third). Returns false when the collection
   * has no usable temporal field or the window is the full range.
   */
  applyTemporalWindow: (fromFrac: number, toFrac: number) => boolean;
  /** Drop any tour-applied temporal window (step transition / tour-end cleanup). */
  clearTemporalFilter: () => void;
  /** true → 2D + density contours; false → back to the 3D galaxy. */
  setDensityView: (on: boolean) => void;
  setActivePanel: (panel: 'controls' | 'search' | 'analytics' | null) => void;
  setShowLabels: (value: boolean) => void;
  getLoadedCollection: () => string | null;
  getColorByField: () => string | null;
  /**
   * SAE feature index of the currently selected point (`metadata.index` on
   * feature-map collections), or null when nothing suitable is selected.
   * Used by the "Inspect SAE" tour's Explore segment to hand off to /sae.
   */
  getSelectedFeatureIndex: () => number | null;
}

/**
 * Library-agnostic step shape shared by every tour. `A` is the page's anchor
 * union, `R` the imperative runtime its page hands to prepare hooks —
 * `TourController` renders any instantiation.
 */
export interface TourStepDefinitionBase<A extends string, R> {
  id: string;
  anchor: A;
  title: string;
  body: string;
  /** Let pointer events through the spotlight (rotate/zoom the plot). */
  allowInteraction?: boolean;
  placement?: 'auto' | 'center' | 'bottom' | 'left' | 'right';
  /** Ceiling for `prepare` (react-joyride `beforeTimeout`), ms. */
  prepareTimeoutMs?: number;
  /**
   * Hide joyride's built-in waiting spinner for this step. Set on steps whose
   * prepare switches collections: the page renders its own full-screen loader
   * there, and two spinners animate on top of each other without this.
   */
  suppressWaitLoader?: boolean;
  /** State setup run before the step is shown; the tour waits for it. */
  prepare?: (runtime: R) => Promise<void>;
}

/** The Explore tour's concrete step type. */
export type TourStepDefinition = TourStepDefinitionBase<TourAnchor, TourRuntime>;

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
 * Re-apply the tour preset and wait for its collection + colouring to land.
 * Normally a no-op (the tour start already applied it); it's the idempotent
 * safety net for direct `?tour=1` entries that beat the manifest load, and for
 * Back-navigation from the finale's collection switch.
 */
async function ensureTourCollection(runtime: TourRuntime): Promise<void> {
  runtime.applyPreset(TOUR_PRESET_ID);
  await waitFor(
    () =>
      runtime.getLoadedCollection() === TOUR_COLLECTION &&
      runtime.getColorByField() !== null,
    25000,
  );
}

/**
 * The seven demo tour steps. Steps prepare their own state programmatically
 * and narrate the visible outcome — the user is never asked to operate a
 * control mid-tour. The semantic search step targets ONLY the tour
 * collection (`TOUR_COLLECTION`, EMNLP): one Gemini embed call per tour run,
 * a deliberate cost — no other collection is ever auto-queried. The SAE
 * feature-search step is free (DuckDB over stored activations).
 */
export const TOUR_STEPS: TourStepDefinition[] = [
  {
    id: 'map',
    anchor: 'plotSide',
    title: 'A map of meaning',
    body:
      'Every point is one of 13,980 EMNLP paper abstracts, placed by a language model so ' +
      'that distance mirrors meaning: nearby points say similar things. The 60 colored ' +
      'clusters are research topics named by an LLM — labels mark their centers, haze ' +
      'traces their extent. Drag to rotate, scroll to zoom — the tour will wait.',
    allowInteraction: true,
    placement: 'left',
    // Tour start switches to the tour collection: the plot target is behind
    // the page's own loader while joyride waits for it to appear.
    suppressWaitLoader: true,
  },
  {
    id: 'search',
    anchor: 'searchInput',
    title: 'Search by meaning',
    body:
      `We're searching "${TOUR_SEARCH_QUERY}" — the query is embedded into the same space, ` +
      'so as results land the matching abstracts glow by similarity even when they share ' +
      'no words with it, and the camera dives to the best match. The Search panel adds ' +
      'substring and metadata filtering on top.',
    prepareTimeoutMs: 30000,
    // The only awaited work is the collection wait, which sits behind the
    // page's own loader; the search itself is fire-and-forget (below).
    suppressWaitLoader: true,
    prepare: async (runtime) => {
      // First prepared step: it owns the wait for the tour collection.
      await ensureTourCollection(runtime);
      // Hard guard: auto-search is allowed against the tour collection only.
      if (runtime.getLoadedCollection() !== TOUR_COLLECTION) return;
      runtime.setShowLabels(true);
      // Deliberately NOT awaited: the tooltip appears immediately, the search
      // input's own spinner (inside the spotlight) shows progress, and the
      // glow + camera dive land as a reveal while the user reads. A stale
      // result can't leak into later steps — clearSearch invalidates
      // in-flight requests.
      void runtime.runSearch(TOUR_SEARCH_QUERY).catch(() => {});
    },
  },
  {
    id: 'feature-search',
    anchor: 'featureSearch',
    title: 'Search by the model’s features',
    body:
      `A different kind of search: this corpus also carries Gemma’s SAE activations. ` +
      `We typed “${TOUR_FEATURE_QUERY}”, matched the model’s own ` +
      '“humor and jokes” feature, and the abstracts where it fires strongest ' +
      'light up — the computational-humor papers surface without sharing a keyword. ' +
      'This runs entirely on stored activations: no model in the loop.',
    placement: 'right',
    prepareTimeoutMs: 15000,
    prepare: async (runtime) => {
      // Same hard guard as the semantic step: tour collection only.
      if (runtime.getLoadedCollection() !== TOUR_COLLECTION) return;
      // Swap glows: drop the semantic constellation before the feature ranking.
      runtime.clearSearch();
      runtime.setActivePanel('search');
      // The Feature Search section renders once the hasActivations probe
      // resolves and the panel slides in — wait for it to be measurable.
      await waitFor(() => {
        if (typeof document === 'undefined') return true;
        const el = document.querySelector(TOUR_ANCHORS.featureSearch);
        return el !== null && el.getBoundingClientRect().left >= 0;
      }, 4000);
      await runtime.runFeatureSearch(TOUR_FEATURE_QUERY);
      await delay(150);
    },
  },
  {
    id: 'focus-topic',
    anchor: 'plotSide',
    title: 'One topic in focus',
    body:
      'We just selected a single research topic: every other cluster fades to a whisper and ' +
      'the camera reframes on what remains — its haze and label stay live. Any legend row, ' +
      'analytics row, or cluster label isolates the same way.',
    allowInteraction: true,
    placement: 'left',
    prepare: async (runtime) => {
      runtime.clearSearch();
      runtime.setActivePanel(null);
      // Topics arrive via their own query after the collection loads; on a
      // fast path they're long since present and the first call isolates.
      // Otherwise poll until they land (each failed attempt is a no-op).
      if (runtime.isolateFirstTopic() === null) {
        await waitFor(() => runtime.isolateFirstTopic() !== null, 5000, 250);
      }
      // The muting auto-refit animates toward the isolated cluster; give it a
      // beat so the spotlight appears over a view already in motion.
      await delay(400);
    },
  },
  {
    id: 'temporal',
    // Card on the right edge with the rest of the map-narrating steps — the
    // step's payoff is regions going dark mid-map, which a chart-anchored
    // tooltip would sit right on top of. The prepare still waits on the
    // temporalChart element so the timeline is on-screen when the card shows.
    anchor: 'plotSide',
    title: 'Travel through time',
    body:
      'Every abstract carries its year, so the Analytics panel (left) grows a timeline. We ' +
      'narrowed it to the earliest years of the corpus — later work fades in the map, and ' +
      'whole regions go dark: those research topics did not exist yet. Drag the handles to ' +
      'scrub; double-click resets.',
    allowInteraction: true,
    placement: 'left',
    prepare: async (runtime) => {
      // One idea per step: drop the topic isolation, keep only the time window.
      runtime.clearTopicSelection();
      // Back-navigation from the 2D density step re-enters here — restore 3D.
      runtime.setDensityView(false);
      runtime.setActivePanel('analytics');
      runtime.applyTemporalWindow(0, 1 / 3);
      // The panel is always mounted, slid offscreen; wait out its transition
      // so the spotlight measures the on-screen position. (Headless test runs
      // have no document — treat the chart as already in place.)
      await waitFor(() => {
        if (typeof document === 'undefined') return true;
        const el = document.querySelector(TOUR_ANCHORS.temporalChart);
        return el !== null && el.getBoundingClientRect().left >= 0;
      }, 2000);
      await delay(150);
    },
  },
  {
    id: 'density',
    anchor: 'plotSide',
    title: 'The flat map',
    body:
      'The same space, flattened to 2D with density contours: ink pools where abstracts ' +
      'concentrate, tinted by topic. Rotation becomes panning — this is the view built for ' +
      'scale, and one click in Controls flips any collection between galaxy and map.',
    allowInteraction: true,
    placement: 'left',
    prepareTimeoutMs: 30000,
    // Back-navigation from the finale re-enters via a collection switch that
    // sits behind the page's own loader.
    suppressWaitLoader: true,
    prepare: async (runtime) => {
      await ensureTourCollection(runtime);
      runtime.clearTopicSelection();
      runtime.clearTemporalFilter();
      // The Analytics panel stays open on purpose: its distributions keep
      // describing the same points now pooling as 2D density ink.
      runtime.setDensityView(true);
      // The 2D plot mounts fresh — let it draw before the spotlight measures.
      await delay(500);
    },
  },
  {
    id: 'finale',
    anchor: 'plotSide',
    title: 'Every collection is a new space',
    body:
      'Last stop: 1,000 tweets mapped into emotional topics — a different corpus, the same ' +
      'instrument. Its search model runs entirely inside this Space, so query as much as ' +
      'you like. That was the tour. The map is yours; replay it or pick another mission ' +
      'from the ? button up top.',
    allowInteraction: true,
    placement: 'left',
    prepareTimeoutMs: 30000,
    suppressWaitLoader: true,
    prepare: async (runtime) => {
      runtime.setActivePanel(null);
      // The preset restores 3D and pins densityMode off (see tourPresets).
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
