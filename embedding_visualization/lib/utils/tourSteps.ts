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
  /** The `?` mission-menu button (demo builds only — absent elsewhere). */
  introButton: '[data-tour="intro-button"]',
  searchInput: '[data-tour="search-input"]',
  panelAnalytics: '[data-tour="panel-analytics"]',
  temporalChart: '[data-tour="temporal-chart"]',
  featureSearch: '[data-tour="feature-search"]',
  probeSection: '[data-tour="probe-section"]',
  /** ProbeSection's settings popover content — mounted only while open. */
  probeSettings: '[data-tour="probe-settings"]',
  /** The probe row whose field is the active colouring (at most one). */
  probeActive: '[data-tour="probe-active"]',
  // Reserved:
  toggleControls: '[data-tour="toggle-controls"]',
  toggleSearch: '[data-tour="toggle-search"]',
  toggleAnalytics: '[data-tour="toggle-analytics"]',
  panelControls: '[data-tour="panel-controls"]',
  panelSearch: '[data-tour="panel-search"]',
} as const;

export type TourAnchor = keyof typeof TOUR_ANCHORS;

/** Optional adjustment applied on top of the default camera framing. */
export interface CameraViewAdjustment {
  /** Orbit around the vertical axis, degrees. Positive = counterclockwise from above. */
  azimuthDeg?: number;
  /** Tilt, degrees. Negative lowers the camera toward the horizon (looks "downwards"). */
  elevationDeg?: number;
  /** Eye-distance multiplier: 0.75 = 25% closer than the default framing. */
  zoom?: number;
  /** Vertical pan (eye + target together). Negative = camera down / scene up. */
  panZ?: number;
  /** Animation length, ms (default 1200). Longer = a slow cinematic move. */
  durationMs?: number;
  /**
   * Apply the adjustment to the LIVE camera instead of the default framing.
   * Use for moves that must compose with wherever the user (or a search
   * fly-to) left the camera — e.g. "20% closer than right now" (`zoom: 0.8`).
   * Default-relative zoom cannot express that: after a search dive the camera
   * is far closer than any default multiple, so every absolute target reads
   * as zooming back out.
   */
  relative?: boolean;
}

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
  /**
   * Animate the 3D camera to the default wide framing, optionally adjusted
   * (orbit / tilt / zoom / vertical pan — see `CameraViewAdjustment`; flip a
   * sign if a view reads rotated the wrong way). Tours use it both to undo a
   * search fly-to and to open a collection at the angle where its structure
   * is actually visible.
   */
  resetCamera: (view?: CameraViewAdjustment) => void;
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
  /**
   * Recolor the map by a metadata field (null → the uncoloured single-hue
   * view). `scaleType` picks the scale family; `opts.scaleName` pins a
   * specific named scale (otherwise the family default applies — the
   * sequential default is the rainbow-like sinebow); `centerZero` pins a
   * diverging scale's neutral midpoint at 0 (probe residuals — mirrors
   * ProbeSection).
   */
  setColorBy: (
    field: string | null,
    scaleType?: 'sequential' | 'diverging' | 'categorical',
    opts?: { scaleName?: string; centerZero?: boolean },
  ) => void;
  /** Whether `field` is currently a known Color By option (probe fields load async). */
  hasColorField: (field: string) => boolean;
  /** Toggle the nebula haze overlay (per-category density glow). */
  setNebulaMode: (on: boolean) => void;
  /** Toggle topic-name labels at cluster centroids. */
  setShowClusterLabels: (on: boolean) => void;
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
  /**
   * Overrides the primary button's caption for this step. The default is
   * "Next" ("Done" on the last step) — a chained tour's handoff step sets
   * "Next" so finishing segment 1 doesn't read as the end of the tour.
   */
  primaryLabel?: string;
  /** Let pointer events through the spotlight (rotate/zoom the plot). */
  allowInteraction?: boolean;
  /**
   * Extra anchors to ring-highlight while the step's tooltip is shown.
   * joyride cuts only ONE spotlight per step, so a step that points at
   * several controls (the finale: collection selector + `?` mission menu,
   * on different header rows in demo builds) lists them here instead —
   * `TourController` draws its own spotlight-style rings over each, on top
   * of whatever overlay the step has. Anchors that aren't mounted (the `?`
   * button outside demo builds) are silently skipped.
   */
  highlightAnchors?: A[];
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

/** Promise that resolves after `ms` (shared by every tour's settle waits). */
export const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Wait for a step's anchor element to be mounted and laid out.
 *
 * **Any prepare on a step whose anchor mounts late MUST call this.**
 * react-joyride polls for a not-yet-mounted target only on steps WITHOUT a
 * `before` hook (`useLifecycleEffect` effect 2: the polling branch is
 * `else if (!beforeRef.current)`). Adding a prepare opts the step out of that
 * polling entirely — the step goes straight to READY when the hook resolves,
 * and a missing target is then treated as TARGET_NOT_FOUND, which silently
 * advances to the next step. So a prepare on the tour's first step turned the
 * built-in wait for the collection load into "skip step 1".
 *
 * The `left >= 0` check also covers the panels, which are always mounted and
 * slid off-screen: an anchor inside one is in the DOM long before it's visible.
 */
export function waitForAnchor(selector: string, timeoutMs: number): Promise<boolean> {
  return waitFor(() => {
    // Headless test runs have no document — treat the anchor as in place.
    if (typeof document === 'undefined') return true;
    const el = document.querySelector(selector);
    return el !== null && el.getBoundingClientRect().left >= 0;
  }, timeoutMs);
}

/**
 * Scroll a step's anchor into the middle of its scroll container. Used for
 * anchors below the fold of a panel — joyride only auto-scrolls to the step's
 * own target, and the map-narrating steps deliberately anchor on the plot.
 */
export function scrollAnchorIntoView(selector: string): void {
  if (typeof document === 'undefined') return;
  const el = document.querySelector(selector);
  if (!el) return;
  const reducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  el.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'center' });
}

/**
 * Apply a preset and poll until its collection is the loaded one, re-applying
 * on every tick — a direct `?tour=` cold entry can beat the collections
 * manifest, in which case the early calls no-op. Returns whether the
 * collection landed within `timeoutMs`.
 */
export async function ensurePresetCollection(
  runtime: Pick<TourRuntime, 'applyPreset' | 'getLoadedCollection'>,
  presetId: string,
  collection: string,
  timeoutMs: number,
  intervalMs = 1000,
): Promise<boolean> {
  runtime.applyPreset(presetId);
  return waitFor(() => {
    if (runtime.getLoadedCollection() === collection) return true;
    runtime.applyPreset(presetId);
    return false;
  }, timeoutMs, intervalMs);
}

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
      'Every point is one of 13,980 EMNLP paper abstracts, embedded by a language model so ' +
      'that distance mirrors meaning: nearby points are semantically similar. The 60 colored ' +
      'clusters are research topics labeled by an LLM. Drag to rotate, scroll to zoom, click to inspect.',
    allowInteraction: true,
    placement: 'left',
    // Tour start switches to the tour collection: the plot target is behind
    // the page's own loader while joyride waits for it to appear.
    suppressWaitLoader: true,
    prepareTimeoutMs: 35000,
    prepare: async (runtime) => {
      // Closing the panel is a no-op going forward (panels start closed); it
      // matters on Back from the search step, which opens the Search panel.
      runtime.setActivePanel(null);
      // MANDATORY, not a nicety: having any prepare at all disables joyride's
      // own target polling (see waitForAnchor), and this step's anchor only
      // mounts once the tour collection has loaded. Without this wait the
      // first step is dropped as TARGET_NOT_FOUND and the tour opens on step 2.
      await waitForAnchor(TOUR_ANCHORS.plotSide, 30000);
    },
  },
  {
    id: 'search',
    anchor: 'searchInput',
    title: 'Search by meaning',
    body:
      `We're semantically searching "${TOUR_SEARCH_QUERY}". ` +
      'The matching abstracts glow according to similarity in the original space even when they share ' +
      'no words with the query, and the camera dives to the best match. The Search panel ' +
      'adds text search and metadata filtering.',
    prepareTimeoutMs: 30000,
    // The only awaited work is the collection wait, which sits behind the
    // page's own loader; the search itself is fire-and-forget (below).
    suppressWaitLoader: true,
    prepare: async (runtime) => {
      // First prepared step: it owns the wait for the tour collection.
      await ensureTourCollection(runtime);
      // Hard guard: auto-search is allowed against the tour collection only.
      if (runtime.getLoadedCollection() !== TOUR_COLLECTION) return;
      // The panel is an absolutely-positioned overlay, so opening it doesn't
      // move the header input the spotlight is anchored to — no reposition
      // wait needed (unlike the feature-search step, whose anchor lives
      // inside the panel). It also stays open into that next step.
      runtime.setActivePanel('search');
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
    title: 'Search by the model’s SAE features',
    body:
      `An experimental kind of search: this corpus also carries Gemma Scope 2 SAE activations. ` +
      `We typed “${TOUR_FEATURE_QUERY}”, matched the model’s own ` +
      '“humor and jokes” feature, and the abstracts where it fires strongest ' +
      'light up: the computational-humor papers surface. ' +
      'SAE search runs entirely on stored activations without reloading the model.',
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
      await waitForAnchor(TOUR_ANCHORS.featureSearch, 4000);
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
      'the camera centers on what remains. Any click on the legend row, ' +
      'analytics row, or cluster label filters the same way. Shift-click adds to the selection, double click resets.',
    allowInteraction: true,
    placement: 'left',
    prepare: async (runtime) => {
      runtime.clearSearch();
      // The Analytics category list is where the isolation is legible as data
      // (rows for the surviving topic) and it's one of the surfaces the body
      // names as clickable. It sits far left; this step's card is pinned to
      // the plot's right edge, so they never collide. The 400 ms settle below
      // covers the panel's 300 ms slide-in — and the anchor is outside the
      // panel, so nothing needs to wait on it.
      runtime.setActivePanel('analytics');
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
      'Every abstract carries its year, so the Analytics panel (left) displays a timeline with the yearly count. We ' +
      'narrowed it to the earliest years of the corpus: later work fades, and ' +
      'whole topics go silent: those research topics did not exist yet. Drag the handles to ' +
      'change intervals; double-click resets.',
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
      // so the chart is on-screen before we scroll to it.
      await waitForAnchor(TOUR_ANCHORS.temporalChart, 2000);
      // The category list above it is tall enough to push the timeline below
      // the panel's fold — joyride only auto-scrolls to a step's own target,
      // and this step deliberately anchors on the plot instead.
      scrollAnchorIntoView(TOUR_ANCHORS.temporalChart);
      // Covers both the panel transition and the smooth scroll.
      await delay(400);
    },
  },
  {
    id: 'density',
    anchor: 'plotSide',
    title: 'The flat map',
    body:
      'The same space, flattened to 2D with density contouring: ink pools where abstracts ' +
      'concentrate, tinted by topic. Rotation becomes panning: this is the view built for ' +
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
    // Card on the plot's right edge (vertically centered → just under the
    // legend), NOT over the map it invites the user to explore. The two ways
    // on from here — the collection selector and the `?` mission menu — get
    // ring highlights instead of a spotlight: `allowInteraction` drops the
    // overlay (map + header stay fully live), and joyride could only cut one
    // spotlight anyway while these sit on different header rows in demo builds.
    anchor: 'plotSide',
    highlightAnchors: ['collectionSelector', 'introButton'],
    title: 'Explore other collections',
    body:
      'Explore different corpus: 1,000 tweets embedded with MiniLM and mapped into emotional topics. ' +
      'Its search model runs entirely inside this Space, so query as much as ' +
      'you like. Switch corpus from the highlighted selector up top, and replay ' +
      'the tour or pick another mission from the ? button.',
    allowInteraction: true,
    placement: 'left',
    prepareTimeoutMs: 30000,
    suppressWaitLoader: true,
    prepare: async (runtime) => {
      // Parting view doubles as an invitation to keep exploring: the Controls
      // panel is open so "how to draw" is discoverable the moment the tour ends.
      runtime.setActivePanel('controls');
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
