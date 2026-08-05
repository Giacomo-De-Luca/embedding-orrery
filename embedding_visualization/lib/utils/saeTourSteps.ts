import type { TourAnchor, TourRuntime, TourStepDefinitionBase } from './tourSteps';
import { TOUR_ANCHORS, delay, ensurePresetCollection, waitFor, waitForAnchor } from './tourSteps';
import { SAE_MAP_COLLECTION, SAE_MAP_PRESET_ID } from './tourPresets';

/**
 * "Inspect SAE" tour: two segments chained across a page navigation (joyride
 * cannot survive one).
 *
 * Segment 1 (`SAE_MAP_TOUR_STEPS`) runs on the Explore page over the SAE
 * label map — every point is a feature — and ends by handing off to /sae with
 * the top search match's feature pre-selected (`saeInspectPath`).
 *
 * Segment 2 (`SAE_TOUR_STEPS`) runs on /sae: feature anatomy, activation
 * examples, the pre-recorded steered chat, and the link back to the map.
 * A direct `/sae?tour=sae` entry (no featureIndex) still works — the anatomy
 * step falls back to running the search itself.
 */

/** The demo's explorable SAE — the pair used for steering in the paper. */
export const SAE_TOUR_MODEL_ID = 'gemma-3-4b-it';
export const SAE_TOUR_SAE_ID = '9-gemmascope-2-res-16k';

/**
 * The auto-run query, searched on BOTH segments. The label map and the /sae
 * label collection are MiniLM-embedded — queries run on the local model, no
 * metered API involved, safe to run per tour.
 */
export const SAE_TOUR_QUERY = 'poetry';

/** Entry URL for the tour: it starts on the Explore page's label map. */
export function saeTourPath(): string {
  return '/?tour=sae';
}

/**
 * Segment-2 handoff URL: pins the pair explicitly (the demo seed carries a
 * second, hidden features-only pair, so default model selection must never
 * be relied on), deep-links the chosen feature, and carries the one-shot
 * `tour=sae` trigger that /sae latches at mount.
 */
export function saeInspectPath(featureIndex?: number | null): string {
  const params = new URLSearchParams({
    modelId: SAE_TOUR_MODEL_ID,
    saeId: SAE_TOUR_SAE_ID,
  });
  if (featureIndex != null) params.set('featureIndex', String(featureIndex));
  params.set('tour', 'sae');
  return `/sae?${params.toString()}`;
}

// ── Segment 1: the label map on Explore ────────────────────────────────────

export type SaeMapTourStepDefinition = TourStepDefinitionBase<TourAnchor, TourRuntime>;

export const SAE_MAP_TOUR_STEPS: SaeMapTourStepDefinition[] = [
  {
    id: 'sae-label-map',
    // Right-edge anchor + placement left: whole-plot steps must not park
    // their card mid-screen over the very map they narrate (same convention
    // as the Explore tour's plot steps).
    anchor: 'plotSide',
    title: 'A map of the model’s representations',
    body:
      'Each of these 15,854 points is a ' +
      'sparse-autoencoder feature from layer 9 of gemma-3-4b-it, a direction in its residual ' +
      'stream. The points are displayed using the MiniLM embeddings of the Neuronpedia’s autointerpreter labels.',
    allowInteraction: true,
    placement: 'left',
    prepareTimeoutMs: 30000,
    // The collection switch sits behind the page's own full-screen loader.
    suppressWaitLoader: true,
    prepare: async (runtime) => {
      const loaded = await ensurePresetCollection(
        runtime,
        SAE_MAP_PRESET_ID,
        SAE_MAP_COLLECTION,
        25000,
      );
      // Prepared steps opt out of joyride's own target polling (see
      // waitForAnchor): the plot anchor mounts after the collection load.
      await waitForAnchor(TOUR_ANCHORS.plotSide, 20_000);
      // The label map's constellation sits off-axis at the default framing:
      // orbit right (negative — live-tuned; +30 turned the wrong way), tilt
      // like a small upward DRAG (negative elevation — dragging up lowers the
      // camera; +25 read as dragging down), and pan slightly up (positive)
      // to center it.
      if (loaded) runtime.resetCamera({ azimuthDeg: -50, elevationDeg: -25, panZ: 0.12 });
    },
  },
  {
    id: 'sae-map-search',
    anchor: 'searchInput',
    title: 'Semantically search features',
    body:
      `We're searching "${SAE_TOUR_QUERY}". The query is embedded by MiniLM running ` +
      'inside this Space, so the features with similar labels light up ' +
      'and the camera dives to the best match.',
    placement: 'bottom',
    // First search cold-starts the Space's MiniLM slot; generous ceiling.
    prepareTimeoutMs: 30000,
    suppressWaitLoader: true,
    prepare: async (runtime) => {
      // Hard guard: auto-search is allowed against the label map only.
      if (runtime.getLoadedCollection() !== SAE_MAP_COLLECTION) return;
      // Fire-and-forget (same reveal pattern as the Explore tour): the search
      // input's own spinner shows progress while the user reads the tooltip.
      void runtime.runSearch(SAE_TOUR_QUERY).catch(() => {});
    },
  },
  {
    id: 'sae-right-click',
    anchor: 'plotSide',
    title: 'Right-click to inspect a feature',
    body:
      'Every point here can be inspected: right-click one and choose "View Feature" to jump ' +
      'to its page in the SAE explorer. Press Next and we’ll do that with the top ' +
      `"${SAE_TOUR_QUERY}" match. The tour continues in the SAE page.`,
    // Last step of THIS joyride instance, but not of the tour — the Done
    // caption would contradict the "press Next" copy and read as the end.
    primaryLabel: 'Next',
    allowInteraction: true,
    placement: 'left',
    prepareTimeoutMs: 15000,
    prepare: async (runtime) => {
      // The search auto-selects its best match; wait for it so Done can hand
      // its feature index to /sae. Timing out is fine — the handoff then goes
      // without a preselected feature and segment 2 searches on its own.
      await waitFor(() => runtime.getSelectedFeatureIndex() !== null, 12000);
    },
  },
];

/**
 * Step-counter continuity across the navigation: segment 1 shows steps
 * 1…3 of the combined total, segment 2 shows 4…7 (TourController's
 * progressOffset/progressTotal props).
 */
export const SAE_TOUR_STEP_OFFSET = SAE_MAP_TOUR_STEPS.length;

// ── Segment 2: the /sae Feature Explorer ───────────────────────────────────

/** data-tour anchor names → CSS selectors on the /sae page. */
export const SAE_TOUR_ANCHORS = {
  header: '[data-tour="sae-header"]',
  searchInput: '[data-tour="sae-search-input"]',
  results: '[data-tour="sae-results"]',
  detail: '[data-tour="sae-detail"]',
  activations: '[data-tour="sae-activations"]',
  chat: '[data-tour="sae-chat"]',
  mapLink: '[data-tour="sae-map-link"]',
} as const;

export type SaeTourAnchor = keyof typeof SAE_TOUR_ANCHORS;

/** Imperative surface the /sae page hands to the tour's prepare hooks. */
export interface SaeTourRuntime {
  setSearchMode: (mode: 'text' | 'semantic') => void;
  /** Run a semantic feature search; resolves when the fan-out has landed. */
  runSemanticSearch: (query: string) => Promise<void>;
  /** Open the top semantic result in the detail pane; false when empty. */
  openFirstResult: () => boolean;
  /** True when a feature is already open in the detail pane (deep link). */
  hasSelectedFeature: () => boolean;
  /** True once the detail pane has its feature loaded (not loading). */
  isDetailLoaded: () => boolean;
  getResultCount: () => number;
  /** Slide the steered-chat sidebar open. */
  openChat: () => void;
  /** True once the demo chat fixture has resolved (even if empty). */
  isChatReady: () => boolean;
  /** Load the first pre-recorded session into the chat; false when none. */
  loadFirstDemoSession: () => Promise<boolean>;
}

export type SaeTourStepDefinition = TourStepDefinitionBase<SaeTourAnchor, SaeTourRuntime>;

/**
 * Window event the steered-chat step fires to close ChatPanel's lateral
 * history list (open by default in demo builds): with both the history and a
 * replayed conversation showing, the un-resized divider leaves the thread too
 * cramped to read. An event because `showHistory` is ChatPanel-local state
 * (same pattern as PROBE_SHOWCASE_EVENT in probeTourSteps.ts).
 */
export const SAE_CHAT_HISTORY_EVENT = 'orrery:sae-chat-history';
export interface SaeChatHistoryDetail {
  open: boolean;
  /** Also rewind the thread to its first message (a replayed session opens scrolled to the bottom). */
  scrollToTop?: boolean;
}

/**
 * The /sae segment. Arriving from the Explore segment the feature is already
 * deep-linked open; the anatomy step's prepare only runs the search fallback
 * for direct `/sae?tour=sae` entries.
 */
export const SAE_TOUR_STEPS: SaeTourStepDefinition[] = [
  {
    id: 'feature-anatomy',
    anchor: 'detail',
    title: 'The anatomy of a feature',
    body:
      'This is the feature we just opened from the map. Its card shows the label, how ' +
      'often it fires (density), and which output tokens it pushes toward or suppresses ' +
      '(top and bottom logits).',
    placement: 'left',
    // Cold-start ceiling for the fallback search (local MiniLM embed).
    prepareTimeoutMs: 30000,
    prepare: async (runtime) => {
      if (!runtime.hasSelectedFeature()) {
        // Direct entry without a deep-linked feature: search and open here.
        runtime.setSearchMode('semantic');
        await runtime.runSemanticSearch(SAE_TOUR_QUERY);
        if (!runtime.openFirstResult()) return;
      }
      await waitFor(runtime.isDetailLoaded, 10000);
      await delay(200);
    },
  },
  {
    id: 'see-it-fire',
    anchor: 'activations',
    title: 'What activates the feature',
    body:
      'These are passages from The Pile Uncopyrighted where the feature activates: the highlight marks where ' +
      'and how strongly. The interval view samples documents by activation-deciles, sketching the ' +
      'feature’s full range.',
    placement: 'left',
  },
  {
    id: 'steered-chat',
    anchor: 'chat',
    title: 'Steer the model',
    body:
      'Features are vectors in the original model space: amplifying one steers what the model writes. ' +
      'This conversation was generated by the engine with steering applied; live ' +
      'generation is off in the demo. Multiple features can be steered at once, compared to baseline, and the conversations are automatically saved and can be exported.',
    placement: 'left',
    prepareTimeoutMs: 15000,
    prepare: async (runtime) => {
      runtime.openChat();
      // The fixture fetch resolves fast (static file); tolerate it anyway.
      await waitFor(runtime.isChatReady, 8000);
      await runtime.loadFirstDemoSession();
      // Close the history list (with it open the conversation column is too
      // cramped to read unless the user resizes the divider) and rewind the
      // replayed thread to its opening message.
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent<SaeChatHistoryDetail>(SAE_CHAT_HISTORY_EVENT, {
            detail: { open: false, scrollToTop: true },
          }),
        );
      }
      // Let the sidebar's slide-in transition land before measuring.
      await delay(400);
    },
  },
  {
    id: 'back-to-the-map',
    anchor: 'mapLink',
    title: 'Back to the galaxy',
    body:
      'This link returns to the label galaxy we started on. Right-click on it ' +
      'to go back to the 3D exploration. Prompt highlighting and prompt search, available in the full version, are disabled in the CPU-only demo. This concludes the SAE tour; the ? button reopens the mission menu.',
    placement: 'bottom',
  },
];

/** Combined length of both segments — the step counter's denominator. */
export const SAE_TOUR_TOTAL_STEPS = SAE_MAP_TOUR_STEPS.length + SAE_TOUR_STEPS.length;
