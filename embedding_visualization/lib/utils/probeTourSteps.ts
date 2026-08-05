import type { TourAnchor, TourRuntime, TourStepDefinitionBase } from './tourSteps';
import { TOUR_ANCHORS, delay, ensurePresetCollection, waitFor, waitForAnchor } from './tourSteps';
import { PROBE_COLLECTION, PROBE_PRESET_ID } from './tourPresets';

/**
 * "Decode human ratings" tour (`?tour=probe`): embedding-space probing on the
 * Glasgow Norms — the paper's probing experiment as a guided walk.
 *
 * Narrative: humans rated 4,682 words on nine psychological dimensions →
 * a linear probe reads concreteness back out of the raw vectors (R² = 0.80
 * held-out) → residuals show where model and humans disagree → valence shows
 * the affective dimensions decode too → the Analytics panel holds the rest.
 *
 * Every step is FREE and read-only: probes and per-word scores are trained
 * offline and shipped in the demo seed; the tour only recolors by fields the
 * `useProbes` layer merges client-side. There is deliberately NO search step
 * — the collection is Gemini-embedded, and auto-searching it would spend
 * metered quota (the standing rule: only `TOUR_COLLECTION` and locally
 * embedded collections are ever auto-queried).
 *
 * Probe fields land asynchronously (their own queries after the collection
 * loads), so recoloring steps gate on `hasColorField` first.
 */

export type ProbeTourStepDefinition = TourStepDefinitionBase<TourAnchor, TourRuntime>;

/**
 * Window event the probing tour fires to showcase the fit form: ProbeSection
 * listens and responds by selecting `detail.kind` and opening its settings
 * popover. An event (not a prop) because the section sits four components
 * deep in the Analytics sidebar and the showcase is a demo-tour-only concern.
 */
export const PROBE_SHOWCASE_EVENT = 'orrery:probe-showcase';
export interface ProbeShowcaseDetail {
  kind: string;
}

/** The tour's headline probe (best held-out R² of the ridge family). */
export const PROBE_SCORE_FIELD = 'probe_concreteness_ridge_score';
export const PROBE_RESIDUAL_FIELD = 'probe_concreteness_ridge_residual';
/** The second-act probe: an affective norm decoding almost as well. */
export const PROBE_VALENCE_FIELD = 'probe_valence_ridge_score';

/** Ceiling for the probe-scores fetch after the collection lands. */
const PROBE_FIELDS_TIMEOUT_MS = 15_000;

export const PROBE_TOUR_STEPS: ProbeTourStepDefinition[] = [
  {
    id: 'probe-ratings',
    anchor: 'plotSide',
    title: 'How we feel about words',
    body:
      'Displayed the Glasgow Norms dataset embedded with Gemini-Embeddings: 4,682 English words, each rated by 33 partecipants on average ' +
      'on nine psychological dimensions. The dataset is coloured on concreteness, from ' +
      'abstract ideas to things you can hold. Notice the embeddings self-organize ' +
      'across dimension: visually it can be seen with the smoth gradient. How can we quantify how much the embeddings encode the information?',
    allowInteraction: true,
    placement: 'left',
    // Ceiling covers the worst case: 25 s collection wait + 8 s colour wait
    // + settle (prepares aren't cancelled on beforeTimeout).
    prepareTimeoutMs: 40_000,
    // The collection switch sits behind the page's own full-screen loader.
    suppressWaitLoader: true,
    prepare: async (runtime) => {
      const loaded = await ensurePresetCollection(
        runtime,
        PROBE_PRESET_ID,
        PROBE_COLLECTION,
        25_000,
      );
      if (!loaded) return;
      // The preset's concreteness colouring lands via its post-load effect.
      await waitFor(() => runtime.getColorByField() !== null, 8000);
      // Prepared steps opt out of joyride's own target polling (see
      // waitForAnchor): the plot anchor mounts after the collection load.
      await waitForAnchor(TOUR_ANCHORS.plotSide, 20_000);
      // The concreteness gradient only reads as linear from the right angle —
      // at the default framing the axis points at the camera. Orbit ~59° left
      // (45 + 14 after live-tuning; adjust here if a UMAP re-run moves it).
      runtime.resetCamera({ azimuthDeg: 59 });
      await delay(300);
    },
  },
  {
    id: 'probe-score',
    anchor: 'plotSide',
    title: 'Quantifying self-organization',
    body:
      'Same palette, different source: the color is now a linear probe’s (ridge regression) prediction of ' +
      'each word’s concreteness, computed from the raw embedding vectors. ' +
      'The picture barely changes: R² = 0.80 on words the probe’s ' +
      'held out test set. The embedding model was never trained to encode concreteness.',
    allowInteraction: true,
    placement: 'left',
    prepareTimeoutMs: PROBE_FIELDS_TIMEOUT_MS + 5_000,
    prepare: async (runtime) => {
      if (runtime.getLoadedCollection() !== PROBE_COLLECTION) return;
      // Probe scores arrive via their own queries — wait for the field. If
      // they never land (seed without probes), keep the current colouring
      // rather than painting a field no point carries.
      const ready = await waitFor(
        () => runtime.hasColorField(PROBE_SCORE_FIELD),
        PROBE_FIELDS_TIMEOUT_MS,
      );
      if (!ready) return;
      // Open the Analytics panel from here on: it hosts the probing lab the
      // tour is walking toward, and its distribution charts track each
      // recolor along the way.
      runtime.setActivePanel('analytics');
      // Same managua palette as the actual ratings, so the step reads as
      // "the picture barely changed" rather than a jarring recolor.
      runtime.setColorBy(PROBE_SCORE_FIELD, 'diverging', { scaleName: 'managua' });
      await delay(300);
    },
  },
  {
    id: 'probe-residual',
    anchor: 'plotSide',
    title: 'Where model and humans disagree',
    body:
      'The dataset is now colored by the probe’s error: blue means the embedding and the human ' +
      'raters agree, the tints mark words the geometry rates as more — or less — ' +
      'concrete than people do. Disagreement clusters too: metaphors, senses shifted by ' +
      'context, words whose meaning moved since the ratings were collected.',
    allowInteraction: true,
    placement: 'left',
    prepare: async (runtime) => {
      if (runtime.getLoadedCollection() !== PROBE_COLLECTION) return;
      if (!runtime.hasColorField(PROBE_RESIDUAL_FIELD)) return;
      // Diverging scale centered at zero: the midpoint must mean "no error"
      // (same convention as ProbeSection's Residual button).
      runtime.setColorBy(PROBE_RESIDUAL_FIELD, 'diverging', { centerZero: true });
      await delay(300);
    },
  },
  {
    id: 'probe-valence',
    anchor: 'plotSide',
    title: 'Self-organization of affective judgments',
    body:
      'Concreteness could be dismissed as topical. Represented here is the valence dimension: how positive a ' +
      'word feels, decoded from the same vectors at R² = 0.76. Affective judgments, ' +
      'are laid out along linear directions in the space. ',
    allowInteraction: true,
    placement: 'left',
    prepare: async (runtime) => {
      if (runtime.getLoadedCollection() !== PROBE_COLLECTION) return;
      const ready = await waitFor(
        () => runtime.hasColorField(PROBE_VALENCE_FIELD),
        PROBE_FIELDS_TIMEOUT_MS,
      );
      if (!ready) return;
      runtime.setColorBy(PROBE_VALENCE_FIELD, 'diverging', { scaleName: 'managua' });
      // Valence lies along a different direction than concreteness — turn a
      // similar amount again (absolute: 59° + another 59° from where the
      // opening step left the camera).
      runtime.resetCamera({ azimuthDeg: 118 });
      await delay(300);
    },
  },
  {
    id: 'probe-panel',
    anchor: 'probeSection',
    title: 'The probing Lab',
    body:
      'The bottom of the Analytics panel is where probing lives. The probes you just ' +
      'saw were pre-fitted and shipped with the demo; training is switched ' +
      'off here because the public demo is read-only. Open are the settings of the MLP probe, one of seven ' +
      'configurable probes (linear and nonlinear). This concludes the probing tour; the ? ' +
      'button up top has the other missions.',
    // No overlay: the settings popover the showcase opens renders below
    // joyride's overlay z-index and would be dimmed under a spotlight.
    // Rings mark the showcased settings (mounts ~600 ms in — the rings poll,
    // so it lights up when the popover opens) and the probe row whose field
    // is currently colouring the map, not the whole section.
    allowInteraction: true,
    highlightAnchors: ['probeSettings', 'probeActive'],
    placement: 'right',
    prepare: async (runtime) => {
      if (runtime.getLoadedCollection() !== PROBE_COLLECTION) return;
      runtime.setActivePanel('analytics');
      // The panel is always mounted, slid offscreen; wait out the transition
      // so the spotlight measures the on-screen position.
      await waitForAnchor(TOUR_ANCHORS.probeSection, 4000);
      // Showcase the fit form: select MLP and open its settings popover.
      // Deferred past the tooltip mount — Radix dismisses the popover when
      // focus lands outside it, and joyride focuses the tooltip as it opens.
      if (typeof window !== 'undefined') {
        window.setTimeout(() => {
          window.dispatchEvent(
            new CustomEvent<ProbeShowcaseDetail>(PROBE_SHOWCASE_EVENT, {
              detail: { kind: 'mlp' },
            }),
          );
        }, 600);
      }
      await delay(150);
    },
  },
];
