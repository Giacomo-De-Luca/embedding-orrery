import type { TourAnchor, TourRuntime, TourStepDefinitionBase } from './tourSteps';
import { TOUR_ANCHORS, delay, ensurePresetCollection, waitFor, waitForAnchor } from './tourSteps';
import { WORDNET_COLLECTION, WORDNET_PRESET_ID, WORDNET_TOUR_QUERY } from './tourPresets';

/**
 * "WordNet galaxy" tour (`?tour=wordnet`): the scale-and-nebula showcase on
 * the demo's largest collection — 212,478 dictionary senses, the dataset of
 * the paper's opening figure.
 *
 * Narrative: ONE opening step for shape + POS colours (they are the
 * collection's saved default, so the map loads already wearing them — an
 * uncoloured-first reveal was tried twice and always flashed default-then-
 * grey, so the two beats are merged instead) → the nebula view (recolor by
 * topic + haze + cluster labels + a camera step-in: the paper's Figure 1) →
 * semantic search at scale, narrated from the plot edge (the input-spotlight
 * treatment belongs to the base tour) → a parting camera move into the
 * galaxy instead of a flat reset. Every step runs on the Explore page
 * against `WORDNET_COLLECTION`; the search step is FREE (the collection is
 * MiniLM-embedded — the model the Space bakes into its image), and the same
 * hard collection guard as the other tours keeps it from ever querying a
 * metered collection.
 *
 * Weight: this collection is ~15× the EMNLP corpus. The welcome-dialog entry
 * carries the "heaviest view" warning, step 1's copy repeats it, and the
 * load waits get ceilings sized for a cold Space (90 s vs the usual 25 s).
 * The topic recolor builds one trace per topic label (thousands) — heavy,
 * but the exact view the paper's opening figure was captured from.
 */

export type WordnetTourStepDefinition = TourStepDefinitionBase<TourAnchor, TourRuntime>;

/** Load ceiling for the 212k-point collection (payload + plot build). */
const WORDNET_LOAD_TIMEOUT_MS = 90_000;

export const WORDNET_TOUR_STEPS: WordnetTourStepDefinition[] = [
  {
    id: 'wn-shape',
    anchor: 'plotSide',
    title: 'Every English Gloss',
    body:
      'This constellation represents WordNet: 212,478 {word}:{definition} glosses, the demo’s heaviest galaxy, so thanks for ' +
      'waiting. Every ridge is a neighborhood of similar words found by the MiniLM embeddings, ' +
      'and each sense is colored by its Part of Speech: the model only saw definitions, yet ' +
      'nouns, verbs, adjectives, and adverbs sort themselves into continents. ' +
      'Drag to rotate, scroll to zoom, click to semantic search and inspect neighbours.',
    allowInteraction: true,
    placement: 'left',
    prepareTimeoutMs: WORDNET_LOAD_TIMEOUT_MS,
    // The collection switch sits behind the page's own full-screen loader.
    suppressWaitLoader: true,
    prepare: async (runtime) => {
      const loaded = await ensurePresetCollection(
        runtime,
        WORDNET_PRESET_ID,
        WORDNET_COLLECTION,
        WORDNET_LOAD_TIMEOUT_MS - 12_000,
      );
      if (!loaded) return;
      // POS is the saved collection default — the map opens already wearing
      // it (an uncoloured-first reveal always flashed default-then-grey, so
      // shape and colours share this one step). The explicit calls below are
      // no-ops on first entry; they restore the view on Back-nav from the
      // nebula step's topic colouring.
      await waitFor(() => runtime.getColorByField() !== null, 8000);
      runtime.setColorBy('pos', 'categorical');
      runtime.setNebulaMode(false);
      runtime.setShowClusterLabels(false);
      // Prepared steps opt out of joyride's own target polling (see
      // waitForAnchor) — and this 212k-point plot mounts well after the
      // collection flag flips, so wait for the anchor or lose the step.
      await waitForAnchor(TOUR_ANCHORS.plotSide, 30_000);
      await delay(600);
    },
  },
  {
    id: 'wn-nebula',
    anchor: 'plotSide',
    title: 'The Nebula view',
    body:
      'Every sense is tinted by its automatically extracted topic, labeled by Gemini-Flash. ' +
      'Nebula mode wraps each topic in a density haze. ' +
      'This is the view on the paper’s opening figure. It is ' +
      'also the heaviest loading for the space, so give it a moment to bloom (it loads and renders more smoothly locally).',
    allowInteraction: true,
    placement: 'left',
    // Rebuilding 212k points into per-topic traces takes a while.
    prepareTimeoutMs: 45_000,
    prepare: async (runtime) => {
      if (runtime.getLoadedCollection() !== WORDNET_COLLECTION) return;
      runtime.setColorBy('topic_label', 'categorical');
      runtime.setNebulaMode(true);
      runtime.setShowClusterLabels(true);
      // Step into the nebula: ~35% closer, orbited the other way and tilted
      // slightly down (live-tuned constants — adjust here if a re-projection
      // moves the dense region).
      runtime.resetCamera({ zoom: 0.65, azimuthDeg: -20, elevationDeg: -10 });
      await delay(1200);
    },
  },
  {
    id: 'wn-search',
    // Narrated from the plot edge, NOT the search-input spotlight — that
    // treatment belongs to the base tour; here the results themselves are
    // the show, and the card must not cover the glow + dive.
    anchor: 'plotSide',
    title: 'Explore the galaxy',
    body:
      `We're semantically searching "${WORDNET_TOUR_QUERY}", similarly to the paper first figure. ` +
      'The matches ' +
      'glow by similarity and the camera dives to the best one. Try your own words after ' +
      'the tour. MiniLM runs inside the space. ',
    allowInteraction: true,
    placement: 'left',
    // First search may cold-start the Space's MiniLM slot.
    prepareTimeoutMs: 30_000,
    suppressWaitLoader: true,
    prepare: async (runtime) => {
      // Hard guard: auto-search runs against the WordNet collection only.
      if (runtime.getLoadedCollection() !== WORDNET_COLLECTION) return;
      // Fire-and-forget (the shared reveal pattern): the search input's own
      // spinner shows progress while the user reads; results land as a reveal.
      void runtime.runSearch(WORDNET_TOUR_QUERY).catch(() => {});
    },
  },
  {
    id: 'wn-finale',
    anchor: 'plotSide',
    title: 'Encode your own galaxies',
    body:
      'That was the WordNet galaxy. The full-version allows embedding, ' +
      'projecting, extracting topics and computing SAE activations on ' +
      'any local or HF dataset to explore as their own galaxies. This concludes the Nebula tour: the ? button up top has the other missions.',
    allowInteraction: true,
    placement: 'left',
    prepare: async (runtime) => {
      if (runtime.getLoadedCollection() !== WORDNET_COLLECTION) return;
      runtime.clearSearch();
      // Parting shot: a slow move deeper in from WHEREVER the camera is —
      // `relative` composes with the search step's fly-to dive, whose close-in
      // position made every default-relative target read as zooming back out.
      runtime.resetCamera({
        relative: true,
        azimuthDeg: 30,
        elevationDeg: -15,
        zoom: 0.8,
        panZ: -0.05,
        durationMs: 2600,
      });
      await delay(400);
    },
  },
];
