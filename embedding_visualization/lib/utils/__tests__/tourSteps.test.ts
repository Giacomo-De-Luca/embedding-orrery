import { describe, it, expect, vi } from 'vitest';
import {
  TOUR_STEPS,
  TOUR_ANCHORS,
  TOUR_FOCUS_TOPIC,
  FINALE_PRESET_ID,
  waitFor,
  type TourRuntime,
} from '../tourSteps';
import {
  TOUR_COLLECTION,
  TOUR_PRESET_ID,
  TOUR_SEARCH_QUERY,
  TOUR_FEATURE_QUERY,
  TOUR_PRESETS,
  getPreset,
} from '../tourPresets';

function makeRuntime(overrides: Partial<TourRuntime> = {}): TourRuntime {
  return {
    applyPreset: vi.fn(),
    runSearch: vi.fn().mockResolvedValue(undefined),
    runFeatureSearch: vi.fn().mockResolvedValue(true),
    clearSearch: vi.fn(),
    resetCamera: vi.fn(),
    isolateFirstTopic: vi.fn().mockReturnValue('Machine Learning NLP'),
    clearTopicSelection: vi.fn(),
    applyTemporalWindow: vi.fn().mockReturnValue(true),
    clearTemporalFilter: vi.fn(),
    setDensityView: vi.fn(),
    setColorBy: vi.fn(),
    hasColorField: vi.fn().mockReturnValue(true),
    setNebulaMode: vi.fn(),
    setShowClusterLabels: vi.fn(),
    setActivePanel: vi.fn(),
    setShowLabels: vi.fn(),
    getLoadedCollection: () => TOUR_COLLECTION,
    getColorByField: () => 'topic_label',
    getSelectedFeatureIndex: () => null,
    ...overrides,
  };
}

const step = (id: string) => TOUR_STEPS.find((s) => s.id === id)!;

describe('TOUR_STEPS', () => {
  it('has seven steps with unique ids, known anchors, and the requested order', () => {
    expect(TOUR_STEPS.map((s) => s.id)).toEqual([
      'map', 'search', 'feature-search', 'focus-topic', 'temporal', 'density', 'finale',
    ]);
    for (const s of TOUR_STEPS) {
      expect(TOUR_ANCHORS[s.anchor]).toMatch(/^\[data-tour=/);
      for (const extra of s.highlightAnchors ?? []) {
        expect(TOUR_ANCHORS[extra]).toMatch(/^\[data-tour=/);
      }
    }
  });

  it('the map step opens the Controls panel (how-to-draw visible from beat one)', async () => {
    const runtime = makeRuntime();
    await step('map').prepare!(runtime);
    expect(runtime.setActivePanel).toHaveBeenCalledWith('controls');
  });

  it('the map step blocks on its plot anchor mounting', async () => {
    // Regression: react-joyride polls for a missing target ONLY on steps with
    // no `before` hook, so a step that has a prepare must wait for its own
    // anchor. Without this the tour's first step was dropped as
    // TARGET_NOT_FOUND while the collection was still loading.
    let mounted = false;
    vi.stubGlobal('document', {
      querySelector: (selector: string) =>
        mounted && selector === TOUR_ANCHORS.plotSide
          ? { getBoundingClientRect: () => ({ left: 0 }) }
          : null,
    });
    try {
      let done = false;
      const prep = step('map')
        .prepare!(makeRuntime())
        .then(() => {
          done = true;
        });
      await new Promise((resolve) => setTimeout(resolve, 150));
      expect(done).toBe(false);
      mounted = true;
      await prep;
      expect(done).toBe(true);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('the search step (second) owns the collection wait, then queries with labels on', async () => {
    const search = step('search');
    const runtime = makeRuntime();
    await search.prepare!(runtime);
    // First prepared step: the tour-preset safety net moved here from structure.
    expect(runtime.applyPreset).toHaveBeenCalledWith(TOUR_PRESET_ID);
    expect(runtime.setShowLabels).toHaveBeenCalledWith(true);
    // The Search panel is open beside the spotlit header input (the body says so).
    expect(runtime.setActivePanel).toHaveBeenCalledWith('search');
    expect(runtime.runSearch).toHaveBeenCalledWith(TOUR_SEARCH_QUERY);
  });

  it('the search step NEVER auto-queries any collection but the tour one', async () => {
    vi.useFakeTimers();
    try {
      for (const other of ['emotion', 'xkcd_hilbert_gemini', 'wordnet_senses_full']) {
        const runtime = makeRuntime({ getLoadedCollection: () => other });
        const prep = step('search').prepare!(runtime);
        // Let the collection-wait safety net time out (25 s) without searching.
        await vi.advanceTimersByTimeAsync(26000);
        await prep;
        expect(runtime.runSearch).not.toHaveBeenCalled();
        expect(runtime.setShowLabels).not.toHaveBeenCalled();
        expect(runtime.setActivePanel).not.toHaveBeenCalled();
      }
    } finally {
      vi.useRealTimers();
    }
  });

  it('the feature-search step swaps the semantic glow for a free SAE ranking in the search panel', async () => {
    const runtime = makeRuntime();
    await step('feature-search').prepare!(runtime);
    // Fresh canvas: drop the semantic constellation before the feature glow.
    expect(runtime.clearSearch).toHaveBeenCalled();
    expect(runtime.setActivePanel).toHaveBeenCalledWith('search');
    expect(runtime.runFeatureSearch).toHaveBeenCalledWith(TOUR_FEATURE_QUERY);
  });

  it('the feature-search step never fires against a non-tour collection', async () => {
    const runtime = makeRuntime({ getLoadedCollection: () => 'emotion' });
    await step('feature-search').prepare!(runtime);
    expect(runtime.runFeatureSearch).not.toHaveBeenCalled();
    expect(runtime.setActivePanel).not.toHaveBeenCalled();
  });

  it('the focus step isolates the NMT topic in the analytics panel', async () => {
    const runtime = makeRuntime();
    await step('focus-topic').prepare!(runtime);
    // Preferred topic: near the top of the count-sorted category list, so the
    // isolation is visible without scrolling (falls back to the first topic).
    expect(runtime.isolateFirstTopic).toHaveBeenCalledWith(TOUR_FOCUS_TOPIC);
    expect(runtime.clearSearch).toHaveBeenCalled();
    // The category list makes the isolation legible as data.
    expect(runtime.setActivePanel).toHaveBeenCalledWith('analytics');
  });

  it('the temporal step swaps isolation for an early-years window in the analytics panel', async () => {
    const runtime = makeRuntime();
    await step('temporal').prepare!(runtime);
    // One idea per step: the previous step's topic isolation is dropped first.
    expect(runtime.clearTopicSelection).toHaveBeenCalled();
    expect(runtime.setActivePanel).toHaveBeenCalledWith('analytics');
    expect(runtime.applyTemporalWindow).toHaveBeenCalledWith(0, 1 / 3);
    expect(runtime.setDensityView).toHaveBeenCalledWith(false);
  });

  it('the temporal step scrolls the timeline into view inside the panel', async () => {
    // The category list above it can push the chart below the panel's fold,
    // and joyride only auto-scrolls to a step's own target (here: the plot).
    const scrollIntoView = vi.fn();
    vi.stubGlobal('document', {
      querySelector: (selector: string) =>
        selector === TOUR_ANCHORS.temporalChart
          ? { getBoundingClientRect: () => ({ left: 0 }), scrollIntoView }
          : null,
    });
    try {
      await step('temporal').prepare!(makeRuntime());
      // The stub has no ScrollArea ancestor, so the helper's scrollIntoView
      // fallback runs (deterministic 'auto' — smooth proved cancellable).
      expect(scrollIntoView).toHaveBeenCalledWith({ behavior: 'auto', block: 'center' });
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('the finale rings the collection selector and the ? mission button', () => {
    const finale = step('finale');
    // Card off to the right edge, map fully interactive (no overlay) — the
    // highlights come from the controller's rings, one per anchor, because
    // joyride can only cut a single spotlight.
    expect(finale.anchor).toBe('plotSide');
    expect(finale.allowInteraction).toBe(true);
    expect(finale.highlightAnchors).toEqual(['collectionSelector', 'introButton']);
  });

  it('the density step clears the filters and flips to the 2D density view', async () => {
    const runtime = makeRuntime();
    await step('density').prepare!(runtime);
    // Back-navigation from the finale re-enters on the emotion collection.
    expect(runtime.applyPreset).toHaveBeenCalledWith(TOUR_PRESET_ID);
    expect(runtime.clearTemporalFilter).toHaveBeenCalled();
    expect(runtime.clearTopicSelection).toHaveBeenCalled();
    // The Analytics panel deliberately stays open through the density flip.
    expect(runtime.setActivePanel).not.toHaveBeenCalled();
    expect(runtime.setDensityView).toHaveBeenCalledWith(true);
  });

  it('the focus step polls until late-arriving topics land', async () => {
    // Topics come from their own query — simulate them resolving after two
    // empty attempts; the step should keep trying instead of giving up.
    const isolate = vi
      .fn()
      .mockReturnValueOnce(null)
      .mockReturnValueOnce(null)
      .mockReturnValue('Machine Learning NLP');
    const runtime = makeRuntime({ isolateFirstTopic: isolate });
    await step('focus-topic').prepare!(runtime);
    expect(isolate.mock.results.at(-1)?.value).toBe('Machine Learning NLP');
  });

  it('the finale opens the Controls panel and lands on the emotion collection (a real preset)', async () => {
    expect(FINALE_PRESET_ID).toBe('emotion');
    expect(getPreset(FINALE_PRESET_ID)).not.toBeNull();
    const runtime = makeRuntime({
      getLoadedCollection: () => TOUR_PRESETS[FINALE_PRESET_ID].collection,
    });
    await step('finale').prepare!(runtime);
    // The parting view invites exploration: "how to draw" is left discoverable.
    expect(runtime.setActivePanel).toHaveBeenCalledWith('controls');
    expect(runtime.applyPreset).toHaveBeenCalledWith(FINALE_PRESET_ID);
  });

  it('suppresses the joyride wait-loader exactly on the steps where the page shows its own', () => {
    // Collection-switching steps unmount the plot behind a full-page loader —
    // joyride's built-in waiting spinner on top of it reads as a double loader.
    const suppressed = TOUR_STEPS.filter((s) => s.suppressWaitLoader).map((s) => s.id);
    expect(suppressed.sort()).toEqual(['density', 'finale', 'map', 'search']);
  });

  it('plot steps allow interaction and hang their card off the right edge', () => {
    for (const s of TOUR_STEPS) {
      if (s.anchor === 'plot' || s.anchor === 'plotSide') {
        expect(s.allowInteraction).toBe(true);
        // Whole-plot steps must not park the card mid-screen over the view
        // they narrate: they target the right-edge anchor with placement left.
        expect(s.anchor).toBe('plotSide');
        expect(s.placement).toBe('left');
      }
    }
    for (const id of ['search', 'density', 'finale']) {
      expect(step(id).prepareTimeoutMs).toBeGreaterThan(20000);
    }
  });
});

describe('waitFor', () => {
  it('resolves true once the predicate holds', async () => {
    let ready = false;
    setTimeout(() => {
      ready = true;
    }, 30);
    await expect(waitFor(() => ready, 1000, 5)).resolves.toBe(true);
  });

  it('resolves false on timeout without throwing', async () => {
    await expect(waitFor(() => false, 40, 5)).resolves.toBe(false);
  });
});
