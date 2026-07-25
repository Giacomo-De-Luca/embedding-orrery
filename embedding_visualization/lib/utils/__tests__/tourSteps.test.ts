import { describe, it, expect, vi } from 'vitest';
import { TOUR_STEPS, TOUR_ANCHORS, FINALE_PRESET_ID, waitFor, type TourRuntime } from '../tourSteps';
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
    }
  });

  it('the search step (second) owns the collection wait, then queries with labels on', async () => {
    const search = step('search');
    const runtime = makeRuntime();
    await search.prepare!(runtime);
    // First prepared step: the tour-preset safety net moved here from structure.
    expect(runtime.applyPreset).toHaveBeenCalledWith(TOUR_PRESET_ID);
    expect(runtime.setShowLabels).toHaveBeenCalledWith(true);
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

  it('the focus step isolates one topic programmatically with panels closed', async () => {
    const runtime = makeRuntime();
    await step('focus-topic').prepare!(runtime);
    expect(runtime.isolateFirstTopic).toHaveBeenCalled();
    expect(runtime.clearSearch).toHaveBeenCalled();
    expect(runtime.setActivePanel).toHaveBeenCalledWith(null);
  });

  it('the temporal step swaps isolation for an early-years window in the analytics panel', async () => {
    const runtime = makeRuntime();
    await step('temporal').prepare!(runtime);
    // One idea per step: the previous step's topic isolation is dropped first.
    expect(runtime.clearTopicSelection).toHaveBeenCalled();
    expect(runtime.setActivePanel).toHaveBeenCalledWith('analytics');
    expect(runtime.applyTemporalWindow).toHaveBeenCalledWith(0, 1 / 3);
    // Back-navigation from the 2D density step must land back in 3D.
    expect(runtime.setDensityView).toHaveBeenCalledWith(false);
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

  it('the finale closes the panel and lands on the emotion collection (a real preset)', async () => {
    expect(FINALE_PRESET_ID).toBe('emotion');
    expect(getPreset(FINALE_PRESET_ID)).not.toBeNull();
    const runtime = makeRuntime({
      getLoadedCollection: () => TOUR_PRESETS[FINALE_PRESET_ID].collection,
    });
    await step('finale').prepare!(runtime);
    expect(runtime.setActivePanel).toHaveBeenCalledWith(null);
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
