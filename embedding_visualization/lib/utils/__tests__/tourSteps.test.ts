import { describe, it, expect, vi } from 'vitest';
import { TOUR_STEPS, TOUR_ANCHORS, FINALE_PRESET_ID, waitFor, type TourRuntime } from '../tourSteps';
import { TOUR_COLLECTION, TOUR_PRESET_ID, TOUR_SEARCH_QUERY, TOUR_PRESETS, getPreset } from '../tourPresets';

function makeRuntime(overrides: Partial<TourRuntime> = {}): TourRuntime {
  return {
    applyPreset: vi.fn(),
    runSearch: vi.fn().mockResolvedValue(undefined),
    clearSearch: vi.fn(),
    isolateFirstTopic: vi.fn().mockReturnValue('Machine Learning NLP'),
    clearTopicSelection: vi.fn(),
    setActivePanel: vi.fn(),
    setShowLabels: vi.fn(),
    getLoadedCollection: () => TOUR_COLLECTION,
    getColorByField: () => 'topic_label',
    ...overrides,
  };
}

describe('TOUR_STEPS', () => {
  it('has six steps with unique ids and known anchors', () => {
    expect(TOUR_STEPS).toHaveLength(6);
    expect(new Set(TOUR_STEPS.map((s) => s.id)).size).toBe(6);
    for (const step of TOUR_STEPS) {
      expect(TOUR_ANCHORS[step.anchor]).toMatch(/^\[data-tour=/);
    }
  });

  it('the analytics step clears the search glow, then isolates a topic', async () => {
    const analytics = TOUR_STEPS.find((s) => s.id === 'analytics')!;
    const runtime = makeRuntime();
    await analytics.prepare!(runtime);
    expect(runtime.clearSearch).toHaveBeenCalled();
    expect(runtime.isolateFirstTopic).toHaveBeenCalled();
    expect(runtime.setActivePanel).toHaveBeenCalledWith('analytics');
  });

  it('the finale closes the panel and applies the manifold preset (a real id)', async () => {
    expect(getPreset(FINALE_PRESET_ID)).not.toBeNull();
    const finale = TOUR_STEPS.find((s) => s.id === 'finale')!;
    const runtime = makeRuntime({
      getLoadedCollection: () => TOUR_PRESETS[FINALE_PRESET_ID].collection,
      getColorByField: () => 'mapped_colour',
    });
    await finale.prepare!(runtime);
    expect(runtime.setActivePanel).toHaveBeenCalledWith(null);
    expect(runtime.applyPreset).toHaveBeenCalledWith(FINALE_PRESET_ID);
  });

  it('the structure step applies the tour preset (a real id, targeting the tour collection)', () => {
    expect(getPreset(TOUR_PRESET_ID)?.collection).toBe(TOUR_COLLECTION);
    const structure = TOUR_STEPS.find((s) => s.id === 'structure')!;
    const runtime = makeRuntime();
    void structure.prepare!(runtime);
    expect(runtime.applyPreset).toHaveBeenCalledWith(TOUR_PRESET_ID);
  });

  it('the search step queries the tour collection with labels on', async () => {
    const search = TOUR_STEPS.find((s) => s.id === 'search')!;
    const runtime = makeRuntime();
    await search.prepare!(runtime);
    expect(runtime.setShowLabels).toHaveBeenCalledWith(true);
    expect(runtime.runSearch).toHaveBeenCalledWith(TOUR_SEARCH_QUERY);
  });

  it('the search step NEVER auto-queries any collection but the tour one', async () => {
    const search = TOUR_STEPS.find((s) => s.id === 'search')!;
    for (const other of ['emotion', 'xkcd_hilbert_gemini', 'wordnet_senses_full']) {
      const runtime = makeRuntime({ getLoadedCollection: () => other });
      await search.prepare!(runtime);
      expect(runtime.runSearch).not.toHaveBeenCalled();
      expect(runtime.setShowLabels).not.toHaveBeenCalled();
    }
  });

  it('plot steps allow interaction; steps with slow prepares set a timeout ceiling', () => {
    for (const step of TOUR_STEPS) {
      if (step.anchor === 'plot') expect(step.allowInteraction).toBe(true);
    }
    const structure = TOUR_STEPS.find((s) => s.id === 'structure')!;
    expect(structure.prepareTimeoutMs).toBeGreaterThan(20000);
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
