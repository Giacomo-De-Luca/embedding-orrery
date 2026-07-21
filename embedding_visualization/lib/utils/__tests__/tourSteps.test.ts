import { describe, it, expect, vi } from 'vitest';
import { TOUR_STEPS, TOUR_ANCHORS, waitFor, type TourRuntime } from '../tourSteps';
import { DEMO_DEFAULT_COLLECTION, TOUR_PRESET_ID, TOUR_SEARCH_QUERY, getPreset } from '../tourPresets';

function makeRuntime(overrides: Partial<TourRuntime> = {}): TourRuntime {
  return {
    applyPreset: vi.fn(),
    runSearch: vi.fn().mockResolvedValue(undefined),
    setActivePanel: vi.fn(),
    setShowLabels: vi.fn(),
    getLoadedCollection: () => DEMO_DEFAULT_COLLECTION,
    getColorByField: () => 'label',
    ...overrides,
  };
}

describe('TOUR_STEPS', () => {
  it('has five steps with unique ids and known anchors', () => {
    expect(TOUR_STEPS).toHaveLength(5);
    expect(new Set(TOUR_STEPS.map((s) => s.id)).size).toBe(5);
    for (const step of TOUR_STEPS) {
      expect(TOUR_ANCHORS[step.anchor]).toMatch(/^\[data-tour=/);
    }
  });

  it('the structure step applies the tour preset (a real id, targeting the demo collection)', () => {
    expect(getPreset(TOUR_PRESET_ID)?.collection).toBe(DEMO_DEFAULT_COLLECTION);
    const structure = TOUR_STEPS.find((s) => s.id === 'structure')!;
    const runtime = makeRuntime();
    void structure.prepare!(runtime);
    expect(runtime.applyPreset).toHaveBeenCalledWith(TOUR_PRESET_ID);
  });

  it('the search step queries the demo collection with labels on', async () => {
    const search = TOUR_STEPS.find((s) => s.id === 'search')!;
    const runtime = makeRuntime();
    await search.prepare!(runtime);
    expect(runtime.setShowLabels).toHaveBeenCalledWith(true);
    expect(runtime.runSearch).toHaveBeenCalledWith(TOUR_SEARCH_QUERY);
  });

  it('the search step NEVER auto-queries a non-demo collection', async () => {
    const search = TOUR_STEPS.find((s) => s.id === 'search')!;
    const runtime = makeRuntime({ getLoadedCollection: () => 'acl_abstracts_emnlp_findings' });
    await search.prepare!(runtime);
    expect(runtime.runSearch).not.toHaveBeenCalled();
    expect(runtime.setShowLabels).not.toHaveBeenCalled();
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
