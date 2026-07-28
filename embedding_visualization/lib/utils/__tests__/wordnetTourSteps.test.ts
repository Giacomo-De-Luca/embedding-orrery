import { describe, it, expect, vi, afterEach } from 'vitest';
import { WORDNET_TOUR_STEPS } from '../wordnetTourSteps';
import { TOUR_ANCHORS, type TourRuntime } from '../tourSteps';
import { WORDNET_COLLECTION, WORDNET_TOUR_QUERY } from '../tourPresets';

function makeRuntime(overrides: Partial<TourRuntime> = {}): TourRuntime {
  return {
    applyPreset: vi.fn(),
    runSearch: vi.fn().mockResolvedValue(undefined),
    runFeatureSearch: vi.fn().mockResolvedValue(true),
    clearSearch: vi.fn(),
    resetCamera: vi.fn(),
    isolateFirstTopic: vi.fn().mockReturnValue(null),
    clearTopicSelection: vi.fn(),
    applyTemporalWindow: vi.fn().mockReturnValue(false),
    clearTemporalFilter: vi.fn(),
    setDensityView: vi.fn(),
    setColorBy: vi.fn(),
    hasColorField: vi.fn().mockReturnValue(true),
    setNebulaMode: vi.fn(),
    setShowClusterLabels: vi.fn(),
    setActivePanel: vi.fn(),
    setShowLabels: vi.fn(),
    getLoadedCollection: () => WORDNET_COLLECTION,
    getColorByField: () => 'pos',
    getSelectedFeatureIndex: () => null,
    ...overrides,
  };
}

const step = (id: string) => WORDNET_TOUR_STEPS.find((s) => s.id === id)!;

/** Run a prepare under fake timers, flushing its settle delays. */
async function runPrepare(id: string, runtime: TourRuntime, flushMs = 2000) {
  vi.useFakeTimers();
  const prep = step(id).prepare!(runtime);
  await vi.advanceTimersByTimeAsync(flushMs);
  await prep;
}

afterEach(() => {
  vi.useRealTimers();
});

describe('WORDNET_TOUR_STEPS', () => {
  it('has four steps in the shape+POS → nebula → search → finale order', () => {
    expect(WORDNET_TOUR_STEPS.map((s) => s.id)).toEqual([
      'wn-shape',
      'wn-nebula',
      'wn-search',
      'wn-finale',
    ]);
    for (const s of WORDNET_TOUR_STEPS) {
      expect(TOUR_ANCHORS[s.anchor]).toBeDefined();
    }
  });

  it('map-narrating steps use the right-edge anchor with left placement', () => {
    for (const s of WORDNET_TOUR_STEPS) {
      if (s.anchor !== 'plotSide') continue;
      expect(s.placement).toBe('left');
      expect(s.allowInteraction).toBe(true);
    }
  });

  it('wn-shape applies the preset and restores the POS view (shape + colours share one step)', async () => {
    const runtime = makeRuntime();
    await runPrepare('wn-shape', runtime);
    expect(runtime.applyPreset).toHaveBeenCalledWith('wordnet-pos');
    // NEVER setColorBy(null): the uncoloured-first reveal flashed
    // default-then-grey twice before being abandoned — POS is the saved
    // default, and the explicit call restores it on Back-nav from the nebula.
    expect(runtime.setColorBy).toHaveBeenCalledWith('pos', 'categorical');
    expect(runtime.setColorBy).not.toHaveBeenCalledWith(null);
    expect(runtime.setNebulaMode).toHaveBeenCalledWith(false);
    expect(runtime.setShowClusterLabels).toHaveBeenCalledWith(false);
  });

  it('wn-nebula recolors by topic, turns on haze + cluster labels, and steps 25% closer (Figure 1)', async () => {
    const runtime = makeRuntime();
    await runPrepare('wn-nebula', runtime);
    expect(runtime.setColorBy).toHaveBeenCalledWith('topic_label', 'categorical');
    expect(runtime.setNebulaMode).toHaveBeenCalledWith(true);
    expect(runtime.setShowClusterLabels).toHaveBeenCalledWith(true);
    expect(runtime.resetCamera).toHaveBeenCalledWith({
      zoom: 0.65,
      azimuthDeg: -20,
      elevationDeg: -10,
    });

    const elsewhere = makeRuntime({ getLoadedCollection: () => 'emotion' });
    await runPrepare('wn-nebula', elsewhere);
    expect(elsewhere.setNebulaMode).not.toHaveBeenCalled();
  });

  it('wn-search auto-searches the WordNet collection only (hard guard)', async () => {
    const runtime = makeRuntime();
    await runPrepare('wn-search', runtime);
    expect(runtime.runSearch).toHaveBeenCalledWith(WORDNET_TOUR_QUERY);

    const elsewhere = makeRuntime({ getLoadedCollection: () => 'emotion' });
    await runPrepare('wn-search', elsewhere);
    expect(elsewhere.runSearch).not.toHaveBeenCalled();
  });

  it('wn-search narrates from the plot edge, not the input spotlight (base-tour treatment)', () => {
    const search = step('wn-search');
    expect(search.anchor).toBe('plotSide');
    expect(search.placement).toBe('left');
  });

  it('wn-finale swaps the search dive for a slow move INTO the galaxy', async () => {
    const runtime = makeRuntime();
    await runPrepare('wn-finale', runtime, 4000);
    expect(runtime.clearSearch).toHaveBeenCalled();
    // MUST be relative: the search step's fly-to leaves the camera far closer
    // than any default-relative target, which read as zooming back out.
    expect(runtime.resetCamera).toHaveBeenCalledWith({
      relative: true,
      azimuthDeg: 30,
      elevationDeg: -15,
      zoom: 0.8,
      panZ: -0.05,
      durationMs: 2600,
    });
  });

  it('never runs the metered feature search', async () => {
    for (const s of WORDNET_TOUR_STEPS) {
      if (!s.prepare) continue;
      const runtime = makeRuntime();
      await runPrepare(s.id, runtime);
      expect(runtime.runFeatureSearch).not.toHaveBeenCalled();
    }
  });
});
