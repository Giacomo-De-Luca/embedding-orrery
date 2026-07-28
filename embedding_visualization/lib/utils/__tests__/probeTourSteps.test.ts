import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  PROBE_TOUR_STEPS,
  PROBE_SCORE_FIELD,
  PROBE_RESIDUAL_FIELD,
  PROBE_VALENCE_FIELD,
  PROBE_SHOWCASE_EVENT,
} from '../probeTourSteps';
import { TOUR_ANCHORS, type TourRuntime } from '../tourSteps';
import { PROBE_COLLECTION } from '../tourPresets';

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
    getLoadedCollection: () => PROBE_COLLECTION,
    getColorByField: () => 'concreteness',
    getSelectedFeatureIndex: () => null,
    ...overrides,
  };
}

const step = (id: string) => PROBE_TOUR_STEPS.find((s) => s.id === id)!;

/** Run a prepare under fake timers, flushing waits and settle delays. */
async function runPrepare(id: string, runtime: TourRuntime, flushMs = 20_000) {
  vi.useFakeTimers();
  const prep = step(id).prepare!(runtime);
  await vi.advanceTimersByTimeAsync(flushMs);
  await prep;
}

afterEach(() => {
  vi.useRealTimers();
});

describe('PROBE_TOUR_STEPS', () => {
  it('has five steps: ratings → score → residual → valence → panel', () => {
    expect(PROBE_TOUR_STEPS.map((s) => s.id)).toEqual([
      'probe-ratings',
      'probe-score',
      'probe-residual',
      'probe-valence',
      'probe-panel',
    ]);
    for (const s of PROBE_TOUR_STEPS) {
      expect(TOUR_ANCHORS[s.anchor]).toBeDefined();
    }
  });

  it('map-narrating steps use the right-edge anchor with left placement', () => {
    for (const s of PROBE_TOUR_STEPS) {
      if (s.anchor !== 'plotSide') continue;
      expect(s.placement).toBe('left');
      expect(s.allowInteraction).toBe(true);
    }
  });

  it('NEVER auto-searches — the collection is Gemini-embedded (metered)', async () => {
    for (const s of PROBE_TOUR_STEPS) {
      if (!s.prepare) continue;
      const runtime = makeRuntime();
      await runPrepare(s.id, runtime);
      expect(runtime.runSearch).not.toHaveBeenCalled();
      expect(runtime.runFeatureSearch).not.toHaveBeenCalled();
    }
  });

  it('probe-ratings applies the glasgow-norms preset and orbits 45° left', async () => {
    const runtime = makeRuntime();
    await runPrepare('probe-ratings', runtime);
    expect(runtime.applyPreset).toHaveBeenCalledWith('glasgow-norms');
    // At the default framing the concreteness axis points at the camera —
    // the gradient only reads as linear from the side (59° after live-tuning).
    expect(runtime.resetCamera).toHaveBeenCalledWith({ azimuthDeg: 59 });
  });

  it('probe-score waits for the async probe field, then recolors in the ratings palette', async () => {
    const runtime = makeRuntime();
    await runPrepare('probe-score', runtime);
    // The Analytics panel opens here and stays open toward the probing lab.
    expect(runtime.setActivePanel).toHaveBeenCalledWith('analytics');
    expect(runtime.setColorBy).toHaveBeenCalledWith(PROBE_SCORE_FIELD, 'diverging', {
      scaleName: 'managua',
    });
  });

  it('probe-panel drops the overlay and rings the settings + the active probe row', () => {
    const panel = step('probe-panel');
    expect(panel.allowInteraction).toBe(true);
    // The showcased settings popover and the circled active probe — NOT the
    // whole section.
    expect(panel.highlightAnchors).toEqual(['probeSettings', 'probeActive']);
  });

  it('probe-panel showcases the MLP settings via the window event', async () => {
    const dispatched: Event[] = [];
    vi.stubGlobal('window', {
      // Lazy delegation: runPrepare installs fake timers AFTER this stub, so
      // an eagerly-bound real setTimeout would never fire under the fake clock.
      setTimeout: (fn: () => void, ms?: number) => globalThis.setTimeout(fn, ms),
      dispatchEvent: (e: Event) => (dispatched.push(e), true),
    });
    try {
      await runPrepare('probe-panel', makeRuntime());
      expect(dispatched).toHaveLength(1);
      expect(dispatched[0].type).toBe(PROBE_SHOWCASE_EVENT);
      expect((dispatched[0] as CustomEvent).detail).toEqual({ kind: 'mlp' });
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('probe fields that never arrive leave the colouring untouched', async () => {
    const runtime = makeRuntime({ hasColorField: vi.fn().mockReturnValue(false) });
    await runPrepare('probe-score', runtime);
    await runPrepare('probe-residual', runtime);
    await runPrepare('probe-valence', runtime);
    expect(runtime.setColorBy).not.toHaveBeenCalled();
  });

  it('probe-residual applies the zero-centered diverging scale', async () => {
    const runtime = makeRuntime();
    await runPrepare('probe-residual', runtime);
    expect(runtime.setColorBy).toHaveBeenCalledWith(PROBE_RESIDUAL_FIELD, 'diverging', {
      centerZero: true,
    });
  });

  it('probe-valence recolors by the second norm and turns a similar amount again', async () => {
    const runtime = makeRuntime();
    await runPrepare('probe-valence', runtime);
    expect(runtime.setColorBy).toHaveBeenCalledWith(PROBE_VALENCE_FIELD, 'diverging', {
      scaleName: 'managua',
    });
    // Absolute angles: 118 = the opening 59 plus another 59.
    expect(runtime.resetCamera).toHaveBeenCalledWith({ azimuthDeg: 118 });
  });

  it('probe-panel opens the Analytics panel; recoloring steps guard on the collection', async () => {
    const runtime = makeRuntime();
    await runPrepare('probe-panel', runtime);
    expect(runtime.setActivePanel).toHaveBeenCalledWith('analytics');

    for (const id of ['probe-score', 'probe-residual', 'probe-valence', 'probe-panel']) {
      const elsewhere = makeRuntime({ getLoadedCollection: () => 'emotion' });
      await runPrepare(id, elsewhere);
      expect(elsewhere.setColorBy).not.toHaveBeenCalled();
      expect(elsewhere.setActivePanel).not.toHaveBeenCalled();
    }
  });
});
