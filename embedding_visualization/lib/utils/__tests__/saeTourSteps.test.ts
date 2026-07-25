import { describe, it, expect, vi } from 'vitest';
import {
  SAE_MAP_TOUR_STEPS,
  SAE_TOUR_STEPS,
  SAE_TOUR_ANCHORS,
  SAE_TOUR_MODEL_ID,
  SAE_TOUR_SAE_ID,
  SAE_TOUR_QUERY,
  saeTourPath,
  saeInspectPath,
  type SaeTourRuntime,
} from '../saeTourSteps';
import { TOUR_ANCHORS, type TourRuntime } from '../tourSteps';
import { SAE_MAP_COLLECTION, SAE_MAP_PRESET_ID } from '../tourPresets';

// ── Segment 1: Explore label map ───────────────────────────────────────────

function makeMapRuntime(overrides: Partial<TourRuntime> = {}): TourRuntime {
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
    setActivePanel: vi.fn(),
    setShowLabels: vi.fn(),
    getLoadedCollection: () => SAE_MAP_COLLECTION,
    getColorByField: () => null,
    getSelectedFeatureIndex: () => 421,
    ...overrides,
  };
}

const mapStep = (id: string) => SAE_MAP_TOUR_STEPS.find((s) => s.id === id)!;

describe('SAE_MAP_TOUR_STEPS (Explore segment)', () => {
  it('has three steps with unique ids and known Explore anchors', () => {
    expect(SAE_MAP_TOUR_STEPS.map((s) => s.id)).toEqual([
      'sae-label-map', 'sae-map-search', 'sae-right-click',
    ]);
    for (const s of SAE_MAP_TOUR_STEPS) {
      expect(TOUR_ANCHORS[s.anchor]).toMatch(/^\[data-tour=/);
    }
  });

  it('the opening step applies the sae-map preset and waits for its collection', async () => {
    const runtime = makeMapRuntime();
    await mapStep('sae-label-map').prepare!(runtime);
    expect(runtime.applyPreset).toHaveBeenCalledWith(SAE_MAP_PRESET_ID);
  });

  it('the search step queries the label map with the tour query', async () => {
    const runtime = makeMapRuntime();
    await mapStep('sae-map-search').prepare!(runtime);
    expect(runtime.runSearch).toHaveBeenCalledWith(SAE_TOUR_QUERY);
    // Cold MiniLM on the Space needs headroom.
    expect(mapStep('sae-map-search').prepareTimeoutMs).toBeGreaterThanOrEqual(30000);
  });

  it('the search step NEVER auto-queries any collection but the label map', async () => {
    for (const other of ['emotion', 'acl_abstracts_emnlp_findings', 'wordnet_senses_full']) {
      const runtime = makeMapRuntime({ getLoadedCollection: () => other });
      await mapStep('sae-map-search').prepare!(runtime);
      expect(runtime.runSearch).not.toHaveBeenCalled();
    }
  });

  it('the right-click step waits for the auto-selected top match', async () => {
    const runtime = makeMapRuntime();
    await expect(mapStep('sae-right-click').prepare!(runtime)).resolves.toBeUndefined();
  });

  it('the right-click step tolerates a search that never selected a point', async () => {
    vi.useFakeTimers();
    try {
      const runtime = makeMapRuntime({ getSelectedFeatureIndex: () => null });
      const prep = mapStep('sae-right-click').prepare!(runtime);
      await vi.advanceTimersByTimeAsync(13000);
      // Resolves (waitFor times out without throwing) — the handoff then goes
      // without a preselected feature.
      await expect(prep).resolves.toBeUndefined();
    } finally {
      vi.useRealTimers();
    }
  });
});

// ── Segment 2: /sae inspection ─────────────────────────────────────────────

function makeRuntime(overrides: Partial<SaeTourRuntime> = {}): SaeTourRuntime {
  return {
    setSearchMode: vi.fn(),
    runSemanticSearch: vi.fn().mockResolvedValue(undefined),
    openFirstResult: vi.fn().mockReturnValue(true),
    hasSelectedFeature: () => false,
    isDetailLoaded: () => true,
    getResultCount: () => 5,
    openChat: vi.fn(),
    isChatReady: () => true,
    loadFirstDemoSession: vi.fn().mockResolvedValue(true),
    ...overrides,
  };
}

const step = (id: string) => SAE_TOUR_STEPS.find((s) => s.id === id)!;

describe('SAE_TOUR_STEPS (/sae segment)', () => {
  it('has four steps with unique ids and known anchors', () => {
    expect(SAE_TOUR_STEPS.map((s) => s.id)).toEqual([
      'feature-anatomy', 'see-it-fire', 'steered-chat', 'back-to-the-map',
    ]);
    for (const s of SAE_TOUR_STEPS) {
      expect(SAE_TOUR_ANCHORS[s.anchor]).toMatch(/^\[data-tour="sae-/);
    }
  });

  it('the anatomy step skips its fallback search when a feature is deep-linked', async () => {
    const runtime = makeRuntime({ hasSelectedFeature: () => true });
    await step('feature-anatomy').prepare!(runtime);
    expect(runtime.runSemanticSearch).not.toHaveBeenCalled();
    expect(runtime.openFirstResult).not.toHaveBeenCalled();
  });

  it('the anatomy step falls back to searching on a direct /sae?tour=sae entry', async () => {
    const runtime = makeRuntime();
    await step('feature-anatomy').prepare!(runtime);
    expect(runtime.setSearchMode).toHaveBeenCalledWith('semantic');
    expect(runtime.runSemanticSearch).toHaveBeenCalledWith(SAE_TOUR_QUERY);
    expect(runtime.openFirstResult).toHaveBeenCalled();
    // Cold MiniLM on the Space needs headroom.
    expect(step('feature-anatomy').prepareTimeoutMs).toBeGreaterThanOrEqual(30000);
  });

  it('the anatomy step degrades gracefully when the search returned nothing', async () => {
    const runtime = makeRuntime({
      openFirstResult: vi.fn().mockReturnValue(false),
      isDetailLoaded: () => false,
    });
    // Must resolve promptly (no wait on a detail that will never load).
    await expect(step('feature-anatomy').prepare!(runtime)).resolves.toBeUndefined();
  });

  it('the chat step opens the sidebar and replays the first recorded session', async () => {
    const runtime = makeRuntime();
    await step('steered-chat').prepare!(runtime);
    expect(runtime.openChat).toHaveBeenCalled();
    expect(runtime.loadFirstDemoSession).toHaveBeenCalled();
  });

  it('the chat step tolerates an empty fixture (no recorded sessions)', async () => {
    const runtime = makeRuntime({
      loadFirstDemoSession: vi.fn().mockResolvedValue(false),
    });
    await expect(step('steered-chat').prepare!(runtime)).resolves.toBeUndefined();
  });
});

// ── Entry / handoff URLs ───────────────────────────────────────────────────

describe('tour paths', () => {
  it('the tour entry is the Explore page (segment 1 runs on the label map)', () => {
    expect(saeTourPath()).toBe('/?tour=sae');
  });

  it('the handoff pins the paper pair, deep-links the feature, and re-triggers', () => {
    const params = new URLSearchParams(saeInspectPath(421).split('?')[1]);
    expect(params.get('modelId')).toBe(SAE_TOUR_MODEL_ID);
    expect(params.get('saeId')).toBe(SAE_TOUR_SAE_ID);
    expect(params.get('featureIndex')).toBe('421');
    expect(params.get('tour')).toBe('sae');
  });

  it('the handoff omits featureIndex when the map search never landed', () => {
    const params = new URLSearchParams(saeInspectPath(null).split('?')[1]);
    expect(params.has('featureIndex')).toBe(false);
    expect(params.get('tour')).toBe('sae');
  });

  it('targets the demo pair (gemma-3-4b-it, the paper steering SAE)', () => {
    expect(SAE_TOUR_MODEL_ID).toBe('gemma-3-4b-it');
    expect(SAE_TOUR_SAE_ID).toBe('9-gemmascope-2-res-16k');
  });
});
