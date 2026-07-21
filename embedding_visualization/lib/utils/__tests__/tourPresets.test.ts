import { describe, it, expect } from 'vitest';
import {
  TOUR_PRESETS,
  DEMO_DEFAULT_COLLECTION,
  TOUR_PRESET_ID,
  TOUR_COLLECTION,
  getPreset,
  seedInitialColorState,
  resolveInitialCollection,
  presetStoreOps,
} from '../tourPresets';
import { serializeColorScale, deserializeColorScale } from '../colorScaleUrl';

const DEMO_COLLECTIONS = ['emotion', 'xkcd_hilbert_gemini', 'acl_abstracts_emnlp_findings'];

describe('TOUR_PRESETS', () => {
  it('only references real demo collections, with ids matching their keys', () => {
    for (const [key, preset] of Object.entries(TOUR_PRESETS)) {
      expect(preset.id).toBe(key);
      expect(DEMO_COLLECTIONS).toContain(preset.collection);
    }
  });

  it('exposes the preset ids used by the Space README links', () => {
    expect(getPreset('emnlp-topics')?.collection).toBe('acl_abstracts_emnlp_findings');
    expect(getPreset('xkcd-manifold')?.collection).toBe('xkcd_hilbert_gemini');
  });

  it('returns null for unknown or missing ids', () => {
    expect(getPreset('nope')).toBeNull();
    expect(getPreset(null)).toBeNull();
    expect(getPreset(undefined)).toBeNull();
  });

  it('xkcd preset scale round-trips through the URL serializer', () => {
    const scale = getPreset('xkcd-manifold')!.color!.scale!;
    const params = serializeColorScale(scale, undefined);
    expect(
      deserializeColorScale({
        scale: params.scale ?? null,
        scaleName: params.scaleName ?? null,
        color: params.color ?? null,
      }),
    ).toEqual(scale);
  });
});

describe('seedInitialColorState', () => {
  const preset = getPreset('xkcd-manifold');

  it('URL colorBy wins outright — preset color ignored entirely', () => {
    const seeded = seedInitialColorState({
      urlColorBy: 'topic_label',
      urlScale: null,
      urlPalette: null,
      preset,
    });
    expect(seeded).toEqual({ colorBy: 'topic_label', scale: null, palette: null });
  });

  it('uses the preset color block wholesale when no URL colorBy', () => {
    const seeded = seedInitialColorState({
      urlColorBy: null,
      urlScale: null,
      urlPalette: null,
      preset,
    });
    expect(seeded.colorBy).toBe('mapped_colour');
    expect(seeded.scale).toEqual({ type: 'sequential', scaleName: 'xkcdColor' });
  });

  it('all null without URL color or preset', () => {
    expect(
      seedInitialColorState({ urlColorBy: null, urlScale: null, urlPalette: null, preset: null }),
    ).toEqual({ colorBy: null, scale: null, palette: null });
  });
});

describe('resolveInitialCollection', () => {
  const keys = DEMO_COLLECTIONS;

  it('precedence: url > preset > demo default > first key', () => {
    expect(
      resolveInitialCollection({
        urlCollection: 'xkcd_hilbert_gemini',
        presetCollection: 'emotion',
        isDemo: true,
        manifestKeys: keys,
      }),
    ).toBe('xkcd_hilbert_gemini');
    expect(
      resolveInitialCollection({
        urlCollection: null,
        presetCollection: 'acl_abstracts_emnlp_findings',
        isDemo: true,
        manifestKeys: keys,
      }),
    ).toBe('acl_abstracts_emnlp_findings');
    expect(
      resolveInitialCollection({
        urlCollection: null,
        presetCollection: null,
        isDemo: true,
        manifestKeys: keys,
      }),
    ).toBe(DEMO_DEFAULT_COLLECTION);
    expect(
      resolveInitialCollection({
        urlCollection: null,
        presetCollection: null,
        isDemo: false,
        manifestKeys: keys,
      }),
    ).toBe(keys[0]);
  });

  it('skips candidates missing from the manifest', () => {
    expect(
      resolveInitialCollection({
        urlCollection: 'gone',
        presetCollection: 'also-gone',
        isDemo: true,
        manifestKeys: ['other'],
      }),
    ).toBe('other');
  });

  it('returns null on an empty manifest', () => {
    expect(
      resolveInitialCollection({
        urlCollection: null,
        presetCollection: null,
        isDemo: true,
        manifestKeys: [],
      }),
    ).toBeNull();
  });
});

describe('presetStoreOps', () => {
  it('emits method/mode/flags ops for the EMNLP preset in stable order', () => {
    expect(presetStoreOps(getPreset('emnlp-topics')!)).toEqual([
      { kind: 'method', value: 'umap' },
      { kind: 'mode', value: '3d' },
      { kind: 'flag', flag: 'nebulaMode', value: true },
      { kind: 'flag', flag: 'showClusterLabels', value: true },
    ]);
  });

  it('xkcd preset explicitly turns nebula and cluster labels OFF', () => {
    const ops = presetStoreOps(getPreset('xkcd-manifold')!);
    expect(ops).toContainEqual({ kind: 'flag', flag: 'nebulaMode', value: false });
    expect(ops).toContainEqual({ kind: 'flag', flag: 'showClusterLabels', value: false });
  });

  it('emits no flag ops when a preset has none', () => {
    const ops = presetStoreOps({
      id: 'x',
      collection: 'x',
      label: 'x',
      description: 'x',
      method: 'pca',
    });
    expect(ops).toEqual([{ kind: 'method', value: 'pca' }]);
  });
});

describe('tour constants', () => {
  it('demo default stays emotion; every preset colors by topics or manifold', () => {
    expect(DEMO_DEFAULT_COLLECTION).toBe('emotion');
    expect(TOUR_PRESETS.emotion.color?.colorBy).toBe('topic_label');
    expect(TOUR_PRESETS['emnlp-topics'].color?.colorBy).toBe('topic_label');
  });

  it('the tour preset resolves to a real preset targeting the tour collection', () => {
    expect(getPreset(TOUR_PRESET_ID)?.collection).toBe(TOUR_COLLECTION);
    expect(TOUR_COLLECTION).toBe('acl_abstracts_emnlp_findings');
  });
});
