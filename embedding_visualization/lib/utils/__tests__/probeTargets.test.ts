/**
 * Tests for probe target profiling: which probe kinds a metadata field can
 * train, the MLP classifier switch, and the min-class preview. The threshold
 * and label rules mirror the backend's probe_targets.py (same table of cases
 * as unit_tests/test_probe_targets.py).
 */
import { describe, it, expect } from 'vitest';

import {
  ProbeTargetProfile,
  canonicalProbeLabel,
  classCountThreshold,
  fallbackProbeKind,
  isProbeKindAvailable,
  mlpClassifierMode,
  resolveProbeKind,
} from '../probeTargets';

const meta = (field: string, values: unknown[]) => values.map((v) => ({ [field]: v }));

const repeat = (counts: Record<string, number>): string[] =>
  Object.entries(counts).flatMap(([label, n]) => Array<string>(n).fill(label));

// Real class counts from the `combined` collection.
const CATEGORY_COUNTS = {
  'EX:OXYMORON PARADOX': 276,
  ANALOGY: 261,
  'RHETORICAL QUESTION': 221,
  OTHER: 196,
  'EX:CONTEXT SHIFT': 185,
  'IM:FALSE ASSERTION': 117,
  EUPHEMISM: 84,
  HYPERBOLE: 81,
};
const SOURCE_COUNTS = {
  '[SENTIPOLC_16]': 7538,
  '[IRONITA-HSC]': 1740,
  '[TWITTIRO]': 1034,
  '[IRONITA-TW-SENTIPOLC, SENTIPOLC_16]': 877,
  '[Multipico-it-0.75Majority]': 770,
  '[IRONITA-TW-BS, SENTIPOLC_16]': 699,
  '[IRONITA-TW-BS]': 541,
  '[IRONITA-TW-SPINO]': 391,
  '[IRONITA-TWITA]': 223,
  '[IRONITA-TW-SENTIPOLC, SENTIPOLC_16, TWITTIRO]': 191,
  '[IRONITA-TW-BS, TWITTIRO]': 114,
  '[IRONITA-TW-BS, SENTIPOLC_16, TWITTIRO]': 29,
  '[SENTIPOLC_16, TWITTIRO]': 26,
  '[IRONITA-TW-SPINO, TWITTIRO]': 24,
  '[IRONITA-TW-SENTIPOLC]': 13,
  '[IRONITA-TW-SENTIPOLC, TWITTIRO]': 3,
};

describe('canonicalProbeLabel', () => {
  it('renders values the way the backend canonicalises them', () => {
    expect(canonicalProbeLabel(3)).toBe('3');
    expect(canonicalProbeLabel(3.0)).toBe('3');
    expect(canonicalProbeLabel(-0)).toBe('0');
    expect(canonicalProbeLabel(2.5)).toBe('2.5');
    expect(canonicalProbeLabel('03')).toBe('03');
    expect(canonicalProbeLabel('3.0')).toBe('3.0');
    expect(canonicalProbeLabel(true)).toBe('true');
  });

  it('treats empty, null and non-scalar values as unlabelled', () => {
    for (const v of ['', null, undefined, [1, 2], { a: 1 }]) {
      expect(canonicalProbeLabel(v)).toBeNull();
    }
  });
});

describe('classCountThreshold', () => {
  it('ignores float noise and rounds fractional counts up', () => {
    expect(classCountThreshold(1400, 0.01)).toBe(14);
    expect(classCountThreshold(1421, 0.01)).toBe(15);
  });

  it('never drops below the floor of five', () => {
    expect(classCountThreshold(100, 0.01)).toBe(5);
    expect(classCountThreshold(10_000, 0)).toBe(5);
  });
});

describe('ProbeTargetProfile', () => {
  it('profiles a multiclass text field', () => {
    const labels = [...repeat(CATEGORY_COUNTS), ...Array(50).fill(null)];
    const profile = ProbeTargetProfile.fromMetadata(meta('category', labels), 'category');
    expect(profile.nLabelled).toBe(1421);
    expect(profile.nDistinct).toBe(8);
    expect(profile.supportsClassification).toBe(true);
    expect(profile.supportsRegression).toBe(false);
    expect(profile.requiresClassification).toBe(true);
  });

  it('previews kept and excluded classes like the backend', () => {
    const category = ProbeTargetProfile.fromMetadata(
      meta('category', repeat(CATEGORY_COUNTS)),
      'category',
    ).preview(0.01);
    expect(category).toMatchObject({ threshold: 15, kept: 8, tooMany: false, tooFew: false });
    expect(category.excluded).toEqual([]);

    const source = ProbeTargetProfile.fromMetadata(
      meta('source', repeat(SOURCE_COUNTS)),
      'source',
    ).preview(0.01);
    expect(source.threshold).toBe(143);
    expect(source.kept).toBe(10);
    expect(source.excluded.map((e) => e.label)).toEqual(
      Object.keys(SOURCE_COUNTS).filter((k) => SOURCE_COUNTS[k as keyof typeof SOURCE_COUNTS] < 143),
    );
  });

  it('never thresholds a two-value field', () => {
    const profile = ProbeTargetProfile.fromMetadata(
      meta('safety', [...Array(200).fill('safe'), 'unsafe']),
      'safety',
    );
    expect(profile.preview(0.1)).toMatchObject({ kept: 2, excluded: [] });
  });

  it('flags too many kept classes and too few', () => {
    const many = Array.from({ length: 51 }, (_, i) => Array(10).fill(`c${i}`)).flat();
    expect(ProbeTargetProfile.fromMetadata(meta('f', many), 'f').preview(0).tooMany).toBe(true);
    const few = [...Array(100).fill('a'), 'b', 'b', 'c', 'c'];
    expect(ProbeTargetProfile.fromMetadata(meta('f', few), 'f').preview(0).tooFew).toBe(true);
  });

  it('accepts integer numeric fields as both regression and classification targets', () => {
    const profile = ProbeTargetProfile.fromMetadata(meta('label', [0, 1, 2, 3, 4, 5, 3.0]), 'label');
    expect(profile.supportsRegression).toBe(true);
    expect(profile.supportsClassification).toBe(true);
    expect(profile.requiresClassification).toBe(false);
  });

  it('treats non-integer numeric fields with 3+ values as continuous', () => {
    const profile = ProbeTargetProfile.fromMetadata(meta('rating', [0.5, 1.5, 2.25]), 'rating');
    expect(profile.supportsRegression).toBe(true);
    expect(profile.supportsClassification).toBe(false);
  });

  it('parses numeric strings strictly', () => {
    const lenient = ProbeTargetProfile.fromMetadata(meta('f', ['0x1A', '1_000', '3']), 'f');
    expect(lenient.supportsRegression).toBe(false); // text labels, not numbers
    const numericText = ProbeTargetProfile.fromMetadata(meta('f', ['1', '2', '10']), 'f');
    expect(numericText.supportsRegression).toBe(true);
  });

  it('caps classification at 100 distinct text values', () => {
    const wide = Array.from({ length: 101 }, (_, i) => `v${i}`);
    expect(ProbeTargetProfile.fromMetadata(meta('f', wide), 'f').supportsClassification).toBe(false);
  });

  it('rejects single-value fields', () => {
    const profile = ProbeTargetProfile.fromMetadata(meta('f', ['a', 'a', null]), 'f');
    expect(profile.supportsClassification).toBe(false);
    expect(profile.supportsRegression).toBe(false);
  });
});

describe('probe kind availability', () => {
  const continuous = ProbeTargetProfile.fromMetadata(meta('r', [0.5, 1.5, 2.25, 9.75]), 'r');
  const binaryText = ProbeTargetProfile.fromMetadata(meta('s', ['safe', 'unsafe']), 's');
  const multiText = ProbeTargetProfile.fromMetadata(meta('c', ['a', 'b', 'c']), 'c');
  const integerLabels = ProbeTargetProfile.fromMetadata(meta('l', [0, 1, 2, 3]), 'l');

  it('offers regression kinds only where regression is possible', () => {
    expect(isProbeKindAvailable('ridge', continuous)).toBe(true);
    expect(isProbeKindAvailable('ridge', binaryText)).toBe(true);
    expect(isProbeKindAvailable('ridge', integerLabels)).toBe(true);
    expect(isProbeKindAvailable('ridge', multiText)).toBe(false);
  });

  it('offers classifier kinds only where classes exist', () => {
    for (const kind of ['logreg', 'svc'] as const) {
      expect(isProbeKindAvailable(kind, continuous)).toBe(false);
      expect(isProbeKindAvailable(kind, binaryText)).toBe(true);
      expect(isProbeKindAvailable(kind, multiText)).toBe(true);
      expect(isProbeKindAvailable(kind, integerLabels)).toBe(true);
    }
  });

  it('always offers the MLP (it switches between regressor and classifier)', () => {
    for (const profile of [continuous, binaryText, multiText, integerLabels]) {
      expect(isProbeKindAvailable('mlp', profile)).toBe(true);
    }
  });

  it('offers everything when no target is selected', () => {
    expect(isProbeKindAvailable('logreg', null)).toBe(true);
    expect(isProbeKindAvailable('ridge', null)).toBe(true);
  });

  it('falls back to a kind the target supports', () => {
    expect(fallbackProbeKind('logreg', continuous)).toBe('ridge');
    expect(fallbackProbeKind('ridge', multiText)).toBe('logreg');
    expect(fallbackProbeKind('mlp', continuous)).toBe('mlp'); // probe tour relies on this
    expect(fallbackProbeKind('svc', binaryText)).toBe('svc');
  });
});

describe('MLP classifier switch', () => {
  const continuous = ProbeTargetProfile.fromMetadata(meta('r', [0.5, 1.5, 2.25]), 'r');
  const binaryText = ProbeTargetProfile.fromMetadata(meta('s', ['safe', 'unsafe']), 's');
  const multiText = ProbeTargetProfile.fromMetadata(meta('c', ['a', 'b', 'c']), 'c');

  it('is forced on for text targets with 3+ classes and off for continuous ones', () => {
    expect(mlpClassifierMode(multiText)).toBe('forced-on');
    expect(mlpClassifierMode(continuous)).toBe('forced-off');
    expect(mlpClassifierMode(binaryText)).toBe('free');
    expect(mlpClassifierMode(null)).toBe('free');
  });

  it('resolves the trained kind', () => {
    expect(resolveProbeKind('mlp', false, multiText)).toBe('mlp_clf');
    expect(resolveProbeKind('mlp', true, continuous)).toBe('mlp');
    expect(resolveProbeKind('mlp', true, binaryText)).toBe('mlp_clf');
    expect(resolveProbeKind('mlp', false, binaryText)).toBe('mlp');
    expect(resolveProbeKind('logreg', true, multiText)).toBe('logreg');
  });
});
