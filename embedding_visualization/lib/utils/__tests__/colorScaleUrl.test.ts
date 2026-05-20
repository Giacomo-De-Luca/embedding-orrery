/**
 * Tests for color-scheme <-> URL / persisted-default serialization.
 *
 * Covers the ColorScale discriminated-union round-trip, the per-collection
 * default-color-scheme round-trip, and the JSON-string storage format used in
 * collection metadata.
 */

import { describe, it, expect } from 'vitest';
import {
  serializeColorScale,
  deserializeColorScale,
  serializeDefaultColorScheme,
  resolveDefaultColorScheme,
} from '../colorScaleUrl';
import type { ColorScale } from '../../types/types';

describe('serializeColorScale / deserializeColorScale', () => {
  it('round-trips categorical', () => {
    expect(serializeColorScale({ type: 'categorical' }, undefined)).toEqual({ scale: 'categorical' });
    expect(deserializeColorScale({ scale: 'categorical', scaleName: null, color: null }))
      .toEqual({ type: 'categorical' });
  });

  it('round-trips sequential with scaleName', () => {
    const scale: ColorScale = { type: 'sequential', scaleName: 'viridis' };
    const p = serializeColorScale(scale, undefined);
    expect(p).toEqual({ scale: 'sequential', scaleName: 'viridis' });
    expect(deserializeColorScale({ scale: p.scale!, scaleName: p.scaleName ?? null, color: null }))
      .toEqual(scale);
  });

  it('round-trips diverging with scaleName', () => {
    const scale: ColorScale = { type: 'diverging', scaleName: 'blueGold' };
    const p = serializeColorScale(scale, undefined);
    expect(p).toEqual({ scale: 'diverging', scaleName: 'blueGold' });
    expect(deserializeColorScale({ scale: p.scale!, scaleName: p.scaleName ?? null, color: null }))
      .toEqual(scale);
  });

  it('round-trips monochrome baseColor', () => {
    const scale: ColorScale = { type: 'monochrome', baseColor: '#1f77b4' };
    const p = serializeColorScale(scale, undefined);
    expect(p).toEqual({ scale: 'monochrome', color: '#1f77b4' });
    expect(deserializeColorScale({ scale: 'monochrome', scaleName: null, color: '#1f77b4' }))
      .toEqual(scale);
  });

  it('emits palette independently of scale type', () => {
    expect(serializeColorScale({ type: 'sequential', scaleName: 'viridis' }, 'tableau10'))
      .toEqual({ scale: 'sequential', scaleName: 'viridis', palette: 'tableau10' });
    expect(serializeColorScale({ type: 'categorical' }, 'set2'))
      .toEqual({ scale: 'categorical', palette: 'set2' });
  });

  it('returns null for invalid or incomplete params', () => {
    expect(deserializeColorScale({ scale: null, scaleName: null, color: null })).toBeNull();
    expect(deserializeColorScale({ scale: 'sequential', scaleName: null, color: null })).toBeNull();
    expect(deserializeColorScale({ scale: 'diverging', scaleName: null, color: null })).toBeNull();
    expect(deserializeColorScale({ scale: 'monochrome', scaleName: null, color: null })).toBeNull();
    expect(deserializeColorScale({ scale: 'bogus', scaleName: 'x', color: null })).toBeNull();
  });
});

describe('serializeDefaultColorScheme / resolveDefaultColorScheme', () => {
  it('round-trips field + sequential scale + palette', () => {
    const scheme = serializeDefaultColorScheme('score', { type: 'sequential', scaleName: 'viridis' }, 'tableau10');
    expect(scheme).toEqual({ colorBy: 'score', scale: 'sequential', scaleName: 'viridis', palette: 'tableau10' });
    expect(resolveDefaultColorScheme(scheme)).toEqual({
      field: 'score',
      scale: { type: 'sequential', scaleName: 'viridis' },
      palette: 'tableau10',
    });
  });

  it('round-trips categorical with no palette', () => {
    const scheme = serializeDefaultColorScheme('topic_label', { type: 'categorical' }, undefined);
    expect(scheme).toEqual({ colorBy: 'topic_label', scale: 'categorical' });
    expect(resolveDefaultColorScheme(scheme)).toEqual({
      field: 'topic_label',
      scale: { type: 'categorical' },
      palette: null,
    });
  });

  it('returns null when there is no field', () => {
    expect(resolveDefaultColorScheme(null)).toBeNull();
    expect(resolveDefaultColorScheme(undefined)).toBeNull();
    // @ts-expect-error - exercising a malformed object missing colorBy
    expect(resolveDefaultColorScheme({ scale: 'categorical' })).toBeNull();
  });

  it('survives the JSON-string storage round-trip', () => {
    const scheme = serializeDefaultColorScheme('x', { type: 'monochrome', baseColor: '#00aaff' }, undefined);
    const parsed = JSON.parse(JSON.stringify(scheme));
    expect(resolveDefaultColorScheme(parsed)).toEqual({
      field: 'x',
      scale: { type: 'monochrome', baseColor: '#00aaff' },
      palette: null,
    });
  });
});
