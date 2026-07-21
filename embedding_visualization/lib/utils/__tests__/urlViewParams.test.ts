import { describe, it, expect } from 'vitest';
import { mergeViewSearch, shouldDropPreset } from '../urlViewParams';

describe('mergeViewSearch', () => {
  it('overwrites owned params and preserves unknown params', () => {
    const out = mergeViewSearch('?collection=old&foo=1&preset=emnlp-topics', {
      collection: 'emotion',
      colorBy: 'label',
    });
    const params = new URLSearchParams(out.slice(1));
    expect(params.get('collection')).toBe('emotion');
    expect(params.get('colorBy')).toBe('label');
    expect(params.get('foo')).toBe('1');
    expect(params.get('preset')).toBe('emnlp-topics'); // untouched when absent from owned
  });

  it('deletes owned params passed as null (clearing colorBy clears the scheme)', () => {
    const out = mergeViewSearch(
      '?collection=c&colorBy=label&scale=sequential&scaleName=viridis&color=%23fff&palette=tableau10',
      { collection: 'c', colorBy: null, scale: null, scaleName: null, color: null, palette: null },
    );
    expect(out).toBe('?collection=c');
  });

  it('always strips one-shot params (tour, intro)', () => {
    const out = mergeViewSearch('?tour=1&intro=1&collection=c&bar=2', { collection: 'c' });
    const params = new URLSearchParams(out.slice(1));
    expect(params.has('tour')).toBe(false);
    expect(params.has('intro')).toBe(false);
    expect(params.get('bar')).toBe('2');
  });

  it('returns a string equal to the input when nothing changes', () => {
    const search = '?collection=emotion&colorBy=label';
    expect(mergeViewSearch(search, { collection: 'emotion', colorBy: 'label' })).toBe(search);
  });

  it('handles empty search and input without a leading ?', () => {
    expect(mergeViewSearch('', { collection: 'c' })).toBe('?collection=c');
    expect(mergeViewSearch('foo=1', { collection: 'c' })).toBe('?foo=1&collection=c');
    expect(mergeViewSearch('?tour=1', {})).toBe('');
  });

  it('leaves undefined owned keys untouched', () => {
    const out = mergeViewSearch('?preset=xkcd-manifold&collection=xkcd_hilbert_gemini', {
      collection: 'xkcd_hilbert_gemini',
      preset: undefined,
    });
    expect(new URLSearchParams(out.slice(1)).get('preset')).toBe('xkcd-manifold');
  });

  it('deletes preset when passed null', () => {
    const out = mergeViewSearch('?preset=emnlp-topics&collection=emotion', {
      collection: 'emotion',
      preset: null,
    });
    expect(out).toBe('?collection=emotion');
  });
});

describe('shouldDropPreset', () => {
  it('keeps the preset while on its collection', () => {
    expect(shouldDropPreset('emotion', 'emotion')).toBe(false);
  });
  it('drops the preset after navigating to another collection', () => {
    expect(shouldDropPreset('emotion', 'xkcd_hilbert_gemini')).toBe(true);
  });
  it('keeps it while no collection is selected yet', () => {
    expect(shouldDropPreset('emotion', null)).toBe(false);
  });
  it('never drops when there is no active preset', () => {
    expect(shouldDropPreset(undefined, 'emotion')).toBe(false);
  });
});
