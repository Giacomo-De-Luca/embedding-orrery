import { describe, it, expect } from 'vitest';
import { parseCollectionsParams, buildCollectionsSearch } from '../urlState';

describe('parseCollectionsParams', () => {
  it('defaults to huggingface with no params', () => {
    expect(parseCollectionsParams(new URLSearchParams(''))).toEqual({
      tab: 'huggingface',
      collection: null,
    });
  });

  it('reads a valid tab', () => {
    expect(parseCollectionsParams(new URLSearchParams('?tab=sae'))).toEqual({
      tab: 'sae',
      collection: null,
    });
  });

  it('falls back to huggingface on an invalid tab', () => {
    expect(parseCollectionsParams(new URLSearchParams('?tab=bogus'))).toEqual({
      tab: 'huggingface',
      collection: null,
    });
  });

  it('implies the manage tab when only ?collection= is present', () => {
    expect(parseCollectionsParams(new URLSearchParams('?collection=emotion'))).toEqual({
      tab: 'manage',
      collection: 'emotion',
    });
  });

  it('keeps collection when tab=manage is explicit', () => {
    expect(
      parseCollectionsParams(new URLSearchParams('?tab=manage&collection=emotion'))
    ).toEqual({ tab: 'manage', collection: 'emotion' });
  });

  it('ignores collection on non-manage tabs', () => {
    expect(
      parseCollectionsParams(new URLSearchParams('?tab=local&collection=emotion'))
    ).toEqual({ tab: 'local', collection: null });
  });
});

describe('buildCollectionsSearch', () => {
  it('omits the default tab entirely', () => {
    expect(buildCollectionsSearch('', { tab: 'huggingface', collection: null })).toBeNull();
    expect(
      buildCollectionsSearch('?tab=manage', { tab: 'huggingface', collection: null })
    ).toBe('');
  });

  it('returns null when nothing changed', () => {
    expect(
      buildCollectionsSearch('?tab=manage&collection=emotion', {
        tab: 'manage',
        collection: 'emotion',
      })
    ).toBeNull();
    expect(buildCollectionsSearch('?tab=sae', { tab: 'sae', collection: null })).toBeNull();
  });

  it('normalizes an explicit default tab away', () => {
    expect(
      buildCollectionsSearch('?tab=huggingface', { tab: 'huggingface', collection: null })
    ).toBe('');
  });

  it('sets tab and collection for manage', () => {
    expect(
      buildCollectionsSearch('', { tab: 'manage', collection: 'my_data' })
    ).toBe('?tab=manage&collection=my_data');
  });

  it('drops collection when leaving manage', () => {
    expect(
      buildCollectionsSearch('?tab=manage&collection=emotion', {
        tab: 'local',
        collection: 'emotion',
      })
    ).toBe('?tab=local');
  });

  it('drops collection when it is cleared on manage', () => {
    expect(
      buildCollectionsSearch('?tab=manage&collection=emotion', {
        tab: 'manage',
        collection: null,
      })
    ).toBe('?tab=manage');
  });

  it('preserves unrelated params', () => {
    expect(
      buildCollectionsSearch('?foo=bar', { tab: 'manage', collection: 'x' })
    ).toBe('?foo=bar&tab=manage&collection=x');
    expect(
      buildCollectionsSearch('?foo=bar&tab=manage&collection=x', {
        tab: 'huggingface',
        collection: null,
      })
    ).toBe('?foo=bar');
  });

  it('encodes collection names with special characters', () => {
    expect(
      buildCollectionsSearch('', { tab: 'manage', collection: 'a b/c' })
    ).toBe('?tab=manage&collection=a+b%2Fc');
  });
});
