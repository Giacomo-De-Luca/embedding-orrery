import { describe, it, expect } from 'vitest';
import { resolveSiteUrl, SITE_DESCRIPTION, SITE_TITLE } from '../siteMetadata';

describe('resolveSiteUrl', () => {
  it('returns undefined when the origin is unset or blank', () => {
    expect(resolveSiteUrl(undefined)).toBeUndefined();
    expect(resolveSiteUrl(null)).toBeUndefined();
    expect(resolveSiteUrl('')).toBeUndefined();
    expect(resolveSiteUrl('   ')).toBeUndefined();
  });

  it('parses an absolute https origin', () => {
    const url = resolveSiteUrl('https://giacomodeluca-orrery-demo.hf.space');
    expect(url?.origin).toBe('https://giacomodeluca-orrery-demo.hf.space');
  });

  it('keeps an explicit http origin (local compose stacks)', () => {
    expect(resolveSiteUrl('http://localhost:3000')?.href).toBe('http://localhost:3000/');
  });

  it('assumes https for a bare host', () => {
    expect(resolveSiteUrl('my-space.hf.space')?.href).toBe('https://my-space.hf.space/');
  });

  it('trims surrounding whitespace', () => {
    expect(resolveSiteUrl('  https://example.org  ')?.origin).toBe('https://example.org');
  });

  it('rejects non-http schemes and unparsable values', () => {
    expect(resolveSiteUrl('ftp://example.org')).toBeUndefined();
    expect(resolveSiteUrl('not a url')).toBeUndefined();
    expect(resolveSiteUrl('https://')).toBeUndefined();
  });
});

describe('site copy', () => {
  it('names the product in the title and keeps the description card-sized', () => {
    expect(SITE_TITLE).toContain('Orrery');
    // Social cards truncate long descriptions; keep it under the usual ~200-char cut.
    expect(SITE_DESCRIPTION.length).toBeLessThanOrEqual(200);
  });
});
