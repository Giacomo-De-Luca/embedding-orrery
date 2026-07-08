/**
 * Tests for the saeId parse/build helpers. parseSaeId is positional
 * ({layer}-{scope...}-{hookAbbrev}-{width}), so it must handle both the
 * gemma-scope and qwen-scope id schemes; buildSaeId is its inverse and takes
 * the scope segment explicitly instead of hardcoding "gemmascope-2".
 */
import { describe, it, expect } from 'vitest';

import { parseSaeId, buildSaeId, saeIdScope } from '../saeCollections';

describe('parseSaeId', () => {
  it('parses gemma-scope ids', () => {
    expect(parseSaeId('9-gemmascope-2-res-16k')).toEqual({
      layerIndex: 9,
      hookType: 'RESID_POST',
      width: '16k',
    });
  });

  it('parses qwen-scope ids (same positional shape)', () => {
    expect(parseSaeId('14-qwenscope-1-res-32k')).toEqual({
      layerIndex: 14,
      hookType: 'RESID_POST',
      width: '32k',
    });
  });
});

describe('saeIdScope', () => {
  it('extracts the scope segment from a gemma id', () => {
    expect(saeIdScope('9-gemmascope-2-res-16k')).toBe('gemmascope-2');
  });

  it('extracts the scope segment from a qwen id', () => {
    expect(saeIdScope('14-qwenscope-1-res-32k')).toBe('qwenscope-1');
  });

  it('falls back to gemmascope-2 for malformed ids', () => {
    expect(saeIdScope('garbage')).toBe('gemmascope-2');
  });
});

describe('buildSaeId', () => {
  it('defaults to the gemma-scope scheme (legacy behavior)', () => {
    expect(buildSaeId(9, 'RESID_POST', '16k')).toBe('9-gemmascope-2-res-16k');
  });

  it('builds qwen-scope ids when given the scope', () => {
    expect(buildSaeId(14, 'RESID_POST', '32k', 'qwenscope-1')).toBe('14-qwenscope-1-res-32k');
  });

  it('round-trips through parseSaeId + saeIdScope', () => {
    const id = '24-qwenscope-1-res-32k';
    const parsed = parseSaeId(id);
    expect(buildSaeId(parsed.layerIndex, parsed.hookType, parsed.width, saeIdScope(id))).toBe(id);
  });
});
