/**
 * Tests for the saeId helpers. parseSaeId is positional
 * ({layer}-{scope...}-{hookAbbrev}-{width}), so it must handle both the
 * gemma-scope and qwen-scope id schemes; buildSaeId is its inverse and takes
 * the scope segment explicitly instead of hardcoding "gemmascope-2". The
 * label, ordering, grouping and filter helpers must keep SAEs that share a
 * layer and width but sit at different hook sites apart.
 */
import { describe, it, expect } from 'vitest';

import {
  parseSaeId,
  buildSaeId,
  saeIdScope,
  formatSaeLabel,
  describeSae,
  compareSaes,
  groupSaeIdsBySite,
  matchesSaeQuery,
} from '../saeCollections';

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

  it('reads the hook site of MLP and attention SAEs', () => {
    expect(parseSaeId('9-gemmascope-2-mlp-16k').hookType).toBe('MLP_OUT');
    expect(parseSaeId('9-gemmascope-2-att-16k').hookType).toBe('ATTN_OUT');
  });
});

describe('formatSaeLabel', () => {
  it('tells residual and MLP SAEs at the same layer and width apart', () => {
    expect(formatSaeLabel('9-gemmascope-2-res-16k')).toBe('L9 res 16k');
    expect(formatSaeLabel('9-gemmascope-2-mlp-16k')).toBe('L9 mlp 16k');
  });

  it('drops the parts every SAE on screen shares', () => {
    const parsed = parseSaeId('9-gemmascope-2-att-65k');
    expect(formatSaeLabel(parsed, { hook: false })).toBe('L9 65k');
    expect(formatSaeLabel(parsed, { width: false })).toBe('L9 att');
    expect(formatSaeLabel(parsed, { hook: false, width: false })).toBe('L9');
  });
});

describe('describeSae', () => {
  it('spells out the hook site', () => {
    expect(describeSae('9-gemmascope-2-mlp-16k')).toBe('L9 · MLP · 16k');
    expect(describeSae('14-qwenscope-1-res-32k')).toBe('L14 · Residual · 32k');
  });
});

describe('compareSaes', () => {
  it('orders by hook site, then layer, then numeric width', () => {
    const ids = [
      '22-gemmascope-2-res-16k',
      '9-gemmascope-2-att-16k',
      '9-gemmascope-2-res-1m',
      '9-gemmascope-2-mlp-16k',
      '9-gemmascope-2-res-262k',
      '9-gemmascope-2-res-16k',
    ];
    const sorted = [...ids].sort((a, b) => compareSaes(parseSaeId(a), parseSaeId(b)));
    expect(sorted).toEqual([
      '9-gemmascope-2-res-16k',
      '9-gemmascope-2-res-262k',
      '9-gemmascope-2-res-1m',
      '22-gemmascope-2-res-16k',
      '9-gemmascope-2-mlp-16k',
      '9-gemmascope-2-att-16k',
    ]);
  });
});

describe('groupSaeIdsBySite', () => {
  it('groups in site order, skips empty sites and keeps the input order within a group', () => {
    expect(
      groupSaeIdsBySite([
        '22-gemmascope-2-res-16k',
        '9-gemmascope-2-att-16k',
        '9-gemmascope-2-res-16k',
      ]),
    ).toEqual([
      { value: 'RESID_POST', items: ['22-gemmascope-2-res-16k', '9-gemmascope-2-res-16k'] },
      { value: 'ATTN_OUT', items: ['9-gemmascope-2-att-16k'] },
    ]);
  });

  it('returns no groups for no SAEs', () => {
    expect(groupSaeIdsBySite([])).toEqual([]);
  });
});

describe('matchesSaeQuery', () => {
  const mlp = '9-gemmascope-2-mlp-16k';

  it('matches the compact and the spelled-out label, term by term', () => {
    expect(matchesSaeQuery(mlp, 'mlp')).toBe(true);
    expect(matchesSaeQuery(mlp, 'L9 16k')).toBe(true);
    expect(matchesSaeQuery(mlp, '9 MLP')).toBe(true);
    expect(matchesSaeQuery('9-gemmascope-2-res-16k', 'residual')).toBe(true);
    expect(matchesSaeQuery(mlp, '  ')).toBe(true);
  });

  it('rejects a query with a term the SAE lacks', () => {
    expect(matchesSaeQuery(mlp, 'residual')).toBe(false);
    expect(matchesSaeQuery(mlp, 'L9 65k')).toBe(false);
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
