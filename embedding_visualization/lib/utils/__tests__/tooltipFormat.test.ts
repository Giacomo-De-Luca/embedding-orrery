/**
 * Tests for the pure tooltip formatting helpers: hex-color detection,
 * hex normalization, and metadata value formatting (number grouping,
 * float rounding, string capping).
 */

import { describe, it, expect } from 'vitest';
import { isHexColor, normalizeHex, formatMetadataValue } from '../tooltipFormat';

describe('isHexColor', () => {
  it('accepts 3-, 6-, and 8-digit hex with a leading #', () => {
    expect(isHexColor('#ff0000')).toBe(true);
    expect(isHexColor('#AABBCC')).toBe(true);
    expect(isHexColor('#f00')).toBe(true);
    expect(isHexColor('#ff0000ff')).toBe(true);
  });

  it('trims surrounding whitespace', () => {
    expect(isHexColor('  #ff0000 ')).toBe(true);
  });

  it('requires a leading # so hex-digit ids/words are not treated as colors', () => {
    expect(isHexColor('ff0000')).toBe(false);
    expect(isHexColor('123456')).toBe(false);
    expect(isHexColor('deadbeef')).toBe(false);
    expect(isHexColor('facade')).toBe(false);
  });

  it('rejects non-hex strings, wrong lengths, and non-strings', () => {
    expect(isHexColor('hello')).toBe(false);
    expect(isHexColor('#ff00')).toBe(false); // 4 digits
    expect(isHexColor('#gggggg')).toBe(false);
    expect(isHexColor(0xff0000)).toBe(false);
    expect(isHexColor(null)).toBe(false);
    expect(isHexColor(undefined)).toBe(false);
  });
});

describe('normalizeHex', () => {
  it('keeps an existing #', () => {
    expect(normalizeHex('#ff0000')).toBe('#ff0000');
  });
  it('trims whitespace', () => {
    expect(normalizeHex('  #ff0000 ')).toBe('#ff0000');
  });
});

describe('formatMetadataValue', () => {
  it('leaves small integers ungrouped (e.g. years)', () => {
    expect(formatMetadataValue(2019)).toBe('2019');
    expect(formatMetadataValue(0)).toBe('0');
  });

  it('groups large integers', () => {
    expect(formatMetadataValue(12345)).toBe((12345).toLocaleString());
  });

  it('rounds floats to at most 3 fraction digits', () => {
    expect(formatMetadataValue(0.42184)).toBe('0.422');
    expect(formatMetadataValue(1.5)).toBe('1.5');
  });

  it('leaves numeric strings untouched', () => {
    expect(formatMetadataValue('2019')).toBe('2019');
    expect(formatMetadataValue('007')).toBe('007');
  });

  it('stringifies null/undefined to empty', () => {
    expect(formatMetadataValue(null)).toBe('');
    expect(formatMetadataValue(undefined)).toBe('');
  });

  it('caps long strings with an ellipsis', () => {
    const long = 'a'.repeat(250);
    const out = formatMetadataValue(long, 200);
    expect(out).toHaveLength(201); // 200 chars + ellipsis
    expect(out.endsWith('…')).toBe(true);
  });

  it('ignores non-finite numbers, treating them as strings', () => {
    expect(formatMetadataValue(Infinity)).toBe('Infinity');
    expect(formatMetadataValue(NaN)).toBe('NaN');
  });
});
