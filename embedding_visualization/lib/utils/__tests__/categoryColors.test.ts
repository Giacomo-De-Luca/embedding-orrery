/**
 * Tests for category color utilities.
 *
 * These tests verify the color generation and preset
 * system for embedding visualization.
 */

import { describe, it, expect } from 'vitest';
import {
  generateCategoryColors,
  buildCategoryColorMap,
  getCategoryLabel,
  getCategoryDisplayName,
  getUnclusteredValues,
  POS_PRESET,
  CATEGORY_PRESETS,
  MISSING_CATEGORY_COLOR,
  CORRECTNESS_PRESET,
  CORRECTNESS_LABELS,
  buildAliasedCategoryColorMap,
} from '../categoryColors';
import { CategoryKey } from '../categoryKey';
import { CATEGORY_PALETTES, BUILTIN_PALETTE_NAMES, getPaletteColors } from '../categoryPalettes';

describe('generateCategoryColors', () => {
  it('should return correct number of colors for small counts', () => {
    expect(generateCategoryColors(1)).toHaveLength(1);
    expect(generateCategoryColors(5)).toHaveLength(5);
    expect(generateCategoryColors(10)).toHaveLength(10);
  });

  it('should use cosmicGalaxy palette by default', () => {
    const colors = generateCategoryColors(5);
    // First color should be cosmicGalaxy deep nebula teal
    expect(colors[0]).toBe('#0b7285');
  });

  it('should use cosmicGalaxy palette for 11-20 categories', () => {
    const colors = generateCategoryColors(15);
    expect(colors).toHaveLength(15);
    expect(colors[0]).toBe('#0b7285'); // deep nebula teal
    expect(colors[2]).toBe('#d4a017'); // stellar gold
  });

  it('should use explicit palette when specified', () => {
    const colors = generateCategoryColors(5, 'category10');
    expect(colors[0]).toBe('#1f77b4'); // D3 category10 blue
    expect(colors).toHaveLength(5);
  });

  it('should cycle colors for > 20 categories', () => {
    const colors = generateCategoryColors(25);
    expect(colors).toHaveLength(25);
    // Should cycle back to start
    expect(colors[20]).toBe(colors[0]);
  });

  it('should handle edge case of 0 categories', () => {
    const colors = generateCategoryColors(0);
    // Should return at least 1 color
    expect(colors.length).toBeGreaterThanOrEqual(1);
  });

  it('should handle negative numbers', () => {
    const colors = generateCategoryColors(-5);
    // Should return at least 1 color
    expect(colors.length).toBeGreaterThanOrEqual(1);
  });
});

describe('buildCategoryColorMap', () => {
  describe('with POS preset', () => {
    it('should use preset colors for known POS values', () => {
      const colorMap = buildCategoryColorMap('pos', ['n', 'v', 'a']);

      expect(colorMap['n']).toBe('#1f77b4'); // noun - blue
      expect(colorMap['v']).toBe('#ff7f0e'); // verb - orange
      expect(colorMap['a']).toBe('#2ca02c'); // adjective - green
    });

    it('should use preset color for known values and dynamic for unknown', () => {
      const colorMap = buildCategoryColorMap('pos', ['n', 'x', 'unknown']);

      expect(colorMap['n']).toBe('#1f77b4'); // Known - blue
      expect(colorMap['unknown']).toBe('#7f7f7f'); // Explicit unknown - gray (in preset)
      // 'x' is not in the POS preset, so it gets a dynamically generated color
      expect(colorMap['x']).toBeDefined();
      expect(colorMap['x']).not.toBe(colorMap['n']);
    });

    it('should be case-insensitive for field name', () => {
      const colorMap1 = buildCategoryColorMap('pos', ['n']);
      const colorMap2 = buildCategoryColorMap('POS', ['n']);
      const colorMap3 = buildCategoryColorMap('Pos', ['n']);

      expect(colorMap1['n']).toBe(colorMap2['n']);
      expect(colorMap2['n']).toBe(colorMap3['n']);
    });

    it('should recognize alternative field names', () => {
      const colorMap = buildCategoryColorMap('part_of_speech', ['n', 'v']);

      expect(colorMap['n']).toBe('#1f77b4'); // Uses POS preset
      expect(colorMap['v']).toBe('#ff7f0e');
    });
  });

  describe('with dynamic colors', () => {
    it('should generate colors for unknown category fields', () => {
      const colorMap = buildCategoryColorMap('topic', ['science', 'tech', 'art']);

      // Should have 3 unique colors
      expect(Object.keys(colorMap)).toHaveLength(3);

      const colors = Object.values(colorMap);
      const uniqueColors = new Set(colors);
      expect(uniqueColors.size).toBe(3);
    });

    it('should generate consistent colors for same order', () => {
      const colorMap1 = buildCategoryColorMap('topic', ['a', 'b', 'c']);
      const colorMap2 = buildCategoryColorMap('topic', ['a', 'b', 'c']);

      // Same input should produce same output
      expect(colorMap1['a']).toBe(colorMap2['a']);
      expect(colorMap1['b']).toBe(colorMap2['b']);
      expect(colorMap1['c']).toBe(colorMap2['c']);
    });

    it('should handle null category field', () => {
      const colorMap = buildCategoryColorMap(null, ['x', 'y', 'z']);

      // Should still generate colors (no preset lookup)
      expect(Object.keys(colorMap)).toHaveLength(3);
    });

    it('should handle empty values array', () => {
      const colorMap = buildCategoryColorMap('topic', []);

      expect(Object.keys(colorMap)).toHaveLength(0);
    });
  });

  describe('with missing values', () => {
    it('gives Missing the fixed neutral color without taking a palette slot', () => {
      const withMissing = buildCategoryColorMap('category', ['a', 'b', CategoryKey.MISSING]);
      const withoutMissing = buildCategoryColorMap('category', ['a', 'b']);

      expect(withMissing[CategoryKey.MISSING]).toBe(MISSING_CATEGORY_COLOR);
      expect(withMissing.a).toBe(withoutMissing.a);
      expect(withMissing.b).toBe(withoutMissing.b);
    });

    it('leaves preset fields unchanged', () => {
      const colorMap = buildCategoryColorMap('pos', ['n', 'x', CategoryKey.MISSING]);

      expect(colorMap.n).toBe('#1f77b4');
      expect(colorMap.x).toBe(buildCategoryColorMap('pos', ['n', 'x']).x);
      expect(colorMap[CategoryKey.MISSING]).toBe(MISSING_CATEGORY_COLOR);
    });

    it('lets a custom override recolor Missing', () => {
      const colorMap = buildCategoryColorMap('category', ['a', CategoryKey.MISSING], undefined, { [CategoryKey.MISSING]: '#123456' });

      expect(colorMap[CategoryKey.MISSING]).toBe('#123456');
    });
  });
});

describe('getCategoryLabel', () => {
  it('should return human-readable labels for POS', () => {
    expect(getCategoryLabel('pos', 'n')).toBe('Noun');
    expect(getCategoryLabel('pos', 'v')).toBe('Verb');
    expect(getCategoryLabel('pos', 'a')).toBe('Adjective');
    expect(getCategoryLabel('pos', 'r')).toBe('Adverb');
    expect(getCategoryLabel('pos', 's')).toBe('Adj. Satellite');
  });

  it('should return original value for unknown POS values', () => {
    expect(getCategoryLabel('pos', 'x')).toBe('x');
    expect(getCategoryLabel('pos', 'custom')).toBe('custom');
  });

  it('should return original value for unknown fields', () => {
    expect(getCategoryLabel('topic', 'science')).toBe('science');
    expect(getCategoryLabel('category', 'animal')).toBe('animal');
  });

  it('should handle null field name', () => {
    expect(getCategoryLabel(null, 'value')).toBe('value');
  });

  it('should label the Missing key "Missing values" for any field', () => {
    expect(getCategoryLabel('category', CategoryKey.MISSING)).toBe('Missing values');
    expect(getCategoryLabel('pos', CategoryKey.MISSING)).toBe('Missing values');
    expect(getCategoryLabel('topic_label', CategoryKey.MISSING)).toBe('Missing values');
    expect(getCategoryLabel(null, CategoryKey.MISSING)).toBe('Missing values');
  });
});

describe('getCategoryDisplayName', () => {
  it('should return preset name for known fields', () => {
    expect(getCategoryDisplayName('pos')).toBe('Part of Speech');
    expect(getCategoryDisplayName('part_of_speech')).toBe('Part of Speech');
  });

  it('should convert field name to title case for unknown fields', () => {
    expect(getCategoryDisplayName('topic_name')).toBe('Topic Name');
    expect(getCategoryDisplayName('userCategory')).toBe('User Category');
    expect(getCategoryDisplayName('type')).toBe('Type');
  });

  it('should return "Category" for null field', () => {
    expect(getCategoryDisplayName(null)).toBe('Category');
  });
});

describe('POS_PRESET', () => {
  it('should have all WordNet POS types', () => {
    expect(POS_PRESET.colors).toHaveProperty('n');
    expect(POS_PRESET.colors).toHaveProperty('v');
    expect(POS_PRESET.colors).toHaveProperty('a');
    expect(POS_PRESET.colors).toHaveProperty('r');
    expect(POS_PRESET.colors).toHaveProperty('s');
    expect(POS_PRESET.colors).toHaveProperty('unknown');
  });

  it('should have labels for all POS types', () => {
    expect(POS_PRESET.labels).toBeDefined();
    expect(POS_PRESET.labels!['n']).toBe('Noun');
    expect(POS_PRESET.labels!['v']).toBe('Verb');
  });

  it('should have distinct colors for each POS', () => {
    const colors = Object.values(POS_PRESET.colors);
    const uniqueColors = new Set(colors);
    // All colors should be unique (except unknown might match one)
    expect(uniqueColors.size).toBeGreaterThanOrEqual(colors.length - 1);
  });
});

describe('CATEGORY_PRESETS', () => {
  it('should include POS preset', () => {
    expect(CATEGORY_PRESETS).toHaveProperty('pos');
    expect(CATEGORY_PRESETS).toHaveProperty('part_of_speech');
  });

  it('should map alternative names to same preset', () => {
    expect(CATEGORY_PRESETS['pos']).toBe(CATEGORY_PRESETS['part_of_speech']);
  });
});

describe('getUnclusteredValues', () => {
  it('returns the noise tokens for topic fields', () => {
    for (const field of ['topic', 'topic_id', 'topic_label']) {
      const noise = getUnclusteredValues(field);
      expect(noise.has('-1')).toBe(true);
      expect(noise.has('Unclustered')).toBe(true);
    }
  });

  it('returns the noise tokens for subtopic fields too', () => {
    for (const field of ['subtopic', 'subtopic_id', 'subtopic_label']) {
      const noise = getUnclusteredValues(field);
      expect(noise.has('-1')).toBe(true);
      expect(noise.has('Unclustered')).toBe(true);
    }
  });

  it('is case-insensitive on the field name', () => {
    expect(getUnclusteredValues('Subtopic_Label').has('Unclustered')).toBe(true);
  });

  it('returns an empty set for non-topic fields and null', () => {
    expect(getUnclusteredValues('pos').size).toBe(0);
    expect(getUnclusteredValues('category').size).toBe(0);
    expect(getUnclusteredValues(null).size).toBe(0);
  });
});

describe('color validity', () => {
  it('should generate valid hex colors', () => {
    const hexColorRegex = /^#[0-9a-fA-F]{6}$/;

    const colors = generateCategoryColors(20);
    for (const color of colors) {
      expect(color).toMatch(hexColorRegex);
    }
  });

  it('should have valid hex colors in presets', () => {
    const hexColorRegex = /^#[0-9a-fA-F]{6}$/;

    for (const color of Object.values(POS_PRESET.colors)) {
      expect(color).toMatch(hexColorRegex);
    }
  });

  it('should have valid hex colors in all built-in palettes', () => {
    const hexColorRegex = /^#[0-9a-fA-F]{6}$/;

    for (const name of BUILTIN_PALETTE_NAMES) {
      const colors = getPaletteColors(name);
      expect(colors).toBeDefined();
      for (const c of colors!) {
        expect(c).toMatch(hexColorRegex);
      }
    }
  });
});

describe('categoryPalettes registry', () => {
  it('should keep all built-in palette keys stable (persisted in URLs and collection defaults)', () => {
    expect(BUILTIN_PALETTE_NAMES).toEqual(expect.arrayContaining([
      'cosmicGalaxy', 'cosmicGalaxyLight', 'cosmicGalaxyXL',
      'emeraldGalaxy', 'azureGalaxy', 'emberGalaxy', 'violetGalaxy',
      'Galaxy', 'category10', 'category20',
    ]));
  });

  it('cosmicGalaxyXL starts with cosmicGalaxy verbatim, so switching keeps existing colors', () => {
    expect(CATEGORY_PALETTES.cosmicGalaxyXL.colors.slice(0, 20))
      .toEqual([...CATEGORY_PALETTES.cosmicGalaxy.colors]);
  });

  it('should return colors via getPaletteColors', () => {
    const colors = getPaletteColors('cosmicGalaxy');
    expect(colors).toBeDefined();
    expect(colors!.length).toBe(20);
    expect(colors![0]).toBe('#0b7285');
  });

  it('should return undefined for unknown palette names', () => {
    expect(getPaletteColors('nonexistent')).toBeUndefined();
  });
});

describe('probe correctness preset', () => {
  it('colours probe_*_correct fields by suffix', () => {
    const field = 'probe_category_logreg_correct';
    const values = [CORRECTNESS_LABELS.correct, CORRECTNESS_LABELS.wrong, CORRECTNESS_LABELS.excluded];
    const colorMap = buildCategoryColorMap(field, values);
    for (const value of values) expect(colorMap[value]).toBe(CORRECTNESS_PRESET.colors[value]);
    expect(getCategoryLabel(field, CORRECTNESS_LABELS.excluded)).toBe(
      CORRECTNESS_PRESET.labels![CORRECTNESS_LABELS.excluded],
    );
  });
});

describe('buildAliasedCategoryColorMap', () => {
  const alias = {
    sourceField: 'category',
    domain: ['ANALOGY', 'EUPHEMISM', 'HYPERBOLE', 'RARE', CategoryKey.MISSING],
  };
  const sourceMap = buildCategoryColorMap('category', alias.domain);

  it("draws each predicted label in the target field's colour", () => {
    // EUPHEMISM is never predicted and there are no missing predictions, yet
    // HYPERBOLE keeps the target field's palette slot.
    const colorMap = buildAliasedCategoryColorMap(alias, ['ANALOGY', 'HYPERBOLE']);
    expect(colorMap).toEqual({
      ANALOGY: sourceMap.ANALOGY,
      HYPERBOLE: sourceMap.HYPERBOLE,
    });
    expect(buildCategoryColorMap('pred', ['ANALOGY', 'HYPERBOLE']).HYPERBOLE).not.toBe(
      sourceMap.HYPERBOLE,
    );
  });

  it('keeps Missing grey and gives unknown labels fresh slots', () => {
    const colorMap = buildAliasedCategoryColorMap(alias, ['ANALOGY', 'NEW', CategoryKey.MISSING]);
    expect(colorMap[CategoryKey.MISSING]).toBe(MISSING_CATEGORY_COLOR);
    expect(colorMap.NEW).toBe(generateCategoryColors(5)[4]);
  });

  it("applies the target field's overrides, then the field's own", () => {
    const colorMap = buildAliasedCategoryColorMap(
      alias,
      ['ANALOGY', 'HYPERBOLE'],
      undefined,
      { ANALOGY: '#111111', HYPERBOLE: '#222222' },
      { HYPERBOLE: '#333333' },
    );
    expect(colorMap).toEqual({ ANALOGY: '#111111', HYPERBOLE: '#333333' });
  });

  it('follows a preset-backed target field', () => {
    const colorMap = buildAliasedCategoryColorMap({ sourceField: 'pos', domain: ['n', 'v'] }, ['v']);
    expect(colorMap.v).toBe(POS_PRESET.colors.v);
  });

  it('follows the chosen palette', () => {
    const colorMap = buildAliasedCategoryColorMap(alias, ['HYPERBOLE'], 'emberGalaxy');
    expect(colorMap.HYPERBOLE).toBe(buildCategoryColorMap('category', alias.domain, 'emberGalaxy').HYPERBOLE);
  });
});
