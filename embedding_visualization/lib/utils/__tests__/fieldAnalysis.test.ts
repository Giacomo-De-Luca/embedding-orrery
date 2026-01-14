/**
 * Tests for field analysis utilities.
 *
 * These tests verify the automatic detection of label and category
 * fields from arbitrary metadata structures.
 */

import { describe, it, expect } from 'vitest';
import {
  fieldToDisplayName,
  analyzeField,
  computeCategoryFieldOptions,
  detectDisplayConfig,
  getFieldValues,
} from '../fieldAnalysis';

describe('fieldToDisplayName', () => {
  it('should convert "pos" to "Part of Speech"', () => {
    expect(fieldToDisplayName('pos')).toBe('Part of Speech');
  });

  it('should convert snake_case to Title Case', () => {
    expect(fieldToDisplayName('created_at')).toBe('Created At');
    expect(fieldToDisplayName('user_name')).toBe('User Name');
    expect(fieldToDisplayName('first_name_last_name')).toBe('First Name Last Name');
  });

  it('should convert camelCase to Title Case', () => {
    expect(fieldToDisplayName('createdAt')).toBe('Created At');
    expect(fieldToDisplayName('userName')).toBe('User Name');
    expect(fieldToDisplayName('firstName')).toBe('First Name');
  });

  it('should capitalize single words', () => {
    expect(fieldToDisplayName('category')).toBe('Category');
    expect(fieldToDisplayName('type')).toBe('Type');
  });

  it('should handle already capitalized words', () => {
    expect(fieldToDisplayName('ID')).toBe('ID');
    expect(fieldToDisplayName('URL')).toBe('URL');
  });
});

describe('analyzeField', () => {
  const sampleMetadata = [
    { pos: 'n', word: 'cat', count: 10 },
    { pos: 'n', word: 'dog', count: 20 },
    { pos: 'v', word: 'run', count: 15 },
    { pos: 'a', word: 'fast', count: 5 },
    { pos: 'n', word: 'tree', count: 30 },
  ];

  it('should count unique values for a field', () => {
    const result = analyzeField('pos', sampleMetadata);
    expect(result.uniqueCount).toBe(3); // n, v, a
    expect(result.values).toContain('n');
    expect(result.values).toContain('v');
    expect(result.values).toContain('a');
  });

  it('should handle fields with unique values per item', () => {
    const result = analyzeField('word', sampleMetadata);
    expect(result.uniqueCount).toBe(5); // All words are unique
  });

  it('should handle numeric fields', () => {
    const result = analyzeField('count', sampleMetadata);
    expect(result.uniqueCount).toBe(5); // 10, 20, 15, 5, 30
    expect(result.values).toContain('10'); // Converted to string
    expect(result.values).toContain('20');
  });

  it('should filter out null and undefined values', () => {
    const metadataWithNulls = [
      { status: 'active' },
      { status: null },
      { status: undefined },
      { status: 'inactive' },
      { status: '' }, // Empty string should also be filtered
    ];
    const result = analyzeField('status', metadataWithNulls);
    expect(result.uniqueCount).toBe(2); // active, inactive
  });

  it('should sort values alphabetically', () => {
    const result = analyzeField('pos', sampleMetadata);
    expect(result.values).toEqual(['a', 'n', 'v']); // Sorted
  });

  it('should respect sample size limit', () => {
    const largeMetadata = Array.from({ length: 5000 }, (_, i) => ({
      id: `item-${i}`,
      category: `cat-${i % 50}`, // 50 unique categories
    }));

    // With default sample size (1000), we might see fewer unique values
    const result = analyzeField('category', largeMetadata, 100);
    // With 100 samples, we'd see items 0-99, categories cat-0 to cat-49
    expect(result.uniqueCount).toBeLessThanOrEqual(50);
  });
});

describe('computeCategoryFieldOptions', () => {
  const sampleMetadata = [
    { pos: 'n', word: 'cat', source: 'wordnet' },
    { pos: 'n', word: 'dog', source: 'wordnet' },
    { pos: 'v', word: 'run', source: 'wordnet' },
    { pos: 'a', word: 'fast', source: 'wordnet' },
    { pos: 'r', word: 'quickly', source: 'wordnet' },
  ];

  it('should return fields with 2-100 unique values', () => {
    const options = computeCategoryFieldOptions(['pos', 'word', 'source'], sampleMetadata);

    // pos has 4 unique values (n, v, a, r) - should be included
    expect(options.find((o) => o.field === 'pos')).toBeDefined();

    // word has 5 unique values - should be included
    expect(options.find((o) => o.field === 'word')).toBeDefined();

    // source has only 1 unique value - should NOT be included
    expect(options.find((o) => o.field === 'source')).toBeUndefined();
  });

  it('should exclude system fields', () => {
    const metadataWithSystem = [
      { pos: 'n', row_index: 0, pca_2d: '[1,2]', source_split: 'train' },
      { pos: 'v', row_index: 1, pca_2d: '[2,3]', source_split: 'train' },
    ];

    const options = computeCategoryFieldOptions(
      ['pos', 'row_index', 'pca_2d', 'source_split'],
      metadataWithSystem
    );

    // System fields should be excluded
    expect(options.find((o) => o.field === 'row_index')).toBeUndefined();
    expect(options.find((o) => o.field === 'pca_2d')).toBeUndefined();
    expect(options.find((o) => o.field === 'source_split')).toBeUndefined();

    // pos should still be included
    expect(options.find((o) => o.field === 'pos')).toBeDefined();
  });

  it('should sort by unique count (fewer values first)', () => {
    const metadata = [
      { small: 'a', medium: 'x', large: '1' },
      { small: 'b', medium: 'y', large: '2' },
      { small: 'a', medium: 'z', large: '3' },
      { small: 'b', medium: 'x', large: '4' },
      { small: 'a', medium: 'y', large: '5' },
    ];

    const options = computeCategoryFieldOptions(['small', 'medium', 'large'], metadata);

    // small has 2 unique values, medium has 3, large has 5
    // Should be sorted: small, medium, large
    expect(options[0].field).toBe('small');
    expect(options[1].field).toBe('medium');
    expect(options[2].field).toBe('large');
  });
});

describe('detectDisplayConfig', () => {
  it('should prefer known label fields', () => {
    const metadata = [{ word: 'cat', title: 'A Cat', name: 'Fluffy' }];
    const config = detectDisplayConfig(['word', 'title', 'name'], metadata);

    // 'word' should be preferred as it comes first in KNOWN_LABEL_FIELDS
    expect(config.labelField).toBe('word');
  });

  it('should prefer known category fields', () => {
    const metadata = [
      { pos: 'n', category: 'animal', type: 'mammal' },
      { pos: 'n', category: 'animal', type: 'mammal' },
      { pos: 'v', category: 'action', type: 'movement' },
    ];

    const config = detectDisplayConfig(['pos', 'category', 'type'], metadata);

    // 'pos' should be preferred as it comes first in KNOWN_CATEGORY_FIELDS
    expect(config.categoryField).toBe('pos');
    expect(config.categoryValues).toContain('n');
    expect(config.categoryValues).toContain('v');
  });

  it('should skip category fields with only 1 unique value', () => {
    const metadata = [
      { pos: 'n', status: 'active' },
      { pos: 'v', status: 'active' },
      { pos: 'a', status: 'active' },
    ];

    const config = detectDisplayConfig(['pos', 'status'], metadata);

    // status has only 1 unique value, should not be selected
    expect(config.categoryField).toBe('pos');
    expect(config.categoryField).not.toBe('status');
  });

  it('should fallback to any suitable field if no known fields', () => {
    const metadata = [
      { custom_category: 'x', custom_label: 'item1' },
      { custom_category: 'y', custom_label: 'item2' },
      { custom_category: 'x', custom_label: 'item3' },
    ];

    const config = detectDisplayConfig(['custom_category', 'custom_label'], metadata);

    // Should find custom_category as it has 2 unique values
    expect(config.categoryField).toBe('custom_category');
    // No known label field, so labelField should be null
    expect(config.labelField).toBeNull();
  });

  it('should generate display name for category', () => {
    const metadata = [
      { topic_name: 'tech' },
      { topic_name: 'science' },
    ];

    const config = detectDisplayConfig(['topic_name'], metadata);

    expect(config.categoryField).toBe('topic_name');
    expect(config.categoryName).toBe('Topic Name'); // Title cased
  });

  it('should handle empty metadata', () => {
    const config = detectDisplayConfig(['pos', 'word'], []);

    // With no metadata to analyze, category detection falls back
    expect(config.labelField).toBe('word');
    expect(config.categoryField).toBeNull();
  });
});

describe('getFieldValues', () => {
  it('should return sorted unique values for a field', () => {
    const metadata = [
      { status: 'pending' },
      { status: 'active' },
      { status: 'pending' },
      { status: 'completed' },
    ];

    const values = getFieldValues('status', metadata);

    expect(values).toHaveLength(3);
    expect(values).toEqual(['active', 'completed', 'pending']); // Sorted
  });

  it('should handle missing field gracefully', () => {
    const metadata = [{ a: 1 }, { a: 2 }];
    const values = getFieldValues('nonexistent', metadata);

    expect(values).toHaveLength(0);
  });
});
