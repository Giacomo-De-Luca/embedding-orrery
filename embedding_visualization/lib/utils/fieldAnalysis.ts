/**
 * Utilities for analyzing metadata fields and determining
 * which fields are suitable for categorization/coloring.
 */

import type { CategoryFieldOption, DisplayConfig } from '../types/types';

/**
 * Convert field name to human-readable display name (title case).
 */
export function fieldToDisplayName(field: string): string {
  if (field === 'pos') return 'Part of Speech';
  return field
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Analyze a metadata field to determine its unique values.
 */
export function analyzeField(
  fieldName: string,
  itemMetadata: Record<string, unknown>[],
  sampleSize: number = 1000
): { uniqueCount: number; values: string[] } {
  const sample = itemMetadata.slice(0, sampleSize);
  const uniqueValues = new Set<string>();

  for (const meta of sample) {
    const value = meta[fieldName];
    if (value !== null && value !== undefined && value !== '') {
      uniqueValues.add(String(value));
    }
  }

  return {
    uniqueCount: uniqueValues.size,
    values: Array.from(uniqueValues).sort(),
  };
}

/**
 * Fields to exclude from category consideration.
 */
const EXCLUDE_FIELDS = new Set([
  'row_index',
  'source_split',
  'source_file',
  'source_dataset',
  'pca_2d',
  'pca_3d',
  'umap_2d',
  'umap_3d',
]);

/**
 * Known good label field names (priority order).
 */
const KNOWN_LABEL_FIELDS = ['word', 'title', 'name', 'label'];

/**
 * Known good category field names (priority order).
 */
const KNOWN_CATEGORY_FIELDS = ['pos', 'category', 'type', 'class', 'topic'];

/**
 * Compute category field options with unique value counts.
 * Only returns fields suitable for coloring (2-100 unique values).
 */
export function computeCategoryFieldOptions(
  availableFields: string[],
  itemMetadata: Record<string, unknown>[]
): CategoryFieldOption[] {
  const options: CategoryFieldOption[] = [];

  for (const field of availableFields) {
    if (EXCLUDE_FIELDS.has(field)) continue;

    const analysis = analyzeField(field, itemMetadata);

    // Only include fields with 2-100 unique values (good for coloring)
    if (analysis.uniqueCount >= 2 && analysis.uniqueCount <= 100) {
      options.push({
        field,
        uniqueCount: analysis.uniqueCount,
        displayName: fieldToDisplayName(field),
      });
    }
  }

  // Sort by unique count (fewer values = cleaner visualization)
  return options.sort((a, b) => a.uniqueCount - b.uniqueCount);
}

/**
 * Auto-detect display configuration based on available metadata fields.
 *
 * Dynamically analyzes fields to find:
 * - Label field: field with high cardinality (likely unique per item)
 * - Category field: field with low cardinality (2-100 unique values)
 */
export function detectDisplayConfig(
  availableFields: string[],
  itemMetadata: Record<string, unknown>[]
): DisplayConfig {
  // Find label field - prefer known names
  let labelField: string | null = null;
  for (const field of KNOWN_LABEL_FIELDS) {
    if (availableFields.includes(field)) {
      labelField = field;
      break;
    }
  }

  // Find category field - prefer known names with valid cardinality
  let categoryField: string | null = null;
  let categoryValues: string[] = [];

  // First try known category fields
  for (const field of KNOWN_CATEGORY_FIELDS) {
    if (availableFields.includes(field)) {
      const analysis = analyzeField(field, itemMetadata);
      if (analysis.uniqueCount >= 2 && analysis.uniqueCount <= 100) {
        categoryField = field;
        categoryValues = analysis.values;
        break;
      }
    }
  }

  // If no known field found, find any suitable field
  if (!categoryField && itemMetadata.length > 0) {
    const candidates = computeCategoryFieldOptions(availableFields, itemMetadata);
    if (candidates.length > 0) {
      categoryField = candidates[0].field;
      const analysis = analyzeField(categoryField, itemMetadata);
      categoryValues = analysis.values;
    }
  }

  return {
    labelField,
    categoryField,
    categoryValues,
    categoryName: categoryField ? fieldToDisplayName(categoryField) : 'Category',
  };
}

/**
 * Get unique values for a specific field from item metadata.
 */
export function getFieldValues(
  field: string,
  itemMetadata: Record<string, unknown>[]
): string[] {
  const analysis = analyzeField(field, itemMetadata);
  return analysis.values;
}

/**
 * Unified color field option with proper type detection.
 */
export interface ColorFieldOption {
  field: string;
  displayName: string;
  valueType: 'string' | 'numeric' | 'mixed';
  uniqueCount: number;
  recommendedScale: 'categorical' | 'sequential';
}

/**
 * Analyze all fields with proper type detection for coloring.
 *
 * Logic:
 * - String fields → categorical
 * - Numeric fields with <20 unique values → categorical (treat as discrete)
 * - Numeric fields with ≥20 unique values → sequential (continuous)
 */
export function analyzeColorFields(
  availableFields: string[],
  itemMetadata: Record<string, unknown>[],
  sampleSize: number = 500
): ColorFieldOption[] {
  if (itemMetadata.length === 0) return [];

  const sample = itemMetadata.slice(0, Math.min(sampleSize, itemMetadata.length));
  const results: ColorFieldOption[] = [];

  for (const field of availableFields) {
    if (EXCLUDE_FIELDS.has(field)) continue;

    let numericCount = 0;
    let stringCount = 0;
    const uniqueNumbers = new Set<number>();
    const uniqueStrings = new Set<string>();

    for (const meta of sample) {
      const value = meta[field];
      if (value === null || value === undefined || value === '') continue;

      if (typeof value === 'number' && !isNaN(value)) {
        numericCount++;
        uniqueNumbers.add(value);
      } else if (typeof value === 'string') {
        // Try to parse as number
        const parsed = parseFloat(value);
        if (!isNaN(parsed) && isFinite(parsed)) {
          numericCount++;
          uniqueNumbers.add(parsed);
        } else {
          stringCount++;
          uniqueStrings.add(value);
        }
      } else {
        // Treat other types as strings
        stringCount++;
        uniqueStrings.add(String(value));
      }
    }

    const totalValues = numericCount + stringCount;
    if (totalValues === 0) continue;

    // Determine value type
    let valueType: 'string' | 'numeric' | 'mixed';
    let uniqueCount: number;

    if (numericCount > 0 && stringCount === 0) {
      valueType = 'numeric';
      uniqueCount = uniqueNumbers.size;
    } else if (stringCount > 0 && numericCount === 0) {
      valueType = 'string';
      uniqueCount = uniqueStrings.size;
    } else {
      valueType = 'mixed';
      // For mixed, count all unique values
      uniqueCount = uniqueNumbers.size + uniqueStrings.size;
    }

    // Skip fields with too many unique values for categorical, but include for sequential
    // Skip fields with only 1 unique value (no variation to show)
    if (uniqueCount < 2) continue;

    // Determine recommended scale
    let recommendedScale: 'categorical' | 'sequential';
    if (valueType === 'string' || valueType === 'mixed') {
      // String/mixed fields: categorical if reasonable count, otherwise skip
      if (uniqueCount > 100) continue;
      recommendedScale = 'categorical';
    } else {
      // Numeric fields: categorical if <20 unique, sequential otherwise
      recommendedScale = uniqueCount < 20 ? 'categorical' : 'sequential';
    }

    results.push({
      field,
      displayName: fieldToDisplayName(field),
      valueType,
      uniqueCount,
      recommendedScale,
    });
  }

  // Sort: categorical fields first (by unique count), then sequential
  return results.sort((a, b) => {
    if (a.recommendedScale !== b.recommendedScale) {
      return a.recommendedScale === 'categorical' ? -1 : 1;
    }
    return a.uniqueCount - b.uniqueCount;
  });
}
