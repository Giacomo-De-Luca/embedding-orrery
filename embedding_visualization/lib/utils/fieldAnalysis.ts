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
 * Result of analyzing a metadata field.
 */
export interface FieldAnalysisResult {
  uniqueCount: number;       // Infinity if >maxUnique
  values: string[];          // Empty if exceeds max
  min?: number;              // For numeric fields
  max?: number;              // For numeric fields
  isNumeric: boolean;
}

/**
 * Analyze a metadata field to determine its unique values.
 * Scans ALL items but terminates early when unique count exceeds maxUnique.
 * Also tracks min/max for numeric fields (needed for gradient legends).
 */
export function analyzeField(
  fieldName: string,
  itemMetadata: Record<string, unknown>[],
  maxUnique: number = 100
): FieldAnalysisResult {
  const uniqueValues = new Set<string>();
  let exceedsMax = false;
  let min: number | undefined;
  let max: number | undefined;
  let hasNumeric = false;
  let hasNonNumeric = false;

  for (const meta of itemMetadata) {
    const value = meta[fieldName];
    if (value === null || value === undefined || value === '') continue;

    // Track numeric range
    let numericValue: number | null = null;
    if (typeof value === 'number' && !isNaN(value)) {
      numericValue = value;
      hasNumeric = true;
    } else if (typeof value === 'string') {
      const parsed = parseFloat(value);
      if (!isNaN(parsed) && isFinite(parsed)) {
        numericValue = parsed;
        hasNumeric = true;
      } else {
        hasNonNumeric = true;
      }
    } else {
      hasNonNumeric = true;
    }

    // Update min/max for numeric values
    if (numericValue !== null) {
      if (min === undefined || numericValue < min) min = numericValue;
      if (max === undefined || numericValue > max) max = numericValue;
    }

    // Track unique values (with early termination)
    if (!exceedsMax) {
      uniqueValues.add(String(value));
      if (uniqueValues.size > maxUnique) {
        exceedsMax = true;
        // Continue scanning to get full min/max range
      }
    }
  }

  return {
    uniqueCount: exceedsMax ? Infinity : uniqueValues.size,
    values: exceedsMax ? [] : Array.from(uniqueValues).sort(),
    min,
    max,
    isNumeric: hasNumeric && !hasNonNumeric,
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

    const analysis = analyzeField(field, itemMetadata, 100);

    // Only include fields with 2-100 unique values (good for coloring)
    // Skip fields that exceed the max unique count (analysis.uniqueCount === Infinity)
    if (analysis.uniqueCount >= 2 && analysis.uniqueCount !== Infinity) {
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
      const analysis = analyzeField(field, itemMetadata, 100);
      if (analysis.uniqueCount >= 2 && analysis.uniqueCount !== Infinity) {
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
      const analysis = analyzeField(categoryField, itemMetadata, 100);
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
 * Returns empty array if field has >100 unique values.
 */
export function getFieldValues(
  field: string,
  itemMetadata: Record<string, unknown>[],
  maxUnique: number = 100
): string[] {
  const analysis = analyzeField(field, itemMetadata, maxUnique);
  return analysis.values;
}

/**
 * Unified color field option with proper type detection.
 */
export interface ColorFieldOption {
  field: string;
  displayName: string;
  valueType: 'string' | 'numeric' | 'mixed';
  uniqueCount: number;      // Infinity if >100
  recommendedScale: 'categorical' | 'sequential';
  min?: number;             // For numeric fields (needed for gradient legend)
  max?: number;             // For numeric fields (needed for gradient legend)
}

/**
 * Analyze all fields with proper type detection for coloring.
 * Scans ALL items (no sampling) with early termination at >100 unique values.
 *
 * Logic:
 * - String fields with ≤100 unique → categorical
 * - String fields with >100 unique → excluded (not useful for visualization)
 * - Numeric fields with <20 unique values → categorical (treat as discrete)
 * - Numeric fields with ≥20 unique values → sequential (continuous)
 */
export function analyzeColorFields(
  availableFields: string[],
  itemMetadata: Record<string, unknown>[]
): ColorFieldOption[] {
  if (itemMetadata.length === 0) return [];

  const results: ColorFieldOption[] = [];

  for (const field of availableFields) {
    if (EXCLUDE_FIELDS.has(field)) continue;

    // Use the updated analyzeField that scans all items with early termination
    const analysis = analyzeField(field, itemMetadata, 100);

    // Skip fields with only 1 or 0 unique values (no variation to show)
    if (analysis.uniqueCount < 2) continue;

    // Determine value type based on analysis
    const valueType: 'string' | 'numeric' | 'mixed' = analysis.isNumeric ? 'numeric' : 'string';

    // Determine recommended scale
    let recommendedScale: 'categorical' | 'sequential';
    if (valueType === 'string') {
      recommendedScale = 'categorical';
    } else {
      // Numeric fields: categorical if <20 unique, sequential otherwise
      recommendedScale = analysis.uniqueCount < 20 ? 'categorical' : 'sequential';
    }

    results.push({
      field,
      displayName: fieldToDisplayName(field),
      valueType,
      uniqueCount: analysis.uniqueCount,
      recommendedScale,
      min: analysis.min,
      max: analysis.max,
    });
  }

  // Sort: categorical fields first (by unique count), then sequential
  return results.sort((a, b) => {
    if (a.recommendedScale !== b.recommendedScale) {
      return a.recommendedScale === 'categorical' ? -1 : 1;
    }
    // For sequential fields (uniqueCount could be Infinity), sort by field name
    if (a.uniqueCount === Infinity && b.uniqueCount === Infinity) {
      return a.field.localeCompare(b.field);
    }
    return (a.uniqueCount === Infinity ? 101 : a.uniqueCount) -
           (b.uniqueCount === Infinity ? 101 : b.uniqueCount);
  });
}
