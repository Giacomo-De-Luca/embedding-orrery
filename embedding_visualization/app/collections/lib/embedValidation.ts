/** Pure validation + summary helpers for the embed flows (HF / local / re-embed). */

import type { DataType, PortionStrategy } from '@/lib/graphql/mutations';

export interface EmbedValidationInput {
  source: 'hf' | 'local-file' | 'reembed';
  /** HF only */
  datasetId?: string;
  /** local-file only */
  filePath?: string;
  collectionName: string;
  embeddingColumns: string[];
  /** local-file only; changes the column message for VECTOR mode */
  dataType?: DataType;
  portionStrategy?: PortionStrategy;
  rangeStart?: number;
  rangeEnd?: number;
}

/**
 * Returns the list of reasons the embed CTA should be disabled (empty when
 * ready). Messages are user-facing.
 */
export function getEmbedValidationIssues(input: EmbedValidationInput): string[] {
  const issues: string[] = [];

  if (input.source === 'hf' && !(input.datasetId ?? '').includes('/')) {
    issues.push('Dataset ID must be in the form org/dataset');
  }

  if (input.source === 'local-file') {
    if (!input.filePath) {
      issues.push('Provide a file path');
    } else if (!input.filePath.startsWith('/')) {
      issues.push('File path must be absolute (starting with /)');
    }
  }

  if (input.embeddingColumns.length === 0) {
    issues.push(
      input.dataType === 'VECTOR'
        ? 'Select a vector column'
        : 'Select at least one embedding column'
    );
  }

  if (!input.collectionName.trim()) {
    issues.push('Provide a collection name');
  }

  if (
    input.portionStrategy === 'ROW_RANGE' &&
    input.rangeStart !== undefined &&
    input.rangeEnd !== undefined &&
    input.rangeStart >= input.rangeEnd
  ) {
    issues.push('Row range start must be before end');
  }

  return issues;
}

export interface EmbedSummaryInput {
  collectionName: string;
  portionStrategy: PortionStrategy;
  numRows?: number;
  rangeStart?: number;
  rangeEnd?: number;
  totalRows?: number | null;
  modelName?: string;
  enableTopics?: boolean;
  dataType?: DataType;
}

/** One-line config recap shown next to the embed CTA. */
export function buildEmbedSummary(input: EmbedSummaryInput): string {
  const parts: string[] = [];
  parts.push(input.collectionName.trim() || 'unnamed collection');

  switch (input.portionStrategy) {
    case 'ALL':
      parts.push(
        input.totalRows ? `all ${input.totalRows.toLocaleString()} rows` : 'all rows'
      );
      break;
    case 'FIRST_N':
      parts.push(`first ${(input.numRows ?? 0).toLocaleString()} rows`);
      break;
    case 'RANDOM_SAMPLE':
      parts.push(`${(input.numRows ?? 0).toLocaleString()} sampled rows`);
      break;
    case 'ROW_RANGE':
      parts.push(`rows ${input.rangeStart ?? 0}–${input.rangeEnd ?? 0}`);
      break;
  }

  if (input.dataType === 'VECTOR') {
    parts.push('pre-computed vectors');
  } else if (input.modelName) {
    // Short model name: drop the org prefix
    parts.push(input.modelName.split('/').pop() || input.modelName);
  }

  if (input.enableTopics) parts.push('topics on');

  return parts.join(' · ');
}
