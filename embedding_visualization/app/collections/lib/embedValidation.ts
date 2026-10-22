/** Pure validation + summary helpers for the embed flows (HF / local / re-embed). */

import type { DataType, MediaOptionsInput, PortionStrategy } from '@/lib/graphql/mutations';
import { DATA_TYPE_LABELS, type MediaKindCounts } from './mediaColumns';
import type { MediaOptionsBudget } from './mediaOptions';

/** What the Image / Audio / Video / Mixed (MULTIMODAL) checks need. */
export interface MediaValidationInput {
  /** The selected image / audio / video columns. */
  columns: string[];
  /** Kinds of the selected columns. */
  kinds: MediaKindCounts;
  /** Limits and token estimate of the media options; null until the settings load. */
  budget: MediaOptionsBudget | null;
  /** Why the media settings failed to load (null while loading or once loaded). */
  settingsError?: string | null;
  /** The changed media options. */
  overrides: MediaOptionsInput;
  /** The model in effect, for the messages. */
  modelName: string;
  /**
   * What the model does not embed of the job's text and media kinds
   * (`ModelSuggestions.missingModalities`; [] for models outside the catalog,
   * which the backend checks when they load).
   */
  missingModalities: string[];
}

/** "image", "image or video", "image, audio or video". */
function orList(items: string[]): string {
  return items.length <= 2 ? items.join(' or ') : `${items.slice(0, -1).join(', ')} or ${items[items.length - 1]}`;
}

function mediaIssues(media: MediaValidationInput): string[] {
  if (media.columns.length === 0) return ['Select at least one image, audio or video column'];
  const issues: string[] = [];
  if (media.missingModalities.length > 0) {
    const example = media.budget ? `, such as ${media.budget.settings.defaultModel}` : '';
    issues.push(`${media.modelName} does not embed ${orList(media.missingModalities)}: pick a model that does${example}`);
  }
  if (!media.budget) {
    return [
      ...issues,
      media.settingsError ? `Could not load the media settings: ${media.settingsError}` : 'Loading the media settings',
    ];
  }
  return [...issues, ...media.budget.issues(media.budget.effective(media.overrides), media.kinds)];
}

export interface EmbedValidationInput {
  source: 'hf' | 'local-file' | 'reembed';
  /** HF only */
  datasetId?: string;
  /** local-file only */
  filePath?: string;
  collectionName: string;
  /** Text columns (optional in MULTIMODAL mode) */
  embeddingColumns: string[];
  /** Changes the column checks: VECTOR takes a vector column, MULTIMODAL media columns (`media`) */
  dataType?: DataType;
  /** MULTIMODAL only */
  media?: MediaValidationInput;
  portionStrategy?: PortionStrategy;
  rangeStart?: number;
  rangeEnd?: number;
  /** Text template / task prefix problems the backend would reject (`placeholderIssues`). */
  placeholderIssues?: string[];
  /** The task-prefix table is still loading: the input could miss the model's prefixes. */
  taskPrefixesLoading?: boolean;
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

  if (input.dataType === 'MULTIMODAL') {
    if (input.media) issues.push(...mediaIssues(input.media));
  } else if (input.embeddingColumns.length === 0) {
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

  issues.push(...(input.placeholderIssues ?? []));
  if (input.taskPrefixesLoading) {
    issues.push('Loading the task prefixes');
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
  } else {
    if (input.dataType === 'MULTIMODAL') parts.push(DATA_TYPE_LABELS.MULTIMODAL);
    // Short model name: drop the org prefix
    if (input.modelName) parts.push(input.modelName.split('/').pop() || input.modelName);
  }

  if (input.enableTopics) parts.push('topics on');

  return parts.join(' · ');
}
