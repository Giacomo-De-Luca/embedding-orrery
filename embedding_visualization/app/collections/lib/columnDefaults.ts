/**
 * The column selection an embed form starts with once a source's columns and
 * preview rows are known (HuggingFace and local files share it). Pure (no React).
 *
 * The text to embed is picked by meaning, not position:
 * 1. a document-like name (`text`, `abstract`, …, in TEXT_NAMES order) whose
 *    preview values are strings;
 * 2. else the column with the longest prose in the preview: at least half of
 *    its values hold two words of letters and are not structured, so ids,
 *    labels, urls, numbers, timestamps, lists, objects and vectors stored as
 *    text never qualify (nor does text written without spaces, such as
 *    Chinese: its column is picked by name or by hand);
 * 3. else nothing, and the form asks for a column.
 * Media columns and the id column are never picked as text.
 */
import type { DataType } from '@/lib/graphql/mutations';
import { MediaColumns, type MediaKind } from './mediaColumns';

export type TextPickReason = 'name' | 'length';

export interface TextPick {
  column: string;
  reason: TextPickReason;
}

export interface ColumnDefaultsInput {
  columns: string[];
  /** Preview rows ([] when the preview is unavailable: names alone decide). */
  rows: readonly Record<string, unknown>[];
  kinds: Record<string, MediaKind>;
  dataType: DataType;
}

export interface ColumnSelection {
  embedding: string[];
  metadata: string[];
  media: string[];
  template: string;
  /** A column whose name marks it as the id, or null (the form keeps its choice). */
  idColumn: string | null;
  /** How the text column was picked (TEXT only; null when nothing qualified). */
  textPick: TextPick | null;
}

/** A list, object or tuple as the previews stringify them (`[0.1, …]`, `{'a': …}`). */
const STRUCTURED = /^\s*[[{(]/;
/** Two words of letters, in any alphabet. */
const PROSE = /\p{L}\s+\p{L}/u;

export class ColumnDefaults {
  /** Names of document columns, most likely first (compared ignoring case). */
  static readonly TEXT_NAMES = [
    'text', 'document', 'content', 'body', 'abstract', 'article', 'passage', 'context',
    'review', 'description', 'summary', 'sentence', 'caption', 'question',
  ];
  static readonly ID_NAMES = ['id', 'index', 'idx', '_id', 'row_id', 'item_id', 'doc_id', 'feature_id'];
  /** Substrings of a pre-computed vector column's name. */
  static readonly VECTOR_NAMES = ['embedding', 'embeddings', 'vector', 'vectors', 'emb'];

  static select({ columns, rows, kinds, dataType }: ColumnDefaultsInput): ColumnSelection {
    const idColumn = ColumnDefaults.idColumn(columns);
    if (dataType === 'MULTIMODAL') {
      const { media, metadata } = MediaColumns.defaultSelection(columns, kinds);
      return { embedding: [], metadata, media, template: '', idColumn, textPick: null };
    }
    if (dataType === 'VECTOR') {
      const vector = ColumnDefaults.vectorColumn(columns);
      return {
        embedding: vector ? [vector] : [],
        metadata: columns.filter((column) => column !== vector),
        media: [],
        template: '',
        idColumn,
        textPick: null,
      };
    }
    const textPick = ColumnDefaults.textPick(MediaColumns.withoutMedia(columns, kinds), rows);
    const text = textPick?.column ?? null;
    return {
      embedding: text ? [text] : [],
      metadata: columns.filter((column) => column !== text),
      media: [],
      template: text ? `{${text}}` : '',
      idColumn,
      textPick,
    };
  }

  /** The text column to embed (see the module comment), or null. */
  static textPick(columns: readonly string[], rows: readonly Record<string, unknown>[]): TextPick | null {
    const id = ColumnDefaults.idColumn(columns);
    const candidates = columns.filter((column) => column !== id);
    for (const name of ColumnDefaults.TEXT_NAMES) {
      const column = candidates.find((c) => c.toLowerCase() === name);
      if (column && ColumnDefaults.holdsStrings(column, rows)) return { column, reason: 'name' };
    }
    let best: { column: string; length: number } | null = null;
    for (const column of candidates) {
      const values = ColumnDefaults.values(column, rows);
      const prose = values.filter(
        (v): v is string => typeof v === 'string' && !STRUCTURED.test(v) && PROSE.test(v),
      );
      if (prose.length === 0 || prose.length * 2 < values.length) continue;
      const length = prose.reduce((sum, v) => sum + v.length, 0) / prose.length;
      if (!best || length > best.length) best = { column, length };
    }
    return best ? { column: best.column, reason: 'length' } : null;
  }

  /** The first column named like an id, or null. */
  static idColumn(columns: readonly string[]): string | null {
    return columns.find((column) => ColumnDefaults.ID_NAMES.includes(column.toLowerCase())) ?? null;
  }

  /** A column named like a vector, else the first column (null without columns). */
  static vectorColumn(columns: readonly string[]): string | null {
    const named = columns.find((column) =>
      ColumnDefaults.VECTOR_NAMES.some((name) => column.toLowerCase().includes(name)),
    );
    return named ?? columns[0] ?? null;
  }

  /** Why the text column was pre-selected, while it is the whole selection (TEXT only). */
  static note(pick: TextPick | null, selected: readonly string[], dataType: DataType): string | null {
    if (!pick || dataType !== 'TEXT' || selected.length !== 1 || selected[0] !== pick.column) return null;
    return pick.reason === 'name'
      ? `Pre-selected "${pick.column}" by its name.`
      : `Pre-selected "${pick.column}": the longest text in the preview.`;
  }

  /** The column's non-empty preview values. */
  private static values(column: string, rows: readonly Record<string, unknown>[]): unknown[] {
    return rows.map((row) => row[column]).filter((v) => v !== null && v !== undefined && v !== '');
  }

  /** Every non-empty preview value is a string (true without values: the name decides). */
  private static holdsStrings(column: string, rows: readonly Record<string, unknown>[]): boolean {
    return ColumnDefaults.values(column, rows).every((v) => typeof v === 'string');
  }
}
