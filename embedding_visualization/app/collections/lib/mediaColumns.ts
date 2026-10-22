/**
 * Media columns of the Image / Audio / Video / Mixed embed mode (`MULTIMODAL`).
 * A column's kind comes from the source: a HuggingFace feature's dtype
 * ("image" / "audio" / "video") or a local file preview's `columnKinds`, which
 * the backend detects from the cells (magic bytes, extensions). Pure (no React).
 */
import type { DataType } from '@/lib/graphql/mutations';

export type MediaKind = 'image' | 'audio' | 'video';

export const MEDIA_KINDS: readonly MediaKind[] = ['image', 'audio', 'video'];

export const MEDIA_KIND_LABELS: Record<MediaKind, string> = {
  image: 'Image',
  audio: 'Audio',
  video: 'Video',
};

/** How many selected columns hold each kind (absent = none). */
export type MediaKindCounts = Partial<Record<MediaKind, number>>;

export const DATA_TYPE_LABELS: Record<DataType, string> = {
  TEXT: 'Text',
  MULTIMODAL: 'Image / Audio / Video / Mixed',
  VECTOR: 'Pre-computed vectors',
};

export interface ColumnInfo {
  name: string;
  dtype: string;
}

export class MediaColumns {
  static isKind(value: unknown): value is MediaKind {
    return typeof value === 'string' && (MEDIA_KINDS as readonly string[]).includes(value);
  }

  /**
   * The media columns of a source and their kinds: `columnKinds` (local
   * preview) when given, else the columns' dtypes (HuggingFace features).
   */
  static kindsOf(
    columns: readonly ColumnInfo[],
    columnKinds?: Record<string, unknown> | null,
  ): Record<string, MediaKind> {
    const kinds: Record<string, MediaKind> = {};
    for (const column of columns) {
      const kind = columnKinds !== undefined ? columnKinds?.[column.name] : column.dtype;
      if (MediaColumns.isKind(kind)) kinds[column.name] = kind;
    }
    return kinds;
  }

  static counts(selected: readonly string[], kinds: Record<string, MediaKind>): MediaKindCounts {
    const counts: MediaKindCounts = {};
    for (const column of selected) {
      const kind = kinds[column];
      if (kind) counts[kind] = (counts[kind] ?? 0) + 1;
    }
    return counts;
  }

  /** The kinds among the selected columns, in `MEDIA_KINDS` order. */
  static selectedKinds(selected: readonly string[], kinds: Record<string, MediaKind>): MediaKind[] {
    const counts = MediaColumns.counts(selected, kinds);
    return MEDIA_KINDS.filter((kind) => counts[kind]);
  }

  /** The columns that hold no media (text and metadata candidates). */
  static withoutMedia(columns: readonly string[], kinds: Record<string, MediaKind>): string[] {
    return columns.filter((column) => !kinds[column]);
  }

  /**
   * The first selection of a multimodal embed: every media column, no text
   * (it is optional), and every other column as metadata.
   */
  static defaultSelection(
    columns: readonly string[],
    kinds: Record<string, MediaKind>,
  ): { media: string[]; metadata: string[] } {
    return {
      media: columns.filter((column) => kinds[column]),
      metadata: MediaColumns.withoutMedia(columns, kinds),
    };
  }

  /**
   * A local file's columns with a type to show: the media kind from the
   * preview, else the JSON type of the first previewed value.
   */
  static localColumns(
    names: readonly string[],
    rows: readonly Record<string, unknown>[],
    columnKinds?: Record<string, unknown> | null,
  ): ColumnInfo[] {
    return names.map((name) => {
      const kind = columnKinds?.[name];
      if (MediaColumns.isKind(kind)) return { name, dtype: kind };
      const value = rows.map((row) => row[name]).find((v) => v !== null && v !== undefined);
      return { name, dtype: MediaColumns.valueType(value) };
    });
  }

  private static valueType(value: unknown): string {
    if (value === undefined) return 'unknown';
    if (Array.isArray(value)) return 'list';
    const type = typeof value;
    return type === 'string' || type === 'number' || type === 'boolean' || type === 'object' ? type : 'unknown';
  }

  /** Collections embedded from media (they cannot be re-embedded from their stored text). */
  static isMultimodalCollection(metadata?: Record<string, unknown> | null): boolean {
    return metadata?.data_type === 'multimodal';
  }
}
