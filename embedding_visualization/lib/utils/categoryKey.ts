import type { MetadataPoint } from '../types/types';

/** Joins a topic and a subtopic into one nested-colouring key (a control character no label contains). */
const SUBTOPIC_KEY_SEPARATOR = '\u001f';

/** Sorted category values of a colour-by field and the point count of each. */
export interface CategoryData {
  categoryValues: string[];
  categoryCounts: Record<string, number>;
}

/**
 * Grouping identity of a categorical metadata value.
 *
 * Absent, null and empty values share the reserved {@link CategoryKey.MISSING}
 * key, so scatter plots, legend and analytics group, colour, count
 * and mute them as one "Missing" category instead of each view deciding on its
 * own what an absent value is.
 */
export class CategoryKey {
  /** Reserved key for absent, null or empty values; displayed as {@link CategoryKey.MISSING_LABEL}. */
  static readonly MISSING = '__missing__';
  static readonly MISSING_LABEL = 'Missing values';
  /**
   * The item fields a topic extraction contributes (merged from the active
   * extraction at query time, never stored in item metadata).
   */
  static readonly TOPIC_FIELDS = ['topic_id', 'topic_label', 'subtopic_id', 'subtopic_label'] as const;

  static isMissing(raw: unknown): boolean {
    return raw === null || raw === undefined || raw === '';
  }

  static of(raw: unknown): string {
    return CategoryKey.isMissing(raw) ? CategoryKey.MISSING : String(raw);
  }

  /** Topic and subtopic keys for nested colouring; a missing subtopic falls back to its topic. */
  static nested(metadata: Record<string, unknown> | undefined): { topic: string; subtopic: string } {
    const topic = CategoryKey.of(metadata?.topic_label);
    const subtopic = metadata?.subtopic_label;
    return { topic, subtopic: CategoryKey.isMissing(subtopic) ? topic : String(subtopic) };
  }

  /**
   * Key of a subtopic within its topic, for nested colouring: one subtopic
   * label under two topics stays two groups (noise a density cut absorbed into
   * a topic keeps the "Unclustered" subtopic, and must not recolour real noise).
   */
  static subtopicKey(topic: string, subtopic: string): string {
    return `${topic}${SUBTOPIC_KEY_SEPARATOR}${subtopic}`;
  }

  /** Inverse of {@link CategoryKey.subtopicKey}. */
  static splitSubtopicKey(key: string): { topic: string; subtopic: string } {
    const at = key.indexOf(SUBTOPIC_KEY_SEPARATOR);
    return { topic: key.slice(0, at), subtopic: key.slice(at + SUBTOPIC_KEY_SEPARATOR.length) };
  }

  /** {@link CategoryKey.subtopicKey} of a point's nested topic and subtopic. */
  static nestedKey(metadata: Record<string, unknown> | undefined): string {
    const { topic, subtopic } = CategoryKey.nested(metadata);
    return CategoryKey.subtopicKey(topic, subtopic);
  }

  /** Opacity multiplier for a category group: `missingOpacity` for Missing, 1 otherwise. */
  static missingFade(key: string, missingOpacity: number): number {
    return key === CategoryKey.MISSING ? missingOpacity : 1;
  }

  /** Comparator placing the Missing key after every other key; other pairs compare equal. */
  static missingLast(a: string, b: string): number {
    return Number(a === CategoryKey.MISSING) - Number(b === CategoryKey.MISSING);
  }

  /** Plain string order (palette slots follow it), with Missing last. */
  static sort(values: Iterable<string>): string[] {
    return [...values].sort((a, b) => CategoryKey.missingLast(a, b) || (a < b ? -1 : a > b ? 1 : 0));
  }

  /** Sorted category keys of `field` over metadata records (the order {@link CategoryKey.count} lists). */
  static valuesOf(records: Iterable<Record<string, unknown> | undefined>, field: string): string[] {
    const keys = new Set<string>();
    for (const record of records) keys.add(CategoryKey.of(record?.[field]));
    return CategoryKey.sort(keys);
  }

  /** Values and counts of `field` over `points`, skipping the `exclude`d point indices. */
  static count(
    points: readonly MetadataPoint[],
    field: string | null | undefined,
    exclude?: ReadonlySet<number> | null,
  ): CategoryData {
    if (!field) return { categoryValues: [], categoryCounts: {} };
    const counts: Record<string, number> = {};
    for (const point of points) {
      if (exclude?.has(point.index)) continue;
      const key = CategoryKey.of(point.metadata?.[field]);
      counts[key] = (counts[key] ?? 0) + 1;
    }
    return { categoryValues: CategoryKey.sort(Object.keys(counts)), categoryCounts: counts };
  }
}
