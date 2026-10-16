import type { MetadataPoint } from '../types/types';
import { CategoryKey } from './categoryKey';

export interface HiddenPointOptions {
  /** Hide HDBSCAN noise points (topic_id -1 / topic_label "Unclustered"). */
  hideUnclustered: boolean;
  /** Hide points whose colour-by field has no value. */
  hideMissing: boolean;
}

/** Points removed from the scatter plots by the "Hide unclustered points" and "Hide missing values" toggles. */
export class HiddenPoints {
  static isUnclustered(metadata: Record<string, unknown> | undefined): boolean {
    const topicId = metadata?.topic_id;
    const topicLabel = metadata?.topic_label;
    return topicId === -1 || topicId === '-1' || topicLabel === 'Unclustered' || topicLabel === 'unclustered';
  }

  /** Points left visible; returns `points` itself when no toggle applies, keeping memo identity. */
  static filter<T extends MetadataPoint>(points: T[], field: string | null | undefined, options: HiddenPointOptions): T[] {
    const missingField = options.hideMissing ? field : null;
    if (!options.hideUnclustered && !missingField) return points;
    return points.filter(({ metadata }) => {
      if (options.hideUnclustered && HiddenPoints.isUnclustered(metadata)) return false;
      return !(missingField && CategoryKey.isMissing(metadata?.[missingField]));
    });
  }
}
