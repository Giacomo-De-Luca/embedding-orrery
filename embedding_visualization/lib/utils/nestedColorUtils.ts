import { hsl } from 'd3-color';
import type { MetadataPoint, NestedColorMap } from '../types/types';
import {
  buildPinnedCategoryColorMap,
  generateCategoryColors,
  MISSING_CATEGORY_COLOR,
  UNCLUSTERED_COLOR,
  type CategoryColorPin,
} from './categoryColors';
import { CategoryKey } from './categoryKey';

/** Topics with a fixed color, listed in this order after every real topic. */
const FIXED_TOPIC_COLORS = new Map<string, string>([
  ['Unclustered', UNCLUSTERED_COLOR],
  [CategoryKey.MISSING, MISSING_CATEGORY_COLOR],
]);
const FIXED_TOPICS = [...FIXED_TOPIC_COLORS.keys()];

/**
 * Check whether nested color mode is available for the current data.
 * Returns true when colorByField is 'topic_label' and subtopic_label exists in points.
 */
export function isNestedColorAvailable(
  points: MetadataPoint[],
  colorByField: string | null | undefined
): boolean {
  if (colorByField !== 'topic_label' || points.length === 0) return false;
  // Check if at least one point has a subtopic_label
  return points.some(p => !CategoryKey.isMissing(p.metadata?.['subtopic_label']));
}

/** A colour pin for the topic level (the topic-tree preview) and its source field's overrides. */
export interface NestedTopicPinning {
  pin: CategoryColorPin;
  sourceOverrides?: Record<string, string>;
}

/**
 * Build a nested color map: topics get base hues, subtopics get lightness variations.
 *
 * 1. Groups subtopic_label values by their co-occurring topic_label
 * 2. Assigns base hues per topic via generateCategoryColors(), or through
 *    `pinning` (each preview group takes its representative's saved colour)
 * 3. For each topic's subtopics, generates HSL lightness variations (30%-72%)
 * 4. Noise topic ("Unclustered") and points without a topic (Missing) keep fixed colors
 *
 * Subtopic colours and counts are keyed by `CategoryKey.subtopicKey(topic,
 * subtopic)`: noise a density cut absorbed into a topic keeps its
 * "Unclustered" subtopic label and must not recolour the real noise.
 */
export function buildNestedColorMap(
  points: MetadataPoint[],
  palette?: string,
  topicOverrides?: Record<string, string>,
  pinning?: NestedTopicPinning,
): NestedColorMap {
  // Build hierarchy: topic → Set<subtopic>, and counts
  const hierarchySet: Record<string, Set<string>> = {};
  const topicCounts: Record<string, number> = {};
  const subtopicCounts: Record<string, number> = {};

  for (const p of points) {
    const { topic, subtopic } = CategoryKey.nested(p.metadata);

    if (!hierarchySet[topic]) hierarchySet[topic] = new Set();
    hierarchySet[topic].add(subtopic);

    topicCounts[topic] = (topicCounts[topic] ?? 0) + 1;
    const key = CategoryKey.subtopicKey(topic, subtopic);
    subtopicCounts[key] = (subtopicCounts[key] ?? 0) + 1;
  }

  // Sort topics, but put the fixed-color topics (Unclustered, Missing) last
  const fixedRank = (topic: string) => FIXED_TOPICS.indexOf(topic);
  const topics = Object.keys(hierarchySet).sort((a, b) => fixedRank(a) - fixedRank(b) || a.localeCompare(b));

  const hierarchy: Record<string, string[]> = {};
  for (const topic of topics) {
    hierarchy[topic] = Array.from(hierarchySet[topic]).sort();
  }

  // Generate base topic colors (fixed-color topics take no hue)
  const colorTopics = topics.filter(t => !FIXED_TOPIC_COLORS.has(t));
  const pinned = pinning && buildPinnedCategoryColorMap(
    'topic_label', pinning.pin, colorTopics, palette, pinning.sourceOverrides, topicOverrides,
  );
  const baseColors = pinned
    ? colorTopics.map((topic) => pinned[topic])
    : generateCategoryColors(colorTopics.length, palette);

  const topicColors: Record<string, string> = {};
  const subtopicColors: Record<string, string> = {};

  // Fixed-color topics color all of their subtopics alike
  for (const [topic, color] of FIXED_TOPIC_COLORS) {
    if (!hierarchy[topic]) continue;
    topicColors[topic] = color;
    for (const sub of hierarchy[topic]) {
      subtopicColors[CategoryKey.subtopicKey(topic, sub)] = color;
    }
  }

  // Assign hue-based colors for real topics
  colorTopics.forEach((topic, i) => {
    const baseHex = topicOverrides?.[topic] ?? baseColors[i];
    topicColors[topic] = baseHex;

    const subs = hierarchy[topic];
    if (subs.length === 1) {
      // Single subtopic gets the base color
      subtopicColors[CategoryKey.subtopicKey(topic, subs[0])] = baseHex;
    } else {
      // Multiple subtopics: vary lightness around the base hue
      const baseHSL = hsl(baseHex);
      const h = baseHSL.h;
      const s = baseHSL.s;

      subs.forEach((sub, j) => {
        // Spread lightness from 0.30 to 0.72
        const t = subs.length > 1 ? j / (subs.length - 1) : 0.5;
        const l = 0.30 + t * 0.42;
        const subHSL = hsl(h, s, l);
        subtopicColors[CategoryKey.subtopicKey(topic, sub)] = subHSL.formatHex();
      });
    }
  });

  return {
    subtopicColors,
    topicColors,
    hierarchy,
    topicCounts,
    subtopicCounts,
  };
}
