import { hsl } from 'd3-color';
import type { Point2D, Point3D, NestedColorMap } from '../types/types';
import { generateCategoryColors } from './categoryColors';

/**
 * Check whether nested color mode is available for the current data.
 * Returns true when colorByField is 'topic_label' and subtopic_label exists in points.
 */
export function isNestedColorAvailable(
  points: (Point2D | Point3D)[],
  colorByField: string | null | undefined
): boolean {
  if (colorByField !== 'topic_label' || points.length === 0) return false;
  // Check if at least one point has a subtopic_label
  return points.some(p => {
    const sub = p.metadata?.['subtopic_label'];
    return sub !== null && sub !== undefined && sub !== '';
  });
}

/**
 * Build a nested color map: topics get base hues, subtopics get lightness variations.
 *
 * 1. Groups subtopic_label values by their co-occurring topic_label
 * 2. Assigns base hues per topic via generateCategoryColors()
 * 3. For each topic's subtopics, generates HSL lightness variations (30%-72%)
 * 4. Noise topic ("Unclustered") stays gray
 */
export function buildNestedColorMap(
  points: (Point2D | Point3D)[],
  palette?: string
): NestedColorMap {
  // Build hierarchy: topic → Set<subtopic>, and counts
  const hierarchySet: Record<string, Set<string>> = {};
  const topicCounts: Record<string, number> = {};
  const subtopicCounts: Record<string, number> = {};

  for (const p of points) {
    const topic = String(p.metadata?.['topic_label'] ?? 'unknown');
    const subtopic = String(p.metadata?.['subtopic_label'] ?? topic);

    if (!hierarchySet[topic]) hierarchySet[topic] = new Set();
    hierarchySet[topic].add(subtopic);

    topicCounts[topic] = (topicCounts[topic] ?? 0) + 1;
    subtopicCounts[subtopic] = (subtopicCounts[subtopic] ?? 0) + 1;
  }

  // Sort topics, but put Unclustered last
  const topics = Object.keys(hierarchySet).sort((a, b) => {
    if (a === 'Unclustered') return 1;
    if (b === 'Unclustered') return -1;
    return a.localeCompare(b);
  });

  const hierarchy: Record<string, string[]> = {};
  for (const topic of topics) {
    hierarchy[topic] = Array.from(hierarchySet[topic]).sort();
  }

  // Generate base topic colors (exclude Unclustered from hue assignment)
  const colorTopics = topics.filter(t => t !== 'Unclustered');
  const baseColors = generateCategoryColors(colorTopics.length, palette);

  const topicColors: Record<string, string> = {};
  const subtopicColors: Record<string, string> = {};

  // Assign Unclustered = gray
  if (hierarchy['Unclustered']) {
    topicColors['Unclustered'] = '#7f7f7f';
    for (const sub of hierarchy['Unclustered']) {
      subtopicColors[sub] = '#7f7f7f';
    }
  }

  // Assign hue-based colors for real topics
  colorTopics.forEach((topic, i) => {
    const baseHex = baseColors[i];
    topicColors[topic] = baseHex;

    const subs = hierarchy[topic];
    if (subs.length === 1) {
      // Single subtopic gets the base color
      subtopicColors[subs[0]] = baseHex;
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
        subtopicColors[sub] = subHSL.formatHex();
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
