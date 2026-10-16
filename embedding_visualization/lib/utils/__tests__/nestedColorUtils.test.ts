import { describe, expect, it } from 'vitest';
import { buildNestedColorMap, isNestedColorAvailable } from '../nestedColorUtils';
import { generateCategoryColors, MISSING_CATEGORY_COLOR, UNCLUSTERED_COLOR } from '../categoryColors';
import { CategoryKey } from '../categoryKey';

const { MISSING } = CategoryKey;

const points = [
  { index: 0, metadata: { topic_label: 'Arts', subtopic_label: 'Poetry' } },
  { index: 1, metadata: { topic_label: 'Arts', subtopic_label: 'Music' } },
  { index: 2, metadata: { topic_label: 'Science' } },
  { index: 3, metadata: {} },
  { index: 4, metadata: { topic_label: 'Unclustered', subtopic_label: '' } },
];

describe('buildNestedColorMap with missing values', () => {
  const map = buildNestedColorMap(points);

  it('groups points without a topic under Missing, listed after Unclustered', () => {
    expect(Object.keys(map.hierarchy)).toEqual(['Arts', 'Science', 'Unclustered', MISSING]);
    expect(map.hierarchy[MISSING]).toEqual([MISSING]);
    expect(map.topicCounts[MISSING]).toBe(1);
  });

  it('falls back to the topic when a subtopic is missing', () => {
    expect(map.hierarchy.Science).toEqual(['Science']);
    expect(map.hierarchy.Unclustered).toEqual(['Unclustered']);
    expect(map.subtopicCounts.Science).toBe(1);
  });

  it('gives Missing the fixed neutral color and keeps real topic hues unchanged', () => {
    const [arts, science] = generateCategoryColors(2);
    expect(map.topicColors[MISSING]).toBe(MISSING_CATEGORY_COLOR);
    expect(map.subtopicColors[MISSING]).toBe(MISSING_CATEGORY_COLOR);
    expect(map.topicColors.Unclustered).toBe(UNCLUSTERED_COLOR);
    expect(map.topicColors.Arts).toBe(arts);
    expect(map.topicColors.Science).toBe(science);
  });
});

describe('isNestedColorAvailable', () => {
  it('requires at least one present subtopic on the topic field', () => {
    expect(isNestedColorAvailable(points, 'topic_label')).toBe(true);
    expect(isNestedColorAvailable(points.slice(2), 'topic_label')).toBe(false);
    expect(isNestedColorAvailable(points, 'category')).toBe(false);
  });
});
