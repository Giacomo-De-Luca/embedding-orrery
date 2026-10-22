import { describe, expect, it } from 'vitest';
import { buildNestedColorMap, isNestedColorAvailable } from '../nestedColorUtils';
import { buildCategoryColorMap, generateCategoryColors, MISSING_CATEGORY_COLOR, UNCLUSTERED_COLOR } from '../categoryColors';
import { CategoryKey } from '../categoryKey';

const { MISSING } = CategoryKey;
const key = CategoryKey.subtopicKey;

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
    expect(map.subtopicCounts[key('Science', 'Science')]).toBe(1);
  });

  it('gives Missing the fixed neutral color and keeps real topic hues unchanged', () => {
    const [arts, science] = generateCategoryColors(2);
    expect(map.topicColors[MISSING]).toBe(MISSING_CATEGORY_COLOR);
    expect(map.subtopicColors[key(MISSING, MISSING)]).toBe(MISSING_CATEGORY_COLOR);
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

describe('buildNestedColorMap keys subtopics by (topic, subtopic)', () => {
  // A density preview absorbed some noise into "Ten +1": that noise keeps the
  // "Unclustered" subtopic label under a real topic.
  const preview = [
    { index: 0, metadata: { topic_label: 'Ten +1', subtopic_label: 'Ten' } },
    { index: 1, metadata: { topic_label: 'Ten +1', subtopic_label: 'Eleven' } },
    { index: 2, metadata: { topic_label: 'Ten +1', subtopic_label: 'Unclustered' } },
    { index: 3, metadata: { topic_label: 'Unclustered', subtopic_label: 'Unclustered' } },
    { index: 4, metadata: { topic_label: 'Unclustered', subtopic_label: 'Unclustered' } },
  ];
  const map = buildNestedColorMap(preview);

  it('never recolours real noise with an absorbed subtopic', () => {
    expect(map.subtopicColors[key('Unclustered', 'Unclustered')]).toBe(UNCLUSTERED_COLOR);
    expect(map.subtopicColors[key('Ten +1', 'Unclustered')]).not.toBe(UNCLUSTERED_COLOR);
  });

  it('counts the same subtopic label separately under each topic', () => {
    expect(map.subtopicCounts[key('Ten +1', 'Unclustered')]).toBe(1);
    expect(map.subtopicCounts[key('Unclustered', 'Unclustered')]).toBe(2);
    expect(map.hierarchy['Ten +1']).toEqual(['Eleven', 'Ten', 'Unclustered']);
  });
});

describe('buildNestedColorMap with a colour pin', () => {
  const points = [
    { index: 0, metadata: { topic_label: 'Gamma', subtopic_label: 'Gamma' } },
    { index: 1, metadata: { topic_label: 'Alpha +1', subtopic_label: 'Alpha' } },
    { index: 2, metadata: { topic_label: 'Alpha +1', subtopic_label: 'Beta' } },
  ];
  const domain = ['Alpha', 'Beta', 'Gamma'];
  const pin = { sourceField: 'topic_label', domain, valueMap: { 'Alpha +1': 'Alpha' } };

  it("gives preview groups their representative's saved hue", () => {
    const saved = buildCategoryColorMap('topic_label', domain);
    const map = buildNestedColorMap(points, undefined, undefined, { pin });
    expect(map.topicColors['Alpha +1']).toBe(saved.Alpha);
    expect(map.topicColors.Gamma).toBe(saved.Gamma);
    expect(map.subtopicColors[key('Gamma', 'Gamma')]).toBe(saved.Gamma);
  });

  it("honours the representative's saved override", () => {
    const map = buildNestedColorMap(points, undefined, undefined, { pin, sourceOverrides: { Alpha: '#123456' } });
    expect(map.topicColors['Alpha +1']).toBe('#123456');
  });
});
