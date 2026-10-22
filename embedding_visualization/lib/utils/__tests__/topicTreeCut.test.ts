import { describe, expect, it } from 'vitest';
import { TopicMergeTree, TopicTreeCut, TopicTreeView } from '../topicTreeCut';
import type { TopicMergeTreeData } from '../../graphql/mutations';

// Same literal examples as interpretability_backend/unit_tests/test_topic_merge_tree.py
// (LITERAL_TREE and the test_literal_* cases), so the backend cut and the
// Explore topic slider cannot drift apart.
//
// Five base topics (ids 10..14, sizes 5, 3, 8, 2, 1):
//   row 0: leaves 0 + 1 -> node 5 at 0.1
//   row 1: leaves 3 + 4 -> node 6 at 0.2
//   row 2: leaf 2 + node 6 -> node 7 at 0.4
//   row 3: node 5 + node 7 -> node 8 at 0.9
const LITERAL_TREE = new TopicMergeTree({
  linkage: 'average',
  leafIds: [10, 11, 12, 13, 14],
  leafSizes: [5, 3, 8, 2, 1],
  children: [[0, 1], [3, 4], [2, 6], [5, 7]],
  heights: [0.1, 0.2, 0.4, 0.9],
  noiseMerges: new Map([['n-early', 0], ['n-mid', 1], ['n-root', 3]]),
});

describe('TopicTreeCut on the literal tree', () => {
  const cut = new TopicTreeCut(LITERAL_TREE);

  it.each([
    [5, [0, 1, 2, 3, 4]],
    [4, [5, 5, 2, 3, 4]],
    [3, [5, 5, 2, 6, 6]],
    [2, [5, 5, 7, 7, 7]],
    [1, [8, 8, 8, 8, 8]],
  ])('groups the leaves at N=%i', (nTopics, groups) => {
    expect(cut.groups(cut.mergesFor(nTopics))).toEqual(groups);
  });

  it.each([
    [0.05, 0],
    [0.1, 1],
    [0.25, 2],
    [5.0, 4],
  ])('counts the merges within height %f', (height, merges) => {
    expect(cut.mergesWithin(height)).toBe(merges);
  });

  it('represents each group by its largest leaf', () => {
    expect(cut.representatives(cut.groups(cut.mergesFor(2)))).toEqual(new Map([[5, 10], [7, 12]]));
  });

  it('breaks representative ties towards the lowest topic id', () => {
    const tree = new TopicMergeTree({
      linkage: 'average',
      leafIds: [4, 2],
      leafSizes: [3, 3],
      children: [[0, 1]],
      heights: [0.5],
    });
    const tied = new TopicTreeCut(tree);
    expect(tied.representatives(tied.groups(1))).toEqual(new Map([[2, 2]]));
  });

  it.each([
    [5, {}],
    [4, { 'n-early': 5 }],
    [3, { 'n-early': 5, 'n-mid': 6 }],
    [2, { 'n-early': 5, 'n-mid': 7 }],
    [1, { 'n-early': 8, 'n-mid': 8, 'n-root': 8 }],
  ])('absorbs the noise of applied merges at N=%i', (nTopics, absorbed) => {
    expect(Object.fromEntries(cut.absorbedNoise(cut.mergesFor(nTopics)))).toEqual(absorbed);
  });

  it.each([0, 6, -1])('rejects a topic count of %i', (nTopics) => {
    expect(() => cut.mergesFor(nTopics)).toThrow(/between 1 and 5/);
  });

  it('rejects a merge count outside the tree', () => {
    expect(() => cut.roots(5)).toThrow(/between 0 and 4/);
  });

  it('counts the documents under each node', () => {
    expect(LITERAL_TREE.nodeSizes()).toEqual([5, 3, 8, 2, 1, 8, 3, 11, 19]);
  });

  it('absorbs any keys mapped to merge rows', () => {
    expect(cut.absorbed(new Map([[7, 1], [9, 3]]), 2)).toEqual(new Map([[7, 6]]));
  });
});

describe('TopicMergeTree validation (mirrors TestTreeValidation)', () => {
  const base = { linkage: 'average', leafIds: [0, 1, 2], leafSizes: [1, 1, 1] };

  it('rejects a child made by a later row', () => {
    expect(() => new TopicMergeTree({ ...base, children: [[0, 4], [1, 2]], heights: [0.1, 0.2] }))
      .toThrow(/before/);
  });

  it('rejects decreasing heights', () => {
    expect(() => new TopicMergeTree({ ...base, children: [[0, 1], [2, 3]], heights: [0.5, 0.2] }))
      .toThrow(/non-decreasing/);
  });

  it('rejects a node merged twice', () => {
    expect(() => new TopicMergeTree({ ...base, children: [[0, 1], [0, 2]], heights: [0.1, 0.2] }))
      .toThrow(/once/);
  });

  it('rejects noise mapped to a missing merge row', () => {
    expect(() => new TopicMergeTree({
      ...base, children: [[0, 1], [2, 3]], heights: [0.1, 0.2], noiseMerges: new Map([['x', 2]]),
    })).toThrow(/missing merge row/);
  });

  it('keeps the merges when noise is attached', () => {
    const tree = LITERAL_TREE.withNoise(new Map([['x', 2]]));
    expect(Object.fromEntries(tree.noiseMerges)).toEqual({ x: 2 });
    expect(tree.children).toEqual(LITERAL_TREE.children);
  });
});

/** The literal tree as the `topicMergeTree` query returns it. */
function literalGraphQL(overrides: Partial<TopicMergeTreeData> = {}): TopicMergeTreeData {
  return {
    extractionId: 'ext-1',
    linkage: 'density',
    baseLevel: 'subtopic',
    leaves: [
      { topicId: 10, label: 'Ten', size: 5 },
      { topicId: 11, label: 'Eleven', size: 3 },
      { topicId: 12, label: 'Twelve', size: 8 },
      { topicId: 13, label: 'Thirteen', size: 2 },
      { topicId: 14, label: 'Fourteen', size: 1 },
    ],
    merges: [
      { left: 0, right: 1, height: 0.1, size: 8 },
      { left: 3, right: 4, height: 0.2, size: 3 },
      { left: 2, right: 6, height: 0.4, size: 11 },
      { left: 5, right: 7, height: 0.9, size: 19 },
    ],
    noiseCount: 3,
    availableLinkages: ['average', 'ward', 'density'],
    noiseItemIds: ['n-early', 'n-mid', 'n-root'],
    noiseMerges: [0, 1, 3],
    ...overrides,
  };
}

describe('TopicTreeView.fromGraphQL', () => {
  it('rebuilds the tree, labels and noise map of the query result', () => {
    const view = TopicTreeView.fromGraphQL(literalGraphQL());
    expect(view.extractionId).toBe('ext-1');
    expect(view.tree.linkage).toBe('density');
    expect(view.tree.leafIds).toEqual([10, 11, 12, 13, 14]);
    expect(view.tree.children).toEqual(LITERAL_TREE.children);
    expect(view.tree.heights).toEqual(LITERAL_TREE.heights);
    expect(view.tree.nodeSizes()).toEqual(LITERAL_TREE.nodeSizes());
    expect(view.labels.get(12)).toBe('Twelve');
    expect(view.noiseCount).toBe(3);
    expect(view.availableLinkages).toEqual(['average', 'ward', 'density']);
    expect(Object.fromEntries(view.tree.noiseMerges)).toEqual({ 'n-early': 0, 'n-mid': 1, 'n-root': 3 });
  });

  it('names the item fields that hold base topic ids', () => {
    const reduced = TopicTreeView.fromGraphQL(literalGraphQL());
    expect([reduced.baseField, reduced.baseLabelField]).toEqual(['subtopic_id', 'subtopic_label']);
    const unreduced = TopicTreeView.fromGraphQL(literalGraphQL({ baseLevel: 'topic' }));
    expect([unreduced.baseField, unreduced.baseLabelField]).toEqual(['topic_id', 'topic_label']);
  });

  it('has no noise map when the query left noise out', () => {
    const view = TopicTreeView.fromGraphQL(literalGraphQL({ noiseItemIds: null, noiseMerges: null }));
    expect(view.tree.noiseMerges.size).toBe(0);
  });

  it('rejects noise arrays of different lengths', () => {
    expect(() => TopicTreeView.fromGraphQL(literalGraphQL({ noiseMerges: [0, 1] }))).toThrow(/noise/);
  });
});
