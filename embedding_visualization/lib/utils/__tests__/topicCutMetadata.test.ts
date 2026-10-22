import { describe, expect, it } from 'vitest';
import { TopicMergeTree, TopicTreeCut, TopicTreeView } from '../topicTreeCut';
import { TopicTreeGroups } from '../topicCutMetadata';

// The literal tree of topicTreeCut.test.ts (and the backend's
// test_topic_merge_tree.py): base topics 10..14, sizes 5, 3, 8, 2, 1.
//   N=4: {10, 11} {12} {13} {14}      N=3: {10, 11} {12} {13, 14}
//   N=2: {10, 11} {12, 13, 14}        noise: n-early -> row 0, n-mid -> row 1, n-root -> row 3
const LABELS = new Map([[10, 'Ten'], [11, 'Eleven'], [12, 'Twelve'], [13, 'Thirteen'], [14, 'Fourteen']]);

function view(baseLevel: 'topic' | 'subtopic', withNoise = false): TopicTreeView {
  const tree = new TopicMergeTree({
    linkage: withNoise ? 'density' : 'average',
    leafIds: [10, 11, 12, 13, 14],
    leafSizes: [5, 3, 8, 2, 1],
    children: [[0, 1], [3, 4], [2, 6], [5, 7]],
    heights: [0.1, 0.2, 0.4, 0.9],
    noiseMerges: withNoise ? new Map([['n-early', 0], ['n-mid', 1], ['n-root', 3]]) : undefined,
  });
  return new TopicTreeView('ext', baseLevel, tree, LABELS, withNoise ? 3 : 0, ['average', 'ward']);
}

/** An item of an unreduced extraction: its base topic is in topic_id. */
const unreduced = (id: string, label: string, extra: Record<string, unknown> = {}) => ({
  topic_id: id, topic_label: label, ...extra,
});

const fields = (meta: Record<string, unknown>) => ({
  topic_id: meta.topic_id,
  topic_label: meta.topic_label,
  subtopic_id: meta.subtopic_id,
  subtopic_label: meta.subtopic_label,
});

describe('TopicCutPreview groups', () => {
  it('labels each group after its largest member, counting the others', () => {
    const preview = new TopicTreeGroups(view('topic')).cut(2, false);
    expect(preview.size).toBe(2);
    expect([...preview.groups.values()].map((g) => [g.topicId, g.label, g.members])).toEqual([
      ['10', 'Ten +1', 2],
      ['12', 'Twelve +2', 3],
    ]);
  });

  it('maps every merged label any cut can show to its representative (the colour pin)', () => {
    expect(new TopicTreeGroups(view('topic')).representativeLabels()).toEqual({
      'Ten +1': 'Ten',
      'Thirteen +1': 'Thirteen',
      'Twelve +2': 'Twelve',
      'Twelve +4': 'Twelve',
    });
  });

  it('names each group after the representative TopicTreeCut picks for its root', () => {
    const tree = view('topic').tree;
    const cut = new TopicTreeCut(tree);
    for (const n of [1, 2, 3, 4, 5]) {
      const preview = new TopicTreeGroups(view('topic')).cut(n, false);
      const named = [...preview.groups].map(([root, group]) => [root, Number(group.topicId)] as const);
      expect(new Map(named)).toEqual(cut.representatives(cut.groups(cut.mergesFor(n))));
    }
  });

  it('rejects a topic count outside the tree', () => {
    expect(() => new TopicTreeGroups(view('topic')).cut(6, false)).toThrow(/between 1 and 5/);
  });
});

describe('TopicCutPreview.apply on an unreduced extraction', () => {
  const ids = ['a', 'b', 'c', 'd'];
  const metadata = [
    unreduced('10', 'Ten', { year: 1990 }),
    unreduced('11', 'Eleven'),
    unreduced('14', 'Fourteen'),
    unreduced('-1', 'Unclustered'),
  ];

  it('keeps every base topic apart at N = k, adding the subtopic fields', () => {
    const cut = new TopicTreeGroups(view('topic')).cut(5, false).apply(metadata, ids);
    expect(cut.map(fields)).toEqual([
      { topic_id: '10', topic_label: 'Ten', subtopic_id: '10', subtopic_label: 'Ten' },
      { topic_id: '11', topic_label: 'Eleven', subtopic_id: '11', subtopic_label: 'Eleven' },
      { topic_id: '14', topic_label: 'Fourteen', subtopic_id: '14', subtopic_label: 'Fourteen' },
      { topic_id: '-1', topic_label: 'Unclustered', subtopic_id: '-1', subtopic_label: 'Unclustered' },
    ]);
  });

  it("gives merged items the representative's id and the group label", () => {
    const cut = new TopicTreeGroups(view('topic')).cut(2, false).apply(metadata, ids);
    expect(cut.map(fields)).toEqual([
      { topic_id: '10', topic_label: 'Ten +1', subtopic_id: '10', subtopic_label: 'Ten' },
      { topic_id: '10', topic_label: 'Ten +1', subtopic_id: '11', subtopic_label: 'Eleven' },
      { topic_id: '12', topic_label: 'Twelve +2', subtopic_id: '14', subtopic_label: 'Fourteen' },
      { topic_id: '-1', topic_label: 'Unclustered', subtopic_id: '-1', subtopic_label: 'Unclustered' },
    ]);
  });

  it('copies changed items with their other fields and leaves the input alone', () => {
    const cut = new TopicTreeGroups(view('topic')).cut(2, false).apply(metadata, ids);
    expect(cut[0].year).toBe(1990);
    expect(cut[0]).not.toBe(metadata[0]);
    expect(metadata[0]).toEqual(unreduced('10', 'Ten', { year: 1990 }));
  });

  it('reads numeric base ids like their string form', () => {
    const cut = new TopicTreeGroups(view('topic')).cut(4, false).apply([{ topic_id: 11, topic_label: 'Eleven' }], ['a']);
    expect(fields(cut[0])).toEqual({ topic_id: '10', topic_label: 'Ten +1', subtopic_id: '11', subtopic_label: 'Eleven' });
  });
});

describe('TopicCutPreview.apply item identity', () => {
  it('returns a new array but reuses items whose four topic fields do not change', () => {
    const done = { topic_id: '12', topic_label: 'Twelve', subtopic_id: '12', subtopic_label: 'Twelve' };
    const input = [done];
    const cut = new TopicTreeGroups(view('topic')).cut(5, false).apply(input, ['a']);
    expect(cut).not.toBe(input);
    expect(cut[0]).toBe(done);
  });

  it('leaves items without the base field untouched', () => {
    const bare = { year: 2001 };
    const cut = new TopicTreeGroups(view('topic')).cut(2, false).apply([bare], ['a']);
    expect(cut[0]).toBe(bare);
  });

  it('leaves items whose base topic the tree does not know untouched (stale tree)', () => {
    const stale = unreduced('99', 'Gone');
    const cut = new TopicTreeGroups(view('topic')).cut(2, false).apply([stale], ['a']);
    expect(cut[0]).toBe(stale);
  });
});

describe('TopicCutPreview.apply on a reduced extraction', () => {
  it('reads base topics from subtopic_id and replaces the saved topic fields', () => {
    const saved = { topic_id: '0', topic_label: 'Saved group', subtopic_id: '13', subtopic_label: 'Thirteen' };
    const cut = new TopicTreeGroups(view('subtopic')).cut(3, false).apply([saved], ['a']);
    expect(fields(cut[0])).toEqual({
      topic_id: '13', topic_label: 'Thirteen +1', subtopic_id: '13', subtopic_label: 'Thirteen',
    });
  });

  it('turns previously absorbed noise back into noise in a non-density preview', () => {
    // A density Apply with absorbNoise saved this item into a topic but kept
    // its base topic (subtopic) at -1, so the reduction stays reversible.
    const absorbed = { topic_id: '2', topic_label: 'Saved group', subtopic_id: '-1', subtopic_label: 'Unclustered' };
    const cut = new TopicTreeGroups(view('subtopic')).cut(3, false).apply([absorbed], ['n-early']);
    expect(fields(cut[0])).toEqual({
      topic_id: '-1', topic_label: 'Unclustered', subtopic_id: '-1', subtopic_label: 'Unclustered',
    });
  });
});

describe('TopicCutPreview.apply with density noise', () => {
  const ids = ['n-early', 'n-mid', 'n-root'];
  const noise = ids.map(() => unreduced('-1', 'Unclustered'));

  it('moves noise into the groups whose merges were applied', () => {
    const cut = new TopicTreeGroups(view('topic', true)).cut(3, true).apply(noise, ids);
    expect(cut.map(fields)).toEqual([
      { topic_id: '10', topic_label: 'Ten +1', subtopic_id: '-1', subtopic_label: 'Unclustered' },
      { topic_id: '13', topic_label: 'Thirteen +1', subtopic_id: '-1', subtopic_label: 'Unclustered' },
      { topic_id: '-1', topic_label: 'Unclustered', subtopic_id: '-1', subtopic_label: 'Unclustered' },
    ]);
  });

  it('absorbs everything at N = 1', () => {
    const cut = new TopicTreeGroups(view('topic', true)).cut(1, true).apply(noise, ids);
    expect(cut.map((m) => m.topic_label)).toEqual(['Twelve +4', 'Twelve +4', 'Twelve +4']);
  });

  it('keeps noise as noise without absorbNoise', () => {
    const cut = new TopicTreeGroups(view('topic', true)).cut(1, false).apply(noise, ids);
    expect(cut.map((m) => m.topic_id)).toEqual(['-1', '-1', '-1']);
  });

  it('absorbs previously absorbed noise of a reduced extraction again', () => {
    const absorbed = { topic_id: '2', topic_label: 'Saved group', subtopic_id: '-1', subtopic_label: 'Unclustered' };
    const cut = new TopicTreeGroups(view('subtopic', true)).cut(4, true).apply([absorbed], ['n-early']);
    expect(fields(cut[0])).toEqual({
      topic_id: '10', topic_label: 'Ten +1', subtopic_id: '-1', subtopic_label: 'Unclustered',
    });
  });
});
