import { describe, expect, it } from 'vitest';
import { ColumnDefaults } from '../columnDefaults';

const PROSE = 'The committee approved the new budget after a long debate on Tuesday evening.';
const SHORT_PROSE = 'Budget approved';

describe('ColumnDefaults.textPick', () => {
  it('prefers a document-like name over position', () => {
    const rows = [{ id: 'a1', title: SHORT_PROSE, text: PROSE, label: 'politics' }];
    expect(ColumnDefaults.textPick(['id', 'title', 'text', 'label'], rows)).toEqual({ column: 'text', reason: 'name' });
  });

  it('matches names ignoring case', () => {
    expect(ColumnDefaults.textPick(['ID', 'Text'], [{ ID: 'x', Text: PROSE }])).toEqual({ column: 'Text', reason: 'name' });
  });

  it('follows the name priority when several match', () => {
    const rows = [{ question: 'Who approved it?', context: PROSE }];
    expect(ColumnDefaults.textPick(['question', 'context'], rows)?.column).toBe('context');
  });

  it('skips a named column that holds no strings', () => {
    const rows = [{ text: ['tok', 'ens'], sentence: PROSE }];
    expect(ColumnDefaults.textPick(['text', 'sentence'], rows)).toEqual({ column: 'sentence', reason: 'name' });
  });

  it('trusts the name when there is no preview', () => {
    expect(ColumnDefaults.textPick(['id', 'abstract'], [])).toEqual({ column: 'abstract', reason: 'name' });
  });

  it('falls back to the longest prose in the preview', () => {
    const rows = [
      { id: 1, headline: SHORT_PROSE, story: PROSE },
      { id: 2, headline: 'Rain expected', story: `${PROSE} ${PROSE}` },
    ];
    expect(ColumnDefaults.textPick(['id', 'headline', 'story'], rows)).toEqual({ column: 'story', reason: 'length' });
  });

  it('never picks one-word values, urls or numbers by length', () => {
    const rows = [
      { url: 'https://example.com/a/very/long/path/to/some/page/index.html', label: 'positive', score: 4, tweet_text: 'great service' },
      { url: 'https://example.com/another/very/long/path/page.html', label: 'negative', score: 1, tweet_text: 'never again' },
    ];
    expect(ColumnDefaults.textPick(['url', 'label', 'score', 'tweet_text'], rows)).toEqual({
      column: 'tweet_text',
      reason: 'length',
    });
  });

  it('never takes structured text for prose: vectors, objects, timestamps', () => {
    // A TSV vector column, as embed_vectors accepts it.
    const tsv = [{ word: 'apple', definition: 'a round fruit', embedding: '[0.12, -0.03, 0.88, 0.41, 0.07]' }];
    expect(ColumnDefaults.textPick(['word', 'definition', 'embedding'], tsv)).toEqual({ column: 'definition', reason: 'length' });
    // A nested JSONL object, stringified by the local preview.
    const jsonl = [{ tweet: 'loving the update', user: "{'name': 'Ada Lovelace', 'bio': 'Writes notes on engines'}" }];
    expect(ColumnDefaults.textPick(['tweet', 'user'], jsonl)?.column).toBe('tweet');
    // A HuggingFace struct (ms_marco-like), stringified by the HF preview.
    const hf = [{ passages: "{'is_selected': [0, 1], 'passage_text': ['Since 2007, the RBA has ...']}", query: 'what is rba' }];
    expect(ColumnDefaults.textPick(['passages', 'query'], hf)?.column).toBe('query');
    expect(ColumnDefaults.textPick(['created_at', 'score'], [{ created_at: '2024-03-01 12:30:00', score: '4 stars' }])).toBeNull();
  });

  it('counts prose in any alphabet', () => {
    expect(ColumnDefaults.textPick(['id', 'отзыв'], [{ id: 1, отзыв: 'очень хороший сервис' }])).toEqual({
      column: 'отзыв',
      reason: 'length',
    });
  });

  it('never picks the id column', () => {
    const rows = [{ doc_id: 'report 2024 annual summary of operations', note: 'see appendix' }];
    expect(ColumnDefaults.textPick(['doc_id', 'note'], rows)?.column).toBe('note');
  });

  it('picks nothing when no column holds text', () => {
    const rows = [{ x: 0.4, y: 1.2, label: 'cat' }];
    expect(ColumnDefaults.textPick(['x', 'y', 'label'], rows)).toBeNull();
    expect(ColumnDefaults.textPick(['x', 'y'], [])).toBeNull();
  });
});

describe('ColumnDefaults.idColumn', () => {
  it('knows the id names of both tabs, ignoring case', () => {
    expect(ColumnDefaults.idColumn(['text', 'Feature_ID'])).toBe('Feature_ID');
    expect(ColumnDefaults.idColumn(['doc_id', 'text'])).toBe('doc_id');
    expect(ColumnDefaults.idColumn(['text', 'identifier'])).toBeNull();
  });
});

describe('ColumnDefaults.select', () => {
  const kinds = { photo: 'image' as const };

  it('TEXT: the picked column, its template, every other column as metadata', () => {
    const selection = ColumnDefaults.select({
      columns: ['id', 'text', 'label'],
      rows: [{ id: 1, text: PROSE, label: 'news' }],
      kinds: {},
      dataType: 'TEXT',
    });
    expect(selection).toEqual({
      embedding: ['text'],
      metadata: ['id', 'label'],
      media: [],
      template: '{text}',
      idColumn: 'id',
      textPick: { column: 'text', reason: 'name' },
    });
  });

  it('TEXT: nothing to embed when no column holds text', () => {
    const selection = ColumnDefaults.select({
      columns: ['x', 'y'],
      rows: [{ x: 1, y: 2 }],
      kinds: {},
      dataType: 'TEXT',
    });
    expect(selection.embedding).toEqual([]);
    expect(selection.template).toBe('');
    expect(selection.metadata).toEqual(['x', 'y']);
    expect(selection.textPick).toBeNull();
  });

  it('TEXT: media columns are never picked as text', () => {
    const selection = ColumnDefaults.select({
      columns: ['photo', 'caption'],
      rows: [{ photo: 'cats/one.jpg is here', caption: 'a cat' }],
      kinds,
      dataType: 'TEXT',
    });
    expect(selection.embedding).toEqual(['caption']);
  });

  it('MULTIMODAL: every media column, no text', () => {
    const selection = ColumnDefaults.select({
      columns: ['photo', 'caption'],
      rows: [],
      kinds,
      dataType: 'MULTIMODAL',
    });
    expect(selection.media).toEqual(['photo']);
    expect(selection.embedding).toEqual([]);
    expect(selection.template).toBe('');
    expect(selection.metadata).toEqual(['caption']);
    expect(selection.textPick).toBeNull();
  });

  it('VECTOR: a vector-named column, else the first one', () => {
    const named = ColumnDefaults.select({ columns: ['word', 'embedding_768'], rows: [], kinds: {}, dataType: 'VECTOR' });
    expect(named.embedding).toEqual(['embedding_768']);
    expect(named.template).toBe('');
    expect(named.metadata).toEqual(['word']);
    const unnamed = ColumnDefaults.select({ columns: ['word', 'values'], rows: [], kinds: {}, dataType: 'VECTOR' });
    expect(unnamed.embedding).toEqual(['word']);
  });
});

describe('ColumnDefaults.note', () => {
  const pick = { column: 'abstract', reason: 'length' as const };

  it('explains the pick while it is the selection', () => {
    expect(ColumnDefaults.note(pick, ['abstract'], 'TEXT')).toContain('abstract');
    expect(ColumnDefaults.note({ column: 'text', reason: 'name' }, ['text'], 'TEXT')).toContain('name');
  });

  it('goes once the selection changes, and outside text mode', () => {
    expect(ColumnDefaults.note(pick, ['abstract', 'title'], 'TEXT')).toBeNull();
    expect(ColumnDefaults.note(pick, [], 'TEXT')).toBeNull();
    expect(ColumnDefaults.note(pick, ['abstract'], 'MULTIMODAL')).toBeNull();
    expect(ColumnDefaults.note(null, ['abstract'], 'TEXT')).toBeNull();
  });
});
