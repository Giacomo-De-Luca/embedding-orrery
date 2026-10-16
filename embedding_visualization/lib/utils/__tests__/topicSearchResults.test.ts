import { describe, expect, it } from 'vitest';

import type { TopicInfo } from '../../types/types';
import { buildTopicFilters, toTopicSearchResults } from '../topicSearchResults';

const topic = (topicId: number, label: string): TopicInfo => ({
  topicId,
  label,
  count: 10,
  keywords: [],
});

const TOPICS = [topic(-1, 'Unclustered'), topic(0, 'Hate speech'), topic(1, 'Translation'), topic(2, 'Vision')];

describe('toTopicSearchResults', () => {
  it('maps backend rows to topics in backend order', () => {
    const results = toTopicSearchResults(
      [
        { topicId: 1, similarity: 0.6, matchedSubtopicId: null, matchedSubtopicLabel: null },
        { topicId: 0, similarity: 0.3, matchedSubtopicId: 7, matchedSubtopicLabel: 'Toxic comments' },
      ],
      TOPICS,
    );

    expect(results.map(r => r.topic.label)).toEqual(['Translation', 'Hate speech']);
    expect(results[1].matchedSubtopicLabel).toBe('Toxic comments');
    expect(results[0].similarity).toBe(0.6);
  });

  it('normalises display relevance to the top score', () => {
    const results = toTopicSearchResults(
      [
        { topicId: 1, similarity: 0.5, matchedSubtopicId: null, matchedSubtopicLabel: null },
        { topicId: 0, similarity: 0.25, matchedSubtopicId: null, matchedSubtopicLabel: null },
        { topicId: 2, similarity: -0.4, matchedSubtopicId: null, matchedSubtopicLabel: null },
      ],
      TOPICS,
    );

    expect(results.map(r => r.relevance)).toEqual([1, 0.5, 0]);
  });

  it('gives zero relevance when no topic scores above zero', () => {
    const results = toTopicSearchResults(
      [{ topicId: 1, similarity: -0.2, matchedSubtopicId: null, matchedSubtopicLabel: null }],
      TOPICS,
    );

    expect(results[0].relevance).toBe(0);
  });

  it('drops noise and topics missing from the loaded list', () => {
    const results = toTopicSearchResults(
      [
        { topicId: -1, similarity: 0.9, matchedSubtopicId: null, matchedSubtopicLabel: null },
        { topicId: 42, similarity: 0.8, matchedSubtopicId: null, matchedSubtopicLabel: null },
        { topicId: 2, similarity: 0.4, matchedSubtopicId: null, matchedSubtopicLabel: null },
      ],
      TOPICS,
    );

    expect(results.map(r => r.topic.topicId)).toEqual([2]);
    expect(results[0].relevance).toBe(1);
  });
});

describe('buildTopicFilters', () => {
  it('returns no filter without a selection', () => {
    expect(buildTopicFilters(new Set())).toBeUndefined();
  });

  it('uses EQ for a single topic', () => {
    expect(buildTopicFilters(new Set([3]))).toEqual([
      { field: 'topic_id', operator: 'EQ', value: 3 },
    ]);
  });

  it('uses IN for several topics', () => {
    expect(buildTopicFilters(new Set([3, 5]))).toEqual([
      { field: 'topic_id', operator: 'IN', value: [3, 5] },
    ]);
  });
});
