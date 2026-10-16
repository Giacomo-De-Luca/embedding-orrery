import type { FilterInput, TopicInfo } from '../types/types';

/** One row of the backend `topicSemanticSearch` query. */
export interface TopicSimilarity {
  topicId: number;
  similarity: number;
  matchedSubtopicId: number | null;
  matchedSubtopicLabel: string | null;
}

/** A ranked topic as the Topics section displays it. */
export interface TopicSearchResult {
  topic: TopicInfo;
  /** Display bar width, 0-1: similarity relative to the best match (negatives clamp to 0). */
  relevance: number;
  /** Raw (mean-centred) cosine similarity between the query and the topic centroid. */
  similarity: number;
  /** Set when one of the topic's subtopics matched better than the topic as a whole. */
  matchedSubtopicLabel: string | null;
}

const NOISE_TOPIC_ID = -1;

/**
 * Attach loaded topic info to backend similarity rows, keeping backend order.
 * Rows for noise or for topics missing from `topics` are dropped.
 */
export function toTopicSearchResults(
  rows: TopicSimilarity[],
  topics: TopicInfo[],
): TopicSearchResult[] {
  const topicById = new Map(topics.map(t => [t.topicId, t]));
  const kept = rows.filter(r => r.topicId !== NOISE_TOPIC_ID && topicById.has(r.topicId));
  const top = Math.max(0, ...kept.map(r => r.similarity));

  return kept.map(r => ({
    topic: topicById.get(r.topicId)!,
    relevance: top > 0 ? Math.max(0, r.similarity) / top : 0,
    similarity: r.similarity,
    matchedSubtopicLabel: r.matchedSubtopicLabel,
  }));
}

/** Backend filters scoping a search to the selected topics (undefined = no scope). */
export function buildTopicFilters(selectedTopicIds: ReadonlySet<number>): FilterInput[] | undefined {
  if (selectedTopicIds.size === 0) return undefined;
  const ids = Array.from(selectedTopicIds);
  if (ids.length === 1) {
    return [{ field: 'topic_id', operator: 'EQ', value: ids[0] }];
  }
  return [{ field: 'topic_id', operator: 'IN', value: ids }];
}
