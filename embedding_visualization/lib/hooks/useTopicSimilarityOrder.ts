'use client';

import { useQuery } from '@apollo/client/react';
import { TOPIC_SIMILARITY_ORDER } from '../graphql/queries';
import { isTopicLikeField } from '../utils/categoryColors';

interface TopicSimilarityOrderData {
  topicSimilarityOrder: string[] | null;
}

/**
 * Values of a topic field ordered by embedding-centroid similarity (the
 * heatmap's Similarity sort). Null for non-topic fields, while loading, and
 * when the collection's centroids have not been computed.
 */
export function useTopicSimilarityOrder(
  collectionName: string | null | undefined,
  field: string | null,
  enabled: boolean,
): string[] | null {
  const skip = !enabled || !collectionName || !isTopicLikeField(field);
  const { data } = useQuery<TopicSimilarityOrderData>(TOPIC_SIMILARITY_ORDER, {
    variables: { collectionName: collectionName ?? '', field: field ?? '' },
    skip,
  });
  return skip ? null : (data?.topicSimilarityOrder ?? null);
}
