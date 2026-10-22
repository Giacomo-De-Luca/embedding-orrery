'use client';

import type { ApolloClient } from '@apollo/client';
import { useQuery } from '@apollo/client/react';
import { TOPIC_SIMILARITY_ORDER } from '../graphql/queries';
import { AssociationFields } from '../utils/fieldAxis';

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
  const skip = !enabled || !collectionName || !AssociationFields.supportsSimilarity(field);
  const { data } = useQuery<TopicSimilarityOrderData>(TOPIC_SIMILARITY_ORDER, {
    variables: { collectionName: collectionName ?? '', field: field ?? '' },
    skip,
  });
  return skip ? null : (data?.topicSimilarityOrder ?? null);
}

/**
 * Drop every cached similarity order and refetch the active ones: a topic
 * reduction renumbers the topics, so no stored order survives it.
 */
export async function refreshTopicSimilarityOrders(client: ApolloClient): Promise<void> {
  await client.refetchQueries({
    updateCache(cache) {
      cache.evict({ fieldName: 'topicSimilarityOrder' });
      cache.gc();
    },
  });
}
