'use client';

import { useQuery } from '@apollo/client/react';
import { GET_COLLECTION_TOPICS } from '../graphql/queries';
import type { TopicInfo } from '../types/types';

interface CollectionTopicsData {
  collectionTopics: {
    topics: TopicInfo[];
  } | null;
}

/**
 * Topics for a collection from the `collectionTopics` GraphQL query — the
 * canonical DuckDB-backed source. The Explore page previously read a
 * `topic_summary` JSON blob off the collections manifest metadata, but that
 * blob was only ever written by the pre-migration ChromaDB flow: nothing
 * populates it anymore, which left topic search / isolation / legend sync
 * silently empty for every collection. This hook replaces it (the manifest
 * blob remains as a fallback for legacy stores).
 */
export function useCollectionTopics(collectionName: string | null): {
  topics: TopicInfo[] | undefined;
  loading: boolean;
} {
  const { data, loading } = useQuery<CollectionTopicsData>(GET_COLLECTION_TOPICS, {
    variables: { collectionName: collectionName ?? '' },
    skip: !collectionName,
  });
  const topics = data?.collectionTopics?.topics;
  return { topics: topics && topics.length > 0 ? topics : undefined, loading };
}
