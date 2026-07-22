'use client';

import { useEffect, useMemo } from 'react';
import { useQuery } from '@apollo/client/react';
import { GET_COLLECTIONS } from '../graphql/queries';
import type { CollectionsManifest, TopicInfo } from '../types/types';
import type { DefaultColorScheme } from '../utils/colorScaleUrl';

interface GraphQLCollection {
  name: string;
  count: number;
  metadata: Record<string, unknown>;
}

interface CollectionsData {
  collections: GraphQLCollection[];
}

/**
 * Parse topic_summary JSON string from collection metadata into TopicInfo[].
 * topic_summary is stored as a JSON-serialized string in ChromaDB metadata.
 */
function parseTopicSummary(metadata: Record<string, unknown>): TopicInfo[] | undefined {
  const raw = metadata?.topic_summary;
  if (!raw || typeof raw !== 'string') return undefined;
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return undefined;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return parsed.map((t: Record<string, any>) => ({
      topicId: t.topic_id ?? t.topicId,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      keywords: (t.keywords ?? []).map((k: any) =>
        typeof k === 'string' ? { word: k, score: 0 } : { word: k.word, score: k.score ?? 0 }
      ),
      label: t.label ?? null,
      count: t.count ?? 0,
      subtopics: t.subtopics ?? null,
    }));
  } catch {
    return undefined;
  }
}

/**
 * Parse the per-collection default colour scheme from metadata. Stored as a
 * JSON string under `default_color_scheme` (see SaveColorDefaultButton).
 */
function parseDefaultColorScheme(metadata: Record<string, unknown>): DefaultColorScheme | undefined {
  const raw = metadata?.default_color_scheme;
  if (!raw || typeof raw !== 'string') return undefined;
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && typeof parsed.colorBy === 'string') {
      return parsed as DefaultColorScheme;
    }
    return undefined;
  } catch {
    return undefined;
  }
}

export function useCollections() {
  // notifyOnNetworkStatusChange off so retry polls resolve silently instead
  // of flipping consumers between loading and error state every cycle.
  const { data, loading, error, startPolling, stopPolling } =
    useQuery<CollectionsData>(GET_COLLECTIONS, { notifyOnNetworkStatusChange: false });

  // An unreachable backend (cold demo Space 502ing behind nginx, or a dev
  // frontend started before the backend) leaves every collection-gated
  // control dead until a manual reload. Retry in every build while in the
  // failed-with-no-data state; stop for good as soon as the manifest lands.
  useEffect(() => {
    if (data || !error) return;
    startPolling(4000);
    return () => stopPolling();
  }, [data, error, startPolling, stopPolling]);

  // Transform GraphQL response to CollectionsManifest format. Memoized on `data`
  // so the manifest keeps a stable reference across renders (avoids re-running the
  // reduce for every consumer and churning effects that depend on `collections`).
  const collections: CollectionsManifest | null = useMemo(() => (
    data?.collections
      ? data.collections.reduce((acc, col) => {
          const topics = parseTopicSummary(col.metadata);
          acc[col.name] = {
            name: col.name,
            display_name: col.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            count: col.count,
            embedding_dim: (col.metadata?.embedding_dim as number | undefined) || 384,
            timestamp: (col.metadata?.timestamp as string | undefined) || '',
            source_dataset: col.metadata?.source_dataset as string | undefined,
            has_projections: col.metadata?.has_projections as boolean | undefined,
            has_topics: topics !== undefined && topics.length > 0,
            topics,
            defaultColorScheme: parseDefaultColorScheme(col.metadata),
          };
          return acc;
        }, {} as CollectionsManifest)
      : null
  ), [data]);

  return { collections, loading, error: error || null };
}
