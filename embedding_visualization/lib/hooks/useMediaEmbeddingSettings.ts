'use client';

import { useQuery } from '@apollo/client/react';
import { GET_MEDIA_EMBEDDING_SETTINGS } from '../graphql/queries';
import type { MediaEmbeddingSettings } from '../graphql/mutations';

interface MediaSettingsData {
  mediaEmbeddingSettings: MediaEmbeddingSettings;
}

export interface MediaEmbeddingSettingsState {
  /** Defaults, limits and token constants of the media options; null until they load. */
  settings: MediaEmbeddingSettings | null;
  loading: boolean;
  /** Why the query failed (null otherwise); `refetch` retries it. */
  error: string | null;
  refetch: () => void;
}

/**
 * The media options' defaults and limits (config/media_embedding.json),
 * fetched once per page (cache-first) for the Image / Audio / Video / Mixed
 * embed forms (app/collections/lib/mediaOptions.ts turns them into rules).
 */
export function useMediaEmbeddingSettings(): MediaEmbeddingSettingsState {
  const { data, loading, error, refetch } = useQuery<MediaSettingsData>(GET_MEDIA_EMBEDDING_SETTINGS, {
    fetchPolicy: 'cache-first',
    // A retry should show "loading" again, not the old error.
    notifyOnNetworkStatusChange: true,
  });
  return {
    settings: data?.mediaEmbeddingSettings ?? null,
    loading,
    error: error && !loading ? error.message : null,
    refetch: () => void refetch(),
  };
}
