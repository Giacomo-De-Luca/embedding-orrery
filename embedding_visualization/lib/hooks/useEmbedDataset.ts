'use client';

import { useState, useCallback } from 'react';
import { useMutation, useLazyQuery, useApolloClient } from '@apollo/client/react';
import {
  GET_HF_DATASET_INFO,
  GET_HF_DATASET_PREVIEW,
  GET_LOCAL_FILE_INFO,
  GET_LOCAL_FILE_PREVIEW,
  EMBED_HUGGINGFACE_DATASET,
  EMBED_LOCAL_FILE,
  DELETE_COLLECTION,
  UPDATE_COLLECTION_METADATA,
  type HFDatasetInfo,
  type HFDatasetPreview,
  type LocalFileInfo,
  type LocalFilePreview,
  type EmbedDatasetInput,
  type EmbedLocalFileInput,
  type EmbedDatasetResult,
  type UpdateCollectionMetadataResult,
  type TopicConfigInput,
  type ExtractTopicsResult,
} from '../graphql/mutations';
import { GET_COLLECTIONS, EXTRACT_TOPICS } from '../graphql/queries';

// ========== Hook Return Types ==========

export interface UseEmbedDatasetReturn {
  // HuggingFace dataset operations
  fetchHFDatasetInfo: (datasetId: string) => Promise<HFDatasetInfo | null>;
  fetchHFDatasetPreview: (
    datasetId: string,
    config?: string,
    split?: string,
    nRows?: number
  ) => Promise<HFDatasetPreview | null>;
  embedHFDataset: (input: EmbedDatasetInput) => Promise<EmbedDatasetResult | null>;

  // Local file operations
  fetchLocalFileInfo: (filePath: string) => Promise<LocalFileInfo | null>;
  fetchLocalFilePreview: (filePath: string, nRows?: number) => Promise<LocalFilePreview | null>;
  embedLocalFile: (input: EmbedLocalFileInput) => Promise<EmbedDatasetResult | null>;

  // Collection operations
  deleteCollection: (collectionName: string) => Promise<boolean>;
  updateCollectionMetadata: (
    collectionName: string,
    metadata: Record<string, unknown>
  ) => Promise<UpdateCollectionMetadataResult | null>;
  refreshCollections: () => Promise<void>;

  // State
  datasetInfo: HFDatasetInfo | null;
  datasetPreview: HFDatasetPreview | null;
  localFileInfo: LocalFileInfo | null;
  localFilePreview: LocalFilePreview | null;

  // Loading states
  infoLoading: boolean;
  previewLoading: boolean;
  embedLoading: boolean;

  // Errors
  error: string | null;
  clearError: () => void;

  // Last embed result
  lastEmbedResult: EmbedDatasetResult | null;

  // Topic extraction
  extractTopics: (collectionName: string, config?: TopicConfigInput) => Promise<ExtractTopicsResult | null>;
  topicsLoading: boolean;
  lastTopicsResult: ExtractTopicsResult | null;

  // Active job tracking for progress display
  activeJobCollectionName: string | null;
  clearActiveJob: () => void;
}

// ========== Hook Implementation ==========

export function useEmbedDataset(): UseEmbedDatasetReturn {
  const client = useApolloClient();

  // Local state
  const [datasetInfo, setDatasetInfo] = useState<HFDatasetInfo | null>(null);
  const [datasetPreview, setDatasetPreview] = useState<HFDatasetPreview | null>(null);
  const [localFileInfo, setLocalFileInfo] = useState<LocalFileInfo | null>(null);
  const [localFilePreview, setLocalFilePreview] = useState<LocalFilePreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastEmbedResult, setLastEmbedResult] = useState<EmbedDatasetResult | null>(null);

  // Active job tracking for progress display
  const [activeJobCollectionName, setActiveJobCollectionName] = useState<string | null>(null);

  // Loading states
  const [infoLoading, setInfoLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);

  // Lazy queries for HuggingFace
  const [getHFInfo] = useLazyQuery<{ huggingfaceDatasetInfo: HFDatasetInfo }>(
    GET_HF_DATASET_INFO,
    { fetchPolicy: 'network-only' }
  );

  const [getHFPreview] = useLazyQuery<{ huggingfaceDatasetPreview: HFDatasetPreview }>(
    GET_HF_DATASET_PREVIEW,
    { fetchPolicy: 'network-only' }
  );

  // Lazy queries for local files
  const [getLocalInfo] = useLazyQuery<{ localFileInfo: LocalFileInfo }>(
    GET_LOCAL_FILE_INFO,
    { fetchPolicy: 'network-only' }
  );

  const [getLocalPreview] = useLazyQuery<{ localFilePreview: LocalFilePreview }>(
    GET_LOCAL_FILE_PREVIEW,
    { fetchPolicy: 'network-only' }
  );

  // Mutations
  const [embedHFMutation, { loading: embedHFLoading }] = useMutation<
    { embedHuggingfaceDataset: EmbedDatasetResult }
  >(EMBED_HUGGINGFACE_DATASET);

  const [embedLocalMutation, { loading: embedLocalLoading }] = useMutation<
    { embedLocalFile: EmbedDatasetResult }
  >(EMBED_LOCAL_FILE);

  const [deleteMutation] = useMutation<{ deleteCollection: boolean }>(DELETE_COLLECTION);

  const [updateMetadataMutation] = useMutation<
    { updateCollectionMetadata: UpdateCollectionMetadataResult }
  >(UPDATE_COLLECTION_METADATA);

  const [extractTopicsMutation, { loading: topicsLoading }] = useMutation<
    { extractTopics: ExtractTopicsResult }
  >(EXTRACT_TOPICS);

  const [lastTopicsResult, setLastTopicsResult] = useState<ExtractTopicsResult | null>(null);

  // ========== HuggingFace Operations ==========

  const fetchHFDatasetInfo = useCallback(async (datasetId: string): Promise<HFDatasetInfo | null> => {
    setInfoLoading(true);
    setError(null);
    setDatasetInfo(null);

    try {
      const { data, error: queryError } = await getHFInfo({
        variables: { datasetId }
      });

      if (queryError) {
        setError(queryError.message);
        return null;
      }

      const info = data?.huggingfaceDatasetInfo || null;

      if (info?.error) {
        setError(info.error);
      }

      setDatasetInfo(info);
      return info;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch dataset info';
      setError(message);
      return null;
    } finally {
      setInfoLoading(false);
    }
  }, [getHFInfo]);

  const fetchHFDatasetPreview = useCallback(async (
    datasetId: string,
    config?: string,
    split: string = 'train',
    nRows: number = 5
  ): Promise<HFDatasetPreview | null> => {
    setPreviewLoading(true);
    setError(null);
    setDatasetPreview(null);

    try {
      const { data, error: queryError } = await getHFPreview({
        variables: { datasetId, config, split, nRows }
      });

      if (queryError) {
        setError(queryError.message);
        return null;
      }

      const preview = data?.huggingfaceDatasetPreview || null;

      if (preview?.error) {
        setError(preview.error);
      }

      setDatasetPreview(preview);
      return preview;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch dataset preview';
      setError(message);
      return null;
    } finally {
      setPreviewLoading(false);
    }
  }, [getHFPreview]);

  const embedHFDataset = useCallback(async (input: EmbedDatasetInput): Promise<EmbedDatasetResult | null> => {
    setError(null);
    setLastEmbedResult(null);
    setActiveJobCollectionName(input.collectionName);

    try {
      const { data, errors } = await embedHFMutation({
        variables: { input },
        // Longer timeout for embedding operations
        context: {
          fetchOptions: {
            timeout: 600000 // 10 minutes
          }
        }
      });

      if (errors && errors.length > 0) {
        setError(errors.map(e => e.message).join(', '));
        setActiveJobCollectionName(null);
        return null;
      }

      const result = data?.embedHuggingfaceDataset || null;

      if (result?.error) {
        setError(result.error);
      }

      setLastEmbedResult(result);
      setActiveJobCollectionName(null);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to embed dataset';
      setError(message);
      setActiveJobCollectionName(null);
      return null;
    }
  }, [embedHFMutation]);

  // ========== Local File Operations ==========

  const fetchLocalFileInfo = useCallback(async (filePath: string): Promise<LocalFileInfo | null> => {
    setInfoLoading(true);
    setError(null);
    setLocalFileInfo(null);
    console.log('DEBUG: fetchLocalFileInfo started', { filePath });

    try {
      const { data, error: queryError } = await getLocalInfo({
        variables: { filePath }
      });

      if (queryError) {
        setError(queryError.message);
        console.log('DEBUG: fetchLocalFileInfo queryError', { message: queryError.message });
        return null;
      }

      const info = data?.localFileInfo || null;

      console.log('DEBUG: fetchLocalFileInfo result', { info });

      if (info?.error) {
        setError(info.error);
      }

      setLocalFileInfo(info);
      return info;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch file info';
      setError(message);
      return null;
    } finally {
      setInfoLoading(false);
    }
  }, [getLocalInfo]);

  const fetchLocalFilePreview = useCallback(async (
    filePath: string,
    nRows: number = 5
  ): Promise<LocalFilePreview | null> => {
    setPreviewLoading(true);
    setError(null);
    setLocalFilePreview(null);

    try {
      const { data, error: queryError } = await getLocalPreview({
        variables: { filePath, nRows }
      });

      if (queryError) {
        setError(queryError.message);
        return null;
      }

      const preview = data?.localFilePreview || null;

      if (preview?.error) {
        setError(preview.error);
      }

      setLocalFilePreview(preview);
      return preview;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch file preview';
      setError(message);
      return null;
    } finally {
      setPreviewLoading(false);
    }
  }, [getLocalPreview]);

  const embedLocalFile = useCallback(async (input: EmbedLocalFileInput): Promise<EmbedDatasetResult | null> => {
    setError(null);
    setLastEmbedResult(null);
    setActiveJobCollectionName(input.collectionName);

    try {
      const { data, errors } = await embedLocalMutation({
        variables: { input },
        context: {
          fetchOptions: {
            timeout: 600000 // 10 minutes
          }
        }
      });

      if (errors && errors.length > 0) {
        setError(errors.map(e => e.message).join(', '));
        setActiveJobCollectionName(null);
        return null;
      }

      const result = data?.embedLocalFile || null;

      if (result?.error) {
        setError(result.error);
      }

      setLastEmbedResult(result);
      setActiveJobCollectionName(null);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to embed file';
      setError(message);
      setActiveJobCollectionName(null);
      return null;
    }
  }, [embedLocalMutation]);

  // ========== Topic Extraction ==========

  const extractTopics = useCallback(async (
    collectionName: string,
    config?: TopicConfigInput
  ): Promise<ExtractTopicsResult | null> => {
    setError(null);
    setLastTopicsResult(null);

    try {
      const { data, errors } = await extractTopicsMutation({
        variables: { collectionName, config: config || null },
        context: {
          fetchOptions: {
            timeout: 600000 // 10 minutes — topic extraction with LLM can be slow
          }
        }
      });

      if (errors && errors.length > 0) {
        setError(errors.map(e => e.message).join(', '));
        return null;
      }

      const result = data?.extractTopics || null;

      if (result?.error) {
        setError(result.error);
      }

      setLastTopicsResult(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to extract topics';
      setError(message);
      return null;
    }
  }, [extractTopicsMutation]);

  // ========== Collection Operations ==========

  const deleteCollection = useCallback(async (collectionName: string): Promise<boolean> => {
    setError(null);

    try {
      const { data, errors } = await deleteMutation({
        variables: { collectionName }
      });

      if (errors && errors.length > 0) {
        setError(errors.map(e => e.message).join(', '));
        return false;
      }

      return data?.deleteCollection ?? false;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete collection';
      setError(message);
      return false;
    }
  }, [deleteMutation]);

  const updateCollectionMetadata = useCallback(async (
    collectionName: string,
    metadata: Record<string, unknown>
  ): Promise<UpdateCollectionMetadataResult | null> => {
    setError(null);

    try {
      const { data, errors } = await updateMetadataMutation({
        variables: { collectionName, metadata }
      });

      if (errors && errors.length > 0) {
        setError(errors.map(e => e.message).join(', '));
        return null;
      }

      const result = data?.updateCollectionMetadata || null;

      if (result?.error) {
        setError(result.error);
      }

      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to update collection metadata';
      setError(message);
      return null;
    }
  }, [updateMetadataMutation]);

  const refreshCollections = useCallback(async () => {
    // Refetch collections query to update the list
    await client.refetchQueries({
      include: [GET_COLLECTIONS]
    });
  }, [client]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const clearActiveJob = useCallback(() => {
    setActiveJobCollectionName(null);
  }, []);

  return {
    // HuggingFace operations
    fetchHFDatasetInfo,
    fetchHFDatasetPreview,
    embedHFDataset,

    // Local file operations
    fetchLocalFileInfo,
    fetchLocalFilePreview,
    embedLocalFile,

    // Topic extraction
    extractTopics,
    topicsLoading,
    lastTopicsResult,

    // Collection operations
    deleteCollection,
    updateCollectionMetadata,
    refreshCollections,

    // State
    datasetInfo,
    datasetPreview,
    localFileInfo,
    localFilePreview,

    // Loading states
    infoLoading,
    previewLoading,
    embedLoading: embedHFLoading || embedLocalLoading,

    // Errors
    error,
    clearError,

    // Last result
    lastEmbedResult,

    // Active job tracking
    activeJobCollectionName,
    clearActiveJob,
  };
}
