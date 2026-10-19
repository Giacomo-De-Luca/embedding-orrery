'use client';

import { useState, useCallback, useRef } from 'react';
import { useMutation, useLazyQuery, useApolloClient } from '@apollo/client/react';
import { toast } from 'sonner';
import {
  GET_HF_DATASET_INFO,
  GET_HF_DATASET_PREVIEW,
  GET_LOCAL_FILE_INFO,
  GET_LOCAL_FILE_PREVIEW,
  EMBED_HUGGINGFACE_DATASET,
  EMBED_LOCAL_FILE,
  RE_EMBED_DATASET,
  REMOVE_EMBEDDING_JOB,
  DELETE_COLLECTION,
  UPDATE_COLLECTION_METADATA,
  RENAME_TOPIC_LABEL,
  REGENERATE_TOPIC_LABEL,
  type HFDatasetInfo,
  type HFDatasetPreview,
  type LocalFileInfo,
  type LocalFilePreview,
  type EmbedDatasetInput,
  type EmbedLocalFileInput,
  type ReEmbedDatasetInput,
  type EmbedDatasetResult,
  type UpdateCollectionMetadataResult,
  type TopicConfigInput,
  type ExtractTopicsResult,
  type ReduceTopicsInput,
  type ReduceTopicsResult,
  type GenerateLlmLabelsInput,
  type GenerateLlmLabelsResult,
  type RenameTopicLabelResult,
  type ComputeDocumentActivationsResult,
  COMPUTE_DOCUMENT_ACTIVATIONS,
} from '../graphql/mutations';
import { GET_COLLECTIONS, EXTRACT_TOPICS, REDUCE_TOPICS, GENERATE_LLM_LABELS, GET_COLLECTION_TOPICS } from '../graphql/queries';
import { announceJobOutcome, jobOutcome } from '../utils/jobCancellation';

// ========== Hook Return Types ==========

/** Which embed flow produced the current lastEmbedResult. */
export type EmbedSource = 'hf' | 'local' | 'reembed';

/** Toast the outcome of a (potentially long) embed run — the user may be on
 * another tab when it finishes. Inline result/error cards stay as the
 * inspectable record. */
function notifyEmbedOutcome(result: EmbedDatasetResult | null) {
  announceJobOutcome(result, {
    cancelled: (r) => `Embedding of "${r.collectionName}" cancelled with ${r.totalEmbedded.toLocaleString()} items embedded.`,
    failed: (r) => `Embedding failed: ${r.error}`,
    succeeded: (r) =>
      `Embedded ${r.totalEmbedded.toLocaleString()} items into "${r.collectionName}" in ${r.durationSeconds.toFixed(1)}s`,
  }, toast);
}

/** Embedding runs can take minutes: allow 10 before the request times out. */
const EMBED_CONTEXT = { fetchOptions: { timeout: 600000 } };

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

  // Re-embed from existing dataset
  reEmbedDataset: (input: ReEmbedDatasetInput) => Promise<EmbedDatasetResult | null>;

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

  // Last embed result + which flow produced it (tabs stay mounted, so each
  // tab only renders the result belonging to its own flow)
  lastEmbedResult: EmbedDatasetResult | null;
  lastEmbedSource: EmbedSource | null;

  // Topic extraction
  extractTopics: (collectionName: string, config?: TopicConfigInput) => Promise<ExtractTopicsResult | null>;
  topicsLoading: boolean;
  lastTopicsResult: ExtractTopicsResult | null;

  // Topic reduction
  reduceTopics: (input: ReduceTopicsInput) => Promise<ReduceTopicsResult | null>;
  reduceTopicsLoading: boolean;
  lastReduceResult: ReduceTopicsResult | null;

  // LLM label generation
  generateLlmLabels: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>;
  llmLabelsLoading: boolean;
  lastLlmLabelsResult: GenerateLlmLabelsResult | null;

  // Topic label renaming
  renameTopicLabel: (collectionName: string, topicId: number, newLabel: string, isSubtopic?: boolean) => Promise<RenameTopicLabelResult | null>;
  regenerateTopicLabel: (collectionName: string, topicId: number, llmConfig?: string) => Promise<RenameTopicLabelResult | null>;

  // Load previously-extracted topics
  fetchCollectionTopics: (collectionName: string) => Promise<ExtractTopicsResult | null>;

  // Document activations (batch SAE inference)
  computeDocumentActivations: (collectionName: string) => Promise<ComputeDocumentActivationsResult | null>;
  docActivationsLoading: boolean;
  lastDocActivationsResult: ComputeDocumentActivationsResult | null;
  /** Collection whose activations compute is in flight (drives the page-global progress modal) */
  activeDocActivationsCollection: string | null;

  // Job management
  removeEmbeddingJob: (collectionName: string) => Promise<boolean>;

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
  const [lastEmbedSource, setLastEmbedSource] = useState<EmbedSource | null>(null);

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

  const [reEmbedMutation, { loading: reEmbedLoading }] = useMutation<
    { reEmbedDataset: EmbedDatasetResult }
  >(RE_EMBED_DATASET);

  const [removeJobMutation] = useMutation<
    { removeEmbeddingJob: boolean }
  >(REMOVE_EMBEDDING_JOB);

  const [deleteMutation] = useMutation<{ deleteCollection: boolean }>(DELETE_COLLECTION);

  const [updateMetadataMutation] = useMutation<
    { updateCollectionMetadata: UpdateCollectionMetadataResult }
  >(UPDATE_COLLECTION_METADATA);

  const [extractTopicsMutation, { loading: topicsLoading }] = useMutation<
    { extractTopics: ExtractTopicsResult }
  >(EXTRACT_TOPICS);

  const [lastTopicsResult, setLastTopicsResult] = useState<ExtractTopicsResult | null>(null);

  const [reduceTopicsMutation, { loading: reduceTopicsLoading }] = useMutation<
    { reduceTopics: ReduceTopicsResult }
  >(REDUCE_TOPICS);

  const [lastReduceResult, setLastReduceResult] = useState<ReduceTopicsResult | null>(null);

  const [generateLlmLabelsMutation, { loading: llmLabelsLoading }] = useMutation<
    { generateLlmLabels: GenerateLlmLabelsResult }
  >(GENERATE_LLM_LABELS);

  const [lastLlmLabelsResult, setLastLlmLabelsResult] = useState<GenerateLlmLabelsResult | null>(null);

  const [renameTopicLabelMutation] = useMutation<
    { renameTopicLabel: RenameTopicLabelResult }
  >(RENAME_TOPIC_LABEL);

  const [regenerateTopicLabelMutation] = useMutation<
    { regenerateTopicLabel: RenameTopicLabelResult }
  >(REGENERATE_TOPIC_LABEL);

  const [computeDocActMutation, { loading: docActivationsLoading }] = useMutation<
    { computeDocumentActivations: ComputeDocumentActivationsResult }
  >(COMPUTE_DOCUMENT_ACTIVATIONS);

  const [lastDocActivationsResult, setLastDocActivationsResult] = useState<ComputeDocumentActivationsResult | null>(null);
  const [activeDocActivationsCollection, setActiveDocActivationsCollection] = useState<string | null>(null);

  const [getCollectionTopics] = useLazyQuery<
    { collectionTopics: ExtractTopicsResult | null }
  >(GET_COLLECTION_TOPICS, { fetchPolicy: 'network-only' });

  // ========== Embedding runs (HF / local file / re-embed share one flow) ==========

  /**
   * Run one embed mutation: tracks the active job (drives the progress
   * modal), records the result for the tab that started it, surfaces a failure
   * in the error state (a cancel is not one) and toasts the outcome.
   */
  const runEmbed = useCallback(async (
    source: EmbedSource,
    collectionName: string,
    run: () => Promise<{ result: EmbedDatasetResult | null; error?: { message: string } | null }>,
    failureMessage: string,
  ): Promise<EmbedDatasetResult | null> => {
    setError(null);
    setLastEmbedResult(null);
    setLastEmbedSource(source);
    setActiveJobCollectionName(collectionName);
    try {
      const { result, error } = await run();
      if (error) {
        setError(error.message);
        return null;
      }
      if (jobOutcome(result) === 'failed') {
        setError(result!.error!);
      }
      setLastEmbedResult(result);
      notifyEmbedOutcome(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : failureMessage;
      setError(message);
      toast.error(message);
      return null;
    } finally {
      setActiveJobCollectionName(null);
    }
  }, []);

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

  const embedHFDataset = useCallback((input: EmbedDatasetInput) =>
    runEmbed('hf', input.collectionName, async () => {
      const { data, error } = await embedHFMutation({ variables: { input }, context: EMBED_CONTEXT });
      return { result: data?.embedHuggingfaceDataset ?? null, error };
    }, 'Failed to embed dataset'),
  [runEmbed, embedHFMutation]);

  // ========== Local File Operations ==========

  const fetchLocalFileInfo = useCallback(async (filePath: string): Promise<LocalFileInfo | null> => {
    setInfoLoading(true);
    setError(null);
    setLocalFileInfo(null);

    try {
      const { data, error: queryError } = await getLocalInfo({
        variables: { filePath }
      });

      if (queryError) {
        setError(queryError.message);
        return null;
      }

      const info = data?.localFileInfo || null;

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

  const embedLocalFile = useCallback((input: EmbedLocalFileInput) =>
    runEmbed('local', input.collectionName, async () => {
      const { data, error } = await embedLocalMutation({ variables: { input }, context: EMBED_CONTEXT });
      return { result: data?.embedLocalFile ?? null, error };
    }, 'Failed to embed file'),
  [runEmbed, embedLocalMutation]);

  // ========== Re-embed Existing Dataset ==========

  const reEmbedDataset = useCallback((input: ReEmbedDatasetInput) =>
    runEmbed('reembed', input.collectionName, async () => {
      const { data, error } = await reEmbedMutation({ variables: { input }, context: EMBED_CONTEXT });
      return { result: data?.reEmbedDataset ?? null, error };
    }, 'Failed to re-embed dataset'),
  [runEmbed, reEmbedMutation]);

  // ========== Topic Extraction ==========

  const extractTopics = useCallback(async (
    collectionName: string,
    config?: TopicConfigInput
  ): Promise<ExtractTopicsResult | null> => {
    setError(null);
    setLastTopicsResult(null);

    try {
      const { data, error } = await extractTopicsMutation({
        variables: { collectionName, config: config || null },
        context: {
          fetchOptions: {
            timeout: 600000 // 10 minutes — topic extraction with LLM can be slow
          }
        }
      });

      if (error) {
        setError(error.message);
        return null;
      }

      const result = data?.extractTopics || null;

      const outcome = announceJobOutcome(result, {
        cancelled: () => `Topic extraction for "${collectionName}" cancelled. The existing topics are unchanged.`,
        failed: (r) => `Topic extraction failed: ${r.error}`,
        succeeded: (r) => `Extracted ${r.numTopics} topics for "${r.collectionName}" in ${r.durationSeconds.toFixed(1)}s`,
      }, toast);
      // A cancel stored nothing: keep showing the collection's current topics
      if (outcome === 'cancelled') return result;
      if (outcome === 'failed') setError(result!.error!);

      setLastTopicsResult(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to extract topics';
      setError(message);
      toast.error(message);
      return null;
    }
  }, [extractTopicsMutation]);

  // ========== Topic Reduction ==========

  const reduceTopics = useCallback(async (
    input: ReduceTopicsInput
  ): Promise<ReduceTopicsResult | null> => {
    setError(null);
    setLastReduceResult(null);

    try {
      const { data, error } = await reduceTopicsMutation({
        variables: { input },
        context: {
          fetchOptions: {
            timeout: 600000 // 10 minutes
          }
        }
      });

      if (error) {
        setError(error.message);
        return null;
      }

      const result = data?.reduceTopics || null;

      const outcome = announceJobOutcome(result, {
        cancelled: () => `Topic reduction for "${input.collectionName}" cancelled. The existing topics are unchanged.`,
        failed: (r) => `Topic reduction failed: ${r.error}`,
        succeeded: (r) => `Reduced ${r.numTopicsBefore} → ${r.numTopicsAfter} topics for "${r.collectionName}"`,
      }, toast);
      // A cancel stored nothing: the current topics stay active
      if (outcome === 'cancelled') return result;
      if (outcome === 'failed') setError(result!.error!);

      setLastReduceResult(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to reduce topics';
      setError(message);
      toast.error(message);
      return null;
    }
  }, [reduceTopicsMutation]);

  // ========== LLM Label Generation ==========

  const generateLlmLabels = useCallback(async (
    input: GenerateLlmLabelsInput
  ): Promise<GenerateLlmLabelsResult | null> => {
    setError(null);
    setLastLlmLabelsResult(null);

    try {
      const { data, error } = await generateLlmLabelsMutation({
        variables: { input },
        context: {
          fetchOptions: {
            timeout: 600000 // 10 minutes
          }
        }
      });

      if (error) {
        setError(error.message);
        return null;
      }

      const result = data?.generateLlmLabels || null;

      const outcome = announceJobOutcome(result, {
        cancelled: () =>
          `LLM labeling of "${input.collectionName}" cancelled. Labels generated so far are saved; resume it from the Jobs list.`,
        failed: (r) => `LLM labeling failed: ${r.error}`,
        succeeded: (r) => {
          const parts = [`${r.topicsLabeled}/${r.totalTopics} topics`];
          if (r.totalSubtopics > 0) parts.push(`${r.subtopicsLabeled}/${r.totalSubtopics} subtopics`);
          return `Labeled ${parts.join(' and ')} for "${r.collectionName}"`;
        },
      }, toast);
      if (outcome === 'failed') setError(result!.error!);

      setLastLlmLabelsResult(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to generate LLM labels';
      setError(message);
      toast.error(message);
      return null;
    }
  }, [generateLlmLabelsMutation]);

  // ========== Topic Label Renaming ==========

  const lastTopicsResultRef = useRef(lastTopicsResult);
  lastTopicsResultRef.current = lastTopicsResult;

  const renameTopicLabel = useCallback(async (
    collectionName: string,
    topicId: number,
    newLabel: string,
    isSubtopic: boolean = false
  ): Promise<RenameTopicLabelResult | null> => {
    setError(null);

    try {
      const { data, error } = await renameTopicLabelMutation({
        variables: { input: { collectionName, topicId, newLabel, isSubtopic } }
      });

      if (error) {
        setError(error.message);
        return null;
      }

      const result = data?.renameTopicLabel || null;

      if (result?.error) {
        setError(result.error);
        return result;
      }

      // Optimistic update of lastTopicsResult
      if (result && !isSubtopic && lastTopicsResultRef.current) {
        const updatedTopics = lastTopicsResultRef.current.topics.map(t =>
          t.topicId === topicId ? { ...t, label: newLabel } : t
        );
        setLastTopicsResult({ ...lastTopicsResultRef.current, topics: updatedTopics });
      }

      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to rename topic label';
      setError(message);
      return null;
    }
  }, [renameTopicLabelMutation]);

  const regenerateTopicLabel = useCallback(async (
    collectionName: string,
    topicId: number,
    llmConfig?: string
  ): Promise<RenameTopicLabelResult | null> => {
    setError(null);

    try {
      const { data, error } = await regenerateTopicLabelMutation({
        variables: { input: { collectionName, topicId, newLabel: llmConfig || "" } }
      });

      if (error) {
        setError(error.message);
        return null;
      }

      const result = data?.regenerateTopicLabel || null;

      if (result?.error) {
        setError(result.error);
        return result;
      }

      // Optimistic update
      if (result?.newLabel && lastTopicsResultRef.current) {
        const updatedTopics = lastTopicsResultRef.current.topics.map(t =>
          t.topicId === topicId ? { ...t, label: result.newLabel } : t
        );
        setLastTopicsResult({ ...lastTopicsResultRef.current, topics: updatedTopics });
      }

      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to regenerate topic label';
      setError(message);
      return null;
    }
  }, [regenerateTopicLabelMutation]);

  // ========== Fetch Previously-Extracted Topics ==========

  const fetchCollectionTopics = useCallback(async (
    collectionName: string
  ): Promise<ExtractTopicsResult | null> => {
    try {
      const { data, error: queryError } = await getCollectionTopics({
        variables: { collectionName }
      });

      if (queryError) {
        return null;
      }

      const result = data?.collectionTopics || null;
      if (result) {
        setLastTopicsResult(result);
      }
      return result;
    } catch {
      return null;
    }
  }, [getCollectionTopics]);

  // ========== Job Management ==========

  const removeEmbeddingJob = useCallback(async (collectionName: string): Promise<boolean> => {
    try {
      const { data } = await removeJobMutation({ variables: { collectionName } });
      return data?.removeEmbeddingJob ?? false;
    } catch {
      return false;
    }
  }, [removeJobMutation]);

  // ========== Collection Operations ==========

  const deleteCollection = useCallback(async (collectionName: string): Promise<boolean> => {
    setError(null);

    try {
      const { data, error } = await deleteMutation({
        variables: { collectionName }
      });

      if (error) {
        setError(error.message);
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
      const { data, error } = await updateMetadataMutation({
        variables: { collectionName, metadata }
      });

      if (error) {
        setError(error.message);
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

  // ========== Document Activations (batch SAE inference) ==========

  const computeDocumentActivations = useCallback(async (
    collectionName: string,
  ): Promise<ComputeDocumentActivationsResult | null> => {
    setError(null);
    setLastDocActivationsResult(null);
    setActiveDocActivationsCollection(collectionName);

    try {
      const { data, error } = await computeDocActMutation({
        variables: { input: { collectionName } },
      });

      if (error) {
        setError(error.message);
        return null;
      }

      const result = data?.computeDocumentActivations || null;
      const outcome = announceJobOutcome(result, {
        cancelled: (r) =>
          `SAE activations for "${collectionName}" cancelled after ${r.itemsProcessed.toLocaleString()} / ${r.totalItems.toLocaleString()} documents. Run it again to continue where it stopped.`,
        failed: (r) => `SAE activations failed: ${r.error}`,
        succeeded: (r) => `Computed SAE activations for ${r.itemsProcessed.toLocaleString()} documents in "${collectionName}"`,
      }, toast);
      if (outcome === 'failed') setError(result!.error!);

      setLastDocActivationsResult(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to compute document activations';
      setError(message);
      return null;
    } finally {
      setActiveDocActivationsCollection(null);
    }
  }, [computeDocActMutation]);

  return {
    // HuggingFace operations
    fetchHFDatasetInfo,
    fetchHFDatasetPreview,
    embedHFDataset,

    // Local file operations
    fetchLocalFileInfo,
    fetchLocalFilePreview,
    embedLocalFile,

    // Re-embed from existing dataset
    reEmbedDataset,

    // Topic extraction
    extractTopics,
    topicsLoading,
    lastTopicsResult,

    // Topic reduction
    reduceTopics,
    reduceTopicsLoading,
    lastReduceResult,

    // LLM label generation
    generateLlmLabels,
    llmLabelsLoading,
    lastLlmLabelsResult,

    // Topic label renaming
    renameTopicLabel,
    regenerateTopicLabel,

    // Load previously-extracted topics
    fetchCollectionTopics,

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
    embedLoading: embedHFLoading || embedLocalLoading || reEmbedLoading,

    // Errors
    error,
    clearError,

    // Last result
    lastEmbedResult,
    lastEmbedSource,

    // Document activations
    computeDocumentActivations,
    docActivationsLoading,
    lastDocActivationsResult,
    activeDocActivationsCollection,

    // Job management
    removeEmbeddingJob,

    // Active job tracking
    activeJobCollectionName,
    clearActiveJob,
  };
}
